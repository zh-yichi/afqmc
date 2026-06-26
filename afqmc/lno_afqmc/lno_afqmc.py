import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from jax import random

from pyscf import lib
from pyscf.lno import lnoccsd
from pyscf.lno import ulnoccsd
from collections.abc import Iterable

# from afqmc import config
from afqmc.lno_afqmc import prep, tools
from afqmc.lno_afqmc import mod_lnoccsd

from functools import partial
import time, gc, pickle

print = partial(print, flush=True)


def run_lnoafqmc(options, option_file='options.bin'):
    
    with open(option_file, 'wb') as f:
        pickle.dump(options, f)

    if 'pt2' in options['trial']:
        script='script/run_afqmc_pt2.py'
    
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/{script}"
    print(f'AFQMC script: {script}')
    
    os.system(f" python {script} |tee afqmc.out")

def run_afqmc(mf,
              lo_coeff = None, 
              frag_lolist = None,
              nfrozen = 0,
              thresh = 1e-6,
              qmc_options = {}, 
              chol_cut = 1e-5, 
              target_sto_error = 1e-3, 
              run_frag_list = None, 
              atom_group = None,
              plot_las = False,
              ):

    spin_type = prep.kind(lo_coeff)

    if frag_lolist is None:
        if spin_type == "unrestricted":
            raise ValueError("frag_lolist must be provided for unrestricted LNO-AFQMC.")
        print("Fragment list not found. Asign every LO to a fragment.")
        frag_lolist = [[i] for i in range(lo_coeff.shape[1])]

    if spin_type == "unrestricted":
        mlno = ulnoccsd.ULNOCCSD(mf, lo_coeff, frag_lolist, frozen=nfrozen).set(verbose=3)
        mf = mlno._scf
    else:
        mlno = lnoccsd.LNOCCSD(mf, lo_coeff, frag_lolist, frozen=nfrozen).set(verbose=3)
    mlno.lno_thresh = [thresh*10, thresh]
    lno_thresh = mlno.lno_thresh
    lno_type = ['1h','1h']
    eris = mlno.ao2mo()

    nfrag_tot = len(frag_lolist)
    if run_frag_list is None:
        run_frag_list = range(nfrag_tot)
    
    frag_lolist = [frag_lolist[i] for i in run_frag_list]
    nfrag_run = len(frag_lolist)

    lno_pct_occ = [None, None]
    lno_norb = [[None,None]] * nfrag_tot

    seeds = random.randint(random.PRNGKey(qmc_options["seed"]),
                           shape=(nfrag_tot,), 
                           minval=0, 
                           maxval=100*nfrag_tot
                           )
    
    qmc_options["max_error"] = target_sto_error / np.sqrt(nfrag_tot)
    trial_base = qmc_options.get("trial", "")

    las_center = [None]*nfrag_run
    las_size = [None]*nfrag_run
    lno_emp2 = np.zeros(nfrag_run, dtype='float64')
    lno_ecc  = np.zeros(nfrag_run, dtype='float64')
    lno_eqmc = np.zeros(nfrag_run, dtype='float64')
    lno_eqmc_err  = np.zeros(nfrag_run, dtype='float64')
    ccsd_time = np.zeros(nfrag_run, dtype='float64')
    qmc_time = np.zeros(nfrag_run, dtype='float64')

    mol = mf.mol

    # Loop over fragment
    for ifrag, loidx in enumerate(frag_lolist):
        print("\n")
        width = 80
        msg = f" {spin_type} LNO-FRAGMENT {run_frag_list[ifrag]+1}/({nfrag_run},{nfrag_tot}) "
        print(msg.center(width, '='))
        if atom_group is not None:
            loc_ctr = f"{atom_group[run_frag_list[ifrag]]}"
            print(f"Center Atom {loc_ctr}")
        else:
            loc_ctr = None
        
        print(f"PySCF NumPy Threads = {lib.num_threads()}")

        if spin_type == "unrestricted":
            orbloc = [lo_coeff[0][:,loidx[0]], lo_coeff[1][:,loidx[1]]]
            lno_param = [
                [
                    {
                        'thresh': (
                            lno_thresh[i][s] if isinstance(lno_thresh[i], Iterable)
                            else lno_thresh[i]
                        ),
                        'pct_occ': (
                            lno_pct_occ[i][s] if isinstance(lno_pct_occ[i], Iterable)
                            else lno_pct_occ[i]
                        ),
                        'norb': (
                            lno_norb[ifrag][i][s] if isinstance(lno_norb[ifrag][i], Iterable)
                            else lno_norb[ifrag][i]
                        ),
                    } for i in [0, 1]
                ] for s in range(2)
            ]
        else:
            orbloc = lo_coeff[:,loidx]
            lno_param = [{
                'thresh': lno_thresh[i],
                'pct_occ': lno_pct_occ[i],
                'norb': lno_norb[ifrag][i]
                } for i in [0,1]]

        # M = <orbloc|canactocc> (M^dagger M)u = eu
        # u|canactocc> => orbtial in/out the space spanned by |orbloc>
        # uocc_loc = <lno_actocc|orbloc>
        lno_coeff, lno_frozen, uocc_loc, _ = mlno.make_las(eris, orbloc, lno_type, lno_param)

        mo_occ = mlno.mo_occ

        if spin_type == "unrestricted":
            if uocc_loc[0].size > 0 and uocc_loc[1].size == 0:
                lno_elec_type = 'alpha'
            elif uocc_loc[0].size == 0 and uocc_loc[1].size > 0:
                lno_elec_type = 'beta'
            else:
                lno_elec_type = 'mixed'
            print(f'LNO-Frgament Spin Type = {lno_elec_type}')

            if loc_ctr is None:
                ao_max_a = prep.ao_comp(mf, orbloc[0])
                ao_max_b = prep.ao_comp(mf, orbloc[1])
                loc_ctr = ao_max_a + ao_max_b
                print(f"LNO Center {loc_ctr}")

            lno_frozen, maskact = ulnoccsd.get_maskact(lno_frozen, [mo_occ[0].size, mo_occ[1].size])
            occidxa = mo_occ[0] > 1e-10
            occidxb = mo_occ[1] > 1e-10
            moidxa, moidxb = maskact
            nactocc_a = int(np.sum(moidxa & occidxa))
            nactvir_a = int(np.sum(moidxa & ~occidxa))
            nactocc_b = int(np.sum(moidxb & occidxb))
            nactvir_b = int(np.sum(moidxb & ~occidxb))
            nactocc = [nactocc_a, nactocc_b]
            nactvir = [nactvir_a, nactvir_b]
            lno_active_a = np.array([i for i in range(mol.nao) if i not in lno_frozen[0]])
            lno_active_b = np.array([i for i in range(mol.nao) if i not in lno_frozen[1]])
            lno_active = [lno_active_a, lno_active_b]
            lno_tot = [len(lno_active_a), len(lno_active_b)]
            # print(f'LAS alpha: {nactocc_a} occupied, {nactvir_a} virtual')
            print(f'LAS occupied orbitals:  {nactocc}')
            print(f'LAS virtual orbitals:   {nactvir}')
            print(f'LAS total size:         {lno_tot}')
            mcc = ulnoccsd.UCCSD(mf, mo_coeff=lno_coeff, frozen=lno_frozen).set(verbose=1)
        else:
            print(f'LNO-Frgament Spin Type = restricted')
            if loc_ctr is None:
                loc_ctr = prep.ao_comp(mf, orbloc)
                print(f"LNO Center {loc_ctr}")

            lno_frozen, maskact = lnoccsd.get_maskact(lno_frozen, mo_occ.size)
            lno_active = np.array([i for i in range(mol.nao) if i not in lno_frozen])
            nactocc, nactvir = prep.las_size(mf, lno_frozen)
            lno_tot = len(lno_active)
            print(f'LAS occupied orbitals:  {nactocc}')
            print(f'LAS virtual orbitals:   {nactvir}')
            print(f'LAS total size:         {lno_tot}')
            mcc = lnoccsd.CCSD(mf, mo_coeff=lno_coeff, frozen=lno_frozen).set(verbose=1)
        
        if plot_las:
            tools.plot_density(mol, orbloc, lno_coeff, lno_active, spin_type, idx = run_frag_list[ifrag]+1)

        mcc._s1e = mlno._s1e
        mcc._h1e = mlno._h1e
        mcc._vhf = mlno._vhf
        if mlno.kwargs_imp is not None:
            mcc = mcc.set(**mlno.kwargs_imp)
        time0 = time.perf_counter()
        (eorb_mp2, eorb_cc), t1, t2 =\
            mod_lnoccsd.lnoccsd_kernel(mcc, lno_coeff, uocc_loc, mo_occ, maskact)
        time1 = time.perf_counter()
        lnocc_time = time1 - time0

        print(f'LNO-MP2 Orbital Energy:  {eorb_mp2:.8f}')
        print(f'LNO-CCSD Orbital Energy: {eorb_cc:.8f}')
        print(f"LNO-CCSD time:           {lnocc_time:.2f} s")

        las_center[ifrag] = loc_ctr
        las_size[ifrag] = lno_tot
        lno_emp2[ifrag] = eorb_mp2
        lno_ecc[ifrag] = eorb_cc
        ccsd_time[ifrag] = lnocc_time

        # project onto center lo space
        # <lno_actocc|orbloc> <orbloc|lno_actocc>
        if spin_type == "unrestricted":
            prjlo = [uocc_loc[0] @ uocc_loc[0].T.conj(),
                     uocc_loc[1] @ uocc_loc[1].T.conj()]
            qmc_options["trial"] = trial_base
            if 'ad' not in trial_base:
                if lno_elec_type == 'alpha':
                    qmc_options["trial"] += '_alpha'
                elif lno_elec_type == 'beta':
                    qmc_options["trial"] += '_beta'
        else:
            prjlo = uocc_loc @ uocc_loc.T.conj()

        qmc_options["seed"] = seeds[ifrag]
        prep.prep_afqmc_integral(
            mf,
            lno_coeff,
            t1,
            t2,
            lno_frozen,
            prjlo,
            qmc_options,
            chol_cut=chol_cut
            )
        
        run_lnoafqmc(qmc_options)
        outfile = f'fragment.out{run_frag_list[ifrag]+1}'
        os.system(f'mv afqmc.out {outfile}')
        with open(outfile, "r") as f:
            for line in f:
                if "Blocked AFQMC/pt2CCSD Orbital Energy" in line:
                    eorb_afqmc = float(line.split()[-3])
                    eorb_afqmc_err = float(line.split()[-1])
                if "total run time" in line:
                    lnoqmc_time = float(line.split()[-1])
        lno_eqmc[ifrag] = eorb_afqmc
        lno_eqmc_err[ifrag] = eorb_afqmc_err
        qmc_time[ifrag] = lnoqmc_time

        header = f' Fragment{run_frag_list[ifrag]+1} Results '
        width = 80  # pick a consistent total width
        with open(outfile, 'a') as f:
            f.write('\n')
            f.write(f'{header:=^{width}}\n')
            f.write("\t LNO Center " + loc_ctr + "\n")
            f.write('-' * width + '\n')
            f.write(f'\t LNO-Active Space electrons: {nactocc} | orbitals: {nactocc+nactvir} \n')
            f.write(f'\t LNO-MP2 Orbital Energy:   {eorb_mp2:.8f} \n')
            f.write(f'\t LNO-CCSD Orbital Energy:  {eorb_cc:.8f} \n')
            f.write(f'\t LNO-AFQMC Orbital Energy: {eorb_afqmc:.6f} +/- {eorb_afqmc_err:.6f} \n')
            f.write(f'\t LNO-CCSD Time:  {lnocc_time:.2f} \n')
            f.write(f'\t LNO-AFQMC Time: {lnoqmc_time:.2f} \n')
            f.write('=' * width + '\n')
        jax.clear_caches()
        gc.collect()

    las_size = np.array(las_size, dtype=np.int32)
    las_max = las_size.max()
    # convert to list of string for print
    las_size = list(map(lambda row: f"{row}", las_size))
    e_mp2 = np.sum(lno_emp2)
    e_ccsd = np.sum(lno_ecc)
    e_afqmc = np.sum(lno_eqmc)
    e_afqmc_err = np.sqrt(np.sum(lno_eqmc_err**2))
    tot_ccsd_time = np.sum(ccsd_time)
    tot_qmc_time = np.sum(qmc_time)

    with open(f'lno_result.out', 'w') as f:
        width = 100
        f.write('=' * width + '\n')
        f.write(f'{"LNO-AFQMC Results":^{width}}\n')
        f.write('=' * width + '\n')

        f.write(f'{"Frag":>4s}  {"LAS Center":>14s}  {"LAS_SIZE":>8s}  '
                f'{"E(MP2)":>10s}  {"E(CCSD)":>10s}  '
                f'{"E(AFQMC)":>10s}  {"Error":>8s}  '
                f'{"t(CCSD)":>8s}  {"t(AFQMC)":>8s}\n')
        f.write('-' * width + '\n')
        
        for n, i in enumerate(run_frag_list):
            f.write(f"{i+1:4d}  {las_center[n]:>14s}  {las_size[n]:8s}  "
                    f"{lno_emp2[n]:10.8f}  {lno_ecc[n]:10.8f}  "
                    f"{lno_eqmc[n]:10.6f}  {lno_eqmc_err[n]:8.6f}  "
                    f"{ccsd_time[n]:8.2f}  {qmc_time[n]:8.2f}\n")
        
        f.write('-' * width + '\n')

        # f.write(f'{"Summary"} {"LNO Thresh"} {"Max LAS"} '
        #         f'{"E[LNO-MP2]"} {"E[LNO-CCSD]"} '
        #         f'{"E[LNO-AFQMC]"} {"Err[LNO-AFQMC]"}'
        #         f'{"CCSD Time"} {"AFQMC Time"}')
        
        # f.write(f'{lno_thresh} {las_max}'
        #         f'{e_mp2:10.8f}  {e_ccsd:10.8f}  '
        #         f'{e_afqmc:10.6f}  {e_afqmc_err:8.6f}  '
        #         f'{tot_ccsd_time:8.2f}  {tot_qmc_time:8.2f}\n')
        f.write(f'{"Summarize Fragments":^{width}}\n')
        f.write('-' * width + '\n')
        # f.write(f'{"Summary"} \n')
        # lno_thresh_str = f"{lno_thresh}"
        lno_thresh_str = "[" + ", ".join(f"{x:.2e}" for x in lno_thresh) + "]"
        f.write(f'{"LNO-Thresh":<20} {"Max LAS":>8} '
                f'{"E[MP2]":>12} {"E[CCSD]":>12} '
                f'{"E[AFQMC]":>10} {"Err[AFQMC]":>10} '
                f'{"CCSD-Time":>10} {"AFQMC-Time":>10}\n')

        f.write(f'{lno_thresh_str:<20} {las_max:>8} '
                f'{e_mp2:>12.8f} {e_ccsd:>12.8f} '
                f'{e_afqmc:>10.6f} {e_afqmc_err:>10.6f} '
                f'{tot_ccsd_time:>10.2f} {tot_qmc_time:>10.2f}\n')
        
        f.write('=' * width + '\n\n')

        # f.write(f'LNO Threshold:          ({lno_thresh[0]:.2e}, {lno_thresh[1]:.2e})\n')
        # f.write(f'MAX. Orbitals:          {las_max}\n')

    return None