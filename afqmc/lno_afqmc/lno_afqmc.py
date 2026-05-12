import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from jax import random
from pyscf.lno import lnoccsd
from pyscf.lno import ulnoccsd
from collections.abc import Iterable

from afqmc import config
from afqmc.lno_afqmc import prep
from afqmc.lno_afqmc import mod_lnoccsd

from functools import partial
import time, gc, pickle

print = partial(print, flush=True)


def run_lnoafqmc(options, option_file='options.bin'):
    jax.config.update("jax_enable_x64", True)
    
    with open(option_file, 'wb') as f:
        pickle.dump(options, f)

    if options["use_gpu"]:
        print(f'running AFQMC on GPU')
        config.afqmc_config = {"use_gpu": True}
        config.setup_jax()
        gpu_flag = "--use_gpu"
    else:
        print(f'running AFQMC on CPU')
        gpu_flag = ""
    if 'pt2' in options['trial']:
        script='ccsd_pt2/run_afqmc.py'

    else:
        raise NotImplementedError("Only support CCSD_pt and CCSD_pt2 trial.")
    
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/{script}"
    print(f'AFQMC script: {script}')
    
    os.system(
        # f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1;"
        f" python {script} {gpu_flag} |tee afqmc.out"
    )

def run_afqmc(mf,
              lo_coeff = None, 
              lo_coeff_file = 'lo_coeff.npz',
              frag_lolist = None,
              nfrozen = 0,
              thresh = 1e-6,
              qmc_options = {}, 
              chol_cut = 1e-5, 
              target_sto_error = 1e-3, 
              run_frg_list = None, 
              atom_group = None,
              ):
    
    if lo_coeff is None:
        try:
            lo_coeff = np.load(lo_coeff_file)["lo_coeff"]
        except:
            raise ValueError(
                f"lo_coeff was not provided and could not be loaded from '{lo_coeff_file}'")

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

    nfrag = len(frag_lolist)
    if run_frg_list is None:
        run_frg_list = range(nfrag)
    
    frag_lolist = [frag_lolist[i] for i in run_frg_list]

    lno_pct_occ = [None, None]
    lno_norb = [[None,None]] * nfrag

    seeds = random.randint(random.PRNGKey(qmc_options["seed"]),
                           shape=(nfrag,), 
                           minval=0, 
                           maxval=100*nfrag
                           )
    
    qmc_options["max_error"] = target_sto_error / np.sqrt(nfrag)
    trial_base = qmc_options.get("trial", "")

    las_center = [None]*nfrag
    las_size = np.zeros(nfrag, dtype='int32')
    lno_emp2 = np.zeros(nfrag, dtype='float64')
    lno_ecc  = np.zeros(nfrag, dtype='float64')
    lno_eqmc = np.zeros(nfrag, dtype='float64')
    lno_eqmc_err  = np.zeros(nfrag, dtype='float64')
    ccsd_time = np.zeros(nfrag, dtype='float64')
    qmc_time = np.zeros(nfrag, dtype='float64')

    # Loop over fragment
    for ifrag, loidx in enumerate(frag_lolist):
        print("\n")
        width = 80
        msg = f" {spin_type} LNO-FRAGMENT {run_frg_list[ifrag]+1}/{nfrag} "
        print(msg.center(width, '='))
        if atom_group is not None:
            atom_msg = f"{atom_group[ifrag]}"
            print(f"Center Atom {atom_msg}")

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
        # lno_coeff still connected to canonical mo_coeff unitarily

        if spin_type == "unrestricted":
            if uocc_loc[0].size > 0 and uocc_loc[1].size == 0:
                lno_elec_type = 'alpha'
            elif uocc_loc[0].size == 0 and uocc_loc[1].size > 0:
                lno_elec_type = 'beta'
            else:
                lno_elec_type = 'mixed'
            print(f'LNO-Frgament Spin Type = {lno_elec_type}')
            ao_message_a, ao_max_a = prep.ao_comp(mf, orbloc[0])
            ao_message_b, ao_max_b = prep.ao_comp(mf, orbloc[1])
            ao_message = ao_message_a + "\n" + ao_message_b
            ao_max = ao_max_a + ao_max_b
        else:
            ao_message, ao_max = prep.ao_comp(mf, orbloc)

        mo_occ = mlno.mo_occ

        if spin_type == "unrestricted":
            lno_frozen, maskact = ulnoccsd.get_maskact(lno_frozen, [mo_occ[0].size, mo_occ[1].size])
            occidxa = mo_occ[0] > 1e-10
            occidxb = mo_occ[1] > 1e-10
            moidxa, moidxb = maskact
            nactocc_a = int(np.sum(moidxa & occidxa))
            nactvir_a = int(np.sum(moidxa & ~occidxa))
            nactocc_b = int(np.sum(moidxb & occidxb))
            nactvir_b = int(np.sum(moidxb & ~occidxb))
            nactocc = nactocc_a + nactocc_b
            nactvir = nactvir_a + nactvir_b
            print(f'LAS alpha: {nactocc_a} occupied, {nactvir_a} virtual')
            print(f'LAS beta:  {nactocc_b} occupied, {nactvir_b} virtual')
        else:
            lno_frozen, maskact = lnoccsd.get_maskact(lno_frozen, mo_occ.size)
            nactocc, nactvir = prep.las_size(mf, lno_frozen)
            print(f'LAS occupied orbitals: {nactocc}')
            print(f'LAS virtual orbitals: {nactvir}')

        if spin_type == "unrestricted":
            mcc = ulnoccsd.UCCSD(mf, mo_coeff=lno_coeff, frozen=lno_frozen).set(verbose=1)
        else:
            mcc = lnoccsd.CCSD(mf, mo_coeff=lno_coeff, frozen=lno_frozen).set(verbose=1)
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

        print(f"CCSD time: {lnocc_time:.6f} s")
        print(f'LNO-MP2 Orbital Energy: {eorb_mp2:.8f}')
        print(f'LNO-CCSD Orbital Energy: {eorb_cc:.8f}')

        if atom_group:
            las_center[ifrag] = atom_msg
        else:
            las_center[ifrag] = ao_max
        las_size[ifrag] = nactocc + nactvir
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
        outfile = f'fragment.out{run_frg_list[ifrag]+1}'
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

        header = f' Fragment{run_frg_list[ifrag]+1} Results '
        width = 80  # pick a consistent total width
        with open(outfile, 'a') as f:
            f.write('\n')
            f.write(f'{header:=^{width}}\n')
            if atom_group:
                f.write("\t Center Atom " + atom_msg + "\n")
            f.write("\t" + ao_message + "\n")
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

    las_max = las_size.max()
    e_mp2 = np.sum(lno_emp2)
    e_ccsd = np.sum(lno_ecc)
    e_afqmc = np.sum(lno_eqmc)
    e_afqmc_err = np.sqrt(np.sum(lno_eqmc_err**2))
    tot_ccsd_time = np.sum(ccsd_time)
    tot_qmc_time = np.sum(qmc_time)

    with open(f'lno_result.out', 'w') as f:
        width = 110
        f.write('=' * width + '\n')
        f.write(f'{"LNO-AFQMC Results":^{width}}\n')
        f.write('=' * width + '\n')

        f.write(f'{"Frag":>4s}  {"LAS Center":>14s}  {"LAS_SIZE":>8s}  '
                f'{"E(MP2)":>10s}  {"E(CCSD)":>10s}  '
                f'{"E(AFQMC)":>10s}  {"Error":>8s}  '
                f'{"t(CCSD)":>8s}  {"t(AFQMC)":>8s}\n')
        f.write('-' * width + '\n')
        
        for n, i in enumerate(run_frg_list):
            f.write(f"{i+1:4d}  {las_center[n]:>14s}  {las_size[n]:8d}  "
                    f"{lno_emp2[n]:10.8f}  {lno_ecc[n]:10.8f}  "
                    f"{lno_eqmc[n]:10.6f}  {lno_eqmc_err[n]:8.6f}  "
                    f"{ccsd_time[n]:8.2f}  {qmc_time[n]:8.2f}\n")
        
        f.write('-' * width + '\n')

        f.write(f'{"Sum":>4s}  {"":>14s}  {"":>8s}  '
                f'{e_mp2:10.8f}  {e_ccsd:10.8f}  '
                f'{e_afqmc:10.6f}  {e_afqmc_err:8.6f}  '
                f'{tot_ccsd_time:8.2f}  {tot_qmc_time:8.2f}\n')
        f.write('=' * width + '\n\n')

        f.write(f'LNO Threshold:          ({lno_thresh[0]:.2e}, {lno_thresh[1]:.2e})\n')
        f.write(f'MAX. Orbitals:          {las_max}\n')

    return None