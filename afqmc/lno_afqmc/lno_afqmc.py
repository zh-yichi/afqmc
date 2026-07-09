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
from afqmc.lno_afqmc import prep, tools, integral
from afqmc.lno_afqmc import mod_lnoccsd

from pyscf import scf

from functools import partial
import time, gc, pickle

print = partial(print, flush=True)

def get_lnoccsd(mf, lo_coeff, frag_lolist, nfrozen, thresh, spin_type, verbose=3):
    
    if spin_type == "restricted":
        mlno = lnoccsd.LNOCCSD(mf, lo_coeff, frag_lolist, frozen=nfrozen).set(verbose=verbose)
    elif spin_type == "unrestricted":
        mlno = ulnoccsd.ULNOCCSD(mf, lo_coeff, frag_lolist, frozen=nfrozen).set(verbose=verbose)
    
    if isinstance(thresh, float):
        mlno.lno_thresh = [thresh*10, thresh]
    elif isinstance(thresh, (list, tuple)):
        assert len(thresh) == 2
        mlno.lno_thresh = [thresh[0], thresh[1]]

    return mlno

def get_lnoparam(lo_coeff, lno_thresh, lno_pct_occ, lno_norb, loidx, ifrag, spin_type):
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
        
    return orbloc, lno_param

def get_las(mlno, orbloc, uocc_loc, lno_frozen, spin_type, loc_ctr):
    mf = mlno._scf
    mol = mf.mol
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
        print(f'LAS occupied orbitals:  {nactocc}')
        print(f'LAS virtual orbitals:   {nactvir}')
        print(f'LAS total size:         {lno_tot}')
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
    
    return  maskact, lno_active, nactocc, nactvir, lno_tot, lno_elec_type

def lnoccsd_kernel(mlno, lno_coeff, lno_frozen, uocc_loc, maskact, verbose=3):
    mf = mlno._scf
    if isinstance(mf, scf.rhf.RHF):
        mcc = lnoccsd.CCSD(mf, mo_coeff=lno_coeff, frozen=lno_frozen).set(verbose=verbose)
    elif isinstance(mf, scf.uhf.UHF):
        mcc = ulnoccsd.UCCSD(mf, mo_coeff=lno_coeff, frozen=lno_frozen).set(verbose=verbose)
    mcc.conv_tol = 1e-6
    mcc.conv_tol_normt = 1e-5
    (eorb_mp2, eorb_cc), t1, t2 =\
            mod_lnoccsd.lnoccsd_kernel(mcc, lno_coeff, uocc_loc, mlno.mo_occ, maskact)
    return (eorb_mp2, eorb_cc), t1, t2

def run_lnoafqmc(options, option_file='options.bin'):
    
    with open(option_file, 'wb') as f:
        pickle.dump(options, f)

    if 'pt2' in options['trial']:
        script='script/run_lno_afqmc_pt2ccsd.py'
    
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/{script}"
    print(f'AFQMC script: {script}')
    
    os.system(f" python {script} |tee afqmc.out")

def lnoafqmc_kernel(mlno, lno_coeff, uocc_loc, lno_frozen, t1, t2, 
                    chol_cut, frag_idx, seeds, qmc_options):
    
    # if spin_type == "unrestricted":
    #     prjlo = [uocc_loc[0] @ uocc_loc[0].T.conj(),
    #                 uocc_loc[1] @ uocc_loc[1].T.conj()]
    #     qmc_options["trial"] = trial_base
    #     if 'ad' not in trial_base:
    #         if lno_elec_type == 'alpha':
    #             qmc_options["trial"] += '_alpha'
    #         elif lno_elec_type == 'beta':
    #             qmc_options["trial"] += '_beta'
    # else:
    #     prjlo = uocc_loc @ uocc_loc.T.conj()
    # prep.prep_afqmc_integral(
    #     mf,
    #     lno_coeff,
    #     t1,
    #     t2,
    #     lno_frozen,
    #     prjlo,
    #     qmc_options,
    #     chol_cut=chol_cut
    #     )

    mf = mlno._scf
    
    integral.prep_lno_integral(mf, lno_coeff, lno_frozen, uocc_loc, t1, t2, chol_cut)
    
    qmc_options["seed"] = seeds[frag_idx]

    run_lnoafqmc(qmc_options)

    outfile = f'fragment.out{frag_idx+1}'
    os.system(f'mv afqmc.out {outfile}')
    with open(outfile, "r") as f:
        for line in f:
            if "Final AFQMC/pt2CCSD Orbital Energy" in line:
                eorb_afqmc = float(line.split()[-3])
                eorb_afqmc_err = float(line.split()[-1])
    
    return eorb_afqmc, eorb_afqmc_err

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


    mlno = get_lnoccsd(mf, lo_coeff, frag_lolist, nfrozen, thresh, spin_type)
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
    for ifrag, frag_idx in enumerate(run_frag_list):
        
        loidx = frag_lolist[ifrag]

        print("\n")
        width = 80
        msg = f" {spin_type} LNO-FRAGMENT {frag_idx+1}/({nfrag_run},{nfrag_tot}) "
        print(msg.center(width, '='))
        if atom_group is not None:
            loc_ctr = f"{atom_group[frag_idx]}"
            print(f"Center Atom {loc_ctr}")
        else:
            loc_ctr = None
        
        print(f"PySCF NumPy Threads = {lib.num_threads()}")

        orbloc, lno_param = get_lnoparam(lo_coeff, lno_thresh, lno_pct_occ, lno_norb, loidx, ifrag, spin_type)
        lno_coeff, lno_frozen, uocc_loc, _ = mlno.make_las(eris, orbloc, lno_type, lno_param)

        maskact, lno_active, nactocc, nactvir, lno_tot, lno_elec_type = \
            get_las(mlno, orbloc, uocc_loc, lno_frozen, spin_type, loc_ctr)
                
        if plot_las:
            tools.plot_density(mol, orbloc, lno_coeff, lno_active, spin_type, idx=frag_idx+1)

        
        time0 = time.perf_counter()
        (eorb_mp2, eorb_cc), t1, t2 = \
            lnoccsd_kernel(mlno, lno_coeff, lno_frozen, uocc_loc, maskact, verbose=4)
        lnocc_time = time.perf_counter() - time0

        print(f'LNO-MP2 Orbital Energy:  {eorb_mp2:.8f}')
        print(f'LNO-CCSD Orbital Energy: {eorb_cc:.8f}')
        print(f"LNO-CCSD time:           {lnocc_time:.2f} s")

        outfile = f'fragment.out{frag_idx+1}'

        time0 = time.perf_counter()
        eorb_afqmc, eorb_afqmc_err \
            = lnoafqmc_kernel(
                mlno, lno_coeff, uocc_loc, lno_frozen, t1, t2, 
                chol_cut, frag_idx, seeds, qmc_options)
        lnoqmc_time = time.perf_counter() - time0
        
        las_center[ifrag] = loc_ctr
        las_size[ifrag] = lno_tot
        lno_emp2[ifrag] = eorb_mp2
        lno_ecc[ifrag] = eorb_cc
        ccsd_time[ifrag] = lnocc_time
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
            f.write(f'\t LNO-AFQMC Orbital Energy: {eorb_afqmc:.5f} +/- {eorb_afqmc_err:.5f} \n')
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
                    f"{lno_eqmc[n]:10.5f}  {lno_eqmc_err[n]:8.5f}  "
                    f"{ccsd_time[n]:8.2f}  {qmc_time[n]:8.2f}\n")
        
        f.write('-' * width + '\n')

        f.write(f'{"Summarize Fragments":^{width}}\n')
        f.write('-' * width + '\n')

        lno_thresh_str = "[" + ", ".join(f"{x:.2e}" for x in lno_thresh) + "]"
        f.write(f'{"LNO-Thresh":<20} {"Max LAS":>8} '
                f'{"E[MP2]":>12} {"E[CCSD]":>12} '
                f'{"E[AFQMC]":>10} {"Err[AFQMC]":>10} '
                f'{"CCSD-Time":>10} {"AFQMC-Time":>10}\n')

        f.write(f'{lno_thresh_str:<20} {las_max:>8} '
                f'{e_mp2:>12.8f} {e_ccsd:>12.8f} '
                f'{e_afqmc:>10.5f} {e_afqmc_err:>10.5f} '
                f'{tot_ccsd_time:>10.2f} {tot_qmc_time:>10.2f}\n')
        
        f.write('=' * width + '\n\n')

    return None