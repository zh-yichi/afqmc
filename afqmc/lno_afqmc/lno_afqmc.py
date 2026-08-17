import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from jax import random

from pyscf import lib, scf
from pyscf.data import elements
from pyscf.lno import lnoccsd
from pyscf.lno import ulnoccsd
from collections.abc import Iterable

from afqmc.lno_afqmc import prep, tools, integral
from afqmc.lno_afqmc import mod_lnoccsd


from functools import partial
import time, gc, pickle

print = partial(print, flush=True)

def get_lnoccsd(mf, lo_coeff, frag_list, nfrozen, thresh, verbose=3):
    if isinstance(mf, scf.rhf.RHF):
        mlno = lnoccsd.LNOCCSD(mf, lo_coeff, frag_list, frozen=nfrozen).set(verbose=verbose)
    elif isinstance(mf, scf.uhf.UHF):
        mlno = ulnoccsd.ULNOCCSD(mf, lo_coeff, frag_list, frozen=nfrozen).set(verbose=verbose)
    else: 
        raise NotImplementedError('LNO Only Support Restricted and Unrestricted MF!')
    
    if isinstance(thresh, float):
        mlno.lno_thresh = [thresh*10, thresh]
    elif isinstance(thresh, (list, tuple)):
        assert len(thresh) == 2
        mlno.lno_thresh = [thresh[0], thresh[1]]

    return mlno

def get_lnoparam(mf, lo_coeff, lno_thresh, lno_pct_occ, lno_norb, loidx, ifrag):
    if isinstance(mf, scf.rhf.RHF):
        orbloc = lo_coeff[:,loidx]
        lno_param = [{
            'thresh': lno_thresh[i],
            'pct_occ': lno_pct_occ[i],
            'norb': lno_norb[ifrag][i]
            } for i in [0,1]]
    elif isinstance(mf, scf.uhf.UHF):
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
        raise NotImplementedError('LNO Only Support Restricted and Unrestricted MF!')

    return orbloc, lno_param

# def get_las(mlno, orbloc, uocc_loc, lno_frozen, spin_type, loc_ctr):
#     mf = mlno._scf
#     mol = mf.mol
#     mo_occ = mlno.mo_occ
#     if spin_type == "unrestricted":
#         if uocc_loc[0].size > 0 and uocc_loc[1].size == 0:
#             lno_elec_type = 'alpha'
#         elif uocc_loc[0].size == 0 and uocc_loc[1].size > 0:
#             lno_elec_type = 'beta'
#         else:
#             lno_elec_type = 'mixed'

#         if loc_ctr is None:
#             ao_max_a = prep.ao_comp(mf, orbloc[0])
#             ao_max_b = prep.ao_comp(mf, orbloc[1])
#             loc_ctr = ao_max_a + ao_max_b
#             print(f"LNO Center {loc_ctr}")

#         lno_frozen, maskact = ulnoccsd.get_maskact(lno_frozen, [mo_occ[0].size, mo_occ[1].size])
#         occidxa = mo_occ[0] > 1e-10
#         occidxb = mo_occ[1] > 1e-10
#         moidxa, moidxb = maskact
#         nactocc_a = int(np.sum(moidxa & occidxa))
#         nactvir_a = int(np.sum(moidxa & ~occidxa))
#         nactocc_b = int(np.sum(moidxb & occidxb))
#         nactvir_b = int(np.sum(moidxb & ~occidxb))
#         nactocc = [nactocc_a, nactocc_b]
#         nactvir = [nactvir_a, nactvir_b]
#         lno_active_a = np.array([i for i in range(mol.nao) if i not in lno_frozen[0]])
#         lno_active_b = np.array([i for i in range(mol.nao) if i not in lno_frozen[1]])
#         lno_active = [lno_active_a, lno_active_b]
#         lno_tot = [len(lno_active_a), len(lno_active_b)]
#         print(f'LAS occupied orbitals:  {nactocc}')
#         print(f'LAS virtual orbitals:   {nactvir}')
#         print(f'LAS total size:         {lno_tot}')
#     else:
#         lno_elec_type = 'restricted'
#         if loc_ctr is None:
#             loc_ctr = prep.ao_comp(mf, orbloc)
#             print(f"LNO Center {loc_ctr}")

#         lno_frozen, maskact = lnoccsd.get_maskact(lno_frozen, mo_occ.size)
#         lno_active = np.array([i for i in range(mol.nao) if i not in lno_frozen])
#         nactocc, nactvir = prep.las_size(mf, lno_frozen)
#         lno_tot = len(lno_active)
#         print(f'LNO-Frgament Spin Type: {lno_elec_type}')
#         print(f'LAS occupied orbitals:  {nactocc}')
#         print(f'LAS virtual orbitals:   {nactvir}')
#         print(f'LAS total size:         {lno_tot}')
    
#     return  maskact, lno_active, nactocc, nactvir, lno_tot

def lnoccsd_kernel(mlno, lno_coeff, lno_frozen, uocc_loc, maskact, verbose=3):
    mf = mlno._scf
    if isinstance(mf, scf.rhf.RHF):
        mcc = lnoccsd.CCSD(mf, mo_coeff=lno_coeff, frozen=lno_frozen).set(verbose=verbose)
        mcc.conv_tol = 1e-6
        mcc.conv_tol_normt = 3e-5
        ecc_frag, t1, t2 = \
            mod_lnoccsd.rlnoccsd_solver(mcc, lno_coeff, uocc_loc, mlno.mo_occ, maskact)
    elif isinstance(mf, scf.uhf.UHF):
        mcc = ulnoccsd.UCCSD(mf, mo_coeff=lno_coeff, frozen=lno_frozen).set(verbose=verbose)
        mcc.conv_tol = 1e-6
        mcc.conv_tol_normt = 3e-5
        ecc_frag, t1, t2 = \
            mod_lnoccsd.ulnoccsd_solver(mcc, lno_coeff, uocc_loc, mlno.mo_occ, maskact)
    else: 
        raise NotImplementedError('LNO Only Support Restricted and Unrestricted Orbitals!')
    return ecc_frag, t1, t2

def lnomp2_kernel(mlno, lno_coeff, lno_frozen, uocc_loc, maskact, verbose=3):
    # give canonicalized orbitals to MP2
    mf = mlno._scf
    if isinstance(mf, scf.rhf.RHF):
        mcc = lnoccsd.CCSD(mf, mo_coeff=lno_coeff, frozen=lno_frozen).set(verbose=verbose)
        emp_frag = mod_lnoccsd.rlnomp2_solver(mcc, lno_coeff, uocc_loc, mlno.mo_occ, maskact)
    elif isinstance(mf, scf.uhf.UHF):
        mcc = ulnoccsd.UCCSD(mf, mo_coeff=lno_coeff, frozen=lno_frozen).set(verbose=verbose)
        emp_frag = mod_lnoccsd.ulnomp2_solver(mcc, lno_coeff, uocc_loc, mlno.mo_occ, maskact)
    else: 
        raise NotImplementedError('LNO Only Support Restricted and Unrestricted Orbitals!')
    # eorb_mp = mod_lnoccsd.lnomp2_solver(mcc, lno_coeff, uocc_loc, mlno.mo_occ, maskact)
    return emp_frag

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
              lo_coeff, 
              frag_list,
              frag_name,
              lno_thresh = 1e-6,
              qmc_options = {}, 
              chol_cut = 1e-5, 
              target_qmc_err = 1e-3, 
              run_frag = None, 
              nfrozen = None,
              run_mp = True,
              run_cc = True,
              run_qmc = True,
              plot_las = False,
              ):
    
    print("\n ******* LNO-CALCULATION ******* \n")
    print(f"LNO THRESHOLD = {lno_thresh}")

    if nfrozen is None:
        print("LNO freezes at least the chemcore orbitals for each element!")
        nfrozen = elements.chemcore(mf.mol)

    tools.check_span(mf, lo_coeff, nfrozen, thresh=1e-6)

    mlno = get_lnoccsd(mf, lo_coeff, frag_list, nfrozen, lno_thresh)

    lno_thresh = mlno.lno_thresh
    # print(f"LNO THRESHOLD = {mlno.lno_thresh}")
    lno_type = ['1h','1h']
    eris = mlno.ao2mo()

    nfrag_tot = len(frag_list)
    if run_frag is None:
        run_frag = range(nfrag_tot)
    
    frag_list = [frag_list[i] for i in run_frag]
    frag_name = [frag_name[i] for i in run_frag]
    nfrag_run = len(frag_list)

    lno_pct_occ = [None, None]
    lno_norb = [[None,None]] * nfrag_tot

    seeds = random.randint(random.PRNGKey(qmc_options["seed"]),
                           shape=(nfrag_tot,), 
                           minval=0, 
                           maxval=100*nfrag_tot
                           )
    
    qmc_options["max_error"] = target_qmc_err / np.sqrt(nfrag_tot)

    # las_center = [None]*nfrag_run
    lno_size = [None]*nfrag_run
    lno_emp = np.zeros(nfrag_run, dtype='float64')
    lno_ecc  = np.zeros(nfrag_run, dtype='float64')
    lno_eqmc = np.zeros(nfrag_run, dtype='float64')
    lno_eqmc_err  = np.zeros(nfrag_run, dtype='float64')
    lno_cc_time = np.zeros(nfrag_run, dtype='float64')
    lno_qmc_time = np.zeros(nfrag_run, dtype='float64')

    # Loop over fragment
    for ifrag, frag_idx in enumerate(run_frag):
        loidx = frag_list[ifrag]
        print("\n")
        width = 80
        msg = f" LNO-FRAGMENT [{frag_name[ifrag]}] {frag_idx+1}/({nfrag_run},{nfrag_tot}) "
        print(msg.center(width, '='))
        # print(f"Fragment Name - {frag_name[ifrag]}")
        print(f"LNO THRESHOLD - {mlno.lno_thresh}")
        print(f"PySCF NumPy Threads - {lib.num_threads()}")

        orbloc, lno_param = get_lnoparam(mf, lo_coeff, lno_thresh, lno_pct_occ, lno_norb, loidx, ifrag)

        lno_coeff, lno_frozen, uocc_loc, _ \
                    = mlno.make_las(eris, orbloc, lno_type, lno_param)
        
        if isinstance(mlno._scf, scf.rhf.RHF):
            lno_frozen, maskact \
                = lnoccsd.get_maskact(lno_frozen, mlno.mo_occ.size)
        elif isinstance(mlno._scf, scf.uhf.UHF):
            lno_frozen, maskact \
                = ulnoccsd.get_maskact(lno_frozen, [mlno.mo_occ[0].size, mlno.mo_occ[1].size])
        else:
            raise TypeError(f'unsupported mean-field type: {type(mlno._scf)}')

        lno_split, nfrzocc, nactocc, nactvir, nfrzvir = tools.split_lno(mlno, lno_coeff, lno_frozen)
                
        # if plot_las:
        #     tools.plot_density(mf, orbloc, lno_split, idx=frag_idx+1)

        
        time0 = time.perf_counter()
        if run_mp:
            efrag_mp = lnomp2_kernel(mlno, lno_coeff, lno_frozen, uocc_loc, maskact, verbose=3)
        else: efrag_mp = 0.0

        if run_cc:
            efrag_cc, t1, t2 = \
                lnoccsd_kernel(mlno, lno_coeff, lno_frozen, uocc_loc, maskact, verbose=4)
        else: efrag_cc = 0.0
        frag_cc_time = time.perf_counter() - time0

        print(f'LNO-MP2 Fragment Energy:  {efrag_mp:.8f}')
        print(f'LNO-CCSD Fragment Energy: {efrag_cc:.8f}')
        print(f"LNO-CCSD time (s):        {frag_cc_time:.2f}")

        outfile = f'fragment.out{frag_idx+1}'

        time0 = time.perf_counter()
        if run_qmc:
            efrag_qmc, efrag_qmc_err \
                = lnoafqmc_kernel(
                    mlno, lno_coeff, uocc_loc, lno_frozen, t1, t2, 
                    chol_cut, frag_idx, seeds, qmc_options)
        else: efrag_qmc, efrag_qmc_err = 0.0, 0.0
        frag_qmc_time = time.perf_counter() - time0
        
        norb = np.array(nactocc)+np.array(nactvir)
        lno_size[ifrag] = norb
        lno_emp[ifrag] = efrag_mp
        lno_ecc[ifrag] = efrag_cc
        lno_cc_time[ifrag] = frag_cc_time
        lno_eqmc[ifrag] = efrag_qmc
        lno_eqmc_err[ifrag] = efrag_qmc_err
        lno_qmc_time[ifrag] = frag_qmc_time

        header = f' Fragment{run_frag[ifrag]+1} Results '
        width = 80
        with open(outfile, 'a') as f:
            f.write('\n')
            f.write(f'{header:=^{width}}\n')
            f.write("\t LNO Fragment " + frag_name[ifrag] + "\n")
            f.write('-' * width + '\n')
            f.write(f'\t LNO-Active Space electrons: {np.array(nactocc)} | orbitals: {norb} \n')
            f.write(f'\t LNO-MP2 Fragment Energy:    {efrag_mp:.8f} \n')
            f.write(f'\t LNO-CCSD Fragment Energy:   {efrag_cc:.8f} \n')
            f.write(f'\t LNO-AFQMC Fragment Energy:  {efrag_qmc:.5f} +/- {efrag_qmc_err:.5f} \n')
            f.write(f'\t LNO-CCSD Fragment Time:     {frag_cc_time:.2f} \n')
            f.write(f'\t LNO-AFQMC Fragment Time:    {frag_qmc_time:.2f} \n')
            f.write('=' * width + '\n')
        jax.clear_caches()
        gc.collect()

    lno_size = np.array(lno_size, dtype=np.int32)
    lno_max = lno_size.max()
    # convert to list of string for print
    lno_size = list(map(lambda row: f"{row}", lno_size))
    e_mp = np.sum(lno_emp)
    e_cc = np.sum(lno_ecc)
    e_qmc = np.sum(lno_eqmc)
    e_qmc_err = np.sqrt(np.sum(lno_eqmc_err**2))
    tot_ccsd_time = np.sum(lno_cc_time)
    tot_qmc_time = np.sum(lno_qmc_time)

    with open(f'lno_result.out', 'w') as f:
        width = 100
        f.write('=' * width + '\n')
        f.write(f'{"LNO-AFQMC Results":^{width}}\n')
        f.write('=' * width + '\n')

        f.write(f'{"Num":>4s}  {"Fragment":>10s}  {"LAS SIZE":>10s}  '
                f'{"E(MP2)":>10s}  {"E(CCSD)":>10s}  '
                f'{"E(AFQMC)":>10s}  {"Error":>8s}  '
                f'{"t(CCSD)":>8s}  {"t(AFQMC)":>8s}\n')
        f.write('-' * width + '\n')
        
        for n, i in enumerate(run_frag):
            f.write(f"{i+1:4d}  {frag_name[n]:>10s}  {lno_size[n]:10s}  "
                    f"{lno_emp[n]:10.8f}  {lno_ecc[n]:10.8f}  "
                    f"{lno_eqmc[n]:10.5f}  {lno_eqmc_err[n]:8.5f}  "
                    f"{lno_cc_time[n]:8.2f}  {lno_qmc_time[n]:8.2f}\n")
        
        f.write('-' * width + '\n')

        f.write(f'{"Summarize Fragments":^{width}}\n')
        f.write('-' * width + '\n')

        lno_thresh_str = "[" + ", ".join(f"{x:.2e}" for x in lno_thresh) + "]"
        f.write(f'{"LNO-Thresh":<20} {"Max LAS":>8} '
                f'{"E[MP2]":>12} {"E[CCSD]":>12} '
                f'{"E[AFQMC]":>10} {"Err[AFQMC]":>10} '
                f'{"CCSD-Time":>10} {"AFQMC-Time":>10}\n')

        f.write(f'{lno_thresh_str:<20} {lno_max:>8} '
                f'{e_mp:>12.8f} {e_cc:>12.8f} '
                f'{e_qmc:>10.5f} {e_qmc_err:>10.5f} '
                f'{tot_ccsd_time:>10.2f} {tot_qmc_time:>10.2f}\n')
        
        f.write('=' * width + '\n\n')

    return e_mp, e_cc, e_qmc, e_qmc_err, lno_max
