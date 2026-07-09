import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from jax import random

from pyscf import lib
# from pyscf.lno import lnoccsd
# from pyscf.lno import ulnoccsd
# from collections.abc import Iterable

# from afqmc import config
from afqmc.lno_afqmc import prep, lno_afqmc, integral

from functools import partial
import time, gc, pickle

print = partial(print, flush=True)


def run_cfsafqmc(options, option_file='options.bin'):
    
    with open(option_file, 'wb') as f:
        pickle.dump(options, f)

    if 'pt2' in options['trial']:
        script='script/run_cfs_afqmc_pt2ccsd.py'
    
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/{script}"
    print(f'AFQMC script: {script}')
    
    os.system(f" python {script} |tee afqmc.out")


def cfsafqmc_kernel(mlno1, lno_coeff1, uocc_loc1, lno_frozen1, t11, t21,
                    mlno2, lno_coeff2, uocc_loc2, lno_frozen2, t12, t22,
                    chol_cut, frag_idx, seeds, qmc_options):
    
    mf1, mf2 = mlno1._scf, mlno2._scf
    # integral.prep_lno_integral(mf1, lno_coeff1, lno_frozen1, uocc_loc1, t11, t21, chol_cut, 
    #                            amp_file="amplitudes1.npz", chol_file="FCIDUMP_chol1")
    # integral.prep_lno_integral(mf2, lno_coeff2, lno_frozen2, uocc_loc2, t12, t22, chol_cut,
    #                            amp_file="amplitudes2.npz", chol_file="FCIDUMP_chol2")
    
    integral.prep_cfs_integral(
        mf1, mf2,
        lno_coeff1, lno_coeff2, 
        lno_frozen1, lno_frozen2,
        uocc_loc1, uocc_loc2,
        t11, t12,
        t21, t22,
        chol_cut,
        amp_file1="amplitudes1.npz", amp_file2="amplitudes2.npz",
        chol_file1="FCIDUMP_chol1", chol_file2="FCIDUMP_chol2"
        )
    
    qmc_options["seed"] = seeds[frag_idx]
    run_cfsafqmc(qmc_options)

    outfile = f'fragment.out{frag_idx+1}'
    os.system(f'mv afqmc.out {outfile}')

    def _two_floats(line):
        """Return (value, error) from the last two tokens, or None if not numeric."""
        tok = line.split()
        try:
            return float(tok[-2]), float(tok[-1])
        except (ValueError, IndexError):
            return None

    efrag1 = efrag_err1 = None
    efrag2 = efrag_err2 = None
    d12 = d12_err = None
    with open(outfile, "r") as f:
        for line in f:
            s = line.strip()
            if s.startswith("System 1"):
                got = _two_floats(line)
                if got is not None:
                    efrag1, efrag_err1 = got
            elif s.startswith("System 2"):
                got = _two_floats(line)
                if got is not None:
                    efrag2, efrag_err2 = got
            elif s.startswith("Difference (1-2)"):
                got = _two_floats(line)
                if got is not None:
                    d12, d12_err = got

    missing = [name for name, val in
               (("efrag1", efrag1), ("efrag_err1", efrag_err1),
                ("efrag2", efrag2), ("efrag_err2", efrag_err2),
                ("d12", d12), ("d12_err", d12_err)) if val is None]
    if missing:
        raise RuntimeError(
            f"cfsafqmc_kernel: could not parse {missing} from {outfile}")

    return efrag1, efrag_err1, efrag2, efrag_err2, d12, d12_err

def run_afqmc(mf1, mf2,
              lo_coeff1 = None,  lo_coeff2 = None,
              frag_lolist1 = None, frag_lolist2 = None,
              thresh1 = 1e-6, thresh2 = 1e-6,
              nfrozen = 0,
              chol_cut = 1e-5,
              target_sto_error = 1e-3,
              qmc_options = {},
              run_frag_list = None,
              atom_group = None,
              ):

    spin_type1 = prep.kind(lo_coeff1)
    spin_type2 = prep.kind(lo_coeff2)

    assert spin_type1 == spin_type2
    spin_type = spin_type1

    if frag_lolist1 is None:
        raise ValueError("frag_lolist must be provided for CFS-AFQMC.")

    if frag_lolist2 is None:
        raise ValueError("frag_lolist must be provided for CFS-AFQMC.")

    nfrag_tot1 = len(frag_lolist1)
    nfrag_tot2 = len(frag_lolist2)

    assert nfrag_tot1 == nfrag_tot2
    nfrag_tot = nfrag_tot1

    mlno1 = lno_afqmc.get_lnoccsd(mf1, lo_coeff1, frag_lolist1, nfrozen, thresh1, spin_type)
    mlno2 = lno_afqmc.get_lnoccsd(mf2, lo_coeff2, frag_lolist2, nfrozen, thresh2, spin_type)

    lno_thresh1 = mlno1.lno_thresh
    lno_thresh2 = mlno2.lno_thresh
    eris1 = mlno1.ao2mo()
    eris2 = mlno2.ao2mo()

    lno_type = ['1h','1h']

    if run_frag_list is None:
        run_frag_list = range(nfrag_tot)

    frag_lolist1 = [frag_lolist1[i] for i in run_frag_list]
    frag_lolist2 = [frag_lolist2[i] for i in run_frag_list]

    nfrag_run = len(run_frag_list)
    lno_pct_occ = [None, None]
    lno_norb = [[None, None] for _ in range(nfrag_run)]   # FIX: was [[None,None]]*n (aliased)

    # seeds sized to nfrag_tot so a subset run_frag_list still indexes correctly
    # (the kernel indexes seeds by the GLOBAL frag_idx)
    seeds = random.randint(random.PRNGKey(qmc_options["seed"]),
                           shape=(nfrag_tot,),          # FIX: was (nfrag_run,)
                           minval=0,
                           maxval=100*nfrag_tot
                           )

    qmc_options["max_error"] = target_sto_error / np.sqrt(nfrag_tot)
    trial_base = qmc_options.get("trial", "")

    las_ctr = [None]*nfrag_run

    las_size1 = [None]*nfrag_run
    lno_emp1 = np.zeros(nfrag_run, dtype='float64')
    lno_ecc1  = np.zeros(nfrag_run, dtype='float64')
    lno_eqmc1 = np.zeros(nfrag_run, dtype='float64')
    lno_eqmc_err1  = np.zeros(nfrag_run, dtype='float64')
    ccsd_time1 = np.zeros(nfrag_run, dtype='float64')
    qmc_time1 = np.zeros(nfrag_run, dtype='float64')

    las_size2 = [None]*nfrag_run
    lno_emp2 = np.zeros(nfrag_run, dtype='float64')
    lno_ecc2  = np.zeros(nfrag_run, dtype='float64')
    lno_eqmc2 = np.zeros(nfrag_run, dtype='float64')
    lno_eqmc_err2  = np.zeros(nfrag_run, dtype='float64')
    ccsd_time2 = np.zeros(nfrag_run, dtype='float64')
    qmc_time2 = np.zeros(nfrag_run, dtype='float64')

    # correlated (CFS) per-fragment AFQMC difference + its correlated error
    lno_deqmc     = np.zeros(nfrag_run, dtype='float64')
    lno_deqmc_err = np.zeros(nfrag_run, dtype='float64')

    # Loop over fragment
    for ifrag, frag_idx in enumerate(run_frag_list):
        loidx1 = frag_lolist1[ifrag]
        loidx2 = frag_lolist2[ifrag]

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

        orbloc1, lno_param1 = lno_afqmc.get_lnoparam(
            lo_coeff1, lno_thresh1, lno_pct_occ, lno_norb, loidx1, ifrag, spin_type)
        orbloc2, lno_param2 = lno_afqmc.get_lnoparam(
            lo_coeff2, lno_thresh2, lno_pct_occ, lno_norb, loidx2, ifrag, spin_type)

        lno_coeff1, lno_frozen1, uocc_loc1, _ = mlno1.make_las(eris1, orbloc1, lno_type, lno_param1)
        lno_coeff2, lno_frozen2, uocc_loc2, _ = mlno2.make_las(eris2, orbloc2, lno_type, lno_param2)

        maskact1, lno_active1, nactocc1, nactvir1, lno_tot1, lno_elec_type = \
            lno_afqmc.get_las(mlno1, orbloc1, uocc_loc1, lno_frozen1, spin_type, loc_ctr)
        maskact2, lno_active2, nactocc2, nactvir2, lno_tot2, lno_elec_type = \
            lno_afqmc.get_las(mlno2, orbloc2, uocc_loc2, lno_frozen2, spin_type, loc_ctr)

        time0 = time.perf_counter()
        (eorb_mp1, eorb_cc1), t11, t21 = \
            lno_afqmc.lnoccsd_kernel(
                mlno1, lno_coeff1, lno_frozen1, uocc_loc1, maskact1, verbose=0)
        lnocc_time1 = time.perf_counter() - time0

        time0 = time.perf_counter()
        (eorb_mp2, eorb_cc2), t12, t22 = \
            lno_afqmc.lnoccsd_kernel(
                mlno2, lno_coeff2, lno_frozen2, uocc_loc2, maskact2, verbose=0)
        lnocc_time2 = time.perf_counter() - time0

        print(f"{'':20}  {'system 1':>15}  {'system 2':>15}")
        print(f"{'LNO-MP2 Energy:':20}  {eorb_mp1:>15.8f}  {eorb_mp2:>15.8f}")
        print(f"{'LNO-CCSD Energy:':20}  {eorb_cc1:>15.8f}  {eorb_cc2:>15.8f}")
        print(f"{'LNO-CCSD time(s):':20}  {lnocc_time1:>15.2f}  {lnocc_time2:>15.2f}")

        outfile = f'fragment.out{frag_idx+1}'
        time0 = time.perf_counter()
        eorb_afqmc1, eorb_afqmc_err1, eorb_afqmc2, eorb_afqmc_err2, deorb12, deorb12_err \
            = cfsafqmc_kernel(
                mlno1, lno_coeff1, uocc_loc1, lno_frozen1, t11, t21,
                mlno2, lno_coeff2, uocc_loc2, lno_frozen2, t12, t22,
                chol_cut, frag_idx, seeds, qmc_options)
        lnoqmc_time = time.perf_counter() - time0

        # ---- store results --------------------------------------------
        las_ctr[ifrag] = loc_ctr

        # system 1
        las_size1[ifrag]     = lno_tot1
        lno_emp1[ifrag]      = eorb_mp1
        lno_ecc1[ifrag]      = eorb_cc1
        lno_eqmc1[ifrag]     = eorb_afqmc1
        lno_eqmc_err1[ifrag] = eorb_afqmc_err1
        ccsd_time1[ifrag]    = lnocc_time1
        qmc_time1[ifrag]     = lnoqmc_time

        # system 2  (same correlated QMC wall time)
        las_size2[ifrag]     = lno_tot2
        lno_emp2[ifrag]      = eorb_mp2
        lno_ecc2[ifrag]      = eorb_cc2
        lno_eqmc2[ifrag]     = eorb_afqmc2
        lno_eqmc_err2[ifrag] = eorb_afqmc_err2
        ccsd_time2[ifrag]    = lnocc_time2
        qmc_time2[ifrag]     = lnoqmc_time

        # correlated difference straight from the CFS kernel
        lno_deqmc[ifrag]     = deorb12
        lno_deqmc_err[ifrag] = deorb12_err

        # ---- per-fragment energy differences (system 1 - system 2) ----
        d_mp2  = eorb_mp1 - eorb_mp2
        d_ccsd = eorb_cc1 - eorb_cc2
        # AFQMC difference + error now come from the CFS kernel (correlated),
        # NOT sqrt(err1**2+err2**2). deorb12 == eorb_afqmc1 - eorb_afqmc2.

        # ---- per-fragment output file ---------------------------------
        def _las_str(x):
            """'n' for restricted, 'na/nb' for unrestricted."""
            return "/".join(str(int(v)) for v in np.ravel(x))
        def _las_add(a, b):
            """Elementwise sum of two counts (scalar or per-spin), as a string."""
            return _las_str(np.ravel(a) + np.ravel(b))

        header = f' Fragment{run_frag_list[ifrag]+1} Results '
        width = 96
        col = 16
        with open(outfile, 'a') as f:
            f.write('\n')
            f.write(f'{header:=^{width}}\n')
            f.write(f'\t LNO Center {loc_ctr}\n')
            f.write('-' * width + '\n')
            f.write(f'\t{"Quantity":<26}{"System 1":>{col}}{"System 2":>{col}}{"Diff (1-2)":>{col}}\n')
            f.write(f'\t{"LNO-Active electrons":<26}{_las_str(nactocc1):>{col}}{_las_str(nactocc2):>{col}}\n')
            f.write(f'\t{"LNO-Active orbitals":<26}{_las_add(nactocc1, nactvir1):>{col}}{_las_add(nactocc2, nactvir2):>{col}}\n')
            f.write(f'\t{"LNO-MP2 Energy":<26}{eorb_mp1:>{col}.8f}{eorb_mp2:>{col}.8f}{d_mp2:>{col}.8f}\n')
            f.write(f'\t{"LNO-CCSD Energy":<26}{eorb_cc1:>{col}.8f}{eorb_cc2:>{col}.8f}{d_ccsd:>{col}.8f}\n')
            f.write(f'\t{"LNO-AFQMC Energy":<26}{eorb_afqmc1:>{col}.6f}{eorb_afqmc2:>{col}.6f}{deorb12:>{col}.6f}\n')
            f.write(f'\t{"LNO-AFQMC Error":<26}{eorb_afqmc_err1:>{col}.6f}{eorb_afqmc_err2:>{col}.6f}{deorb12_err:>{col}.6f}\n')
            f.write(f'\t{"LNO-CCSD Time (s)":<26}{lnocc_time1:>{col}.2f}{lnocc_time2:>{col}.2f}\n')
            f.write(f'\t{"LNO-AFQMC Time (s)":<26}{lnoqmc_time:>{col}.2f}{lnoqmc_time:>{col}.2f}\n')
            f.write('=' * width + '\n')

        jax.clear_caches()
        gc.collect()

    # ================================================================== #
    #  AFTER the fragment loop: per-fragment table + two-system summary   #
    # ================================================================== #

    # LAS sizes may be scalar (restricted) or (nalpha, nbeta) (unrestricted).
    # Keep raw values for per-fragment display; build an integer array
    # (nfrag, k) for spin-channel reductions.
    # def _las_str(x):
    #     """'n' for restricted, 'na/nb' for unrestricted."""
    #     return "/".join(str(int(v)) for v in np.ravel(x))

    las_arr1 = np.array([np.ravel(s) for s in las_size1], dtype=np.int64)  # (nfrag, k)
    las_arr2 = np.array([np.ravel(s) for s in las_size2], dtype=np.int64)
    las_ctr_str = [str(c) if c is not None else '-' for c in las_ctr]

    # totals per system
    e_mp2_1,  e_mp2_2  = lno_emp1.sum(),  lno_emp2.sum()
    e_ccsd_1, e_ccsd_2 = lno_ecc1.sum(),  lno_ecc2.sum()
    e_qmc_1,  e_qmc_2  = lno_eqmc1.sum(), lno_eqmc2.sum()
    e_qmc_err_1 = np.sqrt(np.sum(lno_eqmc_err1**2))
    e_qmc_err_2 = np.sqrt(np.sum(lno_eqmc_err2**2))
    tot_ccsd_time_1, tot_ccsd_time_2 = ccsd_time1.sum(), ccsd_time2.sum()
    tot_qmc_time = qmc_time1.sum()          # == qmc_time2.sum(): one CFS pass per fragment

    # total differences (MP2/CCSD are deterministic; AFQMC diff is correlated)
    d_mp2_tot     = e_mp2_1  - e_mp2_2
    d_ccsd_tot    = e_ccsd_1 - e_ccsd_2
    d_qmc_tot     = np.sum(lno_deqmc)                      # == e_qmc_1 - e_qmc_2
    # fragments use independent QMC seeds -> per-fragment correlated errors add
    # in quadrature for the total difference error
    d_qmc_err_tot = np.sqrt(np.sum(lno_deqmc_err**2))

    # per-spin LAS reductions -> display strings
    sum_las1 = _las_str(las_arr1.sum(axis=0)); sum_las2 = _las_str(las_arr2.sum(axis=0))
    max_las1 = _las_str(las_arr1.max(axis=0)); max_las2 = _las_str(las_arr2.max(axis=0))

    def _sysblock(las, emp2, eccsd, eqmc, eerr, tcc, tqmc):
        return (f"{las:>8s}  {emp2:>12.8f}  {eccsd:>12.8f}  "
                f"{eqmc:>11.6f}  {eerr:>9.6f}  {tcc:>8.2f}  {tqmc:>8.2f}")

    def _sysblock_hdr():
        return (f"{'LAS':>8s}  {'E(MP2)':>12s}  {'E(CCSD)':>12s}  "
                f"{'E(AFQMC)':>11s}  {'Err':>9s}  {'t(CCSD)':>8s}  {'t(AFQMC)':>8s}")

    def _thresh_str(th):
        return "[" + ", ".join(f"{x:.2e}" for x in np.atleast_1d(th)) + "]"

    def _write_system(f, title, las_size, lno_emp, lno_ecc, lno_eqmc, lno_eqmc_err,
                      ccsd_time, qmc_time, sum_las, e_mp2, e_ccsd, e_qmc, e_qmc_err,
                      tot_ccsd_time):
        hdr = f"{'Frag':>4s}  {'LAS Center':>14s}  {_sysblock_hdr()}"
        w = len(hdr)
        f.write('=' * w + '\n')
        f.write(f'{title:^{w}}\n')
        f.write('=' * w + '\n')
        f.write(hdr + '\n')
        f.write('-' * w + '\n')
        for n, i in enumerate(run_frag_list):
            f.write(f"{i+1:>4d}  {las_ctr_str[n]:>14s}  "
                    f"{_sysblock(_las_str(las_size[n]), lno_emp[n], lno_ecc[n], lno_eqmc[n], lno_eqmc_err[n], ccsd_time[n], qmc_time[n])}\n")
        f.write('-' * w + '\n')
        f.write(f"{'SUM':>4s}  {'':>14s}  "
                f"{_sysblock(sum_las, e_mp2, e_ccsd, e_qmc, e_qmc_err, tot_ccsd_time, tot_qmc_time)}\n")
        f.write('=' * w + '\n\n')

    with open('cfs_results.out', 'w') as f:
        f.write('#' * 60 + '\n')
        f.write(f'{"CFS-AFQMC Correlated Results":^60}\n')
        f.write('#' * 60 + '\n\n')

        # -------- System 1 fragment rows --------
        _write_system(f, "System 1  -  per-fragment results",
                      las_size1, lno_emp1, lno_ecc1, lno_eqmc1, lno_eqmc_err1,
                      ccsd_time1, qmc_time1, sum_las1,
                      e_mp2_1, e_ccsd_1, e_qmc_1, e_qmc_err_1, tot_ccsd_time_1)

        # -------- System 2 fragment rows --------
        _write_system(f, "System 2  -  per-fragment results",
                      las_size2, lno_emp2, lno_ecc2, lno_eqmc2, lno_eqmc_err2,
                      ccsd_time2, qmc_time2, sum_las2,
                      e_mp2_2, e_ccsd_2, e_qmc_2, e_qmc_err_2, tot_ccsd_time_2)

        # -------- Difference (System 1 - System 2) fragment rows --------
        dhdr = (f"{'Frag':>4s}  {'LAS Center':>14s}  "
                f"{'dE(MP2)':>12s}  {'dE(CCSD)':>12s}  {'dE(AFQMC)':>11s}  {'Err(cfs)':>9s}")
        dw = len(dhdr)
        f.write('=' * dw + '\n')
        f.write(f'{"Difference (System 1 - System 2)":^{dw}}\n')
        f.write('=' * dw + '\n')
        f.write(dhdr + '\n')
        f.write('-' * dw + '\n')
        for n, i in enumerate(run_frag_list):
            f.write(f"{i+1:>4d}  {las_ctr_str[n]:>14s}  "
                    f"{lno_emp1[n]-lno_emp2[n]:>12.8f}  {lno_ecc1[n]-lno_ecc2[n]:>12.8f}  "
                    f"{lno_deqmc[n]:>11.6f}  {lno_deqmc_err[n]:>9.6f}\n")
        f.write('-' * dw + '\n')
        f.write(f"{'SUM':>4s}  {'':>14s}  "
                f"{d_mp2_tot:>12.8f}  {d_ccsd_tot:>12.8f}  {d_qmc_tot:>11.6f}  {d_qmc_err_tot:>9.6f}\n")
        f.write('=' * dw + '\n\n')

        # -------- vertical final summary --------
        swidth = 60
        LBL = 18
        def _line(label, value):
            return f"  {label:<{LBL}}{value}\n"

        f.write('=' * swidth + '\n')
        f.write(f'{"CFS-AFQMC Final Summary":^{swidth}}\n')
        f.write('=' * swidth + '\n')

        f.write("System 1\n")
        f.write(_line("LNO-Thresh",     _thresh_str(lno_thresh1)))
        f.write(_line("Max LAS",        max_las1))
        f.write(_line("Sum LAS",        sum_las1))
        f.write(_line("E[MP2]  (Ha)",   f"{e_mp2_1:.8f}"))
        f.write(_line("E[CCSD] (Ha)",   f"{e_ccsd_1:.8f}"))
        f.write(_line("E[AFQMC](Ha)",   f"{e_qmc_1:.6f} +/- {e_qmc_err_1:.6f}"))
        f.write('-' * swidth + '\n')

        f.write("System 2\n")
        f.write(_line("LNO-Thresh",     _thresh_str(lno_thresh2)))
        f.write(_line("Max LAS",        max_las2))
        f.write(_line("Sum LAS",        sum_las2))
        f.write(_line("E[MP2]  (Ha)",   f"{e_mp2_2:.8f}"))
        f.write(_line("E[CCSD] (Ha)",   f"{e_ccsd_2:.8f}"))
        f.write(_line("E[AFQMC](Ha)",   f"{e_qmc_2:.6f} +/- {e_qmc_err_2:.6f}"))
        f.write('-' * swidth + '\n')

        f.write("Difference (1 - 2)\n")
        f.write(_line("dE[MP2]  (Ha)",  f"{d_mp2_tot:.8f}"))
        f.write(_line("dE[CCSD] (Ha)",  f"{d_ccsd_tot:.8f}"))
        f.write(_line("dE[AFQMC](Ha)",  f"{d_qmc_tot:.6f} +/- {d_qmc_err_tot:.6f}"))
        f.write('=' * swidth + '\n')
        f.write("  dE[AFQMC] error is the correlated (CFS) error, summed over\n")
        f.write("  fragments in quadrature (each fragment uses an independent seed).\n")
        f.write('=' * swidth + '\n\n')

    return None