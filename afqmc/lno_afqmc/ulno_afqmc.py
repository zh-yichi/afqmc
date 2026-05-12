import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

from jax import random
import numpy as np
from pyscf import mp
from pyscf.lno import ulnoccsd
from afqmc import config
from functools import partial
from collections.abc import Iterable
import pickle, time, gc, re

print = partial(print, flush=True)

# def run_lnoafqmc(options,
#                  option_file ='options.bin',
#                  script = None):

#     with open(option_file, 'wb') as f:
#         pickle.dump(options, f)

#     use_gpu = options["use_gpu"]
#     if use_gpu:
#         print(f'running AFQMC on GPU')
#         config.afqmc_config = {"use_gpu": True}
#         config.setup_jax()
#         gpu_flag = "--use_gpu"

#     else:
#         print(f'running AFQMC on CPU')
#         gpu_flag = ""

    
#     if script is None:
#         if 'pt2' in options['trial']:
#             script='ccsd_pt2/run_uafqmc_nompi.py'
#         else:
#             raise NotImplementedError("Only support pt2CCSD trial.")
    
#     path = os.path.abspath(__file__)
#     dir_path = os.path.dirname(path)
#     script = f"{dir_path}/{script}"
#     print(f'AFQMC script: {script}')
    
#     os.system(
#         f"python {script} {gpu_flag} |tee afqmc.out"
#         # f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1;"
#         # f"{mpi_prefix} python {script} {gpu_flag} |tee afqmc.out"
#     )

def run_afqmc(mf, 
              options, 
              lo_coeff, 
              frag_lolist,
              nfrozen = 0, 
              thresh = 1e-6, 
              chol_cut = 1e-5, 
              emp2_tot = None,
              use_df = False, 
              lno_type = ['1h']*2, 
              run_frg_list = None, 
              fast = False,
              chunk_chol = False,
              qmc_script = None,
              ):
    
    mlno = ulnoccsd.ULNOCCSD(mf, lo_coeff, frag_lolist, frozen=nfrozen).set(verbose=0)
    mlno.lno_thresh = [thresh*10,thresh]
    lno_thresh = mlno.lno_thresh
    lno_type = ['1h','1h'] if lno_type is None else lno_type
    lno_thresh = [1e-5, 1e-6] if lno_thresh is None else lno_thresh
    lno_pct_occ = None
    lno_norb = None
    # lo_proj_thresh = 1e-10
    # lo_proj_thresh_active = 0.1
    eris = None
    trial = options["trial"]
    nfrag_tot = int(mf.mol.nelectron - 2*nfrozen)
 
    if run_frg_list is None:
        nfrag = len(frag_lolist)
        run_frg_list = range(nfrag)
    
    frag_lolist = [frag_lolist[i] for i in run_frg_list]
    nfrag = len(frag_lolist)
    print(f'Number of LNO-FRAGMENT: {nfrag}')
    if lno_pct_occ is None:
        lno_pct_occ = [None, None]
    if lno_norb is None:
        lno_norb = [[None,None]] * nfrag
    mf = mlno._scf
    mol = mf.mol

    if eris is None: eris = mlno.ao2mo()

    seeds = random.randint(random.PRNGKey(options["seed"]), shape=(nfrag,), minval=0, maxval=100*nfrag)
    options["max_error"] = options["max_error"]/np.sqrt(nfrag)

    # Loop over fragment
    for ifrag, loidx in enumerate(frag_lolist):
        print("\n")
        print(f"======================= RUNNING LNO-FRAGMENT {run_frg_list[ifrag]+1}/{nfrag_tot} ========================")
        if len(loidx) == 2 and isinstance(loidx[0], Iterable): # Unrestricted
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
            lno_param = [{'thresh': lno_thresh[i], 'pct_occ': lno_pct_occ[i],
                            'norb': lno_norb[ifrag][i]} for i in [0,1]]

        lno_coeff, frozen, uocc_loc, _ = mlno.make_las(eris, orbloc, lno_type, lno_param)

        # identify the center electron type
        if uocc_loc[0].size > 0 and uocc_loc[1].size == 0:
            lno_elec_type = 'alpha'
            spin_idx = 0
        elif uocc_loc[0].size == 0 and uocc_loc[1].size > 0:
            lno_elec_type = 'beta'
            spin_idx = 1
        else: lno_elec_type = 'How could it be???'
        print(f'LNO-Electron Type = {lno_elec_type} | spin index = {spin_idx}')

        # identify the center LO's AO component
        print(f'Locating local orbital {loidx[spin_idx]}')
        # print(orbloc[0].shape, orbloc[1].shape)
        S = mol.intor('int1e_ovlp')
        proj = (S @ orbloc[spin_idx])**2
        proj = proj / np.sum(proj, axis=0)
        proj = np.sum(proj, axis=1)
        # print(proj.shape)
        ao_labels = mol.ao_labels()
        ao_threshold = 1e-3
        above = np.where(proj > ao_threshold)[0]
        # sort them by contribution descending
        above = above[np.argsort(proj[above])[::-1]]
        ao_lines = []
        print(f"AOs with contribution > {ao_threshold}")
        ao_lines.append(f"AOs with contribution > {ao_threshold}")
        print(f"{'AO Label':>16s}  {'Amp':>6s}")
        ao_lines.append(f"{'AO Label':>16s}  {'Amp':>6s}")
        for idx in above:
            print(f"{ao_labels[idx]:>16s}  {proj[idx]:6.4f}")
            ao_lines.append(f"{ao_labels[idx]:>16s}  {proj[idx]:6.4f}") 
        ao_message = "\n".join(ao_lines)

        mo_occ = mlno.mo_occ
        frozen, maskact = ulnoccsd.get_maskact(frozen, [mo_occ[0].size, mo_occ[1].size])
        mcc = ulnoccsd.UCCSD(mf, mo_coeff=lno_coeff, frozen=frozen).set(verbose=3)
        mcc._s1e = mlno._s1e
        mcc._h1e = mlno._h1e
        mcc._vhf = mlno._vhf
        if mlno.kwargs_imp is not None:
            mcc = mcc.set(**mlno.kwargs_imp)
        time0 = time.perf_counter()
        (eorb_mp2, eorb_ccsd), t1, t2 =\
            ulno_ccsd(mcc, lno_coeff, uocc_loc, mo_occ, maskact)#, ccsd_t=ccsd_t) # <<< this is on CPU
        time1 = time.perf_counter()
        
        prja = uocc_loc[0] @ uocc_loc[0].T.conj()
        prjb = uocc_loc[1] @ uocc_loc[1].T.conj()
        prjlo = [prja, prjb]
        lnoccsdtime = time1 - time0

        print(f'LNO-MP2 Orbital Energy: {eorb_mp2:.8f}')
        print(f'LNO-CCSD Orbital Energy: {eorb_ccsd:.8f}')
        print(f"LNO-CCSD time: {lnoccsdtime:.6f} s")
        
        options["trial"] = trial

        if 'ad' not in options["trial"]:
            if lno_elec_type == 'alpha':
                options["trial"] += '_alpha'
            elif lno_elec_type == 'beta':
                options["trial"] += '_beta'
            if chunk_chol:
                    options["trial"] += '_chunk'
            elif fast:
                    options["trial"] += '_fast'

        options["seed"] = seeds[ifrag]
        nelec, norb = prep.prep_afqmc_integral(
            mf, 
            lno_coeff, 
            t1, 
            t2, 
            frozen, 
            prjlo, 
            options, 
            chol_cut=chol_cut, 
            use_df=use_df
            )
        
        jax.clear_caches()
        gc.collect()
        run_lnoafqmc(options, script=qmc_script) # >> afqmc.out
        # os.system(f'mv afqmc.out lnoafqmc.out{run_frg_list[ifrag]+1}')
        outfile = f'fragment.out{run_frg_list[ifrag]+1}'
        os.system(f'mv afqmc.out {outfile}')
        with open(outfile, "r") as f:
            for line in f:
                if "Blocked AFQMC/pt2CCSD Orbital Energy" in line:
                    eorb_afqmc = float(line.split()[-3])
                    eorb_afqmc_err = float(line.split()[-1])
                if "total run time" in line:
                    lnoafqmctime = float(line.split()[-1])
        header = f' Fragment{run_frg_list[ifrag]+1} Results '
        width = 80  # pick a consistent total width
        with open(outfile, 'a') as f:
            f.write('\n')
            f.write(f'{header:=^{width}}\n')
            f.write("\t" + ao_message + "\n")
            f.write('-' * width + '\n')
            f.write(f'\t LNO-Active Space electrons: {nelec} | orbitals: {norb} \n')
            f.write(f'\t LNO-MP2 Orbital Energy:   {eorb_mp2:.8f} \n')
            f.write(f'\t LNO-CCSD Orbital Energy:  {eorb_ccsd:.8f} \n')
            f.write(f'\t LNO-AFQMC Orbital Energy: {eorb_afqmc:.6f} +/- {eorb_afqmc_err:.6f} \n')
            f.write(f'\t LNO-CCSD Time:  {lnoccsdtime:.2f} \n')
            f.write(f'\t LNO-AFQMC Time: {lnoafqmctime:.2f} \n')
            f.write('=' * width + '\n')
        jax.clear_caches()
        gc.collect()

    # finish lno loop
    if emp2_tot is None:
        mmp = mp.MP2(mf, frozen=nfrozen)
        emp2_tot = mmp.kernel()[0]

    ao_labels = []
    nelec = np.zeros((nfrag,2),dtype='int32')
    norb = np.zeros((nfrag,2),dtype='int32')
    eorb_mp2 = np.zeros(nfrag,dtype='float64')
    eorb_mp2 = np.zeros(nfrag,dtype='float64')
    eorb_ccsd = np.zeros(nfrag,dtype='float64')
    eorb_qmc = np.zeros(nfrag,dtype='float64')
    eorb_qmc_err = np.zeros(nfrag,dtype='float64')
    ccsd_time = np.zeros(nfrag,dtype='float64')
    qmc_time = np.zeros(nfrag,dtype='float64')
    for n, i in enumerate(run_frg_list):
        with open(f"fragment.out{i+1}", "r") as rf:
            for line in rf:
                if "AOs with contribution" in line:
                    next(rf)
                    largest_ao = next(rf).rsplit(maxsplit=1)[0].strip()
                    ao_labels.append(largest_ao)
                if 'LNO-Active Space' in line:
                    nums = re.findall(r'\d+', line)
                    nelec[n] = np.array([int(nums[0]),int(nums[1])])
                    norb[n] = np.array([int(nums[2]),int(nums[3])])
                if "LNO-MP2 Orbital Energy" in line:
                    eorb_mp2[n] = float(line.split()[-1])
                if "LNO-CCSD Orbital Energy" in line:
                    eorb_ccsd[n] = float(line.split()[-1])
                if "LNO-AFQMC Orbital Energy" in line:
                    eorb_qmc[n] = float(line.split()[-3])
                    eorb_qmc_err[n] = float(line.split()[-1])
                if "LNO-CCSD Time" in line:
                    ccsd_time[n] = float(line.split()[-1])
                if "LNO-AFQMC Time" in line:
                    qmc_time[n] = float(line.split()[-1])

    nelec_avg = (np.mean(nelec[:,0]), np.mean(nelec[:,1]))
    norb_avg = (np.mean(norb[:,0]), np.mean(norb[:,1]))
    e_mp2 = np.sum(eorb_mp2)
    e_ccsd = np.sum(eorb_ccsd)
    e_afqmc = np.sum(eorb_qmc)
    e_afqmc_err = np.sqrt(np.sum(eorb_qmc_err**2))
    tot_ccsd_time = np.sum(ccsd_time)
    tot_qmc_time = np.sum(qmc_time)

    with open(f'lno_result.out', 'w') as f:
        width = 110
        f.write('=' * width + '\n')
        f.write(f'{"LNO-AFQMC Results":^{width}}\n')
        f.write('=' * width + '\n')

        f.write(f'{"Frag":>4s}  {"AO Center":>14s}  '  
                f'{"E(MP2)":>10s}  {"E(CCSD)":>10s}  '
                f'{"E(AFQMC)":>10s}  {"Error":>8s}  '
                f'{"nelec":>9s}  {"norb":>9s}  '
                f'{"t(CCSD)":>8s}  {"t(AFQMC)":>8s}\n')
        f.write('-' * width + '\n')
        
        for n, i in enumerate(run_frg_list):
            f.write(f"{i+1:4d}  {ao_labels[n]:>14s}  "
                    f"{eorb_mp2[n]:10.8f}  {eorb_ccsd[n]:10.8f}  "
                    f"{eorb_qmc[n]:10.6f}  {eorb_qmc_err[n]:8.6f}  "
                    f"{str(nelec[n]):>9s}  {str(norb[n]):>9s}  "
                    f"{ccsd_time[n]:8.2f}  {qmc_time[n]:8.2f}\n")
        
        f.write('-' * width + '\n')

        f.write(f'{"Sum":>4s}  {"":>16s}  '
                f'{e_mp2:10.8f}  {e_ccsd:10.8f}  '
                f'{e_afqmc:10.6f}  {e_afqmc_err:8.6f}  '
                f'{"":>9s}  {"":>9s}  '
                f'{tot_ccsd_time:8.2f}  {tot_qmc_time:8.2f}\n')
        f.write('=' * width + '\n\n')

        f.write(f'LNO Threshold:          ({lno_thresh[0]:.2e}, {lno_thresh[1]:.2e})\n')
        f.write(f'Avg. Electrons:         ({nelec_avg[0]:.1f}, {nelec_avg[1]:.1f})\n')
        f.write(f'Avg. Orbitals:          ({norb_avg[0]:.1f}, {norb_avg[1]:.1f})\n')
        f.write(f'MP2 Correction:         {emp2_tot - e_mp2:12.8f}\n')

    return None