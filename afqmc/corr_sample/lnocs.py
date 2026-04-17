import pickle
import os
import numpy as np
import h5py
from jax import numpy as jnp
from ad_afqmc import pyscf_interface, hamiltonian, propagation, sampling, wavefunctions
from ad_afqmc.lno.afqmc import LNOAFQMC
from pyscf import mcscf
from functools import reduce
_fdot = np.dot
fdot = lambda *args: reduce(_fdot, args)


def prep_lnocs_files(options,mf,mo_coeff,lo_coeff,frozen=None,
                     option_file='options.bin',
                     mo_file='mo_coeff.npz',
                     chol_file='FCIDUMP_chol'):
    
    print("# Preparing CS-LNO-AFQMC calculation")
    
    # cs specific options
    options["n_runs"] = options.get("n_runs",200)
    options["rlx_steps"] = options.get("rlx_steps",3)
    options["prop_steps"] = options.get("prop_steps",20)

    options["dt"] = options.get("dt", 0.005)
    options["n_exp_terms"] = options.get("n_exp_terms",6)
    options["n_walkers"] = options.get("n_walkers", 50)
    options["n_prop_steps"] = options.get("n_prop_steps", 50)
    options["n_ene_blocks"] = options.get("n_ene_blocks", 50)
    options["n_sr_blocks"] = options.get("n_sr_blocks", 50)
    options["n_blocks"] = options.get("n_blocks", 50)
    options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))
    # options["n_eql"] = options.get("n_eql", 1)
    options["trial"] = options.get("trial", "rhf")
    options["walker_type"] = options.get("walker_type", "rhf")
    options["use_gpu"] = options.get("use_gpu", False)
    options["ad_mode"] = options.get("ad_mode", None)
    assert options["ad_mode"] in [None, "forward", "reverse", "2rdm"]
    options["orbital_rotation"] = options.get("orbital_rotation", True)
    options["do_sr"] = options.get("do_sr", False)
    options["symmetry"] = options.get("symmetry", False)
    options["save_walkers"] = options.get("save_walkers", False)
    options["free_projection"] = options.get("free_projection", False)
    options["n_batch"] = options.get("n_batch", 1)
    options["ene0"] = options.get("ene0", 0.0)
    
    with open(option_file, 'wb') as f:
        pickle.dump(options, f)

    mol = mf.mol
    maskocc = mf.mo_occ>1e-10
    nmo = mf.mo_occ.size
    # Convert frozen to 0 bc PySCF solvers do not support frozen=None or empty list
    if frozen is None:
        frozen = 0
    elif isinstance(frozen, (list,tuple,np.ndarray)) and len(frozen) == 0:
        frozen = 0
    if isinstance(frozen, (int,np.integer)):
        maskact = np.hstack([np.zeros(frozen,dtype=bool),
                                np.ones(nmo-frozen,dtype=bool)])
    elif isinstance(frozen, (list,tuple,np.ndarray)):
        maskact = np.array([i not in frozen for i in range(nmo)])
        
    orbfrzocc = mo_coeff[:,~maskact& maskocc]
    orbactocc = mo_coeff[:, maskact& maskocc]
    orbactvir = mo_coeff[:, maskact&~maskocc]
    orbfrzvir = mo_coeff[:,~maskact&~maskocc]
    _, nactocc, nactvir, _ = \
        [orb.shape[1] for orb in [orbfrzocc,orbactocc,orbactvir,orbfrzvir]]
    norb_act = (nactocc+nactvir)
    nelec_act = nactocc*2
    norb_frozen = frozen

    # calculate cholesky integrals
    print('# Generating Cholesky Integrals')
    print('# correlated sampling requires Density Fitting!')
    nao = mol.nao
    act_idx = [i for i in range(nao) if i not in norb_frozen]
    _, chol, _, _ = pyscf_interface.generate_integrals(
            mol,mf.get_hcore(),mo_coeff[:,act_idx],DFbas=mf.with_df.auxmol.basis)

    mc = mcscf.CASSCF(mf, norb_act, nelec_act) 
    mc.frozen = norb_frozen
    nelec = mc.nelecas
    mc.mo_coeff = mo_coeff
    h1e, enuc = mc.get_h1eff()

    nbasis = h1e.shape[-1]
    print("# Finished calculating Cholesky integrals\n")
    print('# Size of the correlation space:')
    print(f'# Number of electrons: {nelec}')
    print(f'# Number of basis functions: {nbasis}')
    print(f'# Number of Cholesky vectors: {chol.shape[0]}')
    print(f'# Active orbitals: {act_idx}')
    print(f'# frozen orbitals: {norb_frozen}')

    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * np.einsum('nik,njk->ij', chol, chol, optimize='optimal')
    h1e_mod = h1e - v0
    chol = chol.reshape((chol.shape[0], -1))

    # write mo coefficients
    trial_coeffs = np.empty((2, nbasis, nbasis))
    q = np.eye(mol.nao- len(norb_frozen))
    trial_coeffs[0] = q
    trial_coeffs[1] = q
    s1e = mf.get_ovlp()
    prjlo = fdot(lo_coeff.T, s1e, orbactocc)
    # np.savez(mo_file,mo_coeff=trial_coeffs)
    np.savez(mo_file,mo_coeff=trial_coeffs,prjlo=prjlo)
    pyscf_interface.write_dqmc(h1e,h1e_mod,chol,sum(nelec),nbasis,enuc,ms=mol.spin,
                               filename=chol_file,mo_coeffs=trial_coeffs)
    
    return None

from ad_afqmc import config

def prep_afqmc(options=None,
               option_file="options.bin",
               mo_file="mo_coeff.npz",
               amp_file="amplitudes.npz",
               chol_file="FCIDUMP_chol"):
    
    if options is None:
        with open(option_file, "rb") as f:
                options = pickle.load(f)

    if options["use_gpu"]:
        config.afqmc_config["use_gpu"] = True
    config.setup_jax()
    MPI = config.setup_comm()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:
        print(f"# Number of MPI ranks: {size}\n#")

    with h5py.File(chol_file, "r") as fh5:
        [nelec, nmo, ms, nchol] = fh5["header"]
        h0 = jnp.array(fh5.get("energy_core"))
        h1 = jnp.array(fh5.get("hcore")).reshape(nmo, nmo)
        chol = jnp.array(fh5.get("chol")).reshape(-1, nmo, nmo)

    assert type(ms) is np.int64
    assert type(nelec) is np.int64
    assert type(nmo) is np.int64
    assert type(nchol) is np.int64
    ms, nelec, nmo, nchol = int(ms), int(nelec), int(nmo), int(nchol)
    nelec_sp = ((nelec + abs(ms)) // 2, (nelec - abs(ms)) // 2)

    norb = nmo

    try:
        with h5py.File("observable.h5", "r") as fh5:
            [observable_constant] = fh5["constant"]
            observable_op = np.array(fh5.get("op")).reshape(nmo, nmo)
            if options["walker_type"] == "uhf":
                observable_op = jnp.array([observable_op, observable_op])
            observable = [observable_op, observable_constant]
    except:
        observable = None

    ham = hamiltonian.hamiltonian(nmo)
    ham_data = {}
    ham_data["h0"] = h0
    ham_data["h1"] = jnp.array([h1, h1])
    ham_data["chol"] = chol.reshape(nchol, -1)
    ham_data["ene0"] = options["ene0"]

    mo_coeff = jnp.array(np.load(mo_file)["mo_coeff"])
    prjlo = jnp.array(np.load(mo_file)["prjlo"])
    wave_data = {}
    wave_data["prjlo"] = prjlo
    wave_data["rdm1"] = jnp.array(
        [
            mo_coeff[0][:, : nelec_sp[0]] @ mo_coeff[0][:, : nelec_sp[0]].T,
            mo_coeff[1][:, : nelec_sp[1]] @ mo_coeff[1][:, : nelec_sp[1]].T,
        ]
    )

    if options["trial"] == "rhf":
        trial = wavefunctions.rhf(norb, nelec_sp, n_batch=options["n_batch"])
        wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0]]
    elif options["trial"] == "uhf":
        trial = wavefunctions.uhf(norb, nelec_sp, n_batch=options["n_batch"])
        wave_data["mo_coeff"] = [
            mo_coeff[0][:, : nelec_sp[0]],
            mo_coeff[1][:, : nelec_sp[1]],
        ]
    elif options["trial"] == "noci":
        with open("dets.pkl", "rb") as f:
            ci_coeffs_dets = pickle.load(f)
        ci_coeffs_dets = [
            jnp.array(ci_coeffs_dets[0]),
            [jnp.array(ci_coeffs_dets[1][0]), jnp.array(ci_coeffs_dets[1][1])],
        ]
        wave_data["ci_coeffs_dets"] = ci_coeffs_dets
        trial = wavefunctions.noci(
            norb, nelec_sp, ci_coeffs_dets[0].size, n_batch=options["n_batch"]
        )
    elif options["trial"] == "cisd":
        try:
            amplitudes = np.load(amp_file)
            ci1 = jnp.array(amplitudes["ci1"])
            ci2 = jnp.array(amplitudes["ci2"])
            trial_wave_data = {"ci1": ci1, "ci2": ci2}
            wave_data.update(trial_wave_data)
            trial = wavefunctions.cisd(norb, nelec_sp, n_batch=options["n_batch"])
        except:
            raise ValueError("Trial specified as cisd, but amplitudes.npz not found.")
    elif options["trial"] == "ucisd":
        try:
            amplitudes = np.load(amp_file)
            ci1a = jnp.array(amplitudes["ci1a"])
            ci1b = jnp.array(amplitudes["ci1b"])
            ci2aa = jnp.array(amplitudes["ci2aa"])
            ci2ab = jnp.array(amplitudes["ci2ab"])
            ci2bb = jnp.array(amplitudes["ci2bb"])
            trial_wave_data = {
                "ci1A": ci1a,
                "ci1B": ci1b,
                "ci2AA": ci2aa,
                "ci2AB": ci2ab,
                "ci2BB": ci2bb,
                "mo_coeff": mo_coeff,
            }
            wave_data.update(trial_wave_data)
            trial = wavefunctions.ucisd(norb, nelec_sp, n_batch=options["n_batch"])
        except:
            raise ValueError("Trial specified as ucisd, but amplitudes.npz not found.")
    else:
        try:
            with open("trial.pkl", "rb") as f:
                [trial, trial_wave_data] = pickle.load(f)
            wave_data.update(trial_wave_data)
            if rank == 0:
                print(f"# Read trial of type {type(trial).__name__} from trial.pkl.")
        except:
            if rank == 0:
                print(
                    "# trial.pkl not found, make sure to construct the trial separately."
                )
            trial = None

    if options["walker_type"] == "rhf":
        if options["symmetry"]:
            ham_data["mask"] = jnp.where(jnp.abs(ham_data["h1"]) > 1.0e-10, 1.0, 0.0)
        else:
            ham_data["mask"] = jnp.ones(ham_data["h1"].shape)
        #print(f'using {options["n_exp_terms"]} exp_terms')
        prop = propagation.propagator_restricted(
            options["dt"], 
            options["n_walkers"], 
            options["n_exp_terms"],
            options["n_batch"]
        )

    elif options["walker_type"] == "uhf":
        if options["symmetry"]:
            ham_data["mask"] = jnp.where(jnp.abs(ham_data["h1"]) > 1.0e-10, 1.0, 0.0)
        else:
            ham_data["mask"] = jnp.ones(ham_data["h1"].shape)
        if options["free_projection"]:
            prop = propagation.propagator_unrestricted(
                options["dt"],
                options["n_walkers"],
                10,
                n_batch=options["n_batch"],
            )
        else:
            prop = propagation.propagator_unrestricted(
                options["dt"],
                options["n_walkers"],
                n_batch=options["n_batch"],
            )

    sampler = sampling.sampler(
        options["n_prop_steps"],
        options["n_ene_blocks"],
        options["n_sr_blocks"],
        options["n_blocks"],
    )

    if rank == 0:
        print(f"# norb: {norb}")
        print(f"# nelec: {nelec_sp}")
        print("#")
        for op in options:
            if options[op] is not None:
                print(f"# {op}: {options[op]}")
        print("#")

    return ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI

# def run_lnocs_afqmc(nproc=None,use_gpu=False):
    
#     path = os.path.abspath(__file__)
#     dir_path = os.path.dirname(path)
#     script = f"{dir_path}/run_lnocs.py"
    
#     if use_gpu:
#         mpi_prefix = ""
#         gpu_flag = "--use_gpu"
#     else:
#         mpi_prefix = "mpirun "
#         gpu_flag = ""

#     if nproc is not None:
#         mpi_prefix += f"-np {nproc} "
#     os.system(
#         f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; {mpi_prefix} python {script} {gpu_flag} |tee lnocs_afqmc.out"
#     )

def procrustes_rotation(A,B):
    # occ Procrustes
    # min_R|AR-B|F the Frobenius norm
    # rotate A to match B
    m = B.T @ A
    u, _, v = np.linalg.svd(m)
    pcsts_r = v.T @ u.T
    Ap = A @ pcsts_r

    return Ap

def f_norm(matrix):
    # Procrustes rotation minimizes the Frobenius norm
    return np.sqrt(np.trace(matrix.T @ matrix))

def check_lo_occ(mfcc,orbloc):
    orbactocc = mfcc.split_mo()[1]
    s1e = mfcc._scf.get_ovlp()
    m = orbloc.T @ s1e @ orbactocc
    lospanerr = abs(m.T @ m - np.eye(m.shape[1])).max()
    if lospanerr > 1e-10:
        raise ValueError(
            'LOs do not fully span the occupied space! Max|<occ|LO><LO|occ>| = %e',
            lospanerr)
    return lospanerr < 1e-10

def run_cs_frags(mf1,mf2,frozen=None,options=None,lno_thresh=1e-5,
                 nproc=None,run_frg_list=None,mp2=True):

    print('# all differences are system2 - system1')
    d_e_mf = mf2.e_tot - mf1.e_tot
    print(f'mean-field energy difference: {d_e_mf}')

    no_type='ie'
    lo_type="pm"

    if isinstance(lno_thresh, list):
        thresh_occ, thresh_vir = lno_thresh
    else:
        thresh_occ = lno_thresh*10
        thresh_vir = lno_thresh

    mfcc1 = LNOAFQMC(mf1,frozen=frozen)
    mfcc2 = LNOAFQMC(mf2,frozen=frozen)

    eris1 = mfcc1.ao2mo()
    orbloc1 = mfcc1.get_lo(lo_type=lo_type)
    check_lo_occ(mfcc1,orbloc1)
    
    frag_lolist1 = [[i] for i in range(orbloc1.shape[1])]
    frag_nonvlist1 = mfcc1.frag_nonvlist
    nfrag = len(frag_lolist1)
    if frag_nonvlist1 is None: frag_nonvlist1 = [[None,None]] * nfrag

    eris2 = mfcc2.ao2mo()
    orbloc2 = mfcc2.get_lo(lo_type=lo_type)
    norm = f_norm(orbloc2-orbloc1)
    print(f'# Frobenius norm before Procrustes Rotation {norm}')
    print('# Performing Procrustes Rotation')
    orbloc2 = procrustes_rotation(orbloc2,orbloc1)
    check_lo_occ(mfcc2,orbloc2)
    norm_p = f_norm(orbloc2-orbloc1)
    print(f'# Frobenius norm after Procrustes Rotation {norm_p}')

    frag_lolist2 = [[i] for i in range(orbloc2.shape[1])]
    frag_nonvlist2 = mfcc2.frag_nonvlist
    nfrag2 = len(frag_lolist2)
    if frag_nonvlist2 is None: frag_nonvlist2 = [[None,None]] * nfrag2

    if nfrag != nfrag2: 
        raise ValueError("number of fragments are different in two system!")
    
    if run_frg_list is None:
        run_frg_list = range(nfrag)
    
    from jax import random
    seeds = random.randint(random.PRNGKey(options["seed"]),
                        shape=(len(run_frg_list),), minval=0, maxval=100*nfrag)

    from ad_afqmc.lno.base import lno
    for ifrag in run_frg_list:
        print(f'\n########### running fragment {ifrag+1} ##########')
        frag_target_nocc1, frag_target_nvir1 = frag_nonvlist1[ifrag]
        fraglo1 = frag_lolist1[ifrag]
        orbfragloc1 = orbloc1[:,fraglo1]
        frag_target_nocc2, frag_target_nvir2 = frag_nonvlist2[ifrag]
        fraglo2 = frag_lolist2[ifrag]
        orbfragloc2 = orbloc2[:,fraglo2]

        # make fpno
        THRESH_INTERNAL = 1e-10
        thresh_pno = [thresh_occ,thresh_vir]
        frozen_mask = mfcc1.get_frozen_mask()
        # frozen_mask2 = mfcc2.get_frozen_mask()
        
        frzfrag1, orbfrag1, can_orbfrag1 \
            = lno.make_fpno1(mfcc1,eris1,orbfragloc1,no_type,
                             THRESH_INTERNAL,thresh_pno,
                             frozen_mask=frozen_mask,
                             frag_target_nocc=frag_target_nocc1,
                             frag_target_nvir=frag_target_nvir1,
                             canonicalize=False)
        
        frzfrag2, orbfrag2, can_orbfrag2 \
            = lno.make_fpno1(mfcc2,eris2,orbfragloc2,no_type,
                             THRESH_INTERNAL, thresh_pno,
                             frozen_mask=frozen_mask,
                             frag_target_nocc=frag_target_nocc2,
                             frag_target_nvir=frag_target_nvir2,
                             canonicalize=False)
        
        #take the larger active space
        if len(frzfrag1) > len(frzfrag2):
            frzfrag = frzfrag2
        else:
            frzfrag = frzfrag1
        
        options["seed"] = seeds[ifrag]
        prep_lnocs_files(options,mf1,orbfrag1,orbfragloc1,frozen=frzfrag,
                  option_file='options.bin',mo_file='mo1.npz',chol_file='chol1')
        prep_lnocs_files(options,mf2,orbfrag2,orbfragloc2,frozen=frzfrag,
                  option_file='options.bin',mo_file='mo2.npz',chol_file='chol2')
        
        use_gpu = options["use_gpu"]
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        script = f"{dir_path}/run_lnocs.py"
        if use_gpu:
            mpi_prefix = ""
            gpu_flag = "--use_gpu"
            nproc = None
            from mpi4py import MPI
            if not MPI.Is_finalized():
                MPI.Finalize()
        else:
            mpi_prefix = "mpirun "
            gpu_flag = ""
            if nproc is not None:
                mpi_prefix += f"-np {nproc} "
        os.system(
            f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1;"
            f"{mpi_prefix} python {script} {gpu_flag} |tee cs_frg{ifrag+1}.out"
        )

        if mp2:
            print('# running fragment MP2')
            from ad_afqmc.lno_ccsd import lno_ccsd
            emp2_1 = lno_ccsd.get_mp2_frg_e(
                mf1,frzfrag,eris1,orbfragloc1,can_orbfrag1)
            emp2_2 = lno_ccsd.get_mp2_frg_e(
                mf2,frzfrag,eris2,orbfragloc2,can_orbfrag2)
            print(f'# cs-lno-fragment 1 mp2 energy: {emp2_1:.8f}')
            print(f'# cs-lno-fragment 2 mp2 energy: {emp2_2:.8f}')
            with open(f'cs_frg{ifrag+1}.out', 'a') as out_file:
                print(f"# lno-mp2 orb_corr1: {emp2_1:.6f}",file=out_file)
                print(f"# lno-mp2 orb_corr2: {emp2_2:.6f}",file=out_file)
    
    if mp2:
        from pyscf import mp
        mmp = mp.MP2(mf1, frozen=frozen)
        emp2_tot1 = mmp.kernel()[0]
        mmp = mp.MP2(mf2, frozen=frozen)
        emp2_tot2 = mmp.kernel()[0]

    write_cslno_results(run_frg_list,mf1,mf2,emp2_tot1,emp2_tot2)

    return None

def write_cslno_results(frags,mf1,mf2,emp21=None,emp22=None):
    with open('results.out', 'w') as out_file:
        print('# frag' \
                '  mp2_orb_en1 afqmc_orb_en1  abs_err1' \
                '  mp2_orb_en2 afqmc_orb_en2  abs_err2' \
                '  d_orb_energy  cs_err  runtime',file=out_file)
        for ifrag in frags:
            with open(f"cs_frg{ifrag+1}.out","r") as read_file:
                for line in read_file:
                    if "orbital energy 1" in line:
                        orb_energy1 = line.split()[4]
                        orb_err1 = line.split()[6]
                    if "orbital energy 2" in line:
                        orb_energy2 = line.split()[4]
                        orb_err2 = line.split()[6]
                    if "correlated d_energy" in line:
                        d_orb_en = line.split()[3]
                        cs_err = line.split()[5]
                    if "lno-mp2 orb_corr1" in line:
                        orb_mp21 = line.split()[3]
                    if "lno-mp2 orb_corr2" in line:
                        orb_mp22 = line.split()[3]
                    if "total run time" in line:
                        runtime = line.split()[4]
                print(f'{ifrag+1:3d}  '
                      f'{orb_mp21}  {orb_energy1}  {orb_err1}  '
                      f'{orb_mp22}  {orb_energy2}  {orb_err2}  '
                      f'{d_orb_en}  {cs_err}  {runtime}', file=out_file)
                
    data = []
    with open('results.out', 'r') as read_file:
        for line in read_file:
            if not line.startswith("#"):
                data = np.hstack((data,line.split()))
                data[data == 'None'] = '0.000000' 

    data = np.array(data.reshape(len(frags),10))
    orb_mp21 = np.array(data[:,1],dtype='float32')
    orb_en1 = np.array(data[:,2],dtype='float32')
    orb_err1 = np.array(data[:,3],dtype='float32')
    orb_mp22 = np.array(data[:,4],dtype='float32')
    orb_en2 = np.array(data[:,5],dtype='float32')
    orb_err2 = np.array(data[:,6],dtype='float32')
    d_orb_en = np.array(data[:,7],dtype='float32')
    cs_err = np.array(data[:,8],dtype='float32')
    orb_time = np.array(data[:,9],dtype='float32')

    tot_corr1 = sum(orb_en1)
    tot_err1 = np.sqrt(sum(orb_err1**2))
    tot_corr2 = sum(orb_en2)
    tot_err2 = np.sqrt(sum(orb_err2**2))
    d_tot_corr = sum(d_orb_en)
    cs_tot_err = np.sqrt(sum(cs_err**2))
    tot_mp21 = sum(orb_mp21)
    tot_mp22 = sum(orb_mp22)
    tot_time = sum(orb_time)

    mp2_cr1 = emp21-tot_mp21
    mp2_cr2 = emp22-tot_mp22
    d_mp2_cr = mp2_cr2-mp2_cr1

    d_e_mf = mf2.e_tot-mf1.e_tot

    with open('results.out', 'a') as out_file:
        out_file.write(f'# final results \n')
        out_file.write(f'# system 1 mf energy: {mf1.e_tot:.10f} \n')
        out_file.write(f'# system 2 mf energy: {mf2.e_tot:.10f} \n')
        out_file.write(f'# mean-field energy difference: {d_e_mf:.10f} \n')
        out_file.write(f'# system 1 lno-afqmc-corr: {tot_corr1:.6f} +/- {tot_err1:.6f} \n')
        out_file.write(f'# system 2 lno-afqmc-corr: {tot_corr2:.6f} +/- {tot_err2:.6f} \n')
        out_file.write(f'# cs-system diff_corr: {d_tot_corr:.6f} +/- {cs_tot_err:.6f} \n')
        out_file.write(f'# mp2 energy difference correction: {d_mp2_cr:.10f} \n')
        out_file.write(f'# totol cs-system energy difference: '
                       f'{d_e_mf+d_tot_corr+d_mp2_cr:.6f} +/- {cs_tot_err:.6f} \n')
        out_file.write(f'# total time: {tot_time:.2f} \n')
    
    return None