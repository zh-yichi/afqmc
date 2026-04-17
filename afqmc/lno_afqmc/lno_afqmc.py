import os, h5py, pickle
import numpy as np
import jax
from jax import numpy as jnp
from jax import random
import opt_einsum as oe
from pyscf.cc.ccsd import CCSD
from pyscf import lib, df, mp
from pyscf.lno import lnoccsd
from ad_afqmc import config, pyscf_interface
from functools import partial
from ad_afqmc.lno_afqmc import propagation, sampling
from ad_afqmc.lno_afqmc import wavefunctions_restricted as lno_wavefunctions
from collections.abc import Iterable
import time
print = partial(print, flush=True)

def lno_ccsd(mcc, mo_coeff, uocc_loc, mo_occ, maskact):

    maskocc = mo_occ>1e-10
    nmo = mo_occ.size

    orbfrzocc = mo_coeff[:,~maskact &  maskocc] 
    orbactocc = mo_coeff[:, maskact &  maskocc]
    orbactvir = mo_coeff[:, maskact & ~maskocc]
    orbfrzvir = mo_coeff[:,~maskact & ~maskocc]
    nfrzocc, nactocc, nactvir, nfrzvir = [orb.shape[1]
                                          for orb in [orbfrzocc,orbactocc,
                                                      orbactvir,orbfrzvir]]
    nlo = uocc_loc.shape[1]
    nactmo = nactocc + nactvir

    if nactocc == 0 or nactvir == 0:
        elcorr_pt2 = elcorr_cc = lib.tag_array(0., spin_comp=np.array((0., 0.)))
        elcorr_cc_t = 0.
    else:
        # solve impurity problem
        imp_eris = mcc.ao2mo()
        if isinstance(imp_eris.ovov, np.ndarray):
            ovov = imp_eris.ovov
        else:
            ovov = imp_eris.ovov[()]
        oovv = ovov.reshape(nactocc,nactvir,nactocc,nactvir).transpose(0,2,1,3)
        ovov = None
        
        # MP2 fragment energy
        t1, t2 = mcc.init_amps(eris=imp_eris)[1:]
        elcorr_pt2 = lnoccsd.get_fragment_energy(oovv, t2, uocc_loc).real

        # CCSD fragment energy
        t1, t2 = mcc.kernel(eris=imp_eris, t1=t1, t2=t2)[1:]
        if not mcc.converged:
            print('# Impurity CCSD did not converge!')

        t2 += lib.einsum('ia,jb->ijab',t1,t1)
        elcorr_cc = lnoccsd.get_fragment_energy(oovv, t2, uocc_loc)
        # t1t1 = lib.einsum('ia,jb->ijab',t1,t1)
        # elcorr_t1 = lnoccsd.get_fragment_energy(oovv, t1t1, uocc_loc)
        # CCSD(T) fragment energy
        # if ccsd_t:
        #     from pyscf.lno.lnoccsd_t import kernel as CCSD_T
        #     t2 -= lib.einsum('ia,jb->ijab',t1,t1)   # restore t2
        #     elcorr_cc_t = CCSD_T(mcc, imp_eris, uocc_loc, t1=t1, t2=t2)
        # else:
        #     elcorr_cc_t = 0.
        #     t2 -= lib.einsum('ia,jb->ijab',t1,t1)

    oovv = imp_eris = mcc = None

    return (elcorr_pt2, elcorr_cc), t1, t2 #, elcorr_t1

def get_veff(mf, dm):
    mol = mf.mol
    print('# Building JK matrix')
    # t0 = time.perf_counter()
    vj, vk = mf.get_jk(mol, dm, hermi=1)
    # t1 = time.perf_counter()
    # print(f"# build JK time: {t1 - t0:.6f} s")
    return 2*vj - vk


@jax.jit
def jk_from_cderi(cderi, dm):
    cderi_dm = oe.contract('gik,kj->gij', cderi, dm, backend='jax')
    vj = oe.contract('gkk,gij->ij', cderi_dm, cderi, backend='jax')
    vk = oe.contract('gik,gkj->ij', cderi_dm, cderi, backend='jax')
    return vj, vk

def get_veff2(mf, dm):
    '''use opt einsum on gpu'''
    dm = jnp.array(dm)
    vj = jnp.zeros(dm.shape)
    vk = jnp.zeros(dm.shape)
    print('# Building JK matrix')
    for i,cderi in enumerate(mf.with_df.loop()):
        print(f'# DF loop {i} number of DF vectors {cderi.shape[0]}')
        cderi = jnp.array(lib.unpack_tril(cderi, axis=-1))
        # cderi = jnp.array(cderi)
        # cderi_dm = oe.contract('gik,kj->gij', cderi, dm, backend='jax')
        # vj += oe.contract('gkk,gij->ij', cderi_dm, cderi, backend='jax')
        # vk += oe.contract('gik,gkj->ij', cderi_dm, cderi, backend='jax')
        dvj, dvk = jk_from_cderi(cderi, dm)
        vj += dvj
        vk += dvk
    # vj, vk = mf.get_jk(mol, dm, hermi=1)
    return 2*vj - vk

def h1e_ras(mf, mo_coeff, ncas, ncore):
    '''
    effective one-electron integral for restricted active space
    ncas = nact_electron/2
    ncore = ncore_electrons/2
    '''
    # note casci undo DF

    mo_core = jnp.array(mo_coeff[:,:ncore])
    mo_cas = jnp.array(mo_coeff[:,ncore:ncore+ncas])

    hcore = jnp.array(mf.get_hcore())
    energy_core = mf.energy_nuc()
    if mo_core.size == 0:
        corevhf = 0.
    else:
        # core_dm = np.dot(mo_core, mo_core.T)
        core_dm = mo_core @ mo_core.T
        time0 = time.perf_counter()
        corevhf = get_veff2(mf, core_dm)
        time1 = time.perf_counter()
        print(f"# build JK time: {time1 - time0:.6f} s")
        energy_core += 2 * oe.contract('ij,ji', core_dm, hcore, backend='jax')
        energy_core += oe.contract('ij,ji', core_dm, corevhf, backend='jax')
        time2 = time.perf_counter()
        print(f"# build ecore time: {time2 - time1:.6f} s")
    h1eff = mo_cas.T @ (hcore+corevhf) @ mo_cas
    time3 = time.perf_counter()
    print(f"# build h1eff time: {time3 - time0:.6f} s")
    return h1eff, energy_core

def prep_afqmc(mf_cc,mo_coeff,t1,t2,frozen,prjlo,
               options,chol_cut=1e-5,
               option_file='options.bin',
               mo_file="mo_coeff.npz",
               amp_file="amplitudes.npz",
               chol_file="FCIDUMP_chol"):
    
    jax.config.update("jax_enable_x64", True)
    
    with open(option_file, 'wb') as f:
        pickle.dump(options, f)
    
    if isinstance(mf_cc, CCSD):
        mf = mf_cc._scf
    else:
        mf = mf_cc

    t2 = t2.transpose(0, 2, 1, 3)
    t1 = np.array(t1)
    np.savez(amp_file,t1=t1,t2=t2)

    print('# Calculating Effective Active Space One-electron Integrals')
    mol = mf.mol
    nocc = np.count_nonzero(mf.mo_occ)
    actfrag = np.array([i for i in range(mol.nao) if i not in frozen])
    frzocc = np.array([i for i in range(nocc) if i in frozen])
    actocc = np.array([i for i in range(nocc) if i in actfrag])
    actvir = np.array([i for i in range(nocc,mol.nao) if i in actfrag])
    nfrzocc = len(frzocc)
    nactocc = len(actocc)
    nactvir = len(actvir)
    nactorb = len(actfrag)
    # print(f'# number of forzen occupied orbitals {nfrzocc}')
    print(f'# number of active occupied orbitals {nactocc}')
    print(f'# number of active virtual orbitals {nactvir}')

    ncas = nactorb
    ncore = nfrzocc
    nelec = nactocc*2
    h1e, enuc = h1e_ras(mf, mo_coeff, ncas, ncore)
    mo_act = mo_coeff[:,actfrag]

    print('# Generating Cholesky Integrals')

    if getattr(mf, "with_df", None) is not None:
        # decompose eri in MO to achieve linear scale over the Auxiliary-field
        print("# Composing AO ERIs from DF basis")
        from pyscf.ao2mo import _ao2mo

        naux = mf.with_df.get_naoaux()
        chol_df = np.zeros((naux,ncas*(ncas+1)//2))
        ijslice = (0, ncas, 0, ncas)
        Lpq = None
        p1 = 0
        # print('# test new algorithm!!!!!!!')
        time0 = time.perf_counter()
        for eri1 in mf.with_df.loop():
            Lpq = _ao2mo.nr_e2(eri1, mo_act, ijslice, aosym='s2', out=Lpq).reshape(-1,ncas,ncas)
            p0, p1 = p1, p1 + Lpq.shape[0]
            # print(eri1.shape)
            # print(Lpq.shape)
            chol_df[p0:p1] = lib.pack_tril(Lpq, axis=-1) # in mo representation
        print(f"# packed chol tensor by DF shape: {chol_df.shape}")
        # chol_df = jnp.array(chol_df)

        # chol_df = df.incore.cholesky_eri(mol, mf.with_df.auxmol.basis) # in ao 
        # chol_df = lib.unpack_tril(chol_df).reshape(chol_df.shape[0], -1)
        # chol_df = chol_df.reshape((-1, mol.nao, mol.nao))
        # chol_df = lib.einsum('pr,grs,sq->gpq',mo_act.T,chol_df,mo_act)
        # eri_df = lib.einsum('gP,gQ->PQ', chol_df, chol_df, optimize='optimal')
        eri_df = oe.contract('gP,gQ->PQ', chol_df, chol_df, backend='jax')
        time1 = time.perf_counter()
        print("# Composing active space MO ERIs from AO ERIs")
        # eri_df = lib.pack_tril(eri_df,axis=0) # pyscf.lib pack the lower triangular
        # eri_df = lib.pack_tril(eri_df,axis=-1)
        # eri_df = eri_df.reshape(ncas**2,ncas**2)
        print("# Decomposing MO ERIs to Cholesky vectors")
        print(f"# Cholesky cutoff is: {chol_cut}")
        chol = pyscf_interface.modified_cholesky(eri_df,max_error=chol_cut)
        chol = lib.unpack_tril(chol,axis=-1)
        chol = chol.reshape(-1,ncas,ncas)
        time2 = time.perf_counter()
        print(f"# build 2-electron integral time: {time1 - time0:.6f} s")
        print(f"# decompose 2-electron integral to CD time: {time2 - time1:.6f} s")
        print(f"# total 2-electron integral time: {time2 - time0:.6f} s")
    else:
        raise  NotImplementedError('Use DF Only!')

    print("# Finished calculating Cholesky integrals")
    print('# Size of the correlation space')
    print(f'# Number of electrons: ({nactocc},{nactocc})')
    print(f'# Number of basis functions: {ncas}')
    print(f'# Cholesky shape: {chol.shape}')

    v0 = 0.5 * oe.contract("gpr,grq->pq", chol, chol, backend="jax")
    h1e_mod = h1e - v0
    chol = chol.reshape((chol.shape[0], -1))
    np.savez(mo_file,prjlo=prjlo)

    write_dqmc(
        h1e,
        h1e_mod,
        chol,
        nelec,
        ncas,
        enuc,
        mf.e_tot,
        filename=chol_file,
    )

    return nelec, ncas

def write_dqmc(
    hcore,
    hcore_mod,
    chol,
    nelec,
    nmo,
    enuc,
    emf,
    filename="FCIDUMP_chol",
):
    hcore = np.array(hcore)
    hcore_mod = np.array(hcore_mod)
    chol = np.array(chol)
    with h5py.File(filename, "w") as fh5:
        fh5["header"] = np.array([nelec, nmo, chol.shape[0]])
        fh5["hcore"] = hcore.flatten()
        fh5["hcore_mod"] = hcore_mod.flatten()
        fh5["chol"] = chol.flatten()
        fh5["energy_core"] = enuc
        fh5["emf"] = emf


def _prep_afqmc(option_file="options.bin",
                mo_file="mo_coeff.npz",
                amp_file="amplitudes.npz",
                chol_file="FCIDUMP_chol"):
    
    jax.config.update("jax_enable_x64", True)
    
    try:
        with open(option_file, "rb") as f:
            options = pickle.load(f)
    except:
        print('# Using default options')
        options = {}

    options["dt"] = options.get("dt", 0.005)
    options["n_exp_terms"] = options.get("n_exp_terms",6)
    options["n_walkers"] = options.get("n_walkers", 50)
    options["n_prop_steps"] = options.get("n_prop_steps", 50)
    options["n_ene_blocks"] = options.get("n_ene_blocks", 10)
    options["n_sr_blocks"] = options.get("n_sr_blocks", 5)
    options["n_blocks"] = options.get("n_blocks", 50)
    options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))
    options["n_eql"] = options.get("n_eql", 3)
    options["walker_type"] = options.get("walker_type", "rhf")
    options["trial"] = options.get("trial", None)
    options["ene0"] = options.get("ene0", 0.0)
    options["n_batch"] = options.get("n_batch", 1)

    with h5py.File(chol_file, "r") as fh5:
        [nelec, nmo, nchol] = fh5["header"]
        h0 = jnp.array(fh5.get("energy_core"))
        emf = jnp.array(fh5.get("emf"))
        h1 = jnp.array(fh5.get("hcore")).reshape(nmo, nmo)
        chol = jnp.array(fh5.get("chol")).reshape(-1, nmo, nmo)
        h1_mod = jnp.array(fh5.get("hcore_mod")).reshape(nmo, nmo)

    assert type(nelec) is np.int64
    assert type(nmo) is np.int64
    assert type(nchol) is np.int64
    nelec, nmo, nchol = int(nelec), int(nmo), int(nchol)
    nelec_sp = (nelec // 2, nelec // 2)
    norb = nmo
    # ham = hamiltonian.hamiltonian(nmo)
    ham_data = {}
    ham_data["h0"] = h0
    ham_data["E0"] = emf
    ham_data["ene0"] = options["ene0"]

    ham_data["h1"] = jnp.array([h1, h1])
    ham_data["h1_mod"] = jnp.array(h1_mod)
    nchol = chol.shape[0]
    ham_data["chol"] = jnp.array(chol.reshape(chol.shape[0], -1))

    wave_data = {}
    wave_data['prjlo'] = jnp.array(np.load(mo_file)["prjlo"])
    mo_coeff = jnp.array(np.eye(nmo))

    if options["trial"] == "rhf":
        trial = lno_wavefunctions.rhf(norb, nelec_sp, n_batch=options["n_batch"])
        wave_data["mo_coeff"] = mo_coeff[:, : nelec_sp[0]]
    elif options["trial"] == "ccsd_pt_ad":
        trial = lno_wavefunctions.ccsd_pt_ad(norb, nelec_sp, n_batch=options["n_batch"])
        amplitudes = np.load(amp_file)
        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])
        prj = wave_data['prjlo']
        wave_data["t1"] = oe.contract('ia,ik->ka',t1, prj, backend='jax')
        wave_data["t2"] = oe.contract('iajb,ik->kajb',t2, prj, backend='jax')
    elif options["trial"] == "ccsd_pt":
        trial = lno_wavefunctions.ccsd_pt(norb, nelec_sp, n_batch=options["n_batch"])
        amplitudes = np.load(amp_file)
        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])
        wave_data["t1"] = oe.contract('ia,ik->ka',t1,wave_data['prjlo'])
        wave_data["t2"] = oe.contract('iajb,ik->kajb',t2,wave_data['prjlo'])
        wave_data["mo_coeff"] = mo_coeff[:, :nocc]
    elif "ccsd_pt2" in options["trial"]:
        from jax import scipy as jsp
        nocc = nelec_sp[0]
        amplitudes = np.load(amp_file)
        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])
        t1_full = np.zeros((norb, norb))
        t1_full[:nocc, nocc:] = t1
        wave_data['exp_t1'] = jsp.linalg.expm(t1_full)
        wave_data['exp_mt1'] = jsp.linalg.expm(-t1_full)
        wave_data["t2"] = oe.contract('iajb,ik->kajb',t2, wave_data['prjlo'], backend='jax')
        wave_data["mo_coeff"] = mo_coeff[:, :nocc]
        # print(t1.shape)
        # print(chol.shape)
        lt1 = oe.contract('ia,gja->gij', t1, chol[:, :nocc, nocc:], backend='jax')
        e0t1orb = 2 * oe.contract('gik,ik,gjj->',lt1, wave_data['prjlo'], lt1, backend='jax') \
                    - oe.contract('gij,gjk,ik->',lt1, lt1, wave_data['prjlo'], backend='jax')
        ham_data['e0t1orb'] = e0t1orb
        trial = lno_wavefunctions.ccsd_pt2(norb, nelec_sp, n_batch = options["n_batch"])
        if "fast" in options["trial"]:
            trial = lno_wavefunctions.ccsd_pt2_fast(norb, nelec_sp, n_batch = options["n_batch"])
        if "ad" in options["trial"]:
            trial = lno_wavefunctions.ccsd_pt2_ad(norb, nelec_sp, n_batch = options["n_batch"])
        
    if options["walker_type"] == "rhf":
        prop = propagation.propagator_restricted(
            options["dt"], 
            options["n_walkers"], 
            options["n_exp_terms"],
            options["n_batch"]
        )

    if  'pt' in options['trial']:
        if '2' in options['trial']:
            sampler = sampling.sampler_pt2(
                options["n_prop_steps"],
                options["n_ene_blocks"],
                options["n_sr_blocks"],
                options["n_blocks"],
                nchol,)
        else:
            sampler = sampling.sampler_pt(
                options["n_prop_steps"],
                options["n_ene_blocks"],
                options["n_sr_blocks"],
                options["n_blocks"],
                nchol,)
    else:
        sampler = sampling.sampler(
                options["n_prop_steps"],
                options["n_ene_blocks"],
                options["n_sr_blocks"],
                options["n_blocks"],
                nchol,)

    return ham_data, prop, trial, wave_data, sampler, options

import os
def run_lnoafqmc(options, option_file='options.bin'):
    jax.config.update("jax_enable_x64", True)
    
    with open(option_file, 'wb') as f:
        pickle.dump(options, f)

    if options["use_gpu"]:
        print(f'# running AFQMC on GPU')
        config.afqmc_config = {"use_gpu": True}
        config.setup_jax()
        gpu_flag = "--use_gpu"
        # mpi_prefix = ""
    else:
        print(f'# running AFQMC on CPU')
        gpu_flag = ""
        # mpi_prefix = "mpirun "
        # if nproc is not None:
        #     mpi_prefix += f"-np {nproc} "
    # if  'cc' in options['trial'] and 'pt' in options['trial']:
    if 'pt2' in options['trial']:
        script='ccsd_pt2/run_afqmc_nompi.py'

    else:
        raise NotImplementedError("Only support CCSD_pt and CCSD_pt2 trial.")
    
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/{script}"
    print(f'# AFQMC script: {script}')
    
    os.system(
        # f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1;"
        f" python {script} {gpu_flag} |tee afqmc.out"
    )

def run_afqmc(mf,
              options, 
              lo_coeff = None, 
              lo_coeff_file = 'lo_coeff.npz',
              nfrozen = 0,
              thresh = 1e-6, 
              chol_cut = 1e-5,
              run_frg_list = None, 
              emp2_tot = None,
              ccsd_verbose = 0,
              ):
    
    if lo_coeff is None:
        try:
            lo_coeff = np.load(lo_coeff_file)["lo_coeff"]
        except:
            raise ValueError(
                f"lo_coeff was not provided and could not be loaded "
                f"from file '{lo_coeff_file}'"
                )
        
    frag_lolist = [[i] for i in range(lo_coeff.shape[1])]
    nfrag_tot = len(frag_lolist)
    
    mlno = lnoccsd.LNOCCSD(mf, lo_coeff, frag_lolist, frozen=nfrozen).set(verbose=ccsd_verbose)
    mlno.lno_thresh = [thresh*10, thresh]
    mlno.lo_proj_thresh = 1e-10
    mlno.lo_proj_thresh_active = thresh #0.1
    lno_thresh = mlno.lno_thresh
    lno_type = ['1h','1h'] # if lno_type is None else lno_type
    # lno_thresh = [1e-5, 1e-6] if lno_thresh is None else lno_thresh
    lno_pct_occ = None
    lno_norb = None
    eris = None

    if run_frg_list is None:
        nfrag = len(frag_lolist)
        run_frg_list = range(nfrag)
    
    frag_lolist = [frag_lolist[i] for i in run_frg_list]
    nfrag = len(frag_lolist)
    if lno_pct_occ is None:
        lno_pct_occ = [None, None]
    if lno_norb is None:
        lno_norb = [[None,None]] * nfrag
    mf = mlno._scf

    if eris is None: eris = mlno.ao2mo()

    seeds = random.randint(random.PRNGKey(options["seed"]),
                        shape=(nfrag,), minval=0, maxval=100*nfrag)
    options["max_error"] = options["max_error"]/np.sqrt(nfrag)

    nelec_list = [None] * nfrag
    norb_list = [None] * nfrag
    eorb_mp2_cc = [None] * nfrag
    # Loop over fragment
    for ifrag,loidx in enumerate(frag_lolist):
        print(f'\n ########### RUNNING LNO-FRAGMENT {run_frg_list[ifrag]+1}/{nfrag_tot} ###########')
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

        lno_coeff, lno_frozen, uocc_loc, _ = mlno.make_las(eris, orbloc, lno_type, lno_param)
        print(f'# frozen LNO orbitals {lno_frozen}, nfrozen = {len(lno_frozen)}')
        mo_occ = mlno.mo_occ
        lno_frozen, maskact = lnoccsd.get_maskact(lno_frozen, mo_occ.size)
        # print(lno_frozen)
        mcc = lnoccsd.CCSD(mf, mo_coeff=lno_coeff, frozen=lno_frozen).set(verbose=ccsd_verbose)
        mcc._s1e = mlno._s1e
        mcc._h1e = mlno._h1e
        mcc._vhf = mlno._vhf
        if mlno.kwargs_imp is not None:
            mcc = mcc.set(**mlno.kwargs_imp)
        time0 = time.perf_counter()
        eorb_mp2_cc[ifrag], t1, t2 =\
            lno_ccsd(mcc, lno_coeff, uocc_loc, mo_occ, maskact)
        time1 = time.perf_counter()
        print(f"# CCSD time: {time1 - time0:.6f} s")
        
        prjlo = uocc_loc @ uocc_loc.T.conj()

        print(f'# LNO-MP2 Orbital Energy: {eorb_mp2_cc[ifrag][0]:.8f}')
        print(f'# LNO-CCSD Orbital Energy: {eorb_mp2_cc[ifrag][1]:.8f}')

        options["seed"] = seeds[ifrag]
        nelec_list[ifrag], norb_list[ifrag] \
            = prep_afqmc(mf,lno_coeff,t1,t2,lno_frozen,prjlo,
                         options,chol_cut=chol_cut)
        run_lnoafqmc(options)
        os.system(f'mv afqmc.out lnoafqmc.out{run_frg_list[ifrag]+1}')

    # finish lno loop
    # if emp2_tot is None:
        # mmp = mp.MP2(mf, frozen=nfrozen)
        # emp2_tot = mmp.kernel()[0]

    eorb_pt = np.zeros(nfrag,dtype='float64')
    eorb_pt_err = np.zeros(nfrag,dtype='float64')
    run_time = np.zeros(nfrag,dtype='float64')

    for n, i in enumerate(run_frg_list):
        with open(f"lnoafqmc.out{i+1}", "r") as readfile:
            for line in readfile:
                # if "Ept (direct observation):" in line:
                if "AFQMC/pt2CCSD energy (Mahalanobis):" in line:
                    eorb_pt[n] = float(line.split()[-3])
                    eorb_pt_err[n] = float(line.split()[-1])
                if "total run time" in line:
                    run_time[n] = float(line.split()[-1])

    nelec_list = np.array(nelec_list)
    norb_list = np.array(norb_list)
    eorb_mp2_cc = np.array(eorb_mp2_cc)
    nelec = np.mean(nelec_list)
    norb = np.mean(norb_list)
    e_mp2 = sum(eorb_mp2_cc[:,0])
    e_ccsd = sum(eorb_mp2_cc[:,1])
    e_afqmc_pt = sum(eorb_pt)
    e_afqmc_pt_err = np.sqrt(sum(eorb_pt_err**2))
    tot_time = sum(run_time)

    with open(f'lno_result.out', 'w') as out_file:
        print('# frag  eorb_mp2  eorb_ccsd  eorb_afqmc/ccsd_pt  nelec  norb  time',
                file=out_file)
        for n, i in enumerate(run_frg_list):
            print(f'{i+1:3d}  '
                  f'{eorb_mp2_cc[n,0]:.8f}  {eorb_mp2_cc[n,1]:.8f}  '
                  f'{eorb_pt[n]:.6f} +/- {eorb_pt_err[n]:.6f}  '
                  f'{nelec_list[n]}  {norb_list[n]}  {run_time[n]:.2f}', file=out_file)
            
        print(f'# LNO Thresh: ({lno_thresh[0]:.2e},{lno_thresh[1]:.2e})',file=out_file)
        print(f'# LNO Average Number of Electrons: {nelec:.1f}',file=out_file)
        print(f'# LNO Average Number of Basis: {norb:.1f}',file=out_file)
        print(f'# LNO-MP2 Energy: {e_mp2:.8f}',file=out_file)
        print(f'# LNO-CCSD Energy: {e_ccsd:.8f}',file=out_file)
        # print(f'# LNO-CCSD(T) Energy: {e_ccsd_pt:.8f}',file=out_file)
        print(f'# LNO-AFQMC/CCSD_PT Energy: {e_afqmc_pt:.6f} +/- {e_afqmc_pt_err:.6f}',file=out_file)
        if emp2_tot is not None:
            print(f'# MP2 Correction: {emp2_tot-e_mp2:.8f}',file=out_file)
        print(f"# total run time: {tot_time:.2f}",file=out_file)

    return None