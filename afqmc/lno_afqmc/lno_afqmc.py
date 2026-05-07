import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from jax import random
from pyscf.lno import lnoccsd

from afqmc import config
from afqmc.lno_afqmc import prep
from afqmc.lno_afqmc import mod_lnoccsd

from functools import partial
import time, gc, pickle

print = partial(print, flush=True)

# def lno_ccsd(mcc, mo_coeff, uocc_loc, mo_occ, maskact):

#     maskocc = mo_occ>1e-10
#     nmo = mo_occ.size

#     orbfrzocc = mo_coeff[:,~maskact &  maskocc] 
#     orbactocc = mo_coeff[:, maskact &  maskocc]
#     orbactvir = mo_coeff[:, maskact & ~maskocc]
#     orbfrzvir = mo_coeff[:,~maskact & ~maskocc]
#     nfrzocc, nactocc, nactvir, nfrzvir = [orb.shape[1]
#                                           for orb in [orbfrzocc,orbactocc,
#                                                       orbactvir,orbfrzvir]]
#     nlo = uocc_loc.shape[1]
#     nactmo = nactocc + nactvir

#     if nactocc == 0 or nactvir == 0:
#         elcorr_pt2 = elcorr_cc = lib.tag_array(0., spin_comp=np.array((0., 0.)))
#         elcorr_cc_t = 0.
#     else:
#         # solve impurity problem
#         imp_eris = mcc.ao2mo()
#         if isinstance(imp_eris.ovov, np.ndarray):
#             ovov = imp_eris.ovov
#         else:
#             ovov = imp_eris.ovov[()]
#         oovv = ovov.reshape(nactocc,nactvir,nactocc,nactvir).transpose(0,2,1,3)
#         ovov = None
        
#         # MP2 fragment energy
#         t1, t2 = mcc.init_amps(eris=imp_eris)[1:]
#         elcorr_pt2 = lnoccsd.get_fragment_energy(oovv, t2, uocc_loc).real

#         # CCSD fragment energy
#         t1, t2 = mcc.kernel(eris=imp_eris, t1=t1, t2=t2)[1:]
#         if not mcc.converged:
#             print('# Impurity CCSD did not converge!')

#         t2 += lib.einsum('ia,jb->ijab',t1,t1)
#         elcorr_cc = lnoccsd.get_fragment_energy(oovv, t2, uocc_loc)

#     oovv = imp_eris = mcc = None

#     return (elcorr_pt2, elcorr_cc), t1, t2 #, elcorr_t1

# def get_veff(mf, dm):
#     mol = mf.mol
#     print('Building JK matrix')
#     vj, vk = mf.get_jk(mol, dm, hermi=1)
#     return 2*vj - vk


# @jax.jit
# def jk_from_cderi(cderi, dm):
#     cderi_dm = oe.contract('gik,kj->gij', cderi, dm, backend='jax')
#     vj = oe.contract('gkk,gij->ij', cderi_dm, cderi, backend='jax')
#     vk = oe.contract('gik,gkj->ij', cderi_dm, cderi, backend='jax')
#     return vj, vk

# def get_veff2(mf, dm):
#     '''use opt einsum on gpu'''
#     dm = jnp.array(dm)
#     vj = jnp.zeros(dm.shape)
#     vk = jnp.zeros(dm.shape)
#     print('Building JK matrix')
#     for i,cderi in enumerate(mf.with_df.loop()):
#         print(f'DF loop {i} number of DF vectors {cderi.shape[0]}')
#         cderi = jnp.array(lib.unpack_tril(cderi, axis=-1))
#         # cderi = jnp.array(cderi)
#         # cderi_dm = oe.contract('gik,kj->gij', cderi, dm, backend='jax')
#         # vj += oe.contract('gkk,gij->ij', cderi_dm, cderi, backend='jax')
#         # vk += oe.contract('gik,gkj->ij', cderi_dm, cderi, backend='jax')
#         dvj, dvk = jk_from_cderi(cderi, dm)
#         vj += dvj
#         vk += dvk
#     # vj, vk = mf.get_jk(mol, dm, hermi=1)
#     return 2*vj - vk

# def h1e_ras(mf, mo_coeff, ncas, ncore):
#     '''
#     effective one-electron integral for restricted active space
#     ncas = nact_electron/2
#     ncore = ncore_electrons/2
#     '''
#     # note casci undo DF

#     mo_core = jnp.array(mo_coeff[:,:ncore])
#     mo_cas = jnp.array(mo_coeff[:,ncore:ncore+ncas])

#     hcore = jnp.array(mf.get_hcore())
#     energy_core = mf.energy_nuc()
#     if mo_core.size == 0:
#         corevhf = 0.
#     else:
#         # core_dm = np.dot(mo_core, mo_core.T)
#         core_dm = mo_core @ mo_core.T
#         time0 = time.perf_counter()
#         corevhf = get_veff2(mf, core_dm)
#         time1 = time.perf_counter()
#         print(f"build JK time: {time1 - time0:.6f} s")
#         energy_core += 2 * oe.contract('ij,ji', core_dm, hcore, backend='jax')
#         energy_core += oe.contract('ij,ji', core_dm, corevhf, backend='jax')
#         time2 = time.perf_counter()
#         print(f"build ecore time: {time2 - time1:.6f} s")
#     h1eff = mo_cas.T @ (hcore+corevhf) @ mo_cas
#     time3 = time.perf_counter()
#     print(f"build h1eff time: {time3 - time0:.6f} s")
#     return h1eff, energy_core

# def prep_afqmc(mf_cc,mo_coeff,t1,t2,frozen,prjlo,
#                options,chol_cut=1e-5,
#                option_file='options.bin',
#                mo_file="mo_coeff.npz",
#                amp_file="amplitudes.npz",
#                chol_file="FCIDUMP_chol"):
    
#     jax.config.update("jax_enable_x64", True)
    
#     with open(option_file, 'wb') as f:
#         pickle.dump(options, f)
    
#     if isinstance(mf_cc, CCSD):
#         mf = mf_cc._scf
#     else:
#         mf = mf_cc

#     t2 = t2.transpose(0, 2, 1, 3)
#     t1 = np.array(t1)
#     np.savez(amp_file,t1=t1,t2=t2)

#     print('Calculating Effective Active Space One-electron Integrals')
#     mol = mf.mol
#     nocc = np.count_nonzero(mf.mo_occ)
#     actfrag = np.array([i for i in range(mol.nao) if i not in frozen])
#     frzocc = np.array([i for i in range(nocc) if i in frozen])
#     actocc = np.array([i for i in range(nocc) if i in actfrag])
#     actvir = np.array([i for i in range(nocc,mol.nao) if i in actfrag])
#     nfrzocc = len(frzocc)
#     nactocc = len(actocc)
#     nactvir = len(actvir)
#     nactorb = len(actfrag)
#     # print(f'# number of forzen occupied orbitals {nfrzocc}')
#     print(f'number of active occupied orbitals {nactocc}')
#     print(f'number of active virtual orbitals {nactvir}')

#     ncas = nactorb
#     ncore = nfrzocc
#     nelec = nactocc*2
#     h1e, enuc = h1e_ras(mf, mo_coeff, ncas, ncore)
#     mo_act = mo_coeff[:,actfrag]

#     print('Generating Cholesky Integrals')

#     if getattr(mf, "with_df", None) is not None:
#         # decompose eri in MO to achieve linear scale over the Auxiliary-field
#         print("Composing AO ERIs from DF basis")
#         from pyscf.ao2mo import _ao2mo

#         naux = mf.with_df.get_naoaux()
#         chol_df = np.zeros((naux,ncas*(ncas+1)//2))
#         ijslice = (0, ncas, 0, ncas)
#         Lpq = None
#         p1 = 0

#         time0 = time.perf_counter()
#         for eri1 in mf.with_df.loop():
#             Lpq = _ao2mo.nr_e2(eri1, mo_act, ijslice, aosym='s2', out=Lpq).reshape(-1,ncas,ncas)
#             p0, p1 = p1, p1 + Lpq.shape[0]
#             # print(eri1.shape)
#             # print(Lpq.shape)
#             chol_df[p0:p1] = lib.pack_tril(Lpq, axis=-1) # in mo representation
#         print(f"packed chol tensor by DF shape: {chol_df.shape}")
#         # chol_df = jnp.array(chol_df)

#         # chol_df = df.incore.cholesky_eri(mol, mf.with_df.auxmol.basis) # in ao 
#         # chol_df = lib.unpack_tril(chol_df).reshape(chol_df.shape[0], -1)
#         # chol_df = chol_df.reshape((-1, mol.nao, mol.nao))
#         # chol_df = lib.einsum('pr,grs,sq->gpq',mo_act.T,chol_df,mo_act)
#         # eri_df = lib.einsum('gP,gQ->PQ', chol_df, chol_df, optimize='optimal')
#         eri_df = oe.contract('gP,gQ->PQ', chol_df, chol_df, backend='jax')
#         time1 = time.perf_counter()
#         print("Composing active space MO ERIs from AO ERIs")
#         # eri_df = lib.pack_tril(eri_df,axis=0) # pyscf.lib pack the lower triangular
#         # eri_df = lib.pack_tril(eri_df,axis=-1)
#         # eri_df = eri_df.reshape(ncas**2,ncas**2)
#         print("Decomposing MO ERIs to Cholesky vectors")
#         print(f"Cholesky cutoff is: {chol_cut}")
#         chol = pyscf_interface.modified_cholesky(eri_df,max_error=chol_cut)
#         chol = lib.unpack_tril(chol,axis=-1)
#         chol = chol.reshape(-1,ncas,ncas)
#         time2 = time.perf_counter()
#         print(f"build 2-electron integral time: {time1 - time0:.6f} s")
#         print(f"Decompose 2-electron integral to CD time: {time2 - time1:.6f} s")
#         print(f"Total 2-electron integral time: {time2 - time0:.6f} s")
#     else:
#         raise  NotImplementedError('Use DF Only!')

#     print("Finished calculating Cholesky integrals")
#     print('Size of the correlation space')
#     print(f'Number of electrons: ({nactocc},{nactocc})')
#     print(f'Number of basis functions: {ncas}')
#     print(f'Cholesky shape: {chol.shape}')

#     v0 = 0.5 * oe.contract("gpr,grq->pq", chol, chol, backend="jax")
#     h1e_mod = h1e - v0
#     chol = chol.reshape((chol.shape[0], -1))
#     np.savez(mo_file,prjlo=prjlo)

#     write_dqmc(
#         h1e,
#         h1e_mod,
#         chol,
#         nelec,
#         ncas,
#         enuc,
#         mf.e_tot,
#         filename=chol_file,
#     )

#     return nelec, ncas

# def write_dqmc(
#     hcore,
#     hcore_mod,
#     chol,
#     nelec,
#     nmo,
#     enuc,
#     emf,
#     filename="FCIDUMP_chol",
# ):
#     hcore = np.array(hcore)
#     hcore_mod = np.array(hcore_mod)
#     chol = np.array(chol)
#     with h5py.File(filename, "w") as fh5:
#         fh5["header"] = np.array([nelec, nmo, chol.shape[0]])
#         fh5["hcore"] = hcore.flatten()
#         fh5["hcore_mod"] = hcore_mod.flatten()
#         fh5["chol"] = chol.flatten()
#         fh5["energy_core"] = enuc
#         fh5["emf"] = emf


# def _prep_afqmc(option_file="options.bin",
#                 mo_file="mo_coeff.npz",
#                 amp_file="amplitudes.npz",
#                 chol_file="FCIDUMP_chol"):
    
#     jax.config.update("jax_enable_x64", True)
    
#     try:
#         with open(option_file, "rb") as f:
#             options = pickle.load(f)
#     except:
#         print('Using default options')
#         options = {}

#     options["dt"] = options.get("dt", 0.005)
#     options["n_exp_terms"] = options.get("n_exp_terms",6)
#     options["n_walkers"] = options.get("n_walkers", 50)
#     options["n_prop_steps"] = options.get("n_prop_steps", 50)
#     options["n_blocks"] = options.get("n_blocks", 500)
#     options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))
#     options["n_eql"] = options.get("n_eql", 3)
#     options["walker_type"] = options.get("walker_type", "rhf")
#     options["trial"] = options.get("trial", None)
#     options["ene0"] = options.get("ene0", 0.0)
#     options["n_batch"] = options.get("n_batch", 1)

#     with h5py.File(chol_file, "r") as fh5:
#         [nelec, nmo, nchol] = fh5["header"]
#         h0 = jnp.array(fh5.get("energy_core"))
#         emf = jnp.array(fh5.get("emf"))
#         h1 = jnp.array(fh5.get("hcore")).reshape(nmo, nmo)
#         chol = jnp.array(fh5.get("chol")).reshape(-1, nmo, nmo)
#         h1_mod = jnp.array(fh5.get("hcore_mod")).reshape(nmo, nmo)

#     assert type(nelec) is np.int64
#     assert type(nmo) is np.int64
#     assert type(nchol) is np.int64
#     nelec, nmo, nchol = int(nelec), int(nmo), int(nchol)
#     nelec_sp = (nelec // 2, nelec // 2)
#     norb = nmo
#     # ham = hamiltonian.hamiltonian(nmo)
#     ham_data = {}
#     ham_data["h0"] = h0
#     ham_data["E0"] = emf
#     ham_data["ene0"] = options["ene0"]

#     ham_data["h1"] = jnp.array([h1, h1])
#     ham_data["h1_mod"] = jnp.array(h1_mod)
#     nchol = chol.shape[0]
#     ham_data["chol"] = jnp.array(chol.reshape(chol.shape[0], -1))

#     wave_data = {}
#     wave_data['prjlo'] = jnp.array(np.load(mo_file)["prjlo"])
#     mo_coeff = jnp.array(np.eye(nmo))

#     if options["trial"] == "rhf":
#         trial = lno_wavefunctions.rhf(norb, nelec_sp, n_batch=options["n_batch"])
#         wave_data["mo_coeff"] = mo_coeff[:, : nelec_sp[0]]
#     elif options["trial"] == "ccsd_pt_ad":
#         trial = lno_wavefunctions.ccsd_pt_ad(norb, nelec_sp, n_batch=options["n_batch"])
#         amplitudes = np.load(amp_file)
#         t1 = jnp.array(amplitudes["t1"])
#         t2 = jnp.array(amplitudes["t2"])
#         prj = wave_data['prjlo']
#         wave_data["t1"] = oe.contract('ia,ik->ka',t1, prj, backend='jax')
#         wave_data["t2"] = oe.contract('iajb,ik->kajb',t2, prj, backend='jax')
#     elif options["trial"] == "ccsd_pt":
#         trial = lno_wavefunctions.ccsd_pt(norb, nelec_sp, n_batch=options["n_batch"])
#         amplitudes = np.load(amp_file)
#         t1 = jnp.array(amplitudes["t1"])
#         t2 = jnp.array(amplitudes["t2"])
#         wave_data["t1"] = oe.contract('ia,ik->ka',t1,wave_data['prjlo'])
#         wave_data["t2"] = oe.contract('iajb,ik->kajb',t2,wave_data['prjlo'])
#         wave_data["mo_coeff"] = mo_coeff[:, :nocc]
#     elif "ccsd_pt2" in options["trial"]:
#         from jax import scipy as jsp
#         nocc = nelec_sp[0]
#         amplitudes = np.load(amp_file)
#         t1 = jnp.array(amplitudes["t1"])
#         t2 = jnp.array(amplitudes["t2"])
#         t1_full = np.zeros((norb, norb))
#         t1_full[:nocc, nocc:] = t1
#         wave_data['exp_t1'] = jsp.linalg.expm(t1_full)
#         wave_data['exp_mt1'] = jsp.linalg.expm(-t1_full)
#         wave_data["t2"] = oe.contract('iajb,ik->kajb',t2, wave_data['prjlo'], backend='jax')
#         wave_data["mo_coeff"] = mo_coeff[:, :nocc]
#         # print(t1.shape)
#         # print(chol.shape)
#         lt1 = oe.contract('ia,gja->gij', t1, chol[:, :nocc, nocc:], backend='jax')
#         e0t1orb = 2 * oe.contract('gik,ik,gjj->',lt1, wave_data['prjlo'], lt1, backend='jax') \
#                     - oe.contract('gij,gjk,ik->',lt1, lt1, wave_data['prjlo'], backend='jax')
#         ham_data['e0t1orb'] = e0t1orb
#         trial = lno_wavefunctions.ccsd_pt2(norb, nelec_sp, n_batch = options["n_batch"])
#         if "fast" in options["trial"]:
#             trial = lno_wavefunctions.ccsd_pt2_fast(norb, nelec_sp, n_batch = options["n_batch"])
#         if "ad" in options["trial"]:
#             trial = lno_wavefunctions.ccsd_pt2_ad(norb, nelec_sp, n_batch = options["n_batch"])
        
#     if options["walker_type"] == "rhf":
#         prop = propagation.propagator_restricted(
#             options["dt"], 
#             options["n_walkers"], 
#             options["n_exp_terms"],
#             options["n_batch"]
#         )

#     if  'pt' in options['trial']:
#         if '2' in options['trial']:
#             sampler = sampling.sampler_pt2(
#                 options["n_prop_steps"],
#                 options["n_blocks"],
#                 nchol,)
#         else:
#             sampler = sampling.sampler_pt(
#                 options["n_prop_steps"],
#                 options["n_blocks"],
#                 nchol,)
#     else:
#         sampler = sampling.sampler(
#                 options["n_prop_steps"],
#                 options["n_blocks"],
#                 nchol,)

#     return ham_data, prop, trial, wave_data, sampler, options

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
              options,
              frag_lolist,
              lo_coeff = None, 
              lo_coeff_file = 'lo_coeff.npz',
              nfrozen = 0,
              thresh = 1e-6, 
              chol_cut = 1e-5,
              run_frg_list = None, 
              atom_group = None,
              emp2_tot = None,
              ):
    
    if lo_coeff is None:
        try:
            lo_coeff = np.load(lo_coeff_file)["lo_coeff"]
        except:
            raise ValueError(
                f"lo_coeff was not provided and could not be loaded "
                f"from file '{lo_coeff_file}'"
                )
    
    mlno = lnoccsd.LNOCCSD(mf, lo_coeff, frag_lolist, frozen=nfrozen).set(verbose=3)
    mlno.lno_thresh = [thresh*10, thresh]
    # mlno.lo_proj_thresh = 1e-10
    # mlno.lo_proj_thresh_active = 0.1
    lno_thresh = mlno.lno_thresh
    lno_type = ['1h','1h'] # if lno_type is None else lno_type
    eris = mlno.ao2mo()

    nfrag = len(frag_lolist)
    if run_frg_list is None:
        run_frg_list = range(nfrag)
    
    frag_lolist = [frag_lolist[i] for i in run_frg_list]
    # nfrag = len(frag_lolist)
    lno_pct_occ = [None, None]
    lno_norb = [[None,None]] * nfrag

    seeds = random.randint(random.PRNGKey(options["seed"]),
                           shape=(nfrag,), 
                           minval=0, 
                           maxval=100*nfrag
                           )
    options["max_error"] = options["max_error"] / np.sqrt(nfrag)

    las_center = [None]*nfrag
    las_size = np.zeros(nfrag, dtype='int32')
    lno_emp2 = np.zeros(nfrag, dtype='float64')
    lno_ecc  = np.zeros(nfrag, dtype='float64')
    lno_eqmc = np.zeros(nfrag, dtype='float64')
    lno_eqmc_err  = np.zeros(nfrag, dtype='float64')
    ccsd_time = np.zeros(nfrag, dtype='float64')
    qmc_time = np.zeros(nfrag, dtype='float64')

    # Loop over fragment
    for ifrag,loidx in enumerate(frag_lolist):
        print("\n")
        width = 80
        msg = f" RUNNING LNO-FRAGMENT {run_frg_list[ifrag]+1}/{nfrag} "
        print(msg.center(width, '='))
        if atom_group is not None:
            atom_msg = f"{atom_group[ifrag]}"
            print(f"Center Atom {atom_msg}")

        orbloc = lo_coeff[:,loidx]
        lno_param = [{'thresh': lno_thresh[i], 'pct_occ': lno_pct_occ[i],
                        'norb': lno_norb[ifrag][i]} for i in [0,1]]
        
        ao_message, ao_max = prep.ao_comp(mf, orbloc)

        # M = <orbloc|canactocc> (M^dagger M)u = eu 
        # u|canactocc> => orbtial in/out the space spanned by |orbloc>
        # uocc_loc = <lno_actocc|orbloc>
        lno_coeff, lno_frozen, uocc_loc, _ = mlno.make_las(eris, orbloc, lno_type, lno_param)
        # lno_coeff still connected to canonical mo_coeff unitarily

        mo_occ = mlno.mo_occ
        lno_frozen, maskact = lnoccsd.get_maskact(lno_frozen, mo_occ.size)
        
        nactocc, nactvir = prep.las_size(mf, lno_frozen)
        print(f'LAS occupied orbitals: {nactocc}')
        print(f'LAS virtual orbitals: {nactvir}')

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
        prjlo = uocc_loc @ uocc_loc.T.conj()

        options["seed"] = seeds[ifrag]
        prep.prep_afqmc_integral(
            mf,
            lno_coeff,
            t1,
            t2,
            lno_frozen,
            prjlo,
            options,
            chol_cut=chol_cut
            )
        
        run_lnoafqmc(options)
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

    # nelec = np.zeros((nfrag,2),dtype='int32')
    # norb = np.zeros((nfrag,2),dtype='int32')
    # eorb_mp2 = np.zeros(nfrag,dtype='float64')
    # eorb_mp2 = np.zeros(nfrag,dtype='float64')
    # eorb_ccsd = np.zeros(nfrag,dtype='float64')
    # eorb_qmc = np.zeros(nfrag,dtype='float64')
    # eorb_qmc_err = np.zeros(nfrag,dtype='float64')
    # ccsd_time = np.zeros(nfrag,dtype='float64')
    # qmc_time = np.zeros(nfrag,dtype='float64')
    # for n, i in enumerate(run_frg_list):
    #     with open(f"fragment.out{i+1}", "r") as rf:
    #         for line in rf:
    #             if "AOs with contribution" in line:
    #                 next(rf)
    #                 largest_ao = next(rf).rsplit(maxsplit=1)[0].strip()
    #                 ao_labels.append(largest_ao)
    #             if 'LNO-Active Space' in line:
    #                 # nums = re.findall(r'\d+', line)
    #                 nelec[n] = np.array([int(nums[0]),int(nums[1])])
    #                 norb[n] = np.array([int(nums[2]),int(nums[3])])
    #             if "LNO-MP2 Orbital Energy" in line:
    #                 eorb_mp2[n] = float(line.split()[-1])
    #             if "LNO-CCSD Orbital Energy" in line:
    #                 eorb_ccsd[n] = float(line.split()[-1])
    #             if "LNO-AFQMC Orbital Energy" in line:
    #                 eorb_qmc[n] = float(line.split()[-3])
    #                 eorb_qmc_err[n] = float(line.split()[-1])
    #             if "LNO-CCSD Time" in line:
    #                 ccsd_time[n] = float(line.split()[-1])
    #             if "LNO-AFQMC Time" in line:
    #                 qmc_time[n] = float(line.split()[-1])

    # nelec_ = (np.mean(nelec[:,0]), np.mean(nelec[:,1]))
    las_max = las_size.max() #(np.mean(norb[:,0]), np.mean(norb[:,1]))
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
        f.write(f'MP2 Correction:         {emp2_tot - e_mp2:12.8f}\n')

    return None