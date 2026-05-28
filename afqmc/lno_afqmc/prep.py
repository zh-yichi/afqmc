import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
from jax import scipy as jsp
import opt_einsum as oe

import h5py, pickle, time
import numpy as np
from pyscf import lib
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD
from functools import partial

from afqmc import cholesky
from afqmc.lno_afqmc import propagation, sampling
from afqmc.lno_afqmc import wavefunctions_restricted as lno_wavefunctions
from afqmc.lno_afqmc import wavefunctions_unrestricted as ulno_wavefunctions

print = partial(print, flush=True)

def ao_comp(mf, orbloc, ao_threshold=1e-2):
    mol = mf.mol
    S = mol.intor('int1e_ovlp')
    proj = (S @ orbloc)**2
    proj = proj / np.sum(proj, axis=0)
    if len(proj.shape) == 2:
        proj = np.sum(proj, axis=1)
    ao_labels = mol.ao_labels()
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
    return ao_message, ao_labels[above[0]]

def las_size(mf, frozen):
    mol = mf.mol
    nocc = np.count_nonzero(mf.mo_occ)
    actfrag = np.array([i for i in range(mol.nao) if i not in frozen])
    # frzocc = np.array([i for i in range(nocc) if i in frozen])
    actocc = np.array([i for i in range(nocc) if i in actfrag])
    actvir = np.array([i for i in range(nocc, mol.nao) if i in actfrag])
    # nfrzocc = len(frzocc)
    nactocc = len(actocc)
    nactvir = len(actvir)
    # nactorb = len(actfrag)
    return nactocc, nactvir

# restricted below #
def get_rveff_cpu(mf, dm):
    '''restricted'''
    mol = mf.mol
    # print('Building JK matrix')
    vj, vk = mf.get_jk(mol, dm, hermi=1)
    return 2*vj - vk

@jax.jit
def rjk_from_cderi(cderi, dm):
    '''restricted'''
    cderi_dm = oe.contract('gik,kj->gij', cderi, dm, backend='jax')
    vj = oe.contract('gkk,gij->ij', cderi_dm, cderi, backend='jax')
    vk = oe.contract('gik,gkj->ij', cderi_dm, cderi, backend='jax')
    return vj, vk

def get_rveff_gpu(mf, dm):
    '''restricted'''
    dm = jnp.array(dm)
    vj = jnp.zeros(dm.shape)
    vk = jnp.zeros(dm.shape)
    # print('Building JK matrix')
    for i,cderi in enumerate(mf.with_df.loop()):
        # print(f'DF loop {i} number of DF vectors {cderi.shape[0]}')
        cderi = jnp.array(lib.unpack_tril(cderi, axis=-1))
        # cderi = jnp.array(cderi)
        # cderi_dm = oe.contract('gik,kj->gij', cderi, dm, backend='jax')
        # vj += oe.contract('gkk,gij->ij', cderi_dm, cderi, backend='jax')
        # vk += oe.contract('gik,gkj->ij', cderi_dm, cderi, backend='jax')
        dvj, dvk = rjk_from_cderi(cderi, dm)
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
        corevhf = get_rveff_gpu(mf, core_dm)
        time1 = time.perf_counter()
        print(f"build JK time: {time1 - time0:.6f} s")
        energy_core += 2 * oe.contract('ij,ji', core_dm, hcore, backend='jax')
        energy_core += oe.contract('ij,ji', core_dm, corevhf, backend='jax')
        time2 = time.perf_counter()
        print(f"build ecore time: {time2 - time1:.6f} s")
    h1eff = mo_cas.T @ (hcore+corevhf) @ mo_cas
    time3 = time.perf_counter()
    print(f"build h1eff time: {time3 - time0:.6f} s")
    return h1eff, energy_core

# unrestricted below #

def get_uveff(mf, dm):
    # dm = np.array(dm)
    mol = mf.mol
    vj, vk = mf.get_jk(mol, dm, hermi=1)
    return vj[0]+vj[1] - vk

@jax.jit
def ujk_from_cderi(cderi, dm_a, dm_b):
    """
    cderi : (g, nao, nao)
    dm_a  : (nao, nao)
    dm_b  : (nao, nao)
    """
    # dm_a, dm_b = dm
    dm_tot = dm_a + dm_b # Coulomb uses total density

    cderi_dm_tot = oe.contract('gik,kj->gij', cderi, dm_tot, backend='jax')
    vj = oe.contract('gkk,gij->ij', cderi_dm_tot, cderi, backend='jax')

    cderi_dm_a = oe.contract('gik,kj->gij', cderi, dm_a, backend='jax')
    cderi_dm_b = oe.contract('gik,kj->gij', cderi, dm_b, backend='jax')

    vk_a = oe.contract('gik,gkj->ij', cderi_dm_a, cderi, backend='jax')
    vk_b = oe.contract('gik,gkj->ij', cderi_dm_b, cderi, backend='jax')

    return vj, vk_a, vk_b

def get_uveff_gpu(mf, dm):
    dm_a, dm_b = dm
    dm_a = jnp.array(dm_a)
    dm_b = jnp.array(dm_b)
    
    vj = jnp.zeros_like(dm_a)
    vk_a = jnp.zeros_like(dm_a)
    vk_b = jnp.zeros_like(dm_b)

    print('Building JK matrix')
    for cderi in mf.with_df.loop():
        # print(f'# number of DF vectors {cderi.shape[0]}')
        cderi = jnp.array(lib.unpack_tril(cderi, axis=-1))
        dvj, dvk_a, dvk_b = ujk_from_cderi(cderi, dm_a, dm_b)
        vj += dvj
        vk_a += dvk_a
        vk_b += dvk_b

    return jnp.array([vj - vk_a, vj - vk_b])

def h1e_uas(mf, mo_coeff, ncas, ncore):
    '''
    effective one-electron integral for unrestricted active space
    ncas = (ncas_a, ncas_b) size of active space
    ncore = (ncore_a, ncore_b) number of core electrons
    '''
    # mf = mf.undo_df() ucasci undo DF

    mo_core = [jnp.array(mo_coeff[0][:,:ncore[0]]),
               jnp.array(mo_coeff[1][:,:ncore[1]])]
    mo_cas = [jnp.array(mo_coeff[0][:,ncore[0]:ncore[0]+ncas[0]]),
              jnp.array(mo_coeff[1][:,ncore[1]:ncore[1]+ncas[1]])]

    hcore = mf.get_hcore()
    hcore = [jnp.array(hcore), jnp.array(hcore)]
    energy_core = mf.energy_nuc()
    if mo_core[0].size == 0 and mo_core[1].size == 0:
        corevhf = (0,0)
    else:
        core_dm = jnp.array([mo_core[0] @ mo_core[0].T, 
                            mo_core[1] @ mo_core[1].T])
        time0 = time.perf_counter()
        corevhf = get_uveff_gpu(mf, core_dm)
        time1 = time.perf_counter()
        energy_core += oe.contract('ij,ji', core_dm[0], hcore[0], backend='jax')
        energy_core += oe.contract('ij,ji', core_dm[1], hcore[1], backend='jax')
        energy_core += oe.contract('ij,ji', core_dm[0], corevhf[0], backend='jax') * .5
        energy_core += oe.contract('ij,ji', core_dm[1], corevhf[1], backend='jax') * .5
        time2 = time.perf_counter()
    h1eff = [jnp.array(mo_cas[0].T @ (hcore[0]+corevhf[0]) @ mo_cas[0]),
             jnp.array(mo_cas[1].T @ (hcore[1]+corevhf[1]) @ mo_cas[1])]
    time3 = time.perf_counter()
    print(f"build JK time: {time1 - time0:.6f} s")
    print(f"build ecore time: {time2 - time1:.6f} s")
    print(f"build h1eff time: {time3 - time0:.6f} s")
    return h1eff, energy_core

def prjmo(prj, s1e, mo):
    # prj and reconstruct mo
    # e.g. |B_p> = |A_q><A_q|B_p>
    #            = C^A_mq C^A(T)_qn|m><n|s> C^B_sp
    mo_rec = prj @ prj.T @ s1e @ mo
    return mo_rec

def common_las(mf, lno_coeff, ncas, ncore, torr=1e-10, print_ao=False, ao_thresh=1e-2):
    print("Constracting cLAS that span both Alpha and Beta active space")
    # time0 = time.perf_counter()
    s1e = mf.get_ovlp()
    lno_acta = lno_coeff[0][:,ncore[0]:ncore[0]+ncas[0]]
    lno_actb = lno_coeff[1][:,ncore[1]:ncore[1]+ncas[1]]
    lno_actaa = lno_coeff[0].T @ s1e @ lno_acta # proj to the complete
    lno_actba = lno_coeff[0].T @ s1e @ lno_actb # alpha basis for orthogonal
    clno_act = np.hstack([lno_actaa,lno_actba]) # common active lno
    print('Naive cLAS Shape: ', clno_act.shape)
    # full_matrices = False gives u that just span the space of clno_act
    u, s, _ = np.linalg.svd(clno_act, full_matrices=False)
    print(f'Orthonormalize cLAS shape: {u.shape}')
    print(f'Smallest cLAS SVD Singular values: {s[-1]}')
    print(f"cLAS projection torr: {torr}")
    for idx in range(lno_acta.shape[1],u.shape[1]+1):
        prj = lno_coeff[0] @ u[:,:idx]
        prj_acta = prjmo(prj,s1e,lno_actb)
        prj_actb = prjmo(prj,s1e,lno_acta)
        losa = abs(prj_acta-lno_actb).max()
        losb = abs(prj_actb-lno_acta).max()
        # print(f"# cLAS projection loss: ({losa:.2e}, {losb:.2e})")
        if losa < torr and losb < torr:
            break
    print(f"Minimum size of cLAS to span both Alpha and Beta LAS: {idx}")
    print(f"cLAS projection loss: ({losa:.2e}, {losb:.2e})")
    # span{|C>} = span{|A>} U span{|B>}
    clas_coeff = lno_coeff[0] @ u[:,:idx] # in ao
    print('True Common LAS Shape: ', clas_coeff.shape)
    a2c = clas_coeff.T @ s1e @ lno_acta # <C|A>
    b2c = clas_coeff.T @ s1e @ lno_actb # <C|B>

    # identify the component of the LAS
    if print_ao:
        proj = (s1e @ clas_coeff)**2
        proj = proj / np.sum(proj, axis=0, keepdims=True)
        proj = np.sum(proj, axis=1)
        ao_labels = mf.mol.ao_labels()

        above = np.where(proj > ao_thresh)[0]
        # sort them by contribution descending
        print(f"Find {len(above)} AOs in cLAS with amplitude > {ao_thresh}")
        above = above[np.argsort(proj[above])[::-1]]
        print(f"{'AO Label':>16s}  {'Amp':>6s}")
        for idx in above:
            print(f"{ao_labels[idx]:>16s}  {proj[idx]:8.4f}")

    return clas_coeff, a2c, b2c


# common stuff 
def pack_symmetric(L):
    """
    L: shape (g, n, n), symmetric in last two indices
    returns: shape (g, n*(n+1)//2)
    """
    n = L.shape[-1]
    iu = jnp.tril_indices(n)
    return L[:, iu[0], iu[1]]

@jax.jit
def cderi2mo_gpu(cderi, mo_coeff):
    cderi_mo = oe.contract('pr,grs,sq->gpq', mo_coeff.T, cderi, mo_coeff, backend='jax')
    return pack_symmetric(cderi_mo)

def cderi2mo_cpu(cderi, mo_coeff):
    cderi_mo = lib.einsum('pr,grs,sq->gpq', mo_coeff.T, cderi, mo_coeff, optimize='optimal')
    return pack_symmetric(cderi_mo)

def compress_cderi_cpu(cderi, thresh=1e-6):
    """
    Perform SVD on cderi (CPU) and keep components with s^2 > thresh.

    Parameters
    ----------
    cderi : np.ndarray (naux, npair)
    thresh : float |Threshold on squared singular values sqaure

    Returns
    -------
    compressed cderi: np.ndarray
    """
    _, s, Vh = np.linalg.svd(cderi, full_matrices=False)
    mask = s**2 > thresh

    s = s[mask]
    Vh = Vh[mask, :]
    cp_cderi = s[:, None] * Vh

    return cp_cderi

@jax.jit
def _svd_gpu(cderi):
    return jnp.linalg.svd(cderi, full_matrices=False)

# @jax.jit
def compress_cderi_gpu(cderi, thresh=1e-6):
    """
    Perform SVD on cderi (GPU via JAX) and keep components with s^2 > thresh.

    Parameters
    ----------
    cderi : jnp.ndarray
        Input matrix (m, n) already on GPU
    thresh : float
        Threshold on squared singular values

    Returns
    -------
    cp_cderi : jnp.ndarray
        Compressed cderi (k, n)
    """

    # SVD on GPU
    _, s, Vh = _svd_gpu(cderi)

    # singular values are sorted descending
    mask = s**2 > thresh
    s = s[mask]

    Vh = Vh[mask, :]
    cp_cderi = s[:, None] * Vh

    return cp_cderi

def r_prep_afqmc_integral(
        mf_cc,
        mo_coeff,
        t1,
        t2,
        frozen,
        prjlo,
        options,
        chol_cut=1e-5,
        option_file='options.bin',
        mo_file="mo_coeff.npz",
        amp_file="amplitudes.npz",
        chol_file="FCIDUMP_chol"
        ):
    
    with open(option_file, 'wb') as f:
        pickle.dump(options, f)
    
    if isinstance(mf_cc, CCSD):
        mf = mf_cc._scf
    else:
        mf = mf_cc

    t2 = t2.transpose(0, 2, 1, 3)
    t1 = np.array(t1)
    np.savez(amp_file,t1=t1,t2=t2)

    print('Calculating Effective Active Space One-electron Integrals')
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
    print(f'number of active occupied orbitals {nactocc}')
    print(f'number of active virtual orbitals {nactvir}')

    ncas = nactorb
    ncore = nfrzocc
    nelec = nactocc*2
    time0 = time.perf_counter()
    h1e, enuc = h1e_ras(mf, mo_coeff, ncas, ncore)
    time1 = time.perf_counter()
    mo_act = mo_coeff[:,actfrag]

    print('Generating Cholesky Integrals')

    if getattr(mf, "with_df", None) is not None:
        print("Composing AO ERIs from DF basis")
        naux = mf.with_df.get_naoaux()
        npair = ncas*(ncas+1)//2
        naux = mf.with_df.get_naoaux()
        cderi_las = np.zeros((naux, npair))
        p1 = 0
        
        time2 = time.perf_counter()
        for cderi in mf.with_df.loop():
            cderi = lib.unpack_tril(cderi, axis=-1)
            cderi = jnp.array(cderi)
            cderi = cderi2mo_gpu(cderi, mo_act)
            p0, p1 = p1, p1 + cderi.shape[0]
            cderi_las[p0:p1] = np.array(cderi)
        time3 = time.perf_counter()

        print(f"Raw CDERI in LAS shape: {cderi_las.shape}")
        print("Compress CDERI into LAS Cholesky Vectors by SVD")
        print(f"Cholesky cutoff: {chol_cut}")
        cderi_las = jnp.array(cderi_las)
        cderi_las = compress_cderi_gpu(cderi_las, thresh=chol_cut)
        cderi_las = np.array(cderi_las)
        cderi_las = lib.unpack_tril(cderi_las, axis=-1)
        time4 = time.perf_counter()

        print(f"Build effective h0 and h1 time: {time1 - time0:.6f} s")
        print(f"Build CDERI in LAS time: {time3 - time2:.6f} s")
        print(f"Compress CDERI to LAS Choleskey Vectors time: {time4 - time3:.6f} s")
        print(f"Build Integral total time: {time4 - time0:.6f} s")
    else:
        raise  NotImplementedError('Only Support Mean-Field Object with DF!')

    print("Finished calculating Cholesky integrals")
    print('Size of the correlation space')
    print(f'Number of electrons: ({nactocc},{nactocc})')
    print(f'Number of basis functions: {ncas}')
    print(f'Cholesky shape: {cderi_las.shape}')

    v0 = 0.5 * oe.contract("gpr,grq->pq", cderi_las, cderi_las, backend="jax")
    h1e_mod = h1e - v0
    cderi_las = cderi_las.reshape((cderi_las.shape[0], -1))
    
    np.savez(mo_file, prjlo=prjlo)

    r_write_dqmc(
        h1e,
        h1e_mod,
        cderi_las,
        nelec,
        ncas,
        enuc,
        mf.e_tot,
        filename=chol_file,
    )

    del cderi_las, h1e, h1e_mod, v0

    return None

def u_prep_afqmc_integral(
        mf_cc,
        mo_coeff,
        t1,
        t2,
        frozen,
        prjlo,
        options,
        chol_cut=1e-5,
        option_file='options.bin',
        mo_file="mo_coeff.npz",
        amp_file="amplitudes.npz",
        chol_file="FCIDUMP_chol",
        ):
    
    
    with open(option_file, 'wb') as f:
        pickle.dump(options, f)
    
    if isinstance(mf_cc, (CCSD, UCCSD)):
        mf = mf_cc._scf
    else:
        mf = mf_cc

    if 'cc' in options['trial']:
        t2aa = t2[0]
        t2aa = (t2aa - t2aa.transpose(0, 1, 3, 2)) / 2
        t2aa = t2aa.transpose(0, 2, 1, 3)
        t2bb = t2[2]
        t2bb = (t2bb - t2bb.transpose(0, 1, 3, 2)) / 2
        t2bb = t2bb.transpose(0, 2, 1, 3)
        t2ab = t2[1]
        t2ab = t2ab.transpose(0, 2, 1, 3)
        t1a = np.array(t1[0])
        t1b = np.array(t1[1])
        np.savez(amp_file,
                 t1a=t1a,
                 t1b=t1b,
                 t2aa=t2aa,
                 t2ab=t2ab,
                 t2bb=t2bb)

    print('Calculating Effective Active Space One-electron Integrals')
    mol = mf.mol
    nocc_a = int(sum(mf.mo_occ[0]))
    actfrag_a = np.array([i for i in range(mol.nao) if i not in frozen[0]])
    frzocc_a = np.array([i for i in range(nocc_a) if i in frozen[0]])
    actocc_a = np.array([i for i in range(nocc_a) if i in actfrag_a])
    actvir_a = np.array([i for i in range(nocc_a,mol.nao) if i in actfrag_a])
    nfrzocc_a = len(frzocc_a)
    nactocc_a = len(actocc_a)
    nactvir_a = len(actvir_a)
    nactorb_a = len(actfrag_a)
    nocc_b = int(sum(mf.mo_occ[1]))
    actfrag_b = np.array([i for i in range(mol.nao) if i not in frozen[1]])
    frzocc_b = np.array([i for i in range(nocc_b) if i in frozen[1]])
    actocc_b = np.array([i for i in range(nocc_b) if i in actfrag_b])
    actvir_b = np.array([i for i in range(nocc_b,mol.nao) if i in actfrag_b])
    nfrzocc_b = len(frzocc_b)
    nactocc_b = len(actocc_b)
    nactvir_b = len(actvir_b)
    nactorb_b = len(actfrag_b)

    ncas = (nactorb_a, nactorb_b)
    ncore = (nfrzocc_a, nfrzocc_b)
    nelec = (nactocc_a, nactocc_b)
    time0 = time.perf_counter()
    h1e, enuc = h1e_uas(mf, mo_coeff, ncas, ncore)
    time1 = time.perf_counter()

    print('Generating Cholesky Integrals')

    if getattr(mf, "with_df", None) is not None:
        time2 = time.perf_counter()
        clas_coeff, a2c, b2c = common_las(mf, mo_coeff, ncas, ncore, torr=1e-9, print_ao=True)
        time3 = time.perf_counter()

        print("Composing AO ERIs from DF basis")
        nclas = clas_coeff.shape[1]
        npair = nclas*(nclas+1)//2
        naux = mf.with_df.get_naoaux()
        cderi_clas = np.zeros((naux, npair))
        p1 = 0

        time4 = time.perf_counter()
        for cderi in mf.with_df.loop():
            cderi = lib.unpack_tril(cderi, axis=-1)
            cderi = jnp.array(cderi)
            cderi = cderi2mo_gpu(cderi, clas_coeff)
            p0, p1 = p1, p1 + cderi.shape[0]
            cderi_clas[p0:p1] = np.array(cderi)
        time5 = time.perf_counter()

        print(f"Raw CDERI in cLAS shape: {cderi_clas.shape}")
        print(f"Cholesky cutoff is: {chol_cut}")
        cderi_clas = jnp.array(cderi_clas)
        cderi_clas = compress_cderi_gpu(cderi_clas, thresh=chol_cut)
        print("Compress CDERI into Cholesky Vectors by SVD")
        cderi_clas = np.array(cderi_clas)
        cderi_clas = lib.unpack_tril(cderi_clas, axis=-1)
        time6 = time.perf_counter()
        cderi_a = cderi2mo_cpu(cderi_clas, a2c)
        cderi_b = cderi2mo_cpu(cderi_clas, b2c)
        cderi_a = lib.unpack_tril(cderi_a, axis=-1)
        cderi_b = lib.unpack_tril(cderi_b, axis=-1)
        time7 = time.perf_counter()
        print(f"Build effective h0 and h1 time: {time1 - time0:.6f} s")
        print(f"Build Common LAS time: {time3 - time2:.6f} s")
        # print(f"# Build DF in clsd time: {time5 - time4:.6f} s")
        print(f"Build CDERI in cLAS time: {time5 - time4:.6f} s")
        print(f"Compress CDERI to Choleskey Vectors time: {time6 - time5:.6f} s")
        print(f"Project Cholesky from cLAS to Alpha and Beta time: {time7 - time6:.6f} s")
        print(f"Build Integral total time: {time7 - time0:.6f} s")

    else:
        raise  NotImplementedError('Only Support Mean-Field Object with DF!')
    
    # v0_a = 0.5 * oe.contract("nik,njk->ij", chola, chola, backend='jax')
    # v0_b = 0.5 * oe.contract("nik,njk->ij", cholb, cholb, backend='jax')
    v0_a = 0.5 * lib.einsum("gik,gjk->ij", cderi_a, cderi_a, optimize='optimal')
    v0_b = 0.5 * lib.einsum("gik,gjk->ij", cderi_b, cderi_b, optimize='optimal')

    h1mod_a = np.array(h1e[0]) - v0_a
    h1mod_b = np.array(h1e[1]) - v0_b

    print("Finished calculating Integrals")
    print('Size of the correlation space: ')
    print(f'Number of electrons: {nelec}')
    print(f'Number of basis functions: {ncas}')
    print(f'Alpha Basis Cholesky shape: {cderi_a.shape}')
    print(f' Beta Basis Cholesky shape: {cderi_b.shape}')
    
    cderi_a = cderi_a.reshape(cderi_a.shape[0], -1)
    cderi_b = cderi_b.reshape(cderi_b.shape[0], -1)
    
    np.savez(mo_file,prja=prjlo[0],prjb=prjlo[1])

    u_write_dqmc(h1e,
                 [h1mod_a,h1mod_b],
                 [cderi_a, cderi_b],
                 nelec,
                 ncas,
                 enuc,
                 mf.e_tot,
                 filename=chol_file
                 )
    
    # Clean up all large arrays before returning
    del cderi_clas, cderi_a, cderi_b
    del h1e, h1mod_a, h1mod_b
    del clas_coeff, a2c, b2c
    del v0_a, v0_b

    return None


def r_write_dqmc(
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
        fh5["system"] = np.array([nelec, nmo])
        fh5["hcore"] = hcore.flatten()
        fh5["hcore_mod"] = hcore_mod.flatten()
        fh5["chol"] = chol.flatten()
        fh5["energy_core"] = enuc
        fh5["emf"] = emf


def u_write_dqmc(
    h1e,
    h1e_mod,
    chol,
    nelec,
    nmo,
    enuc,
    emf,
    filename="FCIDUMP_chol"
    ):

    h1e_a, h1e_b = h1e
    h1mod_a, h1mod_b = h1e_mod
    chol_a, chol_b = chol
    h1e_a = np.array(h1e_a)
    h1e_b = np.array(h1e_b)
    h1mod_a = np.array(h1mod_a)
    h1mod_b = np.array(h1mod_b)
    chol_a = np.array(chol_a)
    chol_b = np.array(chol_b)
    with h5py.File(filename, "w") as fh5:
        fh5["system"] = np.array([nelec[0], nelec[1], nmo[0], nmo[1]])
        fh5["h1e_a"] = h1e_a.flatten()
        fh5["h1e_b"] = h1e_b.flatten()
        fh5["h1mod_a"] = h1mod_a.flatten()
        fh5["h1mod_b"] = h1mod_b.flatten()
        fh5["chol_a"] = chol_a.flatten()
        fh5["chol_b"] = chol_b.flatten()
        fh5["energy_core"] = enuc
        fh5["emf"] = emf


def kind(x):
    """don't support general spin-orbital"""
    if isinstance(x, np.ndarray) and x.ndim == 2:
        return "restricted"
    if (isinstance(x, (tuple, list))
            and len(x) == 2
            and all(isinstance(m, np.ndarray) and m.ndim == 2 for m in x)):
        return "unrestricted"
    return "other"

def prep_afqmc_integral(
        mf_cc,
        mo_coeff,
        t1,
        t2,
        frozen,
        prjlo,
        options,
        chol_cut=1e-5,
        option_file='options.bin',
        mo_file="mo_coeff.npz",
        amp_file="amplitudes.npz",
        chol_file="FCIDUMP_chol"
        ):
    
    spin_type = kind(mo_coeff)
    
    if spin_type == "restricted":
        return r_prep_afqmc_integral(
            mf_cc,
            mo_coeff,
            t1,
            t2,
            frozen,
            prjlo,
            options,
            chol_cut,
            option_file,
            mo_file,
            amp_file,
            chol_file,
            )
    
    elif spin_type == "unrestricted":
        return u_prep_afqmc_integral(
            mf_cc,
            mo_coeff,
            t1,
            t2,
            frozen,
            prjlo,
            options,
            chol_cut,
            option_file,
            mo_file,
            amp_file,
            chol_file,
            )
    
    else: 
        raise NotImplementedError('Only Support Restricted and Unrestricted Now!')

def auto_qmc_options(options={}, spin_type="restricted"):

    options["dt"] = options.get("dt", 0.005)
    options["n_walkers"] = options.get("n_walkers", 300)
    options["n_prop_steps"] = options.get("n_prop_steps", 50)
    options["eql_time"] = options.get("n_eql", 20)
    options["n_blocks"] = options.get("n_blocks", 500)
    options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))
    options["n_batch"] = options.get("n_batch", 1)
    options['max_memory'] = options.get("max_memory", 2000)
    options["nchol_chunk"] = options.get("nchol_chunk", 100)
    options['mix_precision'] = options.get("mix_precision", True)
    options["max_error"] = options.get("max_error", 0.0)
    options["n_exp_terms"] = options.get("n_exp_terms",6)

    if spin_type == "restricted":
        options["walker_type"] = options.get("walker_type", "rhf")
        options["trial"] = options.get("trial", "rhf")
    elif spin_type == "unrestricted":
        options["walker_type"] = options.get("walker_type", "uhf")
        options["trial"] = options.get("trial", "uhf")
    
    return options

def r_prep_afqmc_run(
        options,
        mo_file="mo_coeff.npz",
        amp_file="amplitudes.npz",
        chol_file="FCIDUMP_chol"
        ):

    # with open(option_file, "rb") as f:
    #         options = pickle.load(f)

    # options = auto_qmc_options(options)

    with h5py.File(chol_file, "r") as fh5:
        [nelec, norb] = fh5["system"]
        h0 = jnp.array(fh5.get("energy_core"))
        emf = jnp.array(fh5.get("emf"))
        h1 = jnp.array(fh5.get("hcore")).reshape(norb, norb)
        h1_mod = jnp.array(fh5.get("hcore_mod")).reshape(norb, norb)
        chol = jnp.array(fh5.get("chol")).reshape(-1, norb, norb)

    # assert type(nelec) is np.int64
    # assert type(norb) is np.int64
    # assert type(nchol) is np.int64

    nelec, norb = int(nelec), int(norb)
    nocc = nelec // 2
    nelec_sp = (nocc, nocc)
    nchol = chol.shape[0]

    print("\nQMC System")
    print(f"Number of electrons:       {nelec_sp}")
    print(f"Number of orbitals:         {norb}")
    print(f"Number of Cholesky Vectors: {nchol}")

    options["nchol_chunk"] = cholesky.chunk_chol(
        chol, options["nchol_chunk"], 
        options["max_memory"]/options["n_walkers"]
        )

    ham_data = {}
    ham_data["h0"] = h0
    ham_data["E0"] = emf
    ham_data["h1"] = jnp.array([h1, h1])
    ham_data["h1_mod"] = jnp.array(h1_mod)
    ham_data["chol"] = jnp.array(chol.reshape(chol.shape[0], -1))

    wave_data = {}
    wave_data['prjlo'] = jnp.array(np.load(mo_file)["prjlo"])
    mo_coeff = jnp.array(np.eye(norb))
    wave_data["mo_coeff"] = mo_coeff[:, :nocc]

    if options["trial"] == "rhf":
        trial = lno_wavefunctions.rhf(norb, nelec_sp, n_batch=options["n_batch"])
    elif options["trial"] == "ptccsd_ad":
        trial = lno_wavefunctions.ptccsd_ad(norb, nelec_sp, n_batch=options["n_batch"])
        amplitudes = np.load(amp_file)
        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])
        prj = wave_data['prjlo']
        wave_data["t1"] = oe.contract('ia,ik->ka',t1, prj, backend='jax')
        wave_data["t2"] = oe.contract('iajb,ik->kajb',t2, prj, backend='jax')
    elif options["trial"] == "ptccsd":
        trial = lno_wavefunctions.ptccsd(norb, nelec_sp, n_batch=options["n_batch"])
        amplitudes = np.load(amp_file)
        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])
        wave_data["t1"] = oe.contract('ia,ik->ka',t1,wave_data['prjlo'])
        wave_data["t2"] = oe.contract('iajb,ik->kajb',t2,wave_data['prjlo'])
    elif "pt2ccsd" in options["trial"]:
        amplitudes = np.load(amp_file)
        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])
        t1_full = np.zeros((norb, norb))
        t1_full[:nocc, nocc:] = t1
        wave_data['exp_t1'] = jsp.linalg.expm(t1_full)
        wave_data['exp_mt1'] = jsp.linalg.expm(-t1_full)
        wave_data["t2"] = oe.contract('iajb,ik->kajb',t2, wave_data['prjlo'], backend='jax')
        lt1 = oe.contract('ia,gja->gij', t1, chol[:, :nocc, nocc:], backend='jax')
        e0t1orb = 2 * oe.contract('gik,ik,gjj->',lt1, wave_data['prjlo'], lt1, backend='jax') \
                    - oe.contract('gij,gjk,ik->',lt1, lt1, wave_data['prjlo'], backend='jax')
        ham_data['e0t1orb'] = e0t1orb
        if options["trial"] == "pt2ccsd":
            trial = lno_wavefunctions.pt2ccsd(norb, nelec_sp, 
                                              n_batch=options["n_batch"],
                                              nchol_chunk=options["nchol_chunk"], 
                                              mix_precision=options["mix_precision"],
                                              )
        elif "ad" in options["trial"]:
            trial = lno_wavefunctions.pt2ccsd_ad(norb, nelec_sp, 
                                                 n_batch=options["n_batch"])
        
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
                options["n_blocks"],
                nchol,)
        else:
            sampler = sampling.sampler_pt(
                options["n_prop_steps"],
                options["n_blocks"],
                nchol,)
    else:
        sampler = sampling.sampler(
                options["n_prop_steps"],
                options["n_blocks"],
                nchol,)

    return ham_data, prop, trial, wave_data, sampler, options

def u_prep_afqmc_run(
        options,
        mo_file="mo_coeff.npz",
        amp_file="amplitudes.npz",
        chol_file="FCIDUMP_chol"
        ):

    with h5py.File(chol_file, "r") as fh5:
        [nelec_a, nelec_b, norb_a, norb_b] = fh5["system"]
        h0 = jnp.array(fh5.get("energy_core"))
        emf = jnp.array(fh5.get("emf"))
        h1_a = jnp.array(fh5.get("h1e_a")).reshape(norb_a, norb_a)
        h1_b = jnp.array(fh5.get("h1e_b")).reshape(norb_b, norb_b)
        h1mod_a = jnp.array(fh5.get("h1mod_a")).reshape(norb_a, norb_a)
        h1mod_b = jnp.array(fh5.get("h1mod_b")).reshape(norb_b, norb_b)
        chol_a = jnp.array(fh5.get("chol_a")).reshape(-1, norb_a, norb_a)
        chol_b = jnp.array(fh5.get("chol_b")).reshape(-1, norb_b, norb_b)

    assert chol_a.shape[0] == chol_b.shape[0]

    nelec_a, nelec_b, norb_a, norb_b \
        = int(nelec_a), int(nelec_b), int(norb_a), int(norb_b)
    nelec = (nelec_a, nelec_b)
    norb = (norb_a, norb_b)
    nchol = chol_a.shape[0]

    print("\nQMC System")
    print(f"Number of electrons:        {nelec}")
    print(f"Number of orbitals:         {norb}")
    print(f"Number of Cholesky Vectors: {nchol}")

    options["nchol_chunk"] = cholesky.chunk_chol([chol_a, chol_b], 
                                                 options["nchol_chunk"], 
                                                 options["max_memory"]/options["n_walkers"])

    ham_data = {}
    ham_data["h0"] = h0
    ham_data["E0"] = emf
    ham_data["h1"] = [jnp.array(h1_a), jnp.array(h1_b)]
    ham_data["h1_mod"] = [jnp.array(h1mod_a), jnp.array(h1mod_b)]
    ham_data["chol"] = [chol_a.reshape(chol_a.shape[0], -1),
                        chol_b.reshape(chol_b.shape[0], -1)]

    wave_data = {}
    prja = jnp.array(np.load(mo_file)["prja"])
    prjb = jnp.array(np.load(mo_file)["prjb"])
    wave_data['prjlo'] = [prja, prjb]
    mo_coeff_a = jnp.array(np.eye(norb_a))
    mo_coeff_b = jnp.array(np.eye(norb_b))
    wave_data["mo_coeff"] = [
            mo_coeff_a[:, : nelec[0]],
            mo_coeff_b[:, : nelec[1]],
            ]
    
    # options["nchol_chunk"] = min(options["nchol_chunk"], nchol)

    if options["trial"] == "uhf":
        trial = ulno_wavefunctions.uhf(norb, nelec, n_batch=options["n_batch"])
    elif options["trial"] == "uptccsd_ad":
        trial = ulno_wavefunctions.uptccsd_ad(norb, nelec, n_batch = options["n_batch"])
        amplitudes = np.load(amp_file)
        t1a = jnp.array(amplitudes["t1a"])
        t1b = jnp.array(amplitudes["t1b"])
        t2aa = jnp.array(amplitudes["t2aa"])
        t2ab = jnp.array(amplitudes["t2ab"])
        t2bb = jnp.array(amplitudes["t2bb"])
        prja, prjb = wave_data['prjlo']
        wave_data["t1a"] = oe.contract('ia,ik->ka', t1a, prja, backend='jax')
        wave_data["t1b"] = oe.contract('ia,ik->ka', t1b, prjb, backend='jax')
        wave_data["t2aa"] = oe.contract('iajb,ik->kajb', t2aa, prja, backend='jax')
        wave_data["t2ab"] = oe.contract('iajb,ik->kajb', t2ab, prja, backend='jax')
        wave_data["t2ba"] = oe.contract('jbia,ik->kajb', t2ab, prjb, backend='jax')
        wave_data["t2bb"] = oe.contract('iajb,ik->kajb', t2bb, prjb, backend='jax')
    elif options["trial"] == "uptccsd":
        trial = ulno_wavefunctions.uptccsd(norb, nelec, n_batch = options["n_batch"])
        amplitudes = np.load(amp_file)
        t1a = jnp.array(amplitudes["t1a"])
        t1b = jnp.array(amplitudes["t1b"])
        t2aa = jnp.array(amplitudes["t2aa"])
        t2ab = jnp.array(amplitudes["t2ab"])
        t2bb = jnp.array(amplitudes["t2bb"])
        prja, prjb = wave_data['prjlo']
        wave_data["t1a"] = oe.contract('ia,ik->ka', t1a, prja, backend='jax')
        wave_data["t1b"] = oe.contract('ia,ik->ka', t1b, prjb, backend='jax')
        wave_data["t2aa"] = oe.contract('iajb,ik->kajb', t2aa, prja, backend='jax')
        wave_data["t2ab"] = oe.contract('iajb,ik->kajb', t2ab, prja, backend='jax')
        wave_data["t2ba"] = oe.contract('jbia,ik->kajb', t2ab, prjb, backend='jax')
        wave_data["t2bb"] = oe.contract('iajb,ik->kajb', t2bb, prjb, backend='jax')
    elif "upt2ccsd" in options["trial"]:
        nocca, noccb = nelec
        norba, norbb = norb
        amplitudes = np.load(amp_file)
        t1a = jnp.array(amplitudes["t1a"])
        t1b = jnp.array(amplitudes["t1b"])
        t2aa = jnp.array(amplitudes["t2aa"])
        t2ab = jnp.array(amplitudes["t2ab"])
        t2bb = jnp.array(amplitudes["t2bb"])
        t1a_full = np.zeros((norba, norba))
        t1a_full[:nocca, nocca:] = t1a
        t1b_full = np.zeros((norbb, norbb))
        t1b_full[:noccb, noccb:] = t1b
        wave_data['exp_t1a'] = jsp.linalg.expm(t1a_full)
        wave_data['exp_mt1a'] = jsp.linalg.expm(-t1a_full)
        wave_data['exp_t1b'] = jsp.linalg.expm(t1b_full)
        wave_data['exp_mt1b'] = jsp.linalg.expm(-t1b_full)
        lt1a = oe.contract('ia,gja->gij', t1a, chol_a[:, :nocca, nocca:], backend='jax')
        lt1b = oe.contract('ia,gja->gij', t1b, chol_b[:, :noccb, noccb:], backend='jax')
        # e0t1orb = <exp(T1)HF|H|HF>_i
        e0t1orb_aa = (oe.contract('gik,ik,gjj->',lt1a, prja, lt1a, backend='jax')
                    - oe.contract('gij,gjk,ik->',lt1a, lt1a, prja, backend='jax')) * 0.5
        e0t1orb_ab = oe.contract('gik,ik,gjj->',lt1a, prja, lt1b, backend='jax') * 0.5
        e0t1orb_ba = oe.contract('gik,ik,gjj->',lt1b, prjb, lt1a, backend='jax') * 0.5
        e0t1orb_bb = (oe.contract('gik,ik,gjj->',lt1b, prjb, lt1b, backend='jax')
                    - oe.contract('gij,gjk,ik->',lt1b, lt1b, prjb, backend='jax')) * 0.5
        ham_data['e0t1orb'] = e0t1orb_aa + e0t1orb_ab + e0t1orb_ba + e0t1orb_bb
        if "ad" in options["trial"]:
            trial = ulno_wavefunctions.upt2ccsd_ad(norb, nelec, n_batch = options["n_batch"])
            wave_data["t2aa"] = oe.contract('iajb,ik->kajb', t2aa, prja, backend='jax')
            wave_data["t2ab"] = oe.contract('iajb,ik->kajb', t2ab, prja, backend='jax')
            wave_data["t2ba"] = oe.contract('jbia,ik->kajb', t2ab, prjb, backend='jax')
            wave_data["t2bb"] = oe.contract('iajb,ik->kajb', t2bb, prjb, backend='jax')
        elif "alpha" in options["trial"]:
            trial = ulno_wavefunctions.upt2ccsd_alpha(norb, nelec, 
                                                       n_batch = options["n_batch"], 
                                                       nchol_chunk = options["nchol_chunk"],
                                                       mix_precision = options['mix_precision']
                                                       )
            wave_data["t2aa"] = oe.contract('iajb,ik->kajb', t2aa, prja, backend='jax')
            wave_data["t2ab"] = oe.contract('iajb,ik->kajb', t2ab, prja, backend='jax')
        elif "beta" in options["trial"]:
            trial = ulno_wavefunctions.upt2ccsd_beta(norb, nelec, 
                                                     n_batch = options["n_batch"], 
                                                     nchol_chunk = options["nchol_chunk"],
                                                     mix_precision = options['mix_precision']
                                                     )
            wave_data["t2ba"] = oe.contract('jbia,ik->kajb', t2ab, prjb, backend='jax')
            wave_data["t2bb"] = oe.contract('iajb,ik->kajb', t2bb, prjb, backend='jax')
        elif options["trial"] == "upt2ccsd":
            trial = ulno_wavefunctions.upt2ccsd(norb, nelec, 
                                                n_batch = options["n_batch"],
                                                nchol_chunk=options["nchol_chunk"], 
                                                mix_precision=options["mix_precision"],
                                                )
            wave_data["t2aa"] = oe.contract('iajb,ik->kajb', t2aa, prja, backend='jax')
            wave_data["t2ab"] = oe.contract('iajb,ik->kajb', t2ab, prja, backend='jax')
            wave_data["t2ba"] = oe.contract('jbia,ik->kajb', t2ab, prjb, backend='jax')
            wave_data["t2bb"] = oe.contract('iajb,ik->kajb', t2bb, prjb, backend='jax')

    if options["walker_type"] == "uhf":
        prop = propagation.propagator_unrestricted(
                options["dt"],
                options["n_walkers"],
                n_batch=options["n_batch"],
            )
    if  'pt' in options['trial']:
        if '2' in options['trial']:
            sampler = sampling.sampler_pt2(
                options["n_prop_steps"],
                options["n_blocks"],
                nchol,)
        else:
            sampler = sampling.sampler_pt(
                options["n_prop_steps"],
                options["n_blocks"],
                nchol,)
    else:
        sampler = sampling.sampler(
                options["n_prop_steps"],
                options["n_blocks"],
                nchol,)
    
    del h1_a, h1_b, chol_a, chol_b

    return ham_data, prop, trial, wave_data, sampler, options

def prep_afqmc_run(
    option_file="options.bin",
    mo_file="mo_coeff.npz",
    amp_file="amplitudes.npz",
    chol_file="FCIDUMP_chol"
    ):

    with open(option_file, "rb") as f:
            options = pickle.load(f)
    
    if "u" not in options["walker_type"]:
        spin_type = "restricted"
    elif "u" in options["walker_type"]:
        spin_type = "unrestricted"

    options = auto_qmc_options(options, spin_type)

    if spin_type =="restricted":
        ham_data, prop, trial, wave_data, sampler, options =\
            r_prep_afqmc_run(options, mo_file, amp_file, chol_file)
    
    elif spin_type == "unrestricted":
        ham_data, prop, trial, wave_data, sampler, options =\
            u_prep_afqmc_run(options, mo_file, amp_file, chol_file,)

    print("\nQMC Parameters")
    for op in options:
        if options[op] is not None:
            val = options[op]
            val_str = f"{val:.4g}" if isinstance(val, float) else str(val)
            print(f"{str(op):<15s} - {val_str:>15s}")
    
    return ham_data, prop, trial, wave_data, sampler, options