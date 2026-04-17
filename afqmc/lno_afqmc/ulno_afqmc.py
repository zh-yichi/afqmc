import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
from jax import numpy as jnp
from jax import random
import numpy as np
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD
from pyscf import lib, mp
from pyscf.lno import ulnoccsd
from ad_afqmc import config
from functools import partial
from ad_afqmc.lno_afqmc import propagation, sampling, cholesky
from ad_afqmc.lno_afqmc import wavefunctions_unrestreicted as ulno_wavefunctions
from collections.abc import Iterable
import h5py, pickle, time, gc
import opt_einsum as oe
import re

print = partial(print, flush=True)

def ulno_ccsd(mcc, mo_coeff, uocc_loc, mo_occ, maskact): 

    occidxa = mo_occ[0]>1e-10
    occidxb = mo_occ[1]>1e-10
    # nmo = mo_occ[0].size, mo_occ[1].size
    moidxa, moidxb = maskact

    orbfrzocca = mo_coeff[0][:, ~moidxa &  occidxa]
    orbactocca = mo_coeff[0][:,  moidxa &  occidxa]
    orbactvira = mo_coeff[0][:,  moidxa & ~occidxa]
    orbfrzvira = mo_coeff[0][:, ~moidxa & ~occidxa]
    nfrzocca, nactocca, nactvira, nfrzvira = [orb.shape[1]
                                              for orb in [orbfrzocca,orbactocca,
                                                          orbactvira,orbfrzvira]]
    orbfrzoccb = mo_coeff[1][:, ~moidxb &  occidxb]
    orbactoccb = mo_coeff[1][:,  moidxb &  occidxb]
    orbactvirb = mo_coeff[1][:,  moidxb & ~occidxb]
    orbfrzvirb = mo_coeff[1][:, ~moidxb & ~occidxb]
    nfrzoccb, nactoccb, nactvirb, nfrzvirb = [orb.shape[1]
                                              for orb in [orbfrzoccb,orbactoccb,
                                                          orbactvirb,orbfrzvirb]]
    # nlo = [uocc_loc[0].shape[1], uocc_loc[1].shape[1]]
    prjlo = [uocc_loc[0].T.conj(), uocc_loc[1].T.conj()]
    if nactocca * nactvira == 0 and nactoccb * nactvirb == 0:
        elcorr_pt2 = lib.tag_array(0., spin_comp=np.array((0., 0.)))
        elcorr_cc = lib.tag_array(0., spin_comp=np.array((0., 0.)))
    else:
        # solve CCSD impurity problem
        imp_eris = mcc.ao2mo()
        # MP2 fragment energy
        t1, t2 = mcc.init_amps(eris=imp_eris)[1:]
        elcorr_pt2 = ulnoccsd.get_fragment_energy(imp_eris, t1, t2, prjlo)
        # CCSD fragment energy
        t1, t2 = mcc.kernel(eris=imp_eris, t1=t1, t2=t2)[1:]
        elcorr_cc = ulnoccsd.get_fragment_energy(imp_eris, t1, t2, prjlo)

    return (elcorr_pt2, elcorr_cc), t1, t2

def get_veff(mf, dm):
    # dm = np.array(dm)
    mol = mf.mol
    vj, vk = mf.get_jk(mol, dm, hermi=1)
    return vj[0]+vj[1] - vk


def pack_symmetric(L):
    """
    L: shape (g, n, n), symmetric in last two indices
    returns: shape (g, n*(n+1)//2)
    """
    n = L.shape[-1]
    iu = jnp.tril_indices(n)
    return L[:, iu[0], iu[1]]


# def unpack_symmetric(Lp, n):
#     """
#     Lp: shape (g, n*(n+1)//2)
#     n: matrix dimension
#     returns: shape (g, n, n), symmetric
#     """
#     g = Lp.shape[0]
#     # npair = Lp.shape[-1]
#     # n = 
#     iu = jnp.triu_indices(n)

#     L = jnp.zeros((g, n, n), dtype=Lp.dtype)
#     L = L.at[:, iu[0], iu[1]].set(Lp)
#     # reflect upper triangle to lower triangle
#     L = L + jnp.swapaxes(L, -1, -2) - jnp.diag(jnp.diagonal(L, axis1=-2, axis2=-1))
#     return L

@jax.jit
def jk_from_cderi(cderi, dm_a, dm_b):
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

def get_veff_gpu(mf, dm):
    dm_a, dm_b = dm
    dm_a = jnp.array(dm_a)
    dm_b = jnp.array(dm_b)
    
    vj = jnp.zeros_like(dm_a)
    vk_a = jnp.zeros_like(dm_a)
    vk_b = jnp.zeros_like(dm_b)

    print('# Building JK matrix')
    for cderi in mf.with_df.loop():
        # print(f'# number of DF vectors {cderi.shape[0]}')
        cderi = jnp.array(lib.unpack_tril(cderi, axis=-1))
        dvj, dvk_a, dvk_b = jk_from_cderi(cderi, dm_a, dm_b)
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
    # mo_core = (mo_coeff[0][:,:ncore[0]],
    #            mo_coeff[1][:,:ncore[1]])
    # mo_cas = (mo_coeff[0][:,ncore[0]:ncore[0]+ncas[0]],
    #           mo_coeff[1][:,ncore[1]:ncore[1]+ncas[1]])

    hcore = mf.get_hcore()
    hcore = [jnp.array(hcore), jnp.array(hcore)]
    # hcore = [hcore, hcore]
    energy_core = mf.energy_nuc()
    if mo_core[0].size == 0 and mo_core[1].size == 0:
        corevhf = (0,0)
    else:
        core_dm = jnp.array([mo_core[0] @ mo_core[0].T, 
                            mo_core[1] @ mo_core[1].T])
        # core_dm = (mo_core[0] @ mo_core[0].T, 
        #            mo_core[1] @ mo_core[1].T)
        time0 = time.perf_counter()
        corevhf = get_veff_gpu(mf, core_dm)
        # corevhf = get_veff(mf, core_dm)
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

def prjmo(prj,s1e,mo):
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


@jax.jit
def cderi2mo_gpu(cderi, mo_coeff):
    cderi_mo = oe.contract('pr,grs,sq->gpq', mo_coeff.T, cderi, mo_coeff, backend='jax')
    return pack_symmetric(cderi_mo)

def cderi2mo_cpu(cderi, mo_coeff):
    cderi_mo = lib.einsum('pr,grs,sq->gpq', mo_coeff.T, cderi, mo_coeff, optimize='optimal')
    return pack_symmetric(cderi_mo)

@jax.jit
def get_eri(cderi):
    # cderi_clas = cderi2mo(cderi, clas_coeff)
    eri = oe.contract('gP,gQ->PQ', cderi, cderi, backend='jax')
    return eri

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
    # print(s)
    mask = s**2 > thresh

    s = s[mask]
    Vh = Vh[mask, :]
    # cp_cderi = lib.einsum('s,sP->sP', s, Vh, optimize='optimal')
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
    # print(s)

    # singular values are sorted descending
    mask = s**2 > thresh
    s = s[mask]

    Vh = Vh[mask, :]
    cp_cderi = s[:, None] * Vh

    return cp_cderi


# def compress_cderi_AB_gpu(cderi_a, cderi_b, thresh=1e-6):

#     _, sa, Vha = _svd_gpu(cderi_a)
#     _, sb, Vhb = _svd_gpu(cderi_b)

#     mask = mask = (sa**2 > thresh) & (sb**2 > thresh)

#     sa = sa[mask]
#     sb = sb[mask]

#     Vha = Vha[mask, :]
#     Vhb = Vhb[mask, :]

#     cp_cderi_a = sa[:, None] * Vha
#     cp_cderi_b = sb[:, None] * Vhb

#     return cp_cderi_a, cp_cderi_b


def prep_afqmc(mf_cc,
               mo_coeff,
               t1,
               t2,
               frozen,
               prjlo,
               options,
               chol_cut=1e-5,
               use_df=False,
               option_file='options.bin',
               mo_file="mo_coeff.npz",
               amp_file="amplitudes.npz",
               chol_file="FCIDUMP_chol"
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
        # chol_df = df.incore.cholesky_eri(mol, mf.with_df.auxmol.basis)
        # chol_df = lib.unpack_tril(chol_df).reshape(chol_df.shape[0], -1)
        # chol_df = chol_df.reshape((-1, nao, nao))
        # print(f'# DF Tensor shape: {chol_df.shape}')
        if not use_df:
            time2 = time.perf_counter()
            clas_coeff, a2c, b2c = common_las(mf, mo_coeff, ncas, ncore, torr=1e-9, print_ao=True)
            # clas_coeff = jnp.array(clas_coeff)
            # a2c = jnp.array(a2c)
            # b2c = jnp.array(b2c)
            time3 = time.perf_counter()

            # decompose eri in active LNO to achieve linear scale on the auxilary axis
            print("Composing AO ERIs from DF basis")
            nclas = clas_coeff.shape[1]
            npair = nclas*(nclas+1)//2
            # npair_a = nactorb_a*(nactorb_a+1)//2
            # npair_b = nactorb_b*(nactorb_b+1)//2
            naux = mf.with_df.get_naoaux()
            cderi_clas = np.zeros((naux, npair))
            # cderi_clas = jnp.zeros((naux, npair))
            # cderi_a = np.zeros((naux, npair_a))
            # cderi_b = np.zeros((naux, npair_b))
            # mo_acta = mo_coeff[0][:,ncore[0]:ncore[0]+ncas[0]]
            # mo_actb = mo_coeff[1][:,ncore[1]:ncore[1]+ncas[1]]
            p1 = 0
            time4 = time.perf_counter()
            for cderi in mf.with_df.loop():
                cderi = lib.unpack_tril(cderi, axis=-1)
                cderi = jnp.array(cderi)
                cderi = cderi2mo_gpu(cderi, clas_coeff)
                p0, p1 = p1, p1 + cderi.shape[0]
                cderi_clas[p0:p1] = np.array(cderi)
                # cderi1 = cderi2mo_cpu(cderi, mo_acta)
                # cderi2 = cderi2mo_cpu(cderi, mo_actb)
                # p0, p1 = p1, p1 + cderi.shape[0]
                # cderi_a[p0:p1] = cderi1
                # cderi_b[p0:p1] = cderi2

                # cderi = jnp.array(lib.unpack_tril(cderi, axis=-1))
                # cderi = cderi2mo_gpu(cderi, clas_coeff)
                # p0, p1 = p1, p1 + cderi.shape[0]
                # cderi_clas[p0:p1] = cderi
                # eri_clas += get_eri(cderi)
                # chol_df_clas = pack_symmetric(Lpq)
                # eri_clas += oe.contract('gP,gQ->PQ', chol_df_clas, chol_df_clas, backend='jax')
                # cderi = oe.contract('pr,grs,sq->gpq', clas_coeff.T, cderi, clas_coeff, backend='jax')
                # cderi = pack_symmetric(cderi)
                # eri_clas += oe.contract('gP,gQ->PQ', cderi, cderi, backend='jax')
                # print(f"  {Lpq.shape}")
            
            
            # chol_df_clas = jnp.array(chol_df_clas)
            # print(f"# Packed eri in clas shape: {eri_clas.shape}")
            time5 = time.perf_counter()

            # eri_clas = oe.contract('gP,gQ->PQ', chol_df_clas, chol_df_clas, backend='jax')
            print(f"Raw CDERI in cLAS shape: {cderi_clas.shape}")
            # print(f"# Raw CDERI Alpha shape: {cderi_a.shape}")
            # print(f"# Raw CDERI Beta shape: {cderi_b.shape}")
            # print("# Finish Composing cLAS CDERIs")
            # print("Compress CDERI into Cholesky Vectors")
            # print("# Compress CDERI Alpha and Beta into Cholesky Vectors")
            # print("# Tighter Chol_cutoff is recommended for LNO")
            print(f"Cholesky cutoff is: {chol_cut}")
            # eri_clas = np.array(eri_clas)
            # eri_clas = lib.einsum('gP,gQ->PQ', cderi_clas, cderi_clas, optimize='optimal')
            # cderi_clas = pyscf_interface.modified_cholesky(eri_clas, max_error=chol_cut)
            # cderi_clas = compress_cderi_cpu(cderi_clas, thresh=chol_cut)
            cderi_clas = jnp.array(cderi_clas)
            cderi_clas = compress_cderi_gpu(cderi_clas, thresh=chol_cut)
            print("Compress CDERI into Cholesky Vectors by SVD")
            cderi_clas = np.array(cderi_clas)
            cderi_clas = lib.unpack_tril(cderi_clas, axis=-1)
            # print("Compress CDERI into Cholesky Vectors by DF2CD")
            # cderi_clas = cholesky.df2chol_faster(cderi_clas, max_error=chol_cut)
            # cderi_clas = np.array(cderi_clas)
            # cderi_a = jnp.array(cderi_a)
            # cderi_b = jnp.array(cderi_b)
            # cderi_a, cderi_b = compress_cderi_AB_gpu(cderi_a, cderi_b, thresh=1e-6)
            # cderi_a = np.array(cderi_a)
            # cderi_b = np.array(cderi_b)
            # print(f"# Raw CDERI Alpha shape: {cderi_a.shape}")
            # print(f"# Raw CDERI Beta shape: {cderi_b.shape}")
            time6 = time.perf_counter()
            # chol_clas = jnp.array(chol_clas)
            # chol_clas = unpack_symmetric(chol_clas, nclas)
            # print(f"# Compressed Cholesky Vectors in cLAS shape: {cderi_clas.shape}")
            # cderi_clas = lib.unpack_tril(cderi_clas, axis=-1)
            # cbola = oe.contract('pr,grs,sq->gpq', a2c.T, chol_clas, a2c, backend='jax')
            # cholb = oe.contract('pr,grs,sq->gpq', b2c.T, chol_clas, b2c, backend='jax')
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

        elif use_df:
            raise  NotImplementedError('Uncomment the code below and change a bit')
            # print("# Transform DF Tenor into LNO Basis")
            # chola = lib.einsum('pr,grs,sq->gpq',mo_coeff[0].T,chol_df,mo_coeff[0])
            # cholb = lib.einsum('pr,grs,sq->gpq',mo_coeff[1].T,chol_df,mo_coeff[1])
            # chola = chola[:,ncore[0]:ncore[0]+ncas[0],ncore[0]:ncore[0]+ncas[0]]
            # cholb = cholb[:,ncore[1]:ncore[1]+ncas[1],ncore[1]:ncore[1]+ncas[1]]
            # print(f'# Alpha chol shape: {chola.shape}')
            # print(f'#  Beta chol shape: {cholb.shape}')
    else:
        raise  NotImplementedError('Use DF Only!')
        # eri_clas = ao2mo.kernel(mf.mol,clas_coeff,compact=False)
        # chol_clas = pyscf_interface.modified_cholesky(eri_clas,max_error=chol_cut)
        # chol_clas = chol_clas.reshape((-1, nclas, nclas))
        # chol_a = lib.einsum('pr,grs,sq->gpq',a2c.T,chol_clas,a2c)
        # chol_b = lib.einsum('pr,grs,sq->gpq',b2c.T,chol_clas,b2c)
    
    # v0_a = 0.5 * oe.contract("nik,njk->ij", chola, chola, backend='jax')
    # v0_b = 0.5 * oe.contract("nik,njk->ij", cholb, cholb, backend='jax')
    v0_a = 0.5 * lib.einsum("gik,gjk->ij", cderi_a, cderi_a, optimize='optimal')
    v0_b = 0.5 * lib.einsum("gik,gjk->ij", cderi_b, cderi_b, optimize='optimal')
    # h1mod_a = jnp.array(h1e[0] - v0_a)
    # h1mod_b = jnp.array(h1e[1] - v0_b)
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

    write_dqmc(h1e,[h1mod_a,h1mod_b],[cderi_a, cderi_b],
               nelec,ncas,enuc,mf.e_tot,filename=chol_file)
    
    # Clean up all large arrays before returning
    del cderi_clas, cderi_a, cderi_b
    del h1e, h1mod_a, h1mod_b
    del clas_coeff, a2c, b2c
    del v0_a, v0_b

    return nelec, ncas

def write_dqmc(
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
        fh5["header"] = np.array([nelec[0], nelec[1], nmo[0], nmo[1], chol_a.shape[0]])
        fh5["h1e_a"] = h1e_a.flatten()
        fh5["h1e_b"] = h1e_b.flatten()
        fh5["h1mod_a"] = h1mod_a.flatten()
        fh5["h1mod_b"] = h1mod_b.flatten()
        fh5["chol_a"] = chol_a.flatten()
        fh5["chol_b"] = chol_b.flatten()
        fh5["energy_core"] = enuc
        fh5["emf"] = emf


def _prep_afqmc(option_file="options.bin",
                mo_file="mo_coeff.npz",
                amp_file="amplitudes.npz",
                chol_file="FCIDUMP_chol"):

    with open(option_file, "rb") as f:
        options = pickle.load(f)

    options["dt"] = options.get("dt", 0.005)
    options["n_exp_terms"] = options.get("n_exp_terms", 6)
    options["n_walkers"] = options.get("n_walkers", 300)
    options["n_prop_steps"] = options.get("n_prop_steps", 50)
    options["n_sr_blocks"] = options.get("n_sr_blocks", 1)
    options["n_ene_blocks"] = options.get("n_ene_blocks", 1)
    options["n_blocks"] = options.get("n_blocks", 500)
    options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))
    options["n_eql"] = options.get("n_eql", 3)
    options["walker_type"] = options.get("walker_type", "uhf")
    options["trial"] = options.get("trial", "uccsd_pt2")
    options["n_batch"] = options.get("n_batch", 1)
    options['use_gpu'] = options.get("use_gpu", True)
    options['mix_precision'] = options.get("mix_precision", True)

    if "chunk" in options["trial"]:
        options["nchol_chunk"] = options.get("nchol_chunk", 50)

    with h5py.File(chol_file, "r") as fh5:
        [nelec_a,nelec_b,nmo_a,nmo_b,nchol] = fh5["header"]
        h0 = jnp.array(fh5.get("energy_core"))
        emf = jnp.array(fh5.get("emf"))
        h1_a = jnp.array(fh5.get("h1e_a")).reshape(nmo_a, nmo_a)
        h1_b = jnp.array(fh5.get("h1e_b")).reshape(nmo_b, nmo_b)
        h1mod_a = jnp.array(fh5.get("h1mod_a")).reshape(nmo_a, nmo_a)
        h1mod_b = jnp.array(fh5.get("h1mod_b")).reshape(nmo_b, nmo_b)
        chol_a = jnp.array(fh5.get("chol_a")).reshape(-1, nmo_a, nmo_a)
        chol_b = jnp.array(fh5.get("chol_b")).reshape(-1, nmo_b, nmo_b)

    assert chol_a.shape[0] == chol_b.shape[0]

    nelec_a, nelec_b, nmo_a, nmo_b, nchol \
        = int(nelec_a), int(nelec_b), int(nmo_a), int(nmo_b), int(nchol)
    nelec = (nelec_a, nelec_b)
    norb = (nmo_a, nmo_b)

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
    wave_data['prjlo'] = [prja,prjb]
    mo_coeff_a = jnp.array(np.eye(nmo_a))
    mo_coeff_b = jnp.array(np.eye(nmo_b))
    wave_data["mo_coeff"] = [
            mo_coeff_a[:, : nelec[0]],
            mo_coeff_b[:, : nelec[1]],
            ]

    if options["trial"] == "uhf":
        trial = ulno_wavefunctions.uhf(norb, nelec, n_batch=options["n_batch"])
    elif options["trial"] == "uccsd_pt_ad":
        trial = ulno_wavefunctions.uccsd_pt_ad(norb, nelec, n_batch = options["n_batch"])
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
    elif options["trial"] == "uccsd_pt":
        trial = ulno_wavefunctions.uccsd_pt(norb, nelec, n_batch = options["n_batch"])
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
    elif "uccsd_pt2" in options["trial"]:
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
        from jax import scipy as jsp
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
            trial = ulno_wavefunctions.uccsd_pt2_ad(norb, nelec, n_batch = options["n_batch"])
            wave_data["t2aa"] = oe.contract('iajb,ik->kajb', t2aa, prja, backend='jax')
            wave_data["t2ab"] = oe.contract('iajb,ik->kajb', t2ab, prja, backend='jax')
            wave_data["t2ba"] = oe.contract('jbia,ik->kajb', t2ab, prjb, backend='jax')
            wave_data["t2bb"] = oe.contract('iajb,ik->kajb', t2bb, prjb, backend='jax')
        elif "alpha" in options["trial"]:
            trial = ulno_wavefunctions.uccsd_pt2_alpha(norb, nelec, n_batch = options["n_batch"])
            wave_data["t2aa"] = oe.contract('iajb,ik->kajb', t2aa, prja, backend='jax')
            wave_data["t2ab"] = oe.contract('iajb,ik->kajb', t2ab, prja, backend='jax')
            if "chunk" in options["trial"]:
                trial = ulno_wavefunctions.uccsd_pt2_alpha_chunk(norb, 
                                                                 nelec, 
                                                                 n_batch = options["n_batch"], 
                                                                 nchol_chunk = options["nchol_chunk"],
                                                                 mix_precision = options['mix_precision'],
                                                                 )
            if "fast" in options["trial"]:
                trial = ulno_wavefunctions.uccsd_pt2_alpha_fast(norb, nelec, n_batch = options["n_batch"])
        elif "beta" in options["trial"]:
            trial = ulno_wavefunctions.uccsd_pt2_beta(norb, nelec, n_batch = options["n_batch"])
            wave_data["t2ba"] = oe.contract('jbia,ik->kajb', t2ab, prjb, backend='jax')
            wave_data["t2bb"] = oe.contract('iajb,ik->kajb', t2bb, prjb, backend='jax')
            if "chunk" in options["trial"]:
                trial = ulno_wavefunctions.uccsd_pt2_beta_chunk(norb, 
                                                                nelec, 
                                                                n_batch = options["n_batch"], 
                                                                nchol_chunk = options["nchol_chunk"],
                                                                mix_precision = options['mix_precision']
                                                                )
            if "fast" in options["trial"]:
                trial = ulno_wavefunctions.uccsd_pt2_beta_fast(norb, nelec, n_batch = options["n_batch"])
        else:
            trial = ulno_wavefunctions.uccsd_pt2(norb, nelec, n_batch = options["n_batch"])
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
    
    del h1_a, h1_b, chol_a, chol_b

    return ham_data, prop, trial, wave_data, sampler, options

import os
def run_lnoafqmc(options,
                 option_file ='options.bin',
                 script = None):

    with open(option_file, 'wb') as f:
        pickle.dump(options, f)

    use_gpu = options["use_gpu"]
    if use_gpu:
        print(f'running AFQMC on GPU')
        config.afqmc_config = {"use_gpu": True}
        config.setup_jax()
        gpu_flag = "--use_gpu"

    else:
        print(f'running AFQMC on CPU')
        gpu_flag = ""

    
    if script is None:
        if 'pt2' in options['trial']:
            script='ccsd_pt2/run_uafqmc_nompi.py'
        else:
            raise NotImplementedError("Only support pt2CCSD trial.")
    
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/{script}"
    print(f'AFQMC script: {script}')
    
    os.system(
        f"python {script} {gpu_flag} |tee afqmc.out"
        # f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1;"
        # f"{mpi_prefix} python {script} {gpu_flag} |tee afqmc.out"
    )

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
        nelec, norb = prep_afqmc(
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
                if "Energy (blocking)" in line:
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