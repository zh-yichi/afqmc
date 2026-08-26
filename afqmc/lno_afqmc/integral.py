import jax
jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp

import numpy as np

from afqmc import integral, cholesky
from afqmc.corr_sample import integral as csi
from afqmc.lno_afqmc import tools

from pyscf import lib, scf

import time, h5py

def prjmo(prj, s1e, mo):
    # prj and reconstruct mo
    # e.g. |B_p> = |A_q><A_q|B_p>
    #            = C^A_mq C^A(T)_qn|m><n|s> C^B_sp
    mo_rec = prj @ prj.T @ s1e @ mo
    return mo_rec

# def common_las2(mf, lno_coeff, ncas, ncore, torr=1e-8):
#     print("Constructing cLAS that spans both Alpha and Beta active space")
#     s1e = mf.get_ovlp()
#     lno_acta = lno_coeff[0][:, ncore[0]:ncore[0] + ncas[0]]
#     lno_actb = lno_coeff[1][:, ncore[1]:ncore[1] + ncas[1]]

#     # Express both active spaces in the complete, orthonormal alpha MO basis
#     lno_actaa = lno_coeff[0].T @ s1e @ lno_acta
#     lno_actba = lno_coeff[0].T @ s1e @ lno_actb
#     clno_act = np.hstack([lno_actaa, lno_actba])   # redundant spanning set for the union
#     print('Naive cLAS Shape: ', clno_act.shape)

#     # Left singular vectors span the column space; singular values below torr
#     # are the linear dependence between the alpha and beta active spaces.
#     u, s, _ = np.linalg.svd(clno_act, full_matrices=False)
#     rank = int(np.count_nonzero(s > torr))
#     u = u[:, :rank]

#     print(f'Orthonormalized cLAS shape: {u.shape}')
#     print(f'cLAS singular value threshold: {torr}')
#     print(f'Smallest retained singular value: {s[rank - 1]:.2e}')
#     if rank < s.size:
#         print(f'Largest discarded singular value: {s[rank]:.2e}')
#     else:
#         print('No singular values discarded (alpha and beta spaces disjoint)')
#     print(f"Minimum size of cLAS to span both Alpha and Beta LAS: {rank}")

#     # span{|C>} = span{|A>} U span{|B>}
#     clas_coeff = lno_coeff[0] @ u   # in AO basis
#     print('True Common LAS Shape: ', clas_coeff.shape)
#     a2c = clas_coeff.T @ s1e @ lno_acta   # <C|A>
#     b2c = clas_coeff.T @ s1e @ lno_actb   # <C|B>
#     return clas_coeff, a2c, b2c

def common_las(mf, lno_coeff, ncas, ncore, torr=1e-8):
    print("Constructing cLAS that spans both Alpha and Beta active space")

    lno_acta = lno_coeff[0][:, ncore[0]:ncore[0] + ncas[0]]
    lno_actb = lno_coeff[1][:, ncore[1]:ncore[1] + ncas[1]]
    lno_actc = np.hstack([lno_acta, lno_actb])   # redundant spanning set for the union
    print('Naive cLAS Size: ', lno_actc.shape[1])

    # svd combined orbitals
    s = mf.get_ovlp()                       # AO overlap, (nao, nao)
    G = lno_actc.T @ s @ lno_actc           # MO–MO Gram matrix, (n, n), symmetric PSD
    w, V = np.linalg.eigh(G)                # eigh == SVD for symmetric PSD
    w, V = w[::-1], V[:, ::-1]              # descending
    # tol  = 1e-10 * w[0]                      # relative threshold
    keep = w > torr
    rank = keep.sum()
    print(f"union dimension = {keep.sum()} of {len(w)}")
    # S-orthonormal orbitals spanning span(moa) ∪ span(mob)
    clas_coeff = lno_actc @ (V[:, keep] / np.sqrt(w[keep]))
    # sanity check: physically orthonormal
    assert np.allclose(clas_coeff.T @ s @ clas_coeff, np.eye(rank), atol=torr)

    print(f'Orthonormalized cLAS size: {clas_coeff.shape[1]}')
    print(f'cLAS singular value threshold: {torr}')
    print(f'Smallest retained singular value: {w[keep][-1]:.2e}')
    if rank < w.size:
        print(f'Largest discarded singular value: {w[rank]:.2e}')
    else:
        print('No singular values discarded (alpha and beta spaces disjoint)')
    print(f"Minimum size of cLAS to span both Alpha and Beta LAS: {rank}")

    a2c = clas_coeff.T @ s @ lno_acta   # <C|A>
    b2c = clas_coeff.T @ s @ lno_actb   # <C|B>

    p12a, p21a = tools.mo_span(clas_coeff, s, lno_acta)
    p12b, p21b = tools.mo_span(clas_coeff, s, lno_actb)

    assert p12a < torr
    assert p12b < torr

    print(f"alpha proj comm loss {p12a}")
    print(f"beta proj comm loss  {p12b}")

    return clas_coeff, a2c, b2c

def get_las_idx(mf, lno_frozen):
    mol = mf.mol
    if isinstance(mf, scf.rhf.RHF):
        nocc = np.count_nonzero(mf.mo_occ)
        actfrag = np.array([i for i in range(mol.nao) if i not in lno_frozen])
        frzocc = np.array([i for i in range(nocc) if i in lno_frozen])
        actocc = np.array([i for i in range(nocc) if i in actfrag])
        actvir = np.array([i for i in range(nocc,mol.nao) if i in actfrag])
        nfrzocc = len(frzocc)
        nactocc = len(actocc)
        nactvir = len(actvir)
        nactorb = len(actfrag)
        # ncas = nactorb
        # ncore = nfrzocc
        # nocc = nactocc

    elif isinstance(mf, scf.uhf.UHF):
        nocc_a = int(sum(mf.mo_occ[0]))
        actfrag_a = np.array([i for i in range(mol.nao) if i not in lno_frozen[0]])
        frzocc_a = np.array([i for i in range(nocc_a) if i in lno_frozen[0]])
        actocc_a = np.array([i for i in range(nocc_a) if i in actfrag_a])
        actvir_a = np.array([i for i in range(nocc_a,mol.nao) if i in actfrag_a])
        nfrzocc_a = len(frzocc_a)
        nactocc_a = len(actocc_a)
        nactvir_a = len(actvir_a)
        nactorb_a = len(actfrag_a)
        nocc_b = int(sum(mf.mo_occ[1]))
        actfrag_b = np.array([i for i in range(mol.nao) if i not in lno_frozen[1]])
        frzocc_b = np.array([i for i in range(nocc_b) if i in lno_frozen[1]])
        actocc_b = np.array([i for i in range(nocc_b) if i in actfrag_b])
        actvir_b = np.array([i for i in range(nocc_b,mol.nao) if i in actfrag_b])
        nfrzocc_b = len(frzocc_b)
        nactocc_b = len(actocc_b)
        nactvir_b = len(actvir_b)
        nactorb_b = len(actfrag_b)
        nfrzocc = (nfrzocc_a, nfrzocc_b)
        nactocc = (nactocc_a, nactocc_b)
        nactorb = (nactorb_a, nactorb_b)
        actfrag = (actfrag_a, actfrag_b)

    return nfrzocc, nactocc, nactorb, actfrag

def get_lno_integral(mf, lno_coeff, lno_frozen, chol_cut):
    ncore, nocc, ncas, actfrag = get_las_idx(mf, lno_frozen)
    print('*** Correlation Space Size ***')
    print(f'N Occupied Orbitals: {nocc}')
    print(f'N Active Orbitals:   {ncas}')

    if isinstance(mf, scf.rhf.RHF):
        time0 = time.time()
        h1e, enuc = integral.h1e_ras(mf, lno_coeff, ncas, ncore, useDF=True)
        print(f"Build effective h0 and h1 time: {time.time()-time0:.6f} s")
        
        lno_act = lno_coeff[:,actfrag]
        print("Composing CDERIs from DF")
        time0 = time.time()
        naux = mf.with_df.get_naoaux()
        npair = ncas*(ncas+1)//2
        naux = mf.with_df.get_naoaux()
        cderi_las = np.zeros((naux, npair))
        p1 = 0
        for cderi in mf.with_df.loop():
            cderi = lib.unpack_tril(cderi, axis=-1)
            cderi = cholesky.cderi2mo_gpu(jnp.array(cderi), lno_act)
            p0, p1 = p1, p1 + cderi.shape[0]
            cderi_las[p0:p1] = np.array(cderi)
        print(f"Build CDERI in cLAS time: {time.time() - time0:.6f} s")
        print(f"Raw CDERI in LAS shape: {cderi_las.shape}")
        time0 = time.time()
        # print(f"Compress CDERI into LAS by SVD with cutoff: {chol_cut}")
        # cderi_las = cholesky.compress_cderi_gpu(jnp.array(cderi_las), thresh=chol_cut) # svd
        # cderi_las = cholesky.unpack_symmetric(cderi_las, ncas)
        print(f"Compress CDERI 2 Chol: {chol_cut}")
        chol_full, final_nchol = cholesky.df2chol_gpu(jnp.array(cderi_las), max_error=chol_cut)
        cderi_las = chol_full[:final_nchol]
        print(f"Compress CDERI time: {time.time() - time0:.6f} s")
        print("Finished calculating Integrals")
        print(f'LAS Cholesky shape: {cderi_las.shape}')

    elif isinstance(mf, scf.uhf.UHF):
        time0 = time.time()
        h1e, enuc = integral.h1e_uas(mf, lno_coeff, ncas, ncore, useDF=True)
        print(f"Build effective h0 and h1 time: {time.time()-time0:.6f} s")
        print("Composing CDERIs from DF")
        time0 = time.time()
        clas_coeff, a2c, b2c = common_las(mf, lno_coeff, ncas, ncore, torr=1e-5)
        print(f"Build Common LAS time: {time.time()-time0:.6f} s")
        time0 = time.time()
        nclas = clas_coeff.shape[1]
        npair = nclas*(nclas+1)//2
        naux = mf.with_df.get_naoaux()
        cderi_clas = np.zeros((naux, npair))
        p1 = 0
        for cderi in mf.with_df.loop():
            cderi = lib.unpack_tril(cderi, axis=-1)
            cderi = cholesky.cderi2mo_gpu(jnp.array(cderi), clas_coeff)
            p0, p1 = p1, p1 + cderi.shape[0]
            cderi_clas[p0:p1] = np.array(cderi)
        print(f"Build CDERI in cLAS time: {time.time() - time0:.6f} s")
        print(f"Raw CDERI in cLAS shape: {cderi_clas.shape}")
        print(f"Compress CDERI 2 Chol in cLAS cutoff: {chol_cut}")
        time0 = time.time()
        # cderi_clas = cholesky.compress_cderi_gpu(cderi_clas, thresh=chol_cut) # SVD
        # cderi_clas = cholesky.unpack_symmetric(cderi_clas, nclas)
        chol_full, final_nchol = cholesky.df2chol_gpu(jnp.array(cderi_clas), max_error=chol_cut) # CD
        cderi_clas = chol_full[:final_nchol]
        cderi_a = cholesky.cderi2mo_gpu(cderi_clas, a2c)
        cderi_b = cholesky.cderi2mo_gpu(cderi_clas, b2c)
        cderi_a = cholesky.unpack_symmetric(cderi_a, ncas[0])
        cderi_b = cholesky.unpack_symmetric(cderi_b, ncas[1])
        print(f"Compress CDERI time: {time.time() - time0:.6f} s")
        print("Finished calculating Integrals")
        print(f'LAS Alpha Cholesky shape: {cderi_a.shape}')
        print(f'LAS Beta  Cholesky shape: {cderi_b.shape}')
        cderi_las = (cderi_a, cderi_b)

    return enuc, h1e, cderi_las, nocc, ncas

def get_lnoproj(uocc_loc):
    '''return the LNO fragment projector |I><I| for I in F'''
    if isinstance(uocc_loc, (np.ndarray, jnp.ndarray)):
        prjlo = uocc_loc @ uocc_loc.T.conj()
    elif isinstance(uocc_loc, (list, tuple)):
        prjlo = (uocc_loc[0] @ uocc_loc[0].T.conj(),
                 uocc_loc[1] @ uocc_loc[1].T.conj())
    else:
        raise TypeError(
            f"uocc_loc must be an ndarray or a list/tuple of two, "
            f"got {type(uocc_loc).__name__}."
        )
    return prjlo

def prep_lno_integral(
        mf,
        lno_coeff,
        lno_frozen,
        uocc_loc,
        t1 = None,
        t2 = None,
        chol_cut=1e-5,
        amp_file="amplitudes.npz",
        chol_file="FCIDUMP_chol",
        ):
    
    if getattr(mf, "with_df", None) is not None:
        print('Find Density Fit Teonsers in MF object')
        print('Integrals will be built by DF Tensors')
        useDF = True
    else:
        raise  NotImplementedError('LNO Only Support Mean-Field Object with Density Fitting!')
    
    prjlo = get_lnoproj(uocc_loc)
    
    integral.save_cc_amplitude(t1=t1, t2=t2, amp_file=amp_file)

    print('Calculating Effective Active Space One-electron Integrals')

    h0, h1, chol, nocc, norb = get_lno_integral(mf, lno_coeff, lno_frozen, chol_cut)

    write_integral(nocc, norb, h0, h1, chol, mf.e_tot, prjlo, filename=chol_file)

    return None

def prep_cfs_integral(
        mf1, mf2,
        lno_coeff1, lno_coeff2,
        lno_frozen1, lno_frozen2,
        uocc_loc1, uocc_loc2,
        t11=None, t12=None,
        t21=None, t22=None,
        chol_cut=1e-5,
        amp_file1="amplitudes1.npz", amp_file2="amplitudes2.npz",
        chol_file1="FCIDUMP_chol1", chol_file2="FCIDUMP_chol2",
        ):

    def _require_df(mf, tag):
        if getattr(mf, "with_df", None) is None:
            raise NotImplementedError(
                'LNO Only Support Mean-Field Object with Density Fitting!')
        print(f'{tag}: found density-fit tensors; integrals will be built from DF tensors')
    _require_df(mf1, "mf1")
    _require_df(mf2, "mf2")

    def _norb_from_npair(npair):
        """Invert npair = n(n+1)/2 -> n (per spin channel)."""
        return int(round((np.sqrt(8 * npair + 1) - 1) / 2))

    # total occupied count per system (RHF scalar; UHF (n_alpha, n_beta))
    def _nocc(mf):
        occ = mf.mo_occ
        if np.ndim(occ) == 2:                       # UHF: (2, nmo)
            return (int(np.count_nonzero(occ[0])), int(np.count_nonzero(occ[1])))
        return int(np.count_nonzero(occ))           # RHF
    nocc1 = _nocc(mf1)
    nocc2 = _nocc(mf2)

    # ---- align active MOs, occ and vir blocks SEPARATELY ----
    (lno_coeff1, lno_coeff2, u1occ, u1vir, u2occ, u2vir) \
        = csi.align_mo(lno_coeff1, lno_coeff2, nocc1, nocc2, lno_frozen1, lno_frozen2, report=True)

    # ---------- amplitude + uocc_loc transforms (RHF and UHF) ----------
    #  t1 = (occ, vir)
    #  t2 = (occ, occ, vir, vir)  [ijab]; UHF t2ab = (occ_a, occ_b, vir_a, vir_b)
    def _t1_rot(t1, Uo, Uv):
        return np.einsum('ji,ba,jb->ia', Uo.conj(), Uv.conj(), t1, optimize=True)

    def _t2_rot(t2, Uo_i, Uo_j, Uv_a, Uv_b):
        # each of the 4 indices rotated by its own spin's block rotation
        return np.einsum('ki,lj,ca,db,klcd->ijab',
                         Uo_i.conj(), Uo_j.conj(), Uv_a.conj(), Uv_b.conj(), t2,
                         optimize=True)

    def _transform_amps(t1, t2, uocc, uvir):
        """uocc/uvir: arrays (RHF) or [alpha, beta] lists (UHF).
           RHF: t1 (occ,vir), t2 (occ,occ,vir,vir).
           UHF: t1=(t1a,t1b); t2=(t2aa,t2ab,t2bb),
                t2ab = (occ_a, occ_b, vir_a, vir_b)."""
        if not isinstance(uocc, (list, tuple)):                    # RHF
            t1n = None if t1 is None else _t1_rot(t1, uocc, uvir)
            t2n = None if t2 is None else _t2_rot(t2, uocc, uocc, uvir, uvir)
            return t1n, t2n
        Uoa, Uob = uocc; Uva, Uvb = uvir                          # UHF
        t1n = t2n = None
        if t1 is not None:
            t1a, t1b = t1
            t1n = (_t1_rot(t1a, Uoa, Uva), _t1_rot(t1b, Uob, Uvb))
        if t2 is not None:
            t2aa, t2ab, t2bb = t2
            t2n = (_t2_rot(t2aa, Uoa, Uoa, Uva, Uva),             # i,j,a,b all alpha
                   _t2_rot(t2ab, Uoa, Uob, Uva, Uvb),             # occ_a, occ_b, vir_a, vir_b
                   _t2_rot(t2bb, Uob, Uob, Uvb, Uvb))             # all beta
        return t1n, t2n

    def _transform_uocc_loc(uocc_loc, uocc):
        """uocc_loc = <lno_actocc|lo>; rotate occupied ROW index by uocc = <occ_old|occ_new>."""
        if isinstance(uocc, (list, tuple)):                       # UHF: per-spin pair
            return [uocc[s].conj().T @ uocc_loc[s] for s in (0, 1)]
        return uocc.conj().T @ uocc_loc                           # RHF

    t11, t21 = _transform_amps(t11, t21, u1occ, u1vir)
    t12, t22 = _transform_amps(t12, t22, u2occ, u2vir)

    uocc_loc1 = _transform_uocc_loc(uocc_loc1, u1occ)
    uocc_loc2 = _transform_uocc_loc(uocc_loc2, u2occ)

    prjlo1 = get_lnoproj(uocc_loc1) # uocc_loc = <lno_active_occ|lo>
    prjlo2 = get_lnoproj(uocc_loc2)

    integral.save_cc_amplitude(t1=t11, t2=t21, amp_file=amp_file1)
    integral.save_cc_amplitude(t1=t12, t2=t22, amp_file=amp_file2)

    print('Calculating Effective Active Space One-electron Integrals')
    h01, h11, chol1, nelec1, norb1 = get_lno_integral(mf1, lno_coeff1, lno_frozen1, chol_cut)
    h02, h12, chol2, nelec2, norb2 = get_lno_integral(mf2, lno_coeff2, lno_frozen2, chol_cut)

    if isinstance(chol1, (np.ndarray, jax.Array)):
        nchol1 = chol1.shape[0]
        nchol2 = chol2.shape[0]
        chol1 = cholesky.pack_symmetric(chol1)
        chol2 = cholesky.pack_symmetric(chol2)
        if nchol1 != nchol2:
            print("Pad the smaller Cholesky")
            chol1, chol2 = csi.match_nchol(chol1, chol2)
        assert chol1.shape[0] == chol2.shape[0], "nchol mismatch after match_nchol"
        assert chol1.shape[1] == norb1 * (norb1 + 1) // 2
        assert chol2.shape[1] == norb2 * (norb2 + 1) // 2

        chol1, chol2 = csi.align_gauge(chol1, chol2, report=True)
        chol1 = cholesky.unpack_symmetric(chol1).reshape(chol1.shape[0], -1)
        chol2 = cholesky.unpack_symmetric(chol2).reshape(chol2.shape[0], -1)

    elif isinstance(chol1, (tuple, list)):
        nchol1 = chol1[0].shape[0]
        nchol2 = chol2[0].shape[0]
        chol1a, chol1b = chol1
        chol2a, chol2b = chol2
        chol1a = cholesky.pack_symmetric(chol1a)
        chol1b = cholesky.pack_symmetric(chol1b)
        chol2a = cholesky.pack_symmetric(chol2a)
        chol2b = cholesky.pack_symmetric(chol2b)
        if nchol1 != nchol2:
            print("Pad the smaller Cholesky")
            chol1a, chol2a = csi.match_nchol(chol1a, chol2a)
            chol1b, chol2b = csi.match_nchol(chol1b, chol2b)
        assert chol1a.shape[0] == chol1b.shape[0] == chol2a.shape[0] == chol2b.shape[0], \
            "nchol mismatch across spin components after match_nchol"

        (chol1a,chol1b), (chol2a,chol2b) \
            = csi.align_gauge((chol1a, chol1b), (chol2a, chol2b), report=True)   # ONE shared O
        chol1a = cholesky.unpack_symmetric(chol1a).reshape(chol1a.shape[0], -1)
        chol1b = cholesky.unpack_symmetric(chol1b).reshape(chol1b.shape[0], -1)
        chol2a = cholesky.unpack_symmetric(chol2a).reshape(chol2a.shape[0], -1)
        chol2b = cholesky.unpack_symmetric(chol2b).reshape(chol2b.shape[0], -1)
        chol1 = (chol1a, chol1b)
        chol2 = (chol2a, chol2b)

    else:
        raise TypeError(f"unexpected chol type: {type(chol1)}")

    write_integral(nelec1, norb1, h01, h11, chol1, mf1.e_tot, prjlo1, filename=chol_file1)
    write_integral(nelec2, norb2, h02, h12, chol2, mf2.e_tot, prjlo2, filename=chol_file2)
    return None

def write_integral(nocc, norb, h0, h1, chol, emf, prjlo, filename="FCIDUMP_chol"):
    
    unrestricted = isinstance(h1, (list, tuple))
    
    with h5py.File(filename, "w") as fh5:
        fh5["nocc"] = np.asarray(nocc)
        fh5["norb"]  = np.asarray(norb)
        fh5["h0"]    = h0
        fh5["emf"]   = emf
        if unrestricted:
            for s, tag in enumerate(("a", "b")):
                fh5[f"h1_{tag}"]    = np.asarray(h1[s])
                fh5[f"chol_{tag}"]  = np.asarray(chol[s])
                fh5[f"prjlo_{tag}"] = np.asarray(prjlo[s])
        else:
            fh5["h1"]    = np.asarray(h1)
            fh5["chol"]  = np.asarray(chol)
            fh5["prjlo"] = np.asarray(prjlo)
    
    return None

def load_integral(filename="FCIDUMP_chol"):

    assert jax.config.jax_enable_x64, \
            "x64 is disabled; h0/emf would load as float32. " \
            "Call jax.config.update('jax_enable_x64', True) at startup."
    
    def _recover_int(x):
        # int stored as 0-d -> Python int; pair stored as (2,) -> int array
        arr = np.asarray(x)
        return int(arr) if arr.ndim == 0 else arr.astype(np.int64)
    
    with h5py.File(filename, "r") as fh5:

        nocc = _recover_int(fh5["nocc"][()])
        norb  = _recover_int(fh5["norb"][()])
        h0    = jnp.asarray(fh5["h0"][()], dtype=jnp.float64)
        emf   = jnp.asarray(fh5["emf"][()], dtype=jnp.float64)

        if "h1" in fh5:            # restricted
            h1_np   = fh5["h1"][()]
            chol_np = fh5["chol"][()]
            n = int(round(h1_np.size ** 0.5))     # norb from the data itself
            
            assert n == int(norb)

            h1    = jnp.array(h1_np).reshape(n, n)
            chol  = jnp.array(chol_np).reshape(-1, n, n)
            prjlo = jnp.array(fh5["prjlo"][()])
            nchol = chol.shape[0]
        
        elif "h1_a" in fh5:       # unrestricted, possibly unequal spins
            h1a, h1b = fh5["h1_a"][()], fh5["h1_b"][()]
            na = int(round(h1a.size ** 0.5))
            nb = int(round(h1b.size ** 0.5))
            
            assert (na, nb) == (int(norb[0]), int(norb[1]))

            h1    = (jnp.array(h1a).reshape(na, na),
                     jnp.array(h1b).reshape(nb, nb))
            chol  = (jnp.array(fh5["chol_a"][()]).reshape(-1, na, na),
                     jnp.array(fh5["chol_b"][()]).reshape(-1, nb, nb))
            prjlo = (jnp.array(fh5["prjlo_a"][()]),
                     jnp.array(fh5["prjlo_b"][()]))
            nchola = chol[0].shape[0]
            ncholb = chol[1].shape[0]
            assert nchola ==  ncholb
            nchol = nchola
        
        else:
            raise KeyError(
                f"{filename} has neither an 'h1' (restricted) nor an "
                f"'h1_a'/'h1_b' (unrestricted) integral block."
            )
    
    return nocc, norb, nchol, h0, h1, chol, emf, prjlo