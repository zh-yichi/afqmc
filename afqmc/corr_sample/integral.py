from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD
from pyscf import scf

from afqmc import integral

from typing import Optional, Union

from jax import config
config.update("jax_enable_x64", True)
from jax import numpy as jnp
from jax import jit

import numpy as np

@jit
def right_procrustes(A, B):
    '''
    given matices A, B, find R s.t, min_R||A-BR||F the Frobenius norm
    that is, R rotates (flips) the columns of B to match A element-wise
    '''
    m = B.T.conj() @ A
    u, _, v_dag = jnp.linalg.svd(m)
    R = u @ v_dag
    return R

@jit
def left_procrustes(A, B):
    '''
    given matices A, B, find R s.t, min_R||A-RB||F the Frobenius norm
    that is, R rotates (flips) the rows of B to match A element-wise
    '''
    m = A @ B.T.conj()
    u, _, v_dag = jnp.linalg.svd(m)
    R = u @ v_dag
    return R


def _split_active(nmo, nocc, frozen):
    """From a frozen INDEX list + total occupied count nocc, return the active
    occupied and active virtual column indices."""
    fr = np.array([], dtype=int) if frozen is None else \
         np.atleast_1d(np.asarray(frozen, dtype=int)).ravel()
    is_frozen = np.zeros(nmo, dtype=bool)
    if fr.size:
        is_frozen[fr] = True
    idx = np.arange(nmo)
    occ_act = idx[(idx < nocc)  & ~is_frozen]     # occupied, not frozen
    vir_act = idx[(idx >= nocc) & ~is_frozen]     # virtual, not frozen
    return occ_act, vir_act


def _procrustes_pad(ref, big):
    """Pad ref (nao,kref) with zero cols to kbig, rotate big (nao,kbig)
    onto it -> R (kbig,kbig)."""
    nao, kref = ref.shape
    kbig = big.shape[1]
    if kref == 0 or kbig == 0:
        return np.eye(kbig, dtype=big.dtype)
    A_pad = np.zeros((nao, kbig), dtype=np.result_type(ref.dtype, big.dtype))
    A_pad[:, :kref] = ref
    return np.asarray(right_procrustes(jnp.asarray(A_pad), jnp.asarray(big)))


def _align_block(mo1, mo2, cols1, cols2, report=False, tag=""):
    """Align mo1[:,cols1] vs mo2[:,cols2]: pad the smaller, rotate the larger
    onto it. Writes rotated columns back (in place); returns R1, R2 with the
    reference (smaller) block getting identity. The smaller block is returned
    at its ORIGINAL size -- padding lives only inside _procrustes_pad."""
    a1 = mo1[:, cols1]; a2 = mo2[:, cols2]
    k1, k2 = a1.shape[1], a2.shape[1]
    if k1 <= k2:
        R = _procrustes_pad(a1, a2); R1 = np.eye(k1, dtype=R.dtype); R2 = R
        a1n, a2n, kref = a1, a2 @ R, k1
    else:
        R = _procrustes_pad(a2, a1); R1 = R; R2 = np.eye(k2, dtype=R.dtype)
        a1n, a2n, kref = a1 @ R, a2, k2
    if report:
        ref = a1 if k1 <= k2 else a2
        old = (a2 if k1 <= k2 else a1)[:, :kref]      # larger's first kref cols, pre-rotation
        new = (a2n if k1 <= k2 else a1n)[:, :kref]    # ...post-rotation
        def _stats(X, Y):
            d = np.linalg.norm(X - Y)                 # Frobenius residual
            den = np.linalg.norm(X) * np.linalg.norm(Y)
            ov = np.real(np.vdot(X, Y)) / den if den else 1.0
            return d, ov
        db, ob = _stats(ref, old)
        da, oa = _stats(ref, new)
        print(f"  {tag:12s} k1={k1} k2={k2} shared={kref}  "
              f"||d||_F {db:.3e} -> {da:.3e}   overlap {ob:.4f} -> {oa:.6f}")
    mo1[:, cols1] = a1n; mo2[:, cols2] = a2n
    return R1, R2


def _align_one_spin(mo1, mo2, nocc1, nocc2, frozen1, frozen2, report, tag=""):
    mo1 = np.array(mo1); mo2 = np.array(mo2)
    o1, v1 = _split_active(mo1.shape[1], nocc1, frozen1)
    o2, v2 = _split_active(mo2.shape[1], nocc2, frozen2)
    # occ and vir aligned SEPARATELY -> no occ/vir mixing
    Ro1, Ro2 = _align_block(mo1, mo2, o1, o2, report, tag + " occ")
    Rv1, Rv2 = _align_block(mo1, mo2, v1, v2, report, tag + " vir")
    return mo1, mo2, (Ro1, Rv1), (Ro2, Rv2)


def align_mo(mo_coeff1, mo_coeff2, nocc1, nocc2,
             frozen1=None, frozen2=None, report=False):
    """
    Align two systems' active MOs, occupied and virtual blocks SEPARATELY,
    padding the smaller block and Procrustes-rotating the larger onto it.

    mo_coeff1/2 : (nao, nmo) array (RHF) or [alpha, beta] list/tuple (UHF).
    nocc1/2     : total occupied count (RHF scalar; UHF (n_a, n_b)).
    frozen1/2   : frozen MO index list (RHF) or [alpha_list, beta_list] (UHF).

    Returns:
      mo_coeff1, mo_coeff2 (aligned),
      u1actocc, u1actvir, u2actocc, u2actvir
    where u..occ = <occ_old|occ_new>, u..vir = <vir_old|vir_new> for that
    active block (reference system gets identity). t1 transforms as
      t1_new[i,a] = sum_{jb} uocc*[j,i] uvir*[b,a] t1[j,b]
    and t2 with occ rotations on the occupied indices, vir on the virtual.
    """
    print("Procrustes align (occ & vir separately): rotate larger block onto smaller")
    uhf = isinstance(mo_coeff1, (list, tuple))
    if not uhf:
        mo1, mo2, (Ro1, Rv1), (Ro2, Rv2) = _align_one_spin(
            mo_coeff1, mo_coeff2, nocc1, nocc2, frozen1, frozen2, report, "rhf")
        return mo1, mo2, Ro1, Rv1, Ro2, Rv2

    mo1o, mo2o = [], []
    u1occ, u1vir, u2occ, u2vir = [], [], [], []
    for s, name in ((0, "alpha"), (1, "beta")):
        f1 = None if frozen1 is None else frozen1[s]
        f2 = None if frozen2 is None else frozen2[s]
        m1, m2, (Ro1, Rv1), (Ro2, Rv2) = _align_one_spin(
            mo_coeff1[s], mo_coeff2[s], nocc1[s], nocc2[s], f1, f2, report, name)
        mo1o.append(m1); mo2o.append(m2)
        u1occ.append(Ro1); u1vir.append(Rv1); u2occ.append(Ro2); u2vir.append(Rv2)
    return mo1o, mo2o, u1occ, u1vir, u2occ, u2vir

# def _report_line(A, B, tag):
#     res = np.linalg.norm(A - B)
#     denom = np.linalg.norm(A) * np.linalg.norm(B)
#     ov = np.real(np.vdot(A, B)) / denom if denom else 1.0
#     print(f"  {tag:18s} ||d||_F = {res:.6e}   overlap = {ov:.6f}")

# def _align_blocks(mo1, mo2, nocc1, nocc2, frozen, report, prefix=""):
#     names = ("frozen", "occ", "vir")
#     b1 = (mo1[:, :frozen], mo1[:, frozen:nocc1], mo1[:, nocc1:])
#     b2 = (mo2[:, :frozen], mo2[:, frozen:nocc2], mo2[:, nocc2:])
#     aligned = []
#     for name, a, b in zip(names, b1, b2):
#         ba = b @ np.array(right_procrustes(a, b))
#         if report and a.size:
#             tag = f"{prefix}{name}"
#             if a.shape == b.shape:                 # 'before' needs equal block sizes
#                 _report_line(a, b,  f"{tag} before")
#             _report_line(a, ba, f"{tag} after")
#         aligned.append(ba)
#     return np.hstack(aligned)

# def align_rmo(mo1, mo2, frozen=0, report=False):
#     nocc1 = np.count_nonzero(mf1.mo_occ)
#     nocc2 = np.count_nonzero(mf2.mo_occ)
#     return _align_blocks(mf1.mo_coeff, mf2.mo_coeff, nocc1, nocc2, frozen, report)

# def align_umo(mo1, mo2, frozen=0, report=False):
#     matched = []
#     for s, name in ((0, "alpha "), (1, "beta  ")):
#         nocc1 = np.count_nonzero(mf1.mo_occ[s])
#         nocc2 = np.count_nonzero(mf2.mo_occ[s])
#         matched.append(_align_blocks(mf1.mo_coeff[s], mf2.mo_coeff[s],
#                                      nocc1, nocc2, frozen, report, prefix=name))
#     return np.array(matched)

# def align_mo(mo1, mo2, frozen=0, report=False):
#     print("Procrustes Aligning mo1 and mo2")
#     if isinstance(mo1, (np.ndarray, jax.Array)):
#         return align_rmo(mo1, mo2, frozen, report)
#     elif isinstance(mo1, (tuple, list)):
#         return align_umo(mo1, mo2, frozen, report)

def match_nchol(chol1, chol2):
    """
    Pad the Cholesky tensor with fewer vectors so both share the same nchol.

    Each tensor may be (nchol, norb, norb) or (nchol, norb*(norb+1)//2).
    Only axis 0 (nchol) is padded with zero vectors; the trailing
    dimensions are left as-is, so the two systems may differ in norb.

    Returns (chol1, chol2) with matched nchol.
    """
    chol1 = np.asarray(chol1)
    chol2 = np.asarray(chol2)

    nchol_max = max(chol1.shape[0], chol2.shape[0])

    def pad(chol):
        n_pad = nchol_max - chol.shape[0]
        if n_pad == 0:
            return chol
        pad_width = [(0, n_pad)] + [(0, 0)] * (chol.ndim - 1)
        return np.pad(chol, pad_width, mode="constant", constant_values=0.0)

    return pad(chol1), pad(chol2)

def _pad_cols(m, npair):
    """Zero-pad a (ng, p) matrix up to npair columns (p <= npair)."""
    if m.shape[1] == npair:
        return m
    out = np.zeros((m.shape[0], npair), dtype=m.dtype)
    out[:, :m.shape[1]] = m
    return out

def align_gauge(L1, L2, report=False):
    """
    Align two Cholesky sets to a common auxiliary (g-index) gauge. Assumes ng
    (rows) is already matched. If npair differs, the smaller-npair set is
    zero-padded on its COLUMNS to the larger width, ONE row rotation O (ng x ng)
    is fit to match, and applied to the LARGER set (left/row Procrustes).

    Returns both sets: the smaller one at its ORIGINAL npair (unpadded), the
    larger one row-rotated. Zero-padding the shared fit and the unitary row
    rotation both leave each system's integrals (Lᴴ L) unchanged.

    L1, L2 : (ng, npair) arrays, OR tuples (alpha, beta) that share the g-index
             (UHF) -> one common O across components.
    """
    single = not isinstance(L2, (tuple, list))
    L1s = [np.asarray(a) for a in ([L1] if single else list(L1))]
    L2s = [np.asarray(b) for b in ([L2] if single else list(L2))]
    k = len(L2s)

    p1 = [a.shape[1] for a in L1s]
    p2 = [b.shape[1] for b in L2s]
    pmax = [max(p1[i], p2[i]) for i in range(k)]
    F1 = [_pad_cols(L1s[i], pmax[i]) for i in range(k)]     # column-padded copies (for the fit)
    F2 = [_pad_cols(L2s[i], pmax[i]) for i in range(k)]

    # rotate the set with the LARGER active space onto the smaller (reference)
    rotate_L2 = (sum(p1) <= sum(p2))
    ref = F1 if rotate_L2 else F2
    big = F2 if rotate_L2 else F1

    refc = np.concatenate(ref, axis=1)
    bigc = np.concatenate(big, axis=1)
    ng = refc.shape[0]
    if refc.shape[1] < ng:
        print(f"Warning: shared block has {refc.shape[1]} columns < ng={ng}; "
              f"gauge O underdetermined in the null space.")

    O = np.asarray(left_procrustes(refc, bigc))       # min_O ||ref - O big||
    big_rot = [O @ b for b in big]

    if report:
        def _line(A, B, tag):
            res = np.linalg.norm(A - B)
            den = np.linalg.norm(A) * np.linalg.norm(B)
            ov = np.real(np.vdot(A, B)) / den if den else 1.0
            print(f"{tag:8s} ||d||_F={res:.6e}  overlap={ov:.6f}")
        _line(refc, bigc, "before")
        _line(refc, np.concatenate(big_rot, axis=1), "after")

    # smaller side returned ORIGINAL (unpadded); larger side rotated, cropped
    # back to its own real npair (rotation is on rows, so columns are untouched
    # in count -- the crop just drops the zero-pad columns if any were added).
    if rotate_L2:
        L1o = L1s
        L2o = [big_rot[i][:, :p2[i]] for i in range(k)]
    else:
        L1o = [big_rot[i][:, :p1[i]] for i in range(k)]
        L2o = L2s

    return (L1o[0], L2o[0]) if single else (tuple(L1o), tuple(L2o))

# def align_gauge(L1, L2, report=False):
#     """
#     Orthogonal-Procrustes align L2's shared g-gauge to L1, return aligned L2.
#     L1, L2: 2D arrays (ng, np), OR tuples of such arrays that SHARE the g-index
#             (e.g. (alpha, beta) UHF Cholesky). One unitary O is applied to every
#             component, because cross-component contractions (alpha-beta Coulomb)
#             require a single common rotation.
#     """
#     single = not isinstance(L2, (tuple, list))
#     L1s = [L1] if single else list(L1)
#     L2s = [L2] if single else list(L2)
#     shapes = [b.shape for b in L2s]
#     F1 = [a.reshape(a.shape[0], -1) for a in L1s]   # flatten p-side, keep g
#     F2 = [b.reshape(b.shape[0], -1) for b in L2s]

#     # One shared unitary via a single left-Procrustes over all components:
#     # min_O sum_i ||F1_i - O F2_i||_F  ==  min_O ||[F1_0|F1_1|...] - O [F2_0|F2_1|...]||_F
#     F1c = np.concatenate(F1, axis=1)
#     F2c = np.concatenate(F2, axis=1)
#     O = np.asarray(left_procrustes(F1c, F2c))       # min_O ||F1c - O F2c||
#     F2a = [O @ b for b in F2]

#     if report:
#         def _line(A, B, tag):
#             res = np.linalg.norm(A - B)
#             ov  = np.real(np.vdot(A, B)) / (np.linalg.norm(A) * np.linalg.norm(B))
#             print(f"{tag:8s}  ||L1-L2||_F = {res:.6e}   overlap = {ov:.6f}")
#         A  = np.concatenate([a.ravel() for a in F1])
#         Bb = np.concatenate([b.ravel() for b in F2])
#         Ba = np.concatenate([b.ravel() for b in F2a])
#         _line(A, Bb, "before"); _line(A, Ba, "after")

#     L2a = [b.reshape(sh) for b, sh in zip(F2a, shapes)]
#     return L2a[0] if single else tuple(L2a)

def prep_integral(
    mf_cc1: Union[scf.rhf.RHF, scf.uhf.UHF, CCSD, UCCSD],
    mf_cc2: Union[scf.rhf.RHF, scf.uhf.UHF, CCSD, UCCSD],
    basis_coeff1: Optional[np.ndarray] = None,
    basis_coeff2: Optional[np.ndarray] = None,
    norb_frozen1: int = 0,
    norb_frozen2: int = 0,
    chol_cut: float = 1e-5,
    amp_file1 = "amplitudes1.npz",
    chol_file1 = "FCIDUMP_chol1",
    amp_file2 = "amplitudes2.npz",
    chol_file2 = "FCIDUMP_chol2"
):

    print("\nPreparing AFQMC calculation")

    if isinstance(mf_cc1, (CCSD, UCCSD)):
        mf1 = mf_cc1._scf
        cc1 = mf_cc1
        if cc1.frozen is not None:
            norb_frozen1 = cc1.frozen
            integral.save_cc_amplitude(cc1, amp_file1)
    else:
        mf1 = mf_cc1
    
    if isinstance(mf_cc2, (CCSD, UCCSD)):
        mf2 = mf_cc2._scf
        cc2 = mf_cc2
        if cc2.frozen is not None:
            norb_frozen2 = cc2.frozen
            integral.save_cc_amplitude(cc2, amp_file2)
    else:
        mf2 = mf_cc2

    if isinstance(mf1, scf.rhf.RHF):
        spin_type1 = 'restricted'
    elif isinstance(mf1, scf.uhf.UHF):
        spin_type1 = 'unrestricted'
    
    if isinstance(mf2, scf.rhf.RHF):
        spin_type2 = 'restricted'
    elif isinstance(mf2, scf.uhf.UHF):
        spin_type2 = 'unrestricted'

    assert spin_type1 == spin_type2 # should be able to support different

    # mf1.mo_coeff = match_mocoeff(mf1, mf2)

    enuc1, h1e1, chol1, nelec1, nbasis1, nchol1 \
        = integral.get_chol(mf1, spin_type1, norb_frozen1, chol_cut, basis_coeff1)
    enuc2, h1e2, chol2, nelec2, nbasis2, nchol2 \
        = integral.get_chol(mf2, spin_type2, norb_frozen2, chol_cut, basis_coeff2)
      
    print(f"Original number of Cholesky: {nchol1} vs {nchol2}")
    
    if spin_type1 == 'restricted':
        if nchol1 != nchol2:
            print("Pad the smaller Cholesky")
            chol1, chol2 = match_nchol(chol1, chol2)
        nchol1 = chol1.shape[0]; nchol2 = chol2.shape[0]
        assert nchol1 == nchol2
        chol2 = align_gauge(chol1, chol2, report=True)

    elif spin_type1 == 'unrestricted':
        chol1a, chol1b = chol1
        chol2a, chol2b = chol2
        if nchol1 != nchol2:
            print("Pad the smaller Cholesky")
            chol1a, chol2a = match_nchol(chol1a, chol2a)
            chol1b, chol2b = match_nchol(chol1b, chol2b)
        assert chol1a.shape[0] == chol1b.shape[0] == chol2a.shape[0] == chol2b.shape[0]
        nchol1 = nchol2 = chol1a.shape[0]
        chol1 = (chol1a, chol1b)
        chol2 = align_gauge(chol1, (chol2a, chol2b), report=True)   # ONE shared O
        

    print("Finished calculating Cholesky integrals")
    print(f"{'Active Space:':<22}  {'System 1':>10}  {'System 2':>10}")
    print(f"{'Number of electrons:':<22}  {str(nelec1):>10}  {str(nelec2):>10}")
    print(f"{'Number of basis:':<22}  {str(nbasis1):>10}  {str(nbasis2):>10}")
    print(f"{'Number of Cholesky:':<22}  {str(nchol1):>10}  {str(nchol2):>10}")

    integral.write_integral(
        enuc=enuc1,
        hcore=h1e1,
        chol=chol1,
        nelec=sum(nelec1),
        nmo=nbasis1,
        ms=mf1.mol.spin,
        spin_type=spin_type1,
        filename=chol_file1,
    )

    integral.write_integral(
        enuc=enuc2,
        hcore=h1e2,
        chol=chol2,
        nelec=sum(nelec2),
        nmo=nbasis2,
        ms=mf2.mol.spin,
        spin_type=spin_type2,
        filename=chol_file2,
    )