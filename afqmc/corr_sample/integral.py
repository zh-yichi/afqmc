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

# @jit
# def procrustes_rotation(A, B):
#     # occ Procrustes
#     # min_R|AR-B|F the Frobenius norm
#     # rotate A to match B
#     m = B.T @ A
#     u, _, v = np.linalg.svd(m)
#     pcsts_r = v.T @ u.T
#     Ap = A @ pcsts_r
#     return Ap

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

def _report_line(A, B, tag):
    res = np.linalg.norm(A - B)
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    ov = np.real(np.vdot(A, B)) / denom if denom else 1.0
    print(f"  {tag:18s} ||d||_F = {res:.6e}   overlap = {ov:.6f}")

def _align_blocks(mo1, mo2, nocc1, nocc2, frozen, report, prefix=""):
    names = ("frozen", "occ", "vir")
    b1 = (mo1[:, :frozen], mo1[:, frozen:nocc1], mo1[:, nocc1:])
    b2 = (mo2[:, :frozen], mo2[:, frozen:nocc2], mo2[:, nocc2:])
    aligned = []
    for name, a, b in zip(names, b1, b2):
        ba = b @ np.array(right_procrustes(a, b))
        if report and a.size:
            tag = f"{prefix}{name}"
            if a.shape == b.shape:                 # 'before' needs equal block sizes
                _report_line(a, b,  f"{tag} before")
            _report_line(a, ba, f"{tag} after")
        aligned.append(ba)
    return np.hstack(aligned)

def align_rmo(mf1, mf2, frozen=0, report=False):
    nocc1 = np.count_nonzero(mf1.mo_occ)
    nocc2 = np.count_nonzero(mf2.mo_occ)
    return _align_blocks(mf1.mo_coeff, mf2.mo_coeff, nocc1, nocc2, frozen, report)

def align_umo(mf1, mf2, frozen=0, report=False):
    matched = []
    for s, name in ((0, "alpha "), (1, "beta  ")):
        nocc1 = np.count_nonzero(mf1.mo_occ[s])
        nocc2 = np.count_nonzero(mf2.mo_occ[s])
        matched.append(_align_blocks(mf1.mo_coeff[s], mf2.mo_coeff[s],
                                     nocc1, nocc2, frozen, report, prefix=name))
    return np.array(matched)

def align_mo(mf1, mf2, frozen=0, report=False):
    print("Procrustes Aligning mo2 -> mo1")
    if isinstance(mf1, scf.rhf.RHF):
        return align_rmo(mf1, mf2, frozen, report)
    elif isinstance(mf1, scf.uhf.UHF):
        return align_umo(mf1, mf2, frozen, report)

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

def align_gauge(L1, L2, report=False):
    """
    Orthogonal-Procrustes align L2's shared g-gauge to L1, return aligned L2.
    L1, L2: 2D arrays (ng, np), OR tuples of such arrays that SHARE the g-index
            (e.g. (alpha, beta) UHF Cholesky). One unitary O is applied to every
            component, because cross-component contractions (alpha-beta Coulomb)
            require a single common rotation.
    """
    single = not isinstance(L2, (tuple, list))
    L1s = [L1] if single else list(L1)
    L2s = [L2] if single else list(L2)
    shapes = [b.shape for b in L2s]
    F1 = [a.reshape(a.shape[0], -1) for a in L1s]   # flatten p-side, keep g
    F2 = [b.reshape(b.shape[0], -1) for b in L2s]

    # One shared unitary via a single left-Procrustes over all components:
    # min_O sum_i ||F1_i - O F2_i||_F  ==  min_O ||[F1_0|F1_1|...] - O [F2_0|F2_1|...]||_F
    F1c = np.concatenate(F1, axis=1)
    F2c = np.concatenate(F2, axis=1)
    O = np.asarray(left_procrustes(F1c, F2c))       # min_O ||F1c - O F2c||
    F2a = [O @ b for b in F2]

    if report:
        def _line(A, B, tag):
            res = np.linalg.norm(A - B)
            ov  = np.real(np.vdot(A, B)) / (np.linalg.norm(A) * np.linalg.norm(B))
            print(f"{tag:8s}  ||L1-L2||_F = {res:.6e}   overlap = {ov:.6f}")
        A  = np.concatenate([a.ravel() for a in F1])
        Bb = np.concatenate([b.ravel() for b in F2])
        Ba = np.concatenate([b.ravel() for b in F2a])
        _line(A, Bb, "before"); _line(A, Ba, "after")

    L2a = [b.reshape(sh) for b, sh in zip(F2a, shapes)]
    return L2a[0] if single else tuple(L2a)

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