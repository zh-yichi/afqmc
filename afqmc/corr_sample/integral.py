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
def procrustes_rotation(A, B):
    # occ Procrustes
    # min_R|AR-B|F the Frobenius norm
    # rotate A to match B
    m = B.T @ A
    u, _, v = np.linalg.svd(m)
    pcsts_r = v.T @ u.T
    Ap = A @ pcsts_r
    return Ap

# def procrustes_rotation(A, B, S):
#     # orthogonal R minimizing ||A R - B|| in the S-metric,
#     # i.e. maximizing the orbital overlap Tr(R^T A^T S B)
#     M = A.T @ S @ B
#     assert M.shape[0] == M.shape[1], "occ/vir counts must match between the two systems"
#     U, _, Vt = np.linalg.svd(M)
#     R = U @ Vt
#     return A @ R

def match_rmocoeff(mf1, mf2):
    nocc1 = np.count_nonzero(mf1.mo_occ)
    nocc2 = np.count_nonzero(mf2.mo_occ)
    mo1 = mf1.mo_coeff
    mo2 = mf2.mo_coeff
    mo_occ1 = mo1[:,:nocc1]
    mo_vir1 = mo1[:,nocc1:]
    mo_occ2 = mo2[:,:nocc2]
    mo_vir2 = mo2[:,nocc2:]
    mo_occ1 = np.array(procrustes_rotation(mo_occ1, mo_occ2))
    mo_vir1 = np.array(procrustes_rotation(mo_vir1, mo_vir2))
    mo_coeff1 = np.hstack([mo_occ1, mo_vir1]) # match 1 with 2

    return mo_coeff1

def match_umocoeff(mf1, mf2):
    nocc1_a = np.count_nonzero(mf1.mo_occ[0])
    nocc1_b = np.count_nonzero(mf1.mo_occ[1])
    nocc2_a = np.count_nonzero(mf2.mo_occ[0])
    nocc2_b = np.count_nonzero(mf2.mo_occ[1])
    mo1 = mf1.mo_coeff
    mo2 = mf2.mo_coeff

    mo_occ1_a = mo1[0][:,:nocc1_a]
    mo_vir1_a = mo1[0][:,nocc1_a:]
    mo_occ1_b = mo1[1][:,:nocc1_b]
    mo_vir1_b = mo1[1][:,nocc1_b:]

    mo_occ2_a = mo2[0][:,:nocc2_a]
    mo_vir2_a = mo2[0][:,nocc2_a:]
    mo_occ2_b = mo2[1][:,:nocc2_b]
    mo_vir2_b = mo2[1][:,nocc2_b:]

    mo_occ1_a = np.array(procrustes_rotation(mo_occ1_a, mo_occ2_a))
    mo_vir1_a = np.array(procrustes_rotation(mo_vir1_a, mo_vir2_a))
    mo_occ1_b = np.array(procrustes_rotation(mo_occ1_b, mo_occ2_b))
    mo_vir1_b = np.array(procrustes_rotation(mo_vir1_b, mo_vir2_b))

    mo_coeff1 = [np.hstack([mo_occ1_a, mo_vir1_a]),
                 np.hstack([mo_occ1_b, mo_vir1_b])]

    return mo_coeff1

def match_mocoeff(mf1, mf2):
    print("Procrustes Rotation: mo1 -> mo2")
    if isinstance(mf1, scf.rhf.RHF):
        return match_rmocoeff(mf1, mf2)
    elif isinstance(mf1, scf.uhf.UHF):
        return match_umocoeff(mf1, mf2)

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
            integral.save_cc_amplitude(cc2, amp_file1)
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
    
    print(f"Original number of Cholesky {nchol1} {nchol2}")
    
    if nchol1 != nchol2:
        print("Pad the smaller Cholesky")
        if spin_type1 == 'restricted':
            chol1, chol2 = match_nchol(chol1, chol2)
            nchol1 = chol1.shape[0]
            nchol2 = chol2.shape[0]
            assert nchol1 == nchol2
        elif spin_type1 == 'unrestricted':
            chol1a, chol2a = match_nchol(chol1[0], chol2[0])
            chol1b, chol2b = match_nchol(chol1[1], chol2[1])
            assert chol1a.shape[0] == chol1b.shape[0]
            assert chol2a.shape[0] == chol2b.shape[0]
            assert chol1a.shape[0] == chol2a.shape[0]
            nchol1 = chol1a.shape[0]
            nchol2 = chol2a.shape[0]
            chol1 = (chol1a, chol1b)
            chol2 = (chol2a, chol2b)

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