import numpy as np

from pyscf import lo, scf
from pyscf.data import elements
from pyscf.lno.tools import autofrag_iao

def riao_localization(mf, lo_file=None):
    # IAO localization
    mol = mf.mol
    frozen = elements.chemcore(mol)
    orbocc = mf.mo_coeff[:,frozen:np.count_nonzero(mf.mo_occ)]
    lo_coeff = lo.iao.iao(mol, orbocc)
    lo_coeff = lo.orth.vec_lowdin(lo_coeff, mf.get_ovlp())
    moliao = lo.iao.reference_mol(mol)
    frag_lolist = autofrag_iao(moliao)
    if lo_file is not None:
        np.savez('./lo_coeff.npz', lo_coeff=lo_coeff)
    return lo_coeff, frag_lolist, moliao.elements

def uiao_localization(mf, lo_file=None):
    # IAO localization
    mol = mf.mol
    frozen = elements.chemcore(mol)
    orbocc_a = mf.mo_coeff[0][:,frozen:np.count_nonzero(mf.mo_occ[0])]
    orbocc_b = mf.mo_coeff[1][:,frozen:np.count_nonzero(mf.mo_occ[1])]
    lo_coeff_a = lo.iao.iao(mol, orbocc_a)
    lo_coeff_a = lo.orth.vec_lowdin(lo_coeff_a, mf.get_ovlp())
    lo_coeff_b = lo.iao.iao(mol, orbocc_b)
    lo_coeff_b = lo.orth.vec_lowdin(lo_coeff_b, mf.get_ovlp())
    lo_coeff = [lo_coeff_a, lo_coeff_b]
    moliao = lo.iao.reference_mol(mol)
    frag_lolist = autofrag_iao(moliao)
    frag_lolist = [[i,i] for i in frag_lolist]
    if lo_file is not None:
        np.savez('./lo_coeff.npz', lo_coeff_a=lo_coeff_a,lo_coeff_b=lo_coeff_b)
    return lo_coeff, frag_lolist, moliao.elements

def iao_localization(mf, lo_file=None):
    if isinstance(mf, scf.rhf.RHF):
        return riao_localization(mf, lo_file)
    elif isinstance(mf, scf.uhf.UHF):
        return uiao_localization(mf, lo_file)

def plot_density(mol, orbloc, lno_coeff, lno_active, spin_type, idx):
    from pyscf.tools import cubegen
    # plot density as rho(r) = sum_p |psi_p(r)|^2
    if spin_type == "restricted":
        dm_ctr = orbloc @ orbloc.T
        _ = cubegen.density(mol, f'ctr_density_{idx}.cube', dm_ctr)
        dm_las = lno_coeff[:,lno_active] @ lno_coeff[:,lno_active].T
        _ = cubegen.density(mol, f'las_density_{idx}.cube', dm_las)

    elif spin_type == "unrestricted":
        dm_ctr = orbloc[0] @ orbloc[0].T + orbloc[1] @ orbloc[1].T
        _ = cubegen.density(mol, f'ctr_density_{idx}.cube', dm_ctr)
        dm_las = (lno_coeff[0][:,lno_active[0]] @ lno_coeff[0][:,lno_active[0]].T
                  +lno_coeff[1][:,lno_active[1]] @ lno_coeff[1][:,lno_active[1]].T)
        _ = cubegen.density(mol, f'las_density_{idx}.cube', dm_las)
        
    return None

# def mo_span(mo1, s1e, mo2):
#     '''
#     check if mo1 and mo2 span each other
#     return (mo1 span mo2),  (mo2 span mo1)
#     '''
#     olp11 = mo1.T.conj() @ s1e @ mo1
#     olp12 = mo1.T.conj() @ s1e @ mo2
#     olp22 = mo2.T.conj() @ s1e @ mo2

#     span12 = np.abs(olp12.T.conj() @ olp12 - olp22).max()
#     span21 = np.abs(olp12 @ olp12.T.conj() - olp11).max()
#     # span12 = np.abs(olp12.conj().T @ np.linalg.solve(olp11, olp12) - olp22).max() < thresh
#     # span21 = np.abs(olp12 @ np.linalg.solve(olp22, olp12.conj().T) - olp11).max() < thresh

#     return span12, span21


def mo_span(mo1, s1e, mo2):
    '''
    Measure subspace containment between mo1 and mo2.
    Returns (span12, span21):
      span12 = max-abs residual for  span(mo2) ⊆ span(mo1)   ("mo1 spans mo2")
      span21 = max-abs residual for  span(mo1) ⊆ span(mo2)   ("mo2 spans mo1")
    Small residual => containment holds. Assumes mo1, mo2 orthonormal in the s1e metric.
    '''
    olp11 = mo1.T.conj() @ s1e @ mo1
    olp12 = mo1.T.conj() @ s1e @ mo2
    olp22 = mo2.T.conj() @ s1e @ mo2

    span12 = np.abs(olp12.T.conj() @ olp12 - olp22).max()
    span21 = np.abs(olp12 @ olp12.T.conj() - olp11).max()
    return span12, span21


def check_rspan(lo, s1e, mo, thresh=1e-10):
    # lo passed as mo1 -> span12 tests  mo ⊆ span(lo)  (lo spans at least mo)
    span12, span21 = mo_span(lo, s1e, mo)
    return span12 < thresh, span21 < thresh


def check_uspan(lo, s1e, mo, thresh=1e-10):
    span12_a, span21_a = mo_span(lo[0], s1e, mo[0])
    span12_b, span21_b = mo_span(lo[1], s1e, mo[1])
    return span12_a < thresh and span12_b < thresh, span21_a < thresh and span21_b < thresh


def check_span(mf, lo_coeff_occ, frozen=0, thresh=1e-10):
    s1e = mf.get_ovlp()

    if isinstance(mf, scf.uhf.UHF):
        if isinstance(frozen, int):
            frozen = (frozen, frozen)
        nocc = (np.count_nonzero(mf.mo_occ[0]),
                np.count_nonzero(mf.mo_occ[1]))
        mo_coeff_occ = (mf.mo_coeff[0][:, frozen[0]:nocc[0]],
                        mf.mo_coeff[1][:, frozen[1]:nocc[1]])
        span12, span21= check_uspan(lo_coeff_occ, s1e, mo_coeff_occ, thresh)
    elif isinstance(mf, scf.rhf.RHF):
        nocc = np.count_nonzero(mf.mo_occ)
        mo_coeff_occ = mf.mo_coeff[:, frozen:nocc]
        span12, span21 = check_rspan(lo_coeff_occ, s1e, mo_coeff_occ, thresh)
    else:
        raise TypeError(f'unsupported mean-field type: {type(mf)}')

    print(f'LO occ span the occupied MO occ space - {span12}.\n'
          f'MO occ span the occupied LO occ space - {span21}.')
    
    if not span12:
        raise ValueError(f"LOs DO NOT SPAN MOs, CHECK THE LOCALIZATION!!!")
    
    return None
