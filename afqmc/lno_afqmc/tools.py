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