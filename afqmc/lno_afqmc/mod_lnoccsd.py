# modified lno_ccsd impurity solver from Hongzhou's pyscf-forge

import numpy as np
from pyscf import lib
from pyscf.lno import lnoccsd
from pyscf.lno import ulnoccsd
from afqmc.lno_afqmc.prep import kind

def lnoccsd_kernel(mcc, mo_coeff, uocc_loc, mo_occ, maskact):
    spin_type = kind(mo_coeff)
    if spin_type == "restricted":
        return lnoccsd_solver(mcc, mo_coeff, uocc_loc, mo_occ, maskact)
    elif spin_type == "unrestricted":
        return ulnoccsd_solver(mcc, mo_coeff, uocc_loc, mo_occ, maskact)
    else: 
        raise NotImplementedError('Only Support Restricted and Unrestricted Now!')

def lnoccsd_solver(mcc, mo_coeff, uocc_loc, mo_occ, maskact):

    maskocc = mo_occ>1e-10
    nmo = mo_occ.size

    orbfrzocc = mo_coeff[:,~maskact &  maskocc] 
    orbactocc = mo_coeff[:, maskact &  maskocc]
    orbactvir = mo_coeff[:, maskact & ~maskocc]
    orbfrzvir = mo_coeff[:,~maskact & ~maskocc]
    nfrzocc, nactocc, nactvir, nfrzvir = [orb.shape[1]
                                          for orb in [
                                              orbfrzocc,orbactocc,
                                              orbactvir,orbfrzvir
                                              ]]
    # nlo = uocc_loc.shape[1]
    # nactmo = nactocc + nactvir

    if nactocc == 0 or nactvir == 0:
        elcorr_pt2 = elcorr_cc = lib.tag_array(0., spin_comp=np.array((0., 0.)))
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

    return (elcorr_pt2, elcorr_cc), t1, t2

def ulnoccsd_solver(mcc, mo_coeff, uocc_loc, mo_occ, maskact): 

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