import numpy as np
from pyscf import lib, ao2mo
from ad_afqmc.lno.cc import ccsd
from ad_afqmc.lno.base import lno

def thouless_trans(t1):
    q, r = np.linalg.qr(t1)
    u_ai = r.T
    u_ji = q
    u_occ = np.vstack((u_ji,u_ai))
    u, _, _ = np.linalg.svd(u_occ)
    return u

def get_eri_ao(mf):
    nao = mf.mol.nao
    if getattr(mf,"with_df",None) is not None:
        cderi = mf.with_df._cderi
        cderi = lib.unpack_tril(cderi).reshape(cderi.shape[0], -1)
        cderi = cderi.reshape((-1, nao, nao))
        eri_ao = lib.einsum('lpq,lrs->pqrs', cderi, cderi)
    else:
        eri_ao = mf._eri
        eri_ao = ao2mo.restore(1, eri_ao, nao)
    return eri_ao

def get_eri_mo(eri_ao,oa,va):
    eri_mo = lib.einsum('ip,aq,pqrs,rj,sb->iajb',oa.T,va.T,eri_ao,oa,va)
    return eri_mo

def check_lo_span(lnocc,occloc,virloc):
    _,actocc,actvir,_ = lnocc.split_mo()
    s1e = lnocc._scf.get_ovlp()
    m_occ = occloc.T @ s1e @ actocc
    m_vir = virloc.T @ s1e @ actvir
    occspanerr = abs(m_occ.T @ m_occ - np.eye(m_occ.shape[1])).max()
    virspanerr = abs(m_vir.T @ m_vir - np.eye(m_vir.shape[1])).max()
    if occspanerr > 1e-10:
        raise ValueError(
            'LOs do not fully span the active occupied space! Max|<occ|LO><LO|occ>| = %e',
            occspanerr)
    if virspanerr > 1e-10:
        raise ValueError(
            'LOs do not fully span the active virtual space! Max|<vir|LO><LO|vir>| = %e',
            virspanerr)
    return occspanerr < 1e-10 , virspanerr < 1e-10

def get_lo(lnocc,lo_type='boys'):
    '''get the localized orbitals in the active space'''
    from pyscf import lo
    mol = lnocc._scf.mol
    _,actocc,actvir,_ = lnocc.split_mo()
    if lo_type == 'boys':
        lococc = lo.Boys(mol,actocc).kernel()
        locvir = lo.Boys(mol,actvir).kernel()
    if lo_type == 'pm':
        lococc = lo.PM(mol,actocc).kernel()
        locvir = lo.PM(mol,actvir).kernel()
    print('(loc_occ,loc_vir) span the same space as (occ,vir): '
        ,check_lo_span(lnocc,lococc,locvir))
    return lococc, locvir

def make_lno(mfcc,orbfragloc,thresh_internal,thresh_external):

    if isinstance(thresh_external,(float,int)):
        thresh_ext_occ = thresh_ext_vir = thresh_external
    else:
        thresh_ext_occ, thresh_ext_vir  = thresh_external

    mf = mfcc._scf
    if getattr(mf,'with_df',None) is not None:
        print('Using DF integrals')
        eris = mfcc.ao2mo()
        s1e = eris.s1e if eris.s1e is None else mf.get_ovlp()
        fock = eris.fock  if eris.fock is None else mf.get_fock()
    else:
        print('Using true 4-index integrals')
        eris = None
        s1e = mf.get_ovlp()
        fock = mf.get_fock()
        
    nocc = np.count_nonzero(mf.mo_occ>1e-10)
    nmo = mf.mo_occ.size
    orbocc0, orbocc1, orbvir1, orbvir0 = mfcc.split_mo()
    _, moeocc1, moevir1, _ = mfcc.split_moe()

    lovir = abs(orbfragloc.T @ s1e @ orbvir1).max() > 1e-10
    m = orbfragloc.T @ s1e @ orbocc1 # overlap with all loc act_occs
    uocc1, uocc2 = lno.projection_construction(m, thresh_internal)
    moefragocc1, orbfragocc1 = lno.subspace_eigh(fock, orbocc1@uocc1)
    if lovir:
        m = orbfragloc.T @ s1e @ orbvir1
        uvir1, uvir2 = lno.projection_construction(m, thresh_internal)
        moefragvir1, orbfragvir1 = lno.subspace_eigh(fock, orbvir1@uvir1)

    def moe_Ov(moefragocc):
        return (moefragocc[:,None] - moevir1).reshape(-1)

    eov = moe_Ov(moeocc1)
    # Construct PT2 dm_vv
    u = orbocc1.T @ s1e @ orbfragocc1
    if getattr(mf,'with_df',None) is not None:
        print('Using DF integrals')
        ovov = eris.get_Ovov(u)
    else:
        print('Using true 4-index integrals')
        eri_ao = mf._eri
        nao = mf.mol.nao
        eri_ao = ao2mo.restore(1,eri_ao,nao)
        eri_mo = get_eri_mo(eri_ao,orbocc1,orbvir1)
        ovov = lib.einsum('iI,iajb->Iajb',u,eri_mo)
    eia = moe_Ov(moefragocc1)
    ejb = eov
    e1_or_e2 = 'e1'
    swapidx = 'ab'

    eiajb = (eia[:,None]+ejb).reshape(*ovov.shape)
    t2 = ovov / eiajb

    dmvv = lno.make_rdm1_mp2(t2, 'vv', e1_or_e2, swapidx)
   
    if lovir:
        dmvv = uvir2.T @ dmvv @uvir2

    # Construct PT2 dm_oo
    e1_or_e2 = 'e2'
    swapidx = 'ab'

    dmoo = lno.make_rdm1_mp2(t2, 'oo', e1_or_e2, swapidx)
    dmoo = uocc2.T @ dmoo @ uocc2

    t2 = ovov = eiajb = None
    orbfragocc2, orbfragocc0 \
        = lno.natorb_compression(dmoo, orbocc1, thresh_ext_occ, uocc2)

    can_orbfragocc12 = lno.subspace_eigh(fock, np.hstack([orbfragocc2, orbfragocc1]))[1]
    orbfragocc12 = np.hstack([orbfragocc2, orbfragocc1])
    if lovir:
        orbfragvir2, orbfragvir0 \
            = lno.natorb_compression(dmvv,orbvir1,thresh_ext_vir,uvir2)

        can_orbfragvir12 = lno.subspace_eigh(fock, np.hstack([orbfragvir2, orbfragvir1]))[1]
        orbfragvir12 = np.hstack([orbfragvir2, orbfragvir1])
    else: 
        orbfragvir2, orbfragvir0 = lno.natorb_compression(dmvv,orbvir1,thresh_ext_vir)

        can_orbfragvir12 = lno.subspace_eigh(fock, orbfragvir2)[1]
        orbfragvir12 = orbfragvir2

    lno_orb = np.hstack([orbocc0, orbfragocc0, orbfragocc12,
                         orbfragvir12, orbfragvir0, orbvir0])
    can_orbfrag = np.hstack([orbocc0, orbfragocc0, can_orbfragocc12,
                        can_orbfragvir12, orbfragvir0, orbvir0])
    
    frzfrag = np.hstack([np.arange(orbocc0.shape[1]+orbfragocc0.shape[1]),
                         np.arange(nocc+orbfragvir12.shape[1],nmo)])

    return frzfrag, lno_orb , can_orbfrag

def get_fragment_energy(oovv,t2,prj):
    m = prj.T @ prj
    e_frg_corr = lib.einsum('ijab,kjab,ik->',t2,2*oovv-oovv.transpose(0,1,3,2),m)
    return e_frg_corr

def lno_cc_solver(mf,mo_coeff,lo_coeff,eris=None,frozen=None):
    r''' Solve impurity problem and calculate local correlation energy.'''

    maskocc = mf.mo_occ>1e-10
    nmo = mf.mo_occ.size

    # Convert frozen to 0 bc PySCF solvers do not support frozen=None or empty list
    if frozen is None:
        frozen = 0
    elif isinstance(frozen, (list,tuple,np.ndarray)) and len(frozen) == 0:
        frozen = 0

    if isinstance(frozen, (int,np.integer)):
        maskact = np.hstack([np.zeros(frozen,dtype=bool),
                             np.ones(nmo-frozen,dtype=bool)])
    elif isinstance(frozen, (list,tuple,np.ndarray)):
        maskact = np.array([i not in frozen for i in range(nmo)])
    else:
        raise RuntimeError

    orbfrzocc = mo_coeff[:,~maskact& maskocc]
    orbactocc = mo_coeff[:, maskact& maskocc]
    orbactvir = mo_coeff[:, maskact&~maskocc]
    orbfrzvir = mo_coeff[:,~maskact&~maskocc]
    _, nactocc, nactvir, _ = [orb.shape[1]
                                          for orb in [orbfrzocc,orbactocc,
                                                      orbactvir,orbfrzvir]]

    s1e = mf.get_ovlp() if eris is None else eris.s1e
    prjlo = lo_coeff.T @ s1e @ orbactocc

    # solve impurity problem
    from pyscf import cc
    mcc = cc.CCSD(mf, mo_coeff=mo_coeff, frozen=frozen) #.set(verbose=verbose_imp)
    mcc.ao2mo = ccsd.ccsd_ao2mo.__get__(mcc, mcc.__class__)
    mcc._s1e = s1e
    if eris is not None:
        mcc._h1e = eris.h1e
        mcc._vhf = eris.vhf
    imp_eris = mcc.ao2mo()
    # vhf is assumed to be computed with exxdiv=None and imp_eris.mo_energy is not
    # exxdiv-corrected. We correct it for MP2 energy if mf.exxdiv is 'ewald'.
    # FIXME: Should we correct it for other exxdiv options (e.g., 'vcut_sph')?
    if hasattr(mf, 'exxdiv') and mf.exxdiv == 'ewald':  # PBC HF object
        from pyscf.pbc.cc.ccsd import _adjust_occ
        from pyscf.pbc import tools
        madelung = tools.madelung(mf.cell, mf.kpt)
        imp_eris.mo_energy = _adjust_occ(imp_eris.mo_energy, imp_eris.nocc,
                                         -madelung)
    if isinstance(imp_eris.ovov, np.ndarray):
        ovov = imp_eris.ovov
    else:
        ovov = imp_eris.ovov[()]
    oovv = ovov.reshape(nactocc,nactvir,nactocc,nactvir).transpose(0,2,1,3)
    ovov = None

    # MP2 fragment energy
    t1, t2 = mcc.init_amps(eris=imp_eris)[1:]
    # ecorr_p2 = get_fragment_energy(oovv, t2, prjlo)

    # CCSD fragment energy
    mcc.kernel(eris=imp_eris, t1=t1, t2=t2)
    t1, t2 = mcc.t1, mcc.t2
    t2 += lib.einsum('ia,jb->ijab',t1,t1)
    ecorr_cc = get_fragment_energy(oovv, t2, prjlo)
    t2 -= lib.einsum('ia,jb->ijab',t1,t1) 

    oovv = imp_eris = None

    return mcc, ecorr_cc

def lno_mp2_frg_e(mf,frzfrag,orbfragloc,can_orbfrag):
    
    from pyscf import cc
    mol = mf.mol
    nocc = mol.nelectron // 2 
    nao = mol.nao
    actfrag = np.array([i for i in range(nao) if i not in frzfrag])
    # frzocc = np.array([i for i in range(nocc) if i in frzfrag])
    actocc = np.array([i for i in range(nocc) if i in actfrag])
    actvir = np.array([i for i in range(nocc,nao) if i in actfrag])
    nactocc = len(actocc)
    nactocc = len(actocc)
    nactvir = len(actvir)

    s1e = mf.get_ovlp()
    can_prjlo = orbfragloc.T @ s1e @ can_orbfrag[:,actocc]
    mc = cc.CCSD(mf, mo_coeff=can_orbfrag, frozen=frzfrag)
    mc.ao2mo = ccsd.ccsd_ao2mo.__get__(mc,mc.__class__)
    mc._s1e = s1e
    # mc._h1e = eris.h1e
    # mc._vhf = eris.vhf
    imp_eris = mc.ao2mo()
    if isinstance(imp_eris.ovov, np.ndarray):
        ovov = imp_eris.ovov
    else:
        ovov = imp_eris.ovov[()]
    oovv = ovov.reshape(nactocc,nactvir,nactocc,nactvir).transpose(0,2,1,3)
    ovov = None
    # MP2 fragment energy
    _, t2 = mc.init_amps(eris=imp_eris)[1:]
    emp2 = get_fragment_energy(oovv,t2,can_prjlo)
    
    return emp2
