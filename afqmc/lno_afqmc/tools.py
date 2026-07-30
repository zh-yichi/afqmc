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


from pyscf.lno import lno, ulno
from pyscf.lib import logger
from functools import reduce

def make_rlas(mlno, eris, orbloc, lno_type, lno_param):
    log = logger.new_logger(mlno)
    cput1 = (logger.process_clock(), logger.perf_counter())

    s1e = mlno.s1e

    orboccfrz_core, orbocc, orbvir, orbvirfrz_core = mlno.split_mo_coeff()
    moeocc, moevir = mlno.split_mo_energy()[1:3]

    ''' Projection of LO onto occ and vir
    '''
    uocc_loc = reduce(np.dot, (orbloc.T.conj(), s1e, orbocc)) # <loc|mo_occ>
    # uocc_loc[act], std, frz
    uocc_loc, uocc_std, uocc_orth = \
            lno.projection_construction(uocc_loc, mlno.lo_proj_thresh, mlno.lo_proj_thresh_active)
    if uocc_loc.shape[1] == 0:
        log.error('LOs do not overlap with occupied space. This could be caused '
                  'by either a bad fragment choice or too high of `lo_proj_thresh_active` '
                  '(current value: %s).', mlno.lo_proj_thresh_active)
        raise RuntimeError
    log.info('LO occ proj: %d active | %d standby | %d orthogonal',
             *[u.shape[1] for u in [uocc_loc,uocc_std,uocc_orth]])

    uvir_loc = reduce(np.dot, (orbloc.T.conj(), s1e, orbvir))
    uvir_loc, uvir_std, uvir_orth = \
            lno.projection_construction(uvir_loc, mlno.lo_proj_thresh, mlno.lo_proj_thresh_active)
    log.info('LO vir proj: %d active | %d standby | %d orthogonal',
             *[u.shape[1] for u in [uvir_loc,uvir_std,uvir_orth]])
    if uvir_loc.shape[1] == 0:
        uvir_loc = uvir_std = uvir_orth = None

    ''' LNO construction
    '''
    dmoo = mlno.make_lo_rdm1_occ(eris, moeocc, moevir, uocc_loc, uvir_loc, lno_type[0])
    if mlno._match_oldcode: dmoo *= 0.5 # TO MATCH OLD LNO CODE
    dmoo = reduce(np.dot, (uocc_orth.T.conj(), dmoo, uocc_orth))
    if lno_param[0]['norb'] is not None:
        lno_param[0]['norb'] -= uocc_loc.shape[1] + uocc_std.shape[1]
    uoccact_orth, uoccfrz_orth = lno.natorb_select(dmoo, uocc_orth, **lno_param[0])
    uoccact_orth = uoccact_orth[:,::-1] # for occ. flip the NOs so they are in the
    uoccfrz_orth = uoccfrz_orth[:,::-1] # order of small eigenvalue -> large eigenvalue
    orboccfrz = np.hstack((orboccfrz_core, np.dot(orbocc, uoccfrz_orth)))
    uoccact = np.hstack((uoccact_orth, uocc_std, uocc_loc)) 
    orboccact = np.dot(orbocc, uoccact)
    uoccact_loc = np.linalg.multi_dot((orboccact.T.conj(), s1e, orbloc))
    can_uoccact = lno.subspace_eigh(np.diag(moeocc), np.hstack((uoccact_orth, uocc_std, uocc_loc)))[1]
    can_orboccact = np.dot(orbocc, can_uoccact)
    can_uoccact_loc = np.linalg.multi_dot((can_orboccact.T.conj(), s1e, orbloc))
    cput1 = log.timer_debug1('make_lo_rdm1_occ', *cput1)

    dmvv = mlno.make_lo_rdm1_vir(eris, moeocc, moevir, uocc_loc, uvir_loc, lno_type[1])
    if mlno._match_oldcode: dmvv *= 0.5 # TO MATCH OLD LNO CODE
    if uvir_orth is not None:
        dmvv = reduce(np.dot, (uvir_orth.T.conj(), dmvv, uvir_orth))
        if lno_param[1]['norb'] is not None:
            lno_param[1]['norb'] -= uvir_loc.shape[1] + uvir_std.shape[1]
        uviract_orth, uvirfrz_orth = lno.natorb_select(dmvv, uvir_orth, **lno_param[1])
        orbvirfrz = np.hstack((np.dot(orbvir, uvirfrz_orth), orbvirfrz_core))
        uviract = np.hstack((uvir_loc, uvir_std, uviract_orth)) # vir in decreasing eigenvalue order
        orbviract = np.dot(orbvir, uviract)
        can_uviract = lno.subspace_eigh(np.diag(moevir), np.hstack((uvir_loc, uvir_std, uviract_orth)))[1]
        can_orbviract = np.dot(orbvir, can_uviract)
    else:
        orbviract, orbvirfrz = lno.natorb_select(dmvv, orbvir, **lno_param[1])
        orbvirfrz = np.hstack((orbvirfrz, orbvirfrz_core))
        uviract = reduce(np.dot, (orbvir.T.conj(), s1e, orbviract))
        orbviract = np.dot(orbvir, uviract)
        can_uviract = lno.subspace_eigh(np.diag(moevir), uviract)[1]
        can_orbviract = np.dot(orbvir, can_uviract)
    cput1 = log.timer_debug1('make_lo_rdm1_vir', *cput1)

    ''' LAS construction
    '''
    orbfragall = [orboccfrz, orboccact, orbviract, orbvirfrz]
    can_orbfragall = [orboccfrz, can_orboccact, can_orbviract, orbvirfrz]
    orbfrag = np.hstack(orbfragall)
    can_orbfrag = np.hstack(can_orbfragall)
    norbfragall = np.asarray([x.shape[1] for x in orbfragall])
    locfragall = np.cumsum([0] + norbfragall.tolist()).astype(int)
    frzfrag = np.concatenate((
        np.arange(locfragall[0], locfragall[1]),
        np.arange(locfragall[3], locfragall[4]))).astype(int)
    frag_msg = '%d/%d Occ | %d/%d Vir | %d/%d MOs' % (
                    norbfragall[1], sum(norbfragall[:2]),
                    norbfragall[2], sum(norbfragall[2:4]),
                    sum(norbfragall[1:3]), sum(norbfragall)
                )
    if len(frzfrag) == 0:
        frzfrag = 0

    return orbfrag, can_orbfrag, frzfrag, uoccact_loc, can_uoccact_loc, frag_msg

def make_ulas(mlno, eris, orbloc, lno_type, lno_param):
    """
    Create localized active space for a given set of localized orbitals
    given in orbloc
    """
    log = logger.new_logger(mlno)

    s1e = mlno.s1e

    orboccfrz_core = [None,] * 2
    orbocc = [None,] * 2
    orbvir = [None,] * 2
    orbvirfrz_core = [None,] * 2
    moeocc = [None,] * 2
    moevir = [None,] * 2

    uocc_loc = [None,] * 2
    uocc_std = [None,] * 2
    uocc_orth = [None,] * 2

    mo_splits = mlno.split_mo_coeff()
    moe_splits = mlno.split_mo_energy()
    for s in range(2):
        orboccfrz_core[s], orbocc[s], orbvir[s], orbvirfrz_core[s] = mo_splits[s]
        moeocc[s], moevir[s] = moe_splits[s][1:3]
        
        #####################################
        # Projection of LO onto occ and vir #
        #####################################
        ovlp = reduce(np.dot, (orbloc[s].T.conj(), s1e, orbocc[s]))
        uocc_loc[s], uocc_std[s], uocc_orth[s] = \
            lno.projection_construction(ovlp, mlno.lo_proj_thresh, mlno.lo_proj_thresh_active)
        # NOTE we allow empty fragments
        # if uocc_loc[s].shape[1] == 0:
        #    log.error('LOs do not overlap with occupied space. This could be caused '
        #              'by either a bad fragment choice or too high of `lo_proj_thresh_active` '
        #              '(current value: %s).', mlno.lo_proj_thresh_active)
        #    raise RuntimeError
        log.info('LO occ proj: %d active | %d standby | %d orthogonal',
                 *[u.shape[1] for u in [uocc_loc[s], uocc_std[s], uocc_orth[s]]])

    ####################
    # LNO construction #
    ####################
    if lno_type[0] == lno_type[1] == '1h':
        # NOTE: uvir_loc is not used in 1h/1h, so we pass None
        if getattr(mlno, 'with_df', None):
            dmoo, dmvv = ulno.make_lo_rdm1_1h_df(eris, moeocc, moevir, uocc_loc)
        else:
            dmoo, dmvv = ulno.make_lo_rdm1_1h(eris, moeocc, moevir, uocc_loc)
    else:
        raise NotImplementedError('Unsupported LNO type')
        
    # if mlno._match_oldulno:
    #     dmoo[0],dmoo[1]=dmoo[0]/2.0,dmoo[1]/2.0
    #     dmvv[0],dmvv[1]=dmvv[0]/2.0,dmvv[1]/2.0

    orbfrag = [None,] * 2
    frzfrag = [None,] * 2
    uoccact_loc = [None,] * 2
    can_orbfrag = [None,] * 2
    can_uoccact_loc = [None,] * 2
    frag_msg = ""

    for s in range(2):
        dmoo[s] = reduce(np.dot, (uocc_orth[s].T.conj(), dmoo[s], uocc_orth[s]))

        _param = lno_param[s][0]
        if _param['norb'] is not None:
            _param['norb'] -= uocc_loc[s].shape[1] + uocc_std[s].shape[1]

        uoccact_orth, uoccfrz_orth = lno.natorb_select(dmoo[s], uocc_orth[s], **_param)
        uoccact_orth = uoccact_orth[:,::-1] # for occ. flip the NOs so they are in the
        uoccfrz_orth = uoccfrz_orth[:,::-1] # order of small -> large eigenvalue
        orboccfrz = np.hstack((orboccfrz_core[s], np.dot(orbocc[s], uoccfrz_orth)))
        uoccact = np.hstack((uoccact_orth, uocc_std[s], uocc_loc[s]))
        orboccact = np.dot(orbocc[s], uoccact)
        uoccact_loc[s] = np.linalg.multi_dot((orboccact.T.conj(), s1e, orbloc[s]))
        # canonized
        can_uoccact = lno.subspace_eigh(np.diag(moeocc[s]), np.hstack((uoccact_orth, uocc_std[s], uocc_loc[s])))[1]
        can_orboccact = np.dot(orbocc[s], can_uoccact)
        can_uoccact_loc[s] = np.linalg.multi_dot((can_orboccact.T.conj(), s1e, orbloc[s]))

        orbviract, orbvirfrz = lno.natorb_select(dmvv[s], orbvir[s], **(lno_param[s][1]))
        orbvirfrz = np.hstack((orbvirfrz, orbvirfrz_core[s])) # vir in eigenvalue decreasing order
        uviract = reduce(np.dot, (orbvir[s].T.conj(), s1e, orbviract))
        uviract = uviract
        orbviract = np.dot(orbvir[s], uviract)
        # canonized
        can_uviract = lno.subspace_eigh(np.diag(moevir[s]), uviract)[1]
        can_orbviract = np.dot(orbvir[s], can_uviract)

        ####################
        # LAS construction #
        ####################
        orbfragall = [orboccfrz, orboccact, orbviract, orbvirfrz]
        can_orbfragall = [orboccfrz, can_orboccact, can_orbviract, orbvirfrz]
        orbfrag[s] = np.hstack(orbfragall)
        can_orbfrag[s] = np.hstack(can_orbfragall)
        norbfragall = np.asarray([x.shape[1] for x in orbfragall])
        locfragall = np.cumsum([0] + norbfragall.tolist()).astype(int)
        frzfrag[s] = np.concatenate((
            np.arange(locfragall[0], locfragall[1]),
            np.arange(locfragall[3], locfragall[4]))).astype(int)
        frag_msg += '\nSpin channel %d: %d/%d Occ | %d/%d Vir | %d/%d MOs\n' % (
                        s,
                        norbfragall[1], sum(norbfragall[:2]),
                        norbfragall[2], sum(norbfragall[2:4]),
                        sum(norbfragall[1:3]), sum(norbfragall)
                    )
        if len(frzfrag[s]) == 0:
            frzfrag[s] = 0

    return orbfrag, can_orbfrag, frzfrag, uoccact_loc, can_uoccact_loc, frag_msg

def make_las(mlno, eris, orbloc, lno_type, lno_param):
    if isinstance(mlno._scf, scf.rhf.RHF):
        return make_rlas(mlno, eris, orbloc, lno_type, lno_param)
    elif isinstance(mlno._scf, scf.uhf.UHF):
        return make_ulas(mlno, eris, orbloc, lno_type, lno_param)