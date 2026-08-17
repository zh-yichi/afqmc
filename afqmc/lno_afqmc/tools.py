import re
import numpy as np
from collections import defaultdict

from pyscf import gto, lo, scf
from pyscf.scf import atom_hf
from pyscf.data import elements
from pyscf.lno import tools


def _fix_ecpbas(mol):
    """Repair `_ecpbas` on a mol restored from a chkfile.

    gto.mole.dumps() serializes an empty `_ecpbas` -- zeros((0, BAS_SLOTS)) --
    with .tolist(), which collapses it to a bare []; loads() then rebuilds it
    as shape (0,). Mole.build() only reassigns `_ecpbas` when an ECP is
    present, so the 1D array survives and anything doing `_ecpbas[:, 0]`
    (e.g. pyscf.scf.atom_hf) raises IndexError. Only bites ECP-free molecules.
    """
    if np.asarray(mol._ecpbas).ndim != 2:
        mol._ecpbas = np.zeros((0, gto.BAS_SLOTS), dtype=np.int32)
    return mol

def free_atom_minao(mol, occ_tol=1e-6, sv_tol=1e-8, x2c=None):
    """Free-atom occupied HF orbitals (core+valence) in the uncontracted
    working basis, packaged as a PySCF basis dict for use as `minao`.

    x2c : None -> enable scalar relativity iff the working basis is a
           relativistically recontracted one (-DK, -DKH, x2c-, ANO-RCC, ...);
           True/False to force it on or off.
    """
    _fix_ecpbas(mol)
    if x2c is None:
        x2c = _is_relativistic_basis(mol.basis)

    elems = set(mol.elements)

    # uncontracted working basis, per element
    unc = {sym: gto.uncontract(mol._basis[sym]) for sym in elems}

    ref_basis = {}
    for sym in elems:
        # single-atom mol: carries both the (l, exponent) shell layout and the
        # atomic SCF, so no whole-molecule copy goes through atom_hf
        a1 = gto.M(atom=f'{sym} 0 0 0', basis={sym: unc[sym]},
                   spin=gto.charge(sym) % 2, verbose=0)
        ao_loc = a1.ao_loc_nr()
        shells_by_l = defaultdict(list)           # l -> [(exp, ao_start), ...]
        for ib in range(a1.nbas):
            shells_by_l[a1.bas_angular(ib)].append(
                (float(a1.bas_exp(ib)[0]), ao_loc[ib]))

        # spherically-averaged atomic HF in the uncontracted basis. sfx2c1e
        # only swaps get_hcore, so the angular averaging in .eig() still holds
        # (for a single atom the X2C hcore stays spherically symmetric).
        amf = (atom_hf.AtomHF1e(a1) if a1.nelectron == 1
               else atom_hf.AtomSphAverageRHF(a1))
        if x2c:
            amf = amf.sfx2c1e()
        amf.verbose = 0
        amf.run()

        c, occ = amf.mo_coeff, amf.mo_occ
        occ_cols = c[:, occ > occ_tol]            # occupied core + valence

        shells = []
        for l in sorted(shells_by_l):
            exps   = np.array([e for e, _ in shells_by_l[l]])
            starts = [s for _, s in shells_by_l[l]]
            p, ncomp = len(exps), 2 * l + 1
            # radial coeffs over exponents, all m-components stacked
            A = np.stack([occ_cols[s:s+ncomp, :] for s in starts])  # (p, ncomp, nocc)
            Rrad = A.reshape(p, -1)                                   # (p, ncomp*nocc)
            u, sv, _ = np.linalg.svd(Rrad, full_matrices=False)
            for i in np.where(sv > sv_tol * sv.max())[0]:            # distinct radial fns
                d = u[:, i]
                shells.append([l] + [[float(exps[k]), float(d[k])] for k in range(p)])
        ref_basis[sym] = shells
    return ref_basis

# plain all-electron Dunning cc-pVXZ, incl. aug- and core-valence variants;
# anchored $ so relativistic/PP suffixes (-dk, -pp, -f12) deliberately fail
_CCPVXZ = re.compile(r'^(daug|aug)?ccp(w?c)?v[dtq56]z$')

def _is_ccpvxz_family(basis):
    """True only if EVERY element uses a cc-pVXZ-family basis name."""
    def is_cc(name):
        if not isinstance(name, str):     # parsed/custom basis object -> not cc-pVXZ
            return False
        return bool(_CCPVXZ.match(re.sub(r'[\s\-_]', '', name.lower())))
    if isinstance(basis, str):
        return is_cc(basis)
    if isinstance(basis, dict):
        return bool(basis) and all(is_cc(v) for v in basis.values())
    return False

# relativistically recontracted basis families: Douglas-Kroll (cc-pVTZ-DK,
# cc-pwCVTZ-DK3), DKH (Sapporo-DKH3-TZP), X2C (x2c-TZVPall), ANO-RCC, ZORA,
# Dyall. Matched on the name with whitespace/hyphens/underscores stripped.
_RELATIVISTIC = re.compile(r'(dk\d?$|dkh\d?|^x2c|anorcc|^zora|^dyall)')

def _is_relativistic_basis(basis):
    """True if ANY element uses a relativistically recontracted basis.

    `any` rather than `all`: a mixed input still means the molecular SCF was
    run with scalar relativity, so the free-atom reference should match.
    """
    def is_rel(name):
        if not isinstance(name, str):     # parsed/custom basis object -> unknown
            return False
        return bool(_RELATIVISTIC.search(re.sub(r'[\s\-_]', '', name.lower())))
    if isinstance(basis, str):
        return is_rel(basis)
    if isinstance(basis, dict):
        return any(is_rel(v) for v in basis.values())
    return False

def name_fragments(frag_atmlist, elements, sep=""):
    """
    Build a name for each fragment by tagging every atom with its
    element symbol and its original index.

    frag_atmlist : list of lists of atom indices
    elements     : list mapping atom index -> element symbol
    sep          : string placed between atoms in a fragment name
    """
    return [sep.join(f"{elements[i]}{i}" for i in frag)
            for frag in frag_atmlist]

def riao_fragment(mf, nfrozen, frag_type='atom', more_loc=None, minao='minao'):
    # IAO localization
    mol = mf.mol
    s1e = mf.get_ovlp()
    moliao = lo.iao.reference_mol(mol, minao)
    nocc = np.count_nonzero(mf.mo_occ)
    orbocc = mf.mo_coeff[:,nfrozen:nocc]
    lo_coeff = lo.iao.iao(mol, orbocc, minao=minao)
    lo_coeff = lo.orth.vec_lowdin(lo_coeff, s1e)

    if frag_type == 'atom':
        frag_atmlist = tools.autofrag_atom(moliao, H2heavy=False)
    elif frag_type == 'h2heavy':
        frag_atmlist = tools.autofrag_atom(moliao, H2heavy=True)
    else:
        raise ValueError(f'Unsupported fragment type {str(frag_type)}')

    if more_loc is None:
        frag_list = tools.autofrag_iao(moliao, 'atom', frag_atmlist)
    elif more_loc == 'boys':
        lo_coeff = lo.Boys(mol, lo_coeff).kernel()
        lo_coeff = lo.orth.vec_lowdin(lo_coeff, s1e)
        frag_list = tools.map_lo_to_frag(mol, lo_coeff, frag_atmlist)
    elif more_loc == 'pm':
        lo_coeff = lo.PM(mol, lo_coeff).kernel()
        lo_coeff = lo.orth.vec_lowdin(lo_coeff, s1e)
        frag_list = tools.map_lo_to_frag(mol, lo_coeff, frag_atmlist)
    else:
        raise ValueError(f'Unsupported lo type {str(more_loc)}')

    frag_name = name_fragments(frag_atmlist, moliao.elements, sep="")

    ortho = lo_coeff.conj().T @ s1e @ lo_coeff
    assert np.allclose(ortho, np.eye(ortho.shape[1]), atol=1e-8), \
        f"IAOs not orthonormal: max dev {np.abs(ortho - np.eye(ortho.shape[1])).max():.2e}"
    
    return lo_coeff, frag_list, frag_name

def uiao_fragment(mf, nfrozen, frag_type='atom', more_loc=None, minao='minao'):
    # IAO localization
    mol = mf.mol
    s1e = mf.get_ovlp()
    moliao = lo.iao.reference_mol(mol, minao)
    nocca = np.count_nonzero(mf.mo_occ[0])
    noccb = np.count_nonzero(mf.mo_occ[1])
    orbocc_a = mf.mo_coeff[0][:,nfrozen:nocca]
    orbocc_b = mf.mo_coeff[1][:,nfrozen:noccb]
    lo_coeff_a = lo.iao.iao(mol, orbocc_a, minao)
    lo_coeff_a = lo.orth.vec_lowdin(lo_coeff_a, s1e)
    lo_coeff_b = lo.iao.iao(mol, orbocc_b, minao)
    lo_coeff_b = lo.orth.vec_lowdin(lo_coeff_b, s1e)

    if frag_type == 'atom':
        frag_atmlist = tools.autofrag_atom(moliao, H2heavy=False)
    elif frag_type == 'h2heavy':
        frag_atmlist = tools.autofrag_atom(moliao, H2heavy=True)
    else:
        raise ValueError(f'Unsupported fragment type {str(frag_type)}')

    if more_loc is None:
        lo_coeff = [lo_coeff_a, lo_coeff_b]
        frag_list = tools.autofrag_iao(moliao, 'atom', frag_atmlist)
        frag_list = [[i,i] for i in frag_list]
    elif more_loc == 'boys':
        lo_coeff_a = lo.Boys(mol, lo_coeff_a).kernel()
        lo_coeff_a = lo.orth.vec_lowdin(lo_coeff_a, s1e)
        lo_coeff_b = lo.Boys(mol, lo_coeff_b).kernel()
        lo_coeff_b = lo.orth.vec_lowdin(lo_coeff_b, s1e)
        lo_coeff = [lo_coeff_a, lo_coeff_b]
        frag_list = tools.map_lo_to_frag(mol, lo_coeff, frag_atmlist)
    elif more_loc == 'pm':
        lo_coeff_a = lo.PM(mol, lo_coeff_a).kernel()
        lo_coeff_a = lo.orth.vec_lowdin(lo_coeff_a, s1e)
        lo_coeff_b = lo.PM(mol, lo_coeff_b).kernel()
        lo_coeff_b = lo.orth.vec_lowdin(lo_coeff_b, s1e)
        lo_coeff = [lo_coeff_a, lo_coeff_b]
        frag_list = tools.map_lo_to_frag(mol, lo_coeff, frag_atmlist)
    else:
        raise TypeError(f'Unsupported lo type {str(more_loc)}')

    frag_name = name_fragments(frag_atmlist, moliao.elements, sep="")

    ortho_a = lo_coeff[0].conj().T @ s1e @ lo_coeff[0]
    ortho_b = lo_coeff[1].conj().T @ s1e @ lo_coeff[1]
    assert np.allclose(ortho_a, np.eye(ortho_a.shape[1]), atol=1e-8), \
        f"IAOs not orthonormal: max dev {np.abs(ortho_a - np.eye(ortho_a.shape[1])).max():.2e}"
    assert np.allclose(ortho_b, np.eye(ortho_b.shape[1]), atol=1e-8), \
        f"IAOs not orthonormal: max dev {np.abs(ortho_b - np.eye(ortho_b.shape[1])).max():.2e}"

    return lo_coeff, frag_list, frag_name

def iao_fragment(mf, nfrozen=None, frag_type='h2heavy', more_loc=None,
                 minao='minao', x2c=None):
    mol = mf.mol

    if nfrozen is None:
        nfrozen = elements.chemcore(mol)

    if not _is_ccpvxz_family(mol.basis):
        print('Detected basis set not in the cc-pVXZ family. '
              'Run free atom scf to generate reference basis.')
        if x2c is None:
            x2c = _is_relativistic_basis(mol.basis)
        if x2c:
            print('Detected relativistic basis set. '
                  'Run free atom scf with scalar relativistic (sfX2C1e) effects.')
        minao = free_atom_minao(mol, occ_tol=1e-6, sv_tol=1e-8, x2c=x2c)

    if isinstance(mf, scf.rhf.RHF):
        return riao_fragment(mf, nfrozen, frag_type, more_loc, minao)
    elif isinstance(mf, scf.uhf.UHF):
        return uiao_fragment(mf, nfrozen, frag_type, more_loc, minao)
    else:
        raise TypeError(f'Unsupported mf type {type(mf)}')

# def plot_density(mf, orbloc, lno_split, idx):
#     from pyscf.tools import cubegen
#     # plot density as rho(r) = sum_p |psi_p(r)|^2
#     mol = mf.mol
#     if spin_type == "restricted":
#         dm_ctr = orbloc @ orbloc.T
#         _ = cubegen.density(mol, f'ctr_density_{idx+1}.cube', dm_ctr)
#         dm_las = lno_coeff[:,lno_active] @ lno_coeff[:,lno_active].T
#         _ = cubegen.density(mol, f'las_density_{idx+1}.cube', dm_las)

#     elif spin_type == "unrestricted":
#         dm_ctr = orbloc[0] @ orbloc[0].T + orbloc[1] @ orbloc[1].T
#         _ = cubegen.density(mol, f'ctr_density_{idx+1}.cube', dm_ctr)
#         dm_las = (lno_coeff[0][:,lno_active[0]] @ lno_coeff[0][:,lno_active[0]].T
#                   +lno_coeff[1][:,lno_active[1]] @ lno_coeff[1][:,lno_active[1]].T)
#         _ = cubegen.density(mol, f'las_density_{idx+1}.cube', dm_las)
        
#     return None


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


def check_rspan(lo, s1e, mo):
    # lo passed as mo1 -> span12 tests  mo ⊆ span(lo)  (lo spans at least mo)
    span12, span21 = mo_span(lo, s1e, mo)
    return span12, span21


def check_uspan(lo, s1e, mo):
    span12_a, span21_a = mo_span(lo[0], s1e, mo[0])
    span12_b, span21_b = mo_span(lo[1], s1e, mo[1])
    return (span12_a, span12_b), (span21_a, span21_b)


def check_span(mf, lo_coeff_occ, frozen=0, thresh=1e-6):
    s1e = mf.get_ovlp()

    if isinstance(mf, scf.uhf.UHF):
        if isinstance(frozen, int):
            frozen = (frozen, frozen)
        nocc = (np.count_nonzero(mf.mo_occ[0]),
                np.count_nonzero(mf.mo_occ[1]))
        mo_coeff_occ = (mf.mo_coeff[0][:, frozen[0]:nocc[0]],
                        mf.mo_coeff[1][:, frozen[1]:nocc[1]])
        p12, p21 \
            = check_uspan(lo_coeff_occ, s1e, mo_coeff_occ)
        span12 = p12[0] < thresh and p12[1] < thresh
        span21 = p21[0] < thresh and p21[1] < thresh
    elif isinstance(mf, scf.rhf.RHF):
        nocc = np.count_nonzero(mf.mo_occ)
        mo_coeff_occ = mf.mo_coeff[:, frozen:nocc]
        p12, p21 = check_rspan(lo_coeff_occ, s1e, mo_coeff_occ)
        span12 = p12 < thresh
        span21 = p21 < thresh
    else:
        raise TypeError(f'unsupported mean-field type: {type(mf)}')

    print(f'LO occ span the occupied MO occ space - {span12}.\n'
          f'MO occ span the occupied LO occ space - {span21}.')
    
    if not span12:
        raise ValueError(f"LOs DO NOT SPAN MOs, CHECK THE LOCALIZATION!!!"
                         f"the projection lost are {p12} {p21}")
    
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


def split_lno(mlno, lno_coeff, lno_frozen):
    mf = mlno._scf
    mol = mf.mol
    mo_occ = mlno.mo_occ

    if isinstance(mf, scf.rhf.RHF):
        nocc = np.count_nonzero(mo_occ)

        idx_act    = np.array([i for i in range(mol.nao) if i not in lno_frozen], dtype=int)
        idx_frzocc = np.array([i for i in range(nocc) if i not in idx_act], dtype=int)
        idx_actocc = np.array([i for i in range(nocc) if i in idx_act], dtype=int)
        idx_actvir = np.array([i for i in range(nocc, mol.nao) if i in idx_act], dtype=int)
        idx_frzvir = np.array([i for i in range(nocc, mol.nao) if i not in idx_act], dtype=int)

        lno_frzocc = lno_coeff[:, idx_frzocc]
        lno_actocc = lno_coeff[:, idx_actocc]
        lno_actvir = lno_coeff[:, idx_actvir]
        lno_frzvir = lno_coeff[:, idx_frzvir]

        nfrzocc = len(idx_frzocc)
        nactocc = len(idx_actocc)
        nactvir = len(idx_actvir)
        nfrzvir = len(idx_frzvir)

        # nact    = len(idx_actocc) + len(idx_actvir)

        lno_split = [lno_frzocc, lno_actocc, lno_actvir, lno_frzvir]

    elif isinstance(mf, scf.uhf.UHF):
        nocc_a = np.count_nonzero(mo_occ[0])
        nocc_b = np.count_nonzero(mo_occ[1])

        idx_act_a = np.array([i for i in range(mol.nao) if i not in lno_frozen[0]], dtype=int)
        idx_act_b = np.array([i for i in range(mol.nao) if i not in lno_frozen[1]], dtype=int)

        idx_frzocc_a = np.array([i for i in range(nocc_a) if i not in idx_act_a], dtype=int)
        idx_actocc_a = np.array([i for i in range(nocc_a) if i in idx_act_a], dtype=int)
        idx_actvir_a = np.array([i for i in range(nocc_a, mol.nao) if i in idx_act_a], dtype=int)
        idx_frzvir_a = np.array([i for i in range(nocc_a, mol.nao) if i not in idx_act_a], dtype=int)
        idx_frzocc_b = np.array([i for i in range(nocc_b) if i not in idx_act_b], dtype=int)
        idx_actocc_b = np.array([i for i in range(nocc_b) if i in idx_act_b], dtype=int)
        idx_actvir_b = np.array([i for i in range(nocc_b, mol.nao) if i in idx_act_b], dtype=int)
        idx_frzvir_b = np.array([i for i in range(nocc_b, mol.nao) if i not in idx_act_b], dtype=int)

        lno_frzocc_a = lno_coeff[0][:, idx_frzocc_a]
        lno_actocc_a = lno_coeff[0][:, idx_actocc_a]
        lno_actvir_a = lno_coeff[0][:, idx_actvir_a]
        lno_frzvir_a = lno_coeff[0][:, idx_frzvir_a]
        lno_frzocc_b = lno_coeff[1][:, idx_frzocc_b]
        lno_actocc_b = lno_coeff[1][:, idx_actocc_b]
        lno_actvir_b = lno_coeff[1][:, idx_actvir_b]
        lno_frzvir_b = lno_coeff[1][:, idx_frzvir_b]

        nfrzocc = [len(idx_frzocc_a), len(idx_frzocc_b)]
        nactocc = [len(idx_actocc_a), len(idx_actocc_b)]
        nactvir = [len(idx_actvir_a), len(idx_actvir_b)]
        nfrzvir = [len(idx_frzvir_a), len(idx_frzvir_b)]
        # nact    = [len(idx_actocc_a) + len(idx_actvir_a),
        #            len(idx_actocc_b) + len(idx_actvir_b)] 

        lno_split_a = [lno_frzocc_a, lno_actocc_a, lno_actvir_a, lno_frzvir_a]
        lno_split_b = [lno_frzocc_b, lno_actocc_b, lno_actvir_b, lno_frzvir_b]
        lno_split = [lno_split_a, lno_split_b]

    print(f'nfrozen occupied orbitals:  {nfrzocc}')
    print(f'nactive occupied orbitals:  {nactocc}')
    print(f'nactive virtual orbitals:   {nactvir}')
    print(f'nfrozen virtual orbitals:   {nfrzvir}')

    return lno_split, nfrzocc, nactocc, nactvir, nfrzvir