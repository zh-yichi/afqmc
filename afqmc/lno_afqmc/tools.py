import re, os, glob, h5py
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

    # An ECP basis has no core functions, so the free atom must carry the same
    # pseudopotential as the molecule: without it the atomic SCF would put all
    # Z electrons into a valence-only basis and the reference basis would come
    # back with spurious core-like radial functions.
    ecp = getattr(mol, '_ecp', None) or {}
    first_ia = {}
    for ia, sym in enumerate(mol.elements):
        first_ia.setdefault(sym, ia)

    ref_basis = {}
    for sym in elems:
        atom_ecp = {sym: ecp[sym]} if sym in ecp else {}
        # electrons left on the free atom once the ECP core is removed. Only
        # its parity is used: AtomSphAverageRHF takes its (fractional, sphe-
        # rically averaged) occupations from elements.NRSRHF_CONFIGURATION and
        # never looks at mol.spin -- this just keeps gto.M from complaining
        # about an odd electron count, exactly as pyscf's own get_atm_nrhf does
        nelec = gto.charge(sym) - mol.atom_nelec_core(first_ia[sym])

        # single-atom mol: carries both the (l, exponent) shell layout and the
        # atomic SCF, so no whole-molecule copy goes through atom_hf
        a1 = gto.M(atom=f'{sym} 0 0 0',
                   basis={sym: unc[sym]},
                   ecp=atom_ecp,
                   spin=nelec % 2,
                   verbose=0)
        if a1.nelectron != nelec:
            raise RuntimeError(
                f'free-atom {sym} has {a1.nelectron} electrons but the molecule '
                f'leaves it {nelec}: the ECP did not carry over. Check that '
                f'mol._ecp is keyed by element symbol (found {list(ecp)}).')
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
        # amf.verbose = mol.verbose
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

IAO_FILE_VERSION = 1

def _iao_spin_kind(lo_coeff):
    """'restricted' for a single (nao, nlo) array, 'unrestricted' for a pair."""
    if isinstance(lo_coeff, np.ndarray) and lo_coeff.ndim == 2:
        return 'restricted'
    if (isinstance(lo_coeff, (list, tuple)) and len(lo_coeff) == 2
            and all(isinstance(c, np.ndarray) and c.ndim == 2 for c in lo_coeff)):
        return 'unrestricted'
    raise TypeError('lo_coeff must be a 2D array (restricted) or a pair of '
                    f'2D arrays (unrestricted), got {type(lo_coeff)}')

def save_iao_fragment(filename, lo_coeff, frag_list, frag_name, mol=None, meta=None):
    """Dump the output of `iao_fragment` to an HDF5 file.

    The molecular fingerprint (nuclear charges, geometry, nao) is stored along
    with the orbitals so that `load_iao_fragment` can refuse to hand the IAOs
    to a different molecule.
    """
    kind = _iao_spin_kind(lo_coeff)

    if len(frag_list) != len(frag_name):
        raise ValueError(f'frag_list ({len(frag_list)}) and frag_name '
                         f'({len(frag_name)}) have different lengths')

    # remember whether the caller had plain lists or ndarrays so a
    # save/load round trip returns exactly what iao_fragment would have
    frag0 = frag_list[0] if kind == 'restricted' else frag_list[0][0]
    idx_kind = 'array' if isinstance(frag0, np.ndarray) else 'list'

    with h5py.File(filename, 'w') as fh5:
        fh5.attrs['version'] = IAO_FILE_VERSION
        fh5.attrs['spin'] = kind

        if kind == 'restricted':
            fh5['lo_coeff'] = np.asarray(lo_coeff)
        else:
            fh5['lo_coeff_a'] = np.asarray(lo_coeff[0])
            fh5['lo_coeff_b'] = np.asarray(lo_coeff[1])

        fh5['frag_name'] = np.array(list(frag_name), dtype=h5py.string_dtype())

        grp = fh5.create_group('frag_list')
        grp.attrs['nfrag'] = len(frag_list)
        grp.attrs['idx_kind'] = idx_kind
        for i, frag in enumerate(frag_list):
            if kind == 'restricted':
                grp[f'{i}'] = np.asarray(frag, dtype=np.int64)
            else:
                grp[f'{i}_a'] = np.asarray(frag[0], dtype=np.int64)
                grp[f'{i}_b'] = np.asarray(frag[1], dtype=np.int64)

        if mol is not None:
            mgrp = fh5.create_group('mol')
            mgrp['atom_charges'] = np.asarray(mol.atom_charges())
            mgrp['atom_coords'] = np.asarray(mol.atom_coords())   # bohr
            mgrp.attrs['natm'] = mol.natm
            mgrp.attrs['nao'] = mol.nao
            mgrp.attrs['basis'] = str(mol.basis)[:1024]           # provenance only
            mgrp.attrs['charge'] = mol.charge
            mgrp.attrs['spin'] = mol.spin

        mtgrp = fh5.create_group('meta')
        for key, val in (meta or {}).items():
            mtgrp.attrs[key] = str(val)[:1024]

    return filename

def load_iao_fragment(filename, mol=None, s1e=None, coord_tol=1e-6, ortho_tol=1e-8):
    """Read back what `save_iao_fragment` wrote.

    mol : if given, the stored molecular fingerprint must match it.
    s1e : if given, the loaded IAOs must be orthonormal w.r.t. it.
    """
    with h5py.File(filename, 'r') as fh5:
        version = int(fh5.attrs.get('version', -1))
        if version != IAO_FILE_VERSION:
            raise ValueError(f'{filename}: IAO file version {version} is not '
                             f'readable by this code (expected {IAO_FILE_VERSION})')
        kind = fh5.attrs['spin']

        if kind == 'restricted':
            lo_coeff = np.asarray(fh5['lo_coeff'][()])
            nao = lo_coeff.shape[0]
        else:
            lo_coeff = [np.asarray(fh5['lo_coeff_a'][()]),
                        np.asarray(fh5['lo_coeff_b'][()])]
            nao = lo_coeff[0].shape[0]

        frag_name = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in fh5['frag_name'][()]]

        grp = fh5['frag_list']
        nfrag = int(grp.attrs['nfrag'])
        as_list = grp.attrs.get('idx_kind', 'array') == 'list'

        def _idx(dset):
            idx = np.asarray(dset[()], dtype=np.int64)
            return idx.tolist() if as_list else idx

        if kind == 'restricted':
            frag_list = [_idx(grp[f'{i}']) for i in range(nfrag)]
        else:
            frag_list = [[_idx(grp[f'{i}_a']), _idx(grp[f'{i}_b'])]
                         for i in range(nfrag)]

        stored_mol = dict(fh5['mol'].attrs) if 'mol' in fh5 else None
        if stored_mol is not None:
            stored_mol['atom_charges'] = np.asarray(fh5['mol/atom_charges'][()])
            stored_mol['atom_coords'] = np.asarray(fh5['mol/atom_coords'][()])

        meta = dict(fh5['meta'].attrs) if 'meta' in fh5 else {}

    if len(frag_name) != nfrag:
        raise ValueError(f'{filename}: {nfrag} fragments but {len(frag_name)} names')

    if mol is not None:
        if mol.nao != nao:
            raise ValueError(f'{filename}: IAOs were built in a basis with {nao} '
                             f'AOs but the current mol has {mol.nao}')
        if stored_mol is None:
            print(f'Warning: {filename} carries no molecular fingerprint; '
                  'cannot verify that it belongs to this molecule.')
        else:
            if not np.array_equal(stored_mol['atom_charges'], mol.atom_charges()):
                raise ValueError(f'{filename}: nuclear charges differ from the '
                                 'current molecule')
            dev = np.abs(stored_mol['atom_coords'] - mol.atom_coords()).max()
            if dev > coord_tol:
                raise ValueError(f'{filename}: geometry differs from the current '
                                 f'molecule (max deviation {dev:.2e} bohr > '
                                 f'{coord_tol:.1e})')

    if s1e is not None:
        for c in ([lo_coeff] if kind == 'restricted' else lo_coeff):
            ortho = c.conj().T @ s1e @ c
            dev = np.abs(ortho - np.eye(ortho.shape[1])).max()
            if dev > ortho_tol:
                raise ValueError(f'{filename}: loaded IAOs are not orthonormal '
                                 f'w.r.t. the current overlap (max dev {dev:.2e}). '
                                 'They most likely belong to another molecule '
                                 'or another basis set.')

    return lo_coeff, frag_list, frag_name, meta

def iao_fragment(mf, nfrozen=None, frag_type='h2heavy', more_loc=None,
                 minao='minao', x2c=None, save2=None, read_from=None):
    """Build (or reuse) the IAO fragment input for an LNO calculation.

    save2     : path of an HDF5 file to write (lo_coeff, frag_list, frag_name) to.
    read_from : path of such a file to read instead of rebuilding the IAOs.
                The stored molecule must match `mf.mol`.
    """
    mol = mf.mol

    want = 'unrestricted' if isinstance(mf, scf.uhf.UHF) else 'restricted'

    if read_from is not None:
        if not os.path.isfile(read_from):
            raise FileNotFoundError(
                f'IAO fragment file {read_from} not found. Run the calculation '
                'once with save2=<file> to create it.')
        print(f'Reading IAO fragments from {read_from}')
        lo_coeff, frag_list, frag_name, meta = load_iao_fragment(
            read_from, mol=mol, s1e=mf.get_ovlp())

        kind = _iao_spin_kind(lo_coeff)
        if kind != want:
            raise ValueError(f'{read_from} holds {kind} IAOs but {type(mf).__name__} '
                             f'needs {want} ones')
        for key, val in (('frag_type', frag_type), ('more_loc', more_loc)):
            if key in meta and meta[key] != str(val):
                print(f'Warning: {read_from} was built with {key}={meta[key]}, '
                      f'but {key}={val} was requested. Using the stored fragments.')

        print(f'Loaded {len(frag_name)} IAO fragments: {frag_name}')

        if save2 is not None and os.path.abspath(save2) != os.path.abspath(read_from):
            save_iao_fragment(save2, lo_coeff, frag_list, frag_name, mol=mol, meta=meta)
            print(f'IAO fragments copied to {save2}')

        return lo_coeff, frag_list, frag_name

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
        lo_coeff, frag_list, frag_name = \
            riao_fragment(mf, nfrozen, frag_type, more_loc, minao)
    elif isinstance(mf, scf.uhf.UHF):
        lo_coeff, frag_list, frag_name = \
            uiao_fragment(mf, nfrozen, frag_type, more_loc, minao)
    else:
        raise TypeError(f'Unsupported mf type {type(mf)}')

    if save2 is not None:
        meta = {'frag_type': frag_type,
                'more_loc': more_loc,
                'nfrozen': nfrozen,
                'minao': minao if isinstance(minao, str) else 'free_atom_minao'}
        save_iao_fragment(save2, lo_coeff, frag_list, frag_name, mol=mol, meta=meta)
        print(f'IAO fragments saved to {save2}')

    return lo_coeff, frag_list, frag_name

def plot_density(mf, orbloc, lno_split, ifrag):
    '''
    mf: mean-field object
    orbloc: orbital of the local fragment
    lno_split: local natural orbitals of the local fragment
               splitted into frzocc, actocc, actvir, frzvir
    '''
    from pyscf.tools import cubegen
    # plot density as rho(r) = sum_p |psi_p(r)|^2
    cubedir = './cubefiles'
    os.makedirs(cubedir, exist_ok=True)

    mol = mf.mol
    if isinstance(mf, scf.rhf.RHF):
        dm_frg = orbloc @ orbloc.T
        las_coeff = np.hstack(lno_split[1:3])
        dm_las = las_coeff @ las_coeff.T
        _ = cubegen.density(mol, f'{cubedir}/Fragment_Density_{ifrag}.cube', dm_frg)
        _ = cubegen.density(mol, f'{cubedir}/LocalAS_Density_{ifrag}.cube', dm_las)

    elif isinstance(mf, scf.uhf.UHF):
        dm_frg = orbloc[0] @ orbloc[0].T + orbloc[1] @ orbloc[1].T
        las_coeff_a = np.hstack(lno_split[0][1:3])
        las_coeff_b = np.hstack(lno_split[1][1:3])
        dm_las = (las_coeff_a @ las_coeff_a.T + las_coeff_b @ las_coeff_b.T)
        _ = cubegen.density(mol, f'{cubedir}/Fragment_Density_{ifrag}.cube', dm_frg)
        _ = cubegen.density(mol, f'{cubedir}/LocalAS_Density_{ifrag}.cube', dm_las)
    else:
        raise TypeError(f'Unsupported mf type {type(mf)}')
        
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

    # print(f'nfrozen occupied orbitals:  {nfrzocc}')
    # print(f'nactive occupied orbitals:  {nactocc}')
    # print(f'nactive virtual orbitals:   {nactvir}')
    # print(f'nfrozen virtual orbitals:   {nfrzvir}')

    return lno_split, nfrzocc, nactocc, nactvir, nfrzvir

# regex for one fragment row of the LNO-AFQMC result table:
#   Num  Fragment  LAS SIZE  E(MP2) ...
# where LAS SIZE is either a bare int (restricted) or the numpy repr of the
# (alpha, beta) pair, e.g. "[62 61]" (unrestricted).
_LNO_ROW = re.compile(r'^\s*(\d+)\s+(\S+)\s+(\[[^\]]*\]|\d+)\s+(-?\d+\.\d+)')

def read_lno_size(filename='lno_result.out'):
    """Parse the LAS sizes out of an LNO-AFQMC result file.

    Returns (frag_idx, frag_name, las_size) where `frag_idx` is the 0-based
    fragment index (the printed "Num" minus one), and `las_size` has shape
    (nfrag, nspin) -- nspin = 1 for restricted, 2 for unrestricted.
    """
    frag_idx, frag_name, las_size = [], [], []
    with open(filename, 'r') as f:
        for line in f:
            m = _LNO_ROW.match(line)
            if m is None:
                continue
            frag_idx.append(int(m.group(1)) - 1)
            frag_name.append(m.group(2))
            las_size.append([int(x) for x in m.group(3).strip('[]').split()])

    if not frag_idx:
        raise ValueError(f'no fragment rows found in {filename}')

    nspin = set(map(len, las_size))
    if len(nspin) != 1:
        raise ValueError(f'inconsistent LAS SIZE columns in {filename}')

    return (np.asarray(frag_idx, dtype=np.int32), frag_name,
            np.asarray(las_size, dtype=np.int32))

def sort_frag_by_size(filename='lno_result.out', key='max', reverse=False,
                      return_size=False):
    """Fragment indices ordered from the smallest to the largest LAS.

    key : how to collapse the spin channels of an unrestricted calculation
          into one number -- 'max' (default, matches the cost-determining
          dimension), 'sum', 'alpha' or 'beta'. Ignored if restricted.
    reverse : largest to smallest instead.
    return_size : also return the collapsed sizes in the same order.
    """
    frag_idx, frag_name, las_size = read_lno_size(filename)

    if las_size.shape[1] == 1:
        size = las_size[:, 0]
    elif key == 'max':
        size = las_size.max(axis=1)
    elif key == 'sum':
        size = las_size.sum(axis=1)
    elif key == 'alpha':
        size = las_size[:, 0]
    elif key == 'beta':
        size = las_size[:, 1]
    else:
        raise ValueError(f'unknown key {key}')

    # stable sort so equally sized fragments keep their printed order
    order = np.argsort(size, kind='stable')
    if reverse:
        order = order[::-1]

    idx = frag_idx[order].tolist()
    return (idx, size[order].tolist()) if return_size else idx


# ---------------------------------------------------------------------------
# Collecting per-fragment output files into a single LNO-AFQMC result table
# ---------------------------------------------------------------------------
# `run_afqmc` appends a block like
#
#   ========================= Fragment1 Results ==========================
#   \t LNO Fragment Fe0
#   ----------------------------------------------------------------------
#   \t LNO-Active Space electrons: [13 13] | orbitals: [63 61]
#   \t LNO-MP2 Fragment Energy:    -0.52620714
#   ...
#   ======================================================================
#
# to `fragment.out<N>` (N = 1-based fragment index) as soon as a fragment is
# done, so a job that dies -- or a run split over several jobs -- still leaves
# every finished fragment on disk.
_FRAG_HEAD = re.compile(r'^=+\s*Fragment(\d+)\s+Results\s*=+\s*$')
_FRAG_TAIL = re.compile(r'^=+\s*$')
_FRAG_SIZE = r'(\[[^\]]*\]|\d+)'
_FRAG_NUM = r'(-?(?:\d+\.\d*|\.?\d+)(?:[eE][-+]?\d+)?|[-+]?nan|[-+]?inf)'
_FRAG_FIELD = (
    ('name',   re.compile(r'LNO Fragment\s+(.*?)\s*$')),
    ('las',    re.compile(r'LNO-Active Space electrons:\s*' + _FRAG_SIZE +
                          r'\s*\|\s*orbitals:\s*' + _FRAG_SIZE)),
    ('e_mp',   re.compile(r'LNO-MP2 Fragment Energy:\s*' + _FRAG_NUM)),
    ('e_cc',   re.compile(r'LNO-CCSD Fragment Energy:\s*' + _FRAG_NUM)),
    ('e_qmc',  re.compile(r'LNO-AFQMC Fragment Energy:\s*' + _FRAG_NUM +
                          r'\s*\+/-\s*' + _FRAG_NUM)),
    ('t_cc',   re.compile(r'LNO-CCSD Fragment Time:\s*' + _FRAG_NUM)),
    ('t_wait', re.compile(r'LNO-CCSD Fragment Wait:\s*' + _FRAG_NUM)),
    ('t_qmc',  re.compile(r'LNO-AFQMC Fragment Time:\s*' + _FRAG_NUM)),
)

def read_fragment_out(filename):
    """Parse the "Fragment<N> Results" blocks of one `fragment.out<N>` file.

    Returns a list of dicts, one per complete block, in file order. A file
    that is still being written (or whose fragment crashed) simply yields
    fewer blocks -- no exception. `fragment.out<N>` is opened in append mode,
    so re-running a fragment leaves several blocks behind; the caller is
    expected to keep the last one.
    """
    blocks = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    i, nline = 0, len(lines)
    while i < nline:
        m = _FRAG_HEAD.match(lines[i])
        if m is None:
            i += 1
            continue

        blk = {'num': int(m.group(1)), 'file': filename}
        i += 1
        while i < nline and not _FRAG_TAIL.match(lines[i]):
            if _FRAG_HEAD.match(lines[i]):     # truncated block, restart here
                break
            line = lines[i]
            for key, pat in _FRAG_FIELD:
                mm = pat.search(line)
                if mm is None:
                    continue
                if key == 'name':
                    blk['name'] = mm.group(1)
                elif key == 'las':
                    blk['nelec'] = mm.group(1)
                    blk['norb'] = mm.group(2)
                elif key == 'e_qmc':
                    blk['e_qmc'] = float(mm.group(1))
                    blk['e_qmc_err'] = float(mm.group(2))
                else:
                    blk[key] = float(mm.group(1))
                break
            i += 1

        # only keep blocks that carry the full record
        need = ('name', 'norb', 'e_mp', 'e_cc', 'e_qmc', 'e_qmc_err',
                't_cc', 't_wait', 't_qmc')
        if all(k in blk for k in need):
            blocks.append(blk)

    return blocks

def collect_lno_result(pattern='fragment.out*', outfile='lno_result.out',
                       lno_thresh=None, nfrag_tot=None, verbose=True):
    """Rebuild an `lno_result.out` table from the per-fragment output files.

    Same layout as the file `run_afqmc` writes at the end of a complete run,
    so `read_lno_size` / `sort_frag_by_size` work on it unchanged. Use it when
    the fragments were spread over several jobs, or when a job died before
    reaching the final summary.

    pattern   : glob for the fragment files (or an explicit list of paths).
    outfile   : where to write the table; None only returns the numbers.
    lno_thresh: the LNO threshold(s) of the run -- a float or a pair. The
                fragment files do not record it, so it is printed as "n/a"
                when not given.
    nfrag_tot : total number of fragments expected, used only to report which
                ones are still missing. Defaults to the largest fragment
                number found.
    verbose   : print the collected/skipped/missing files.

    Returns (e_mp, e_cc, e_qmc, e_qmc_err, lno_max), like `run_afqmc`.
    """
    if isinstance(pattern, str):
        files = glob.glob(pattern)
    else:
        files = list(pattern)

    if not files:
        raise ValueError(f'no fragment files match {pattern}')

    # oldest first, so a re-run of a fragment overrides the earlier attempt
    files.sort(key=lambda p: (os.path.getmtime(p), p))

    frags, empty = {}, []
    for fname in files:
        blocks = read_fragment_out(fname)
        if not blocks:
            empty.append(fname)
            continue
        if len(blocks) > 1 and verbose:
            print(f'{fname}: {len(blocks)} result blocks, keeping the last')
        blk = blocks[-1]
        if blk['num'] in frags and verbose:
            print(f"fragment {blk['num']}: {fname} overrides "
                  f"{frags[blk['num']]['file']}")
        frags[blk['num']] = blk

    if not frags:
        raise ValueError(f'no complete fragment results found in {len(files)} '
                         f'file(s) matching {pattern}')

    nums = sorted(frags)
    rows = [frags[n] for n in nums]

    if nfrag_tot is None:
        nfrag_tot = nums[-1]
    missing = [n for n in range(1, nfrag_tot + 1) if n not in frags]

    if verbose:
        print(f'collected {len(rows)} fragment(s) from {len(files)} file(s): '
              f'{nums}')
        if empty:
            print(f'no complete result block in: {", ".join(empty)}')
        if missing:
            print(f'missing fragment(s): {missing}')

    e_mp = float(np.sum([r['e_mp'] for r in rows]))
    e_cc = float(np.sum([r['e_cc'] for r in rows]))
    e_qmc = float(np.sum([r['e_qmc'] for r in rows]))
    e_qmc_err = float(np.sqrt(np.sum([r['e_qmc_err']**2 for r in rows])))
    tot_cc_time = float(np.sum([r['t_cc'] for r in rows]))
    tot_qmc_time = float(np.sum([r['t_qmc'] for r in rows]))
    tot_wait_time = float(np.sum([r['t_wait'] for r in rows]))
    serial_time = tot_cc_time + tot_qmc_time

    lno_max = max(max(int(x) for x in r['norb'].strip('[]').split())
                  for r in rows)

    if lno_thresh is None:
        lno_thresh_str = 'n/a'
    elif np.ndim(lno_thresh) == 0:
        lno_thresh_str = f'{float(lno_thresh):.2e}'
    else:
        lno_thresh_str = '[' + ', '.join(f'{x:.2e}' for x in lno_thresh) + ']'

    if outfile is not None:
        with open(outfile, 'w') as f:
            width = 120
            f.write('=' * width + '\n')
            f.write(f'{"LNO-AFQMC Results":^{width}}\n')
            f.write('=' * width + '\n')

            f.write(f'{"Num":>4s}  {"Fragment":>16s}  {"LAS SIZE":>10s}  '
                    f'{"E(MP2)":>10s}  {"E(CCSD)":>10s}  '
                    f'{"E(AFQMC)":>10s}  {"Error":>8s}  '
                    f'{"t(CCSD)":>8s}  {"t(wait)":>8s}  {"t(AFQMC)":>8s}\n')
            f.write('-' * width + '\n')

            for r in rows:
                f.write(f"{r['num']:4d}  {r['name']:>16s}  {r['norb']:10s}  "
                        f"{r['e_mp']:10.8f}  {r['e_cc']:10.8f}  "
                        f"{r['e_qmc']:10.5f}  {r['e_qmc_err']:8.5f}  "
                        f"{r['t_cc']:8.2f}  {r['t_wait']:8.2f}  "
                        f"{r['t_qmc']:8.2f}\n")

            f.write('-' * width + '\n')

            f.write(f'{"Summarize Fragments":^{width}}\n')
            f.write('-' * width + '\n')

            f.write(f'{"LNO-Thresh":<20} {"Max LAS":>8} '
                    f'{"E[MP2]":>12} {"E[CCSD]":>12} '
                    f'{"E[AFQMC]":>10} {"Err[AFQMC]":>10} '
                    f'{"CCSD-Time":>10} {"AFQMC-Time":>10}\n')

            f.write(f'{lno_thresh_str:<20} {lno_max:>8} '
                    f'{e_mp:>12.8f} {e_cc:>12.8f} '
                    f'{e_qmc:>10.5f} {e_qmc_err:>10.5f} '
                    f'{tot_cc_time:>10.2f} {tot_qmc_time:>10.2f}\n')

            f.write('-' * width + '\n')
            f.write(f'{"Collected Fragments":^{width}}\n')
            f.write('-' * width + '\n')
            f.write(f'{"Fragments collected":<28} {len(rows):>10d} '
                    f'of {nfrag_tot}\n')
            f.write(f'{"Missing fragments":<28} '
                    f'{(str(missing) if missing else "none"):>10s}\n')
            f.write(f'{"Serial equivalent (CPU+GPU)":<28} '
                    f'{serial_time:>10.2f} s\n')
            f.write(f'{"CPU time hidden behind GPU":<28} '
                    f'{tot_cc_time - tot_wait_time:>10.2f} s '
                    f'({100.0*(tot_cc_time - tot_wait_time)/tot_cc_time if tot_cc_time > 0 else 0.0:.1f}%)\n')

            f.write('=' * width + '\n\n')

        if verbose:
            print(f'wrote {outfile}')

    return e_mp, e_cc, e_qmc, e_qmc_err, lno_max
