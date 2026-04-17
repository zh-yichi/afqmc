import time
import os
from functools import partial
from typing import List, Optional, Tuple, Union
import pickle
import jax
import jax.numpy as jnp
from jax import random, lax, jit
import numpy as np
from pyscf import __config__,  df,  lib, mcscf, scf
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD
from ad_afqmc import pyscf_interface, sampling, config
from ad_afqmc.propagation import propagator
from ad_afqmc.wavefunctions import wave_function


modified_cholesky = pyscf_interface.modified_cholesky
generate_integrals = pyscf_interface.generate_integrals
ao2mo_chol_copy = pyscf_interface.ao2mo_chol_copy
ao2mo_chol = pyscf_interface.ao2mo_chol
write_dqmc = pyscf_interface.write_dqmc

#@jit
def fix_len_chunked_cholesky(mol, chol_len, max_error=1e-6, verbose=False, cmax=10):
    """Modified cholesky decomposition of certain length from pyscf eris."""

    nao = mol.nao_nr()
    diag = np.zeros(nao * nao)
    nchol_max = cmax * nao
    # This shape is more convenient for pauxy.
    chol_vecs = np.zeros((nchol_max, nao * nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0, mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2 * l + 1) * nc
        dims.append(nao_per_i)
    # print (dims)
    for i in range(0, mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        buf = mol.intor("int2e_sph", shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag : ndiag + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
        ndiag += di * nao
    nu = np.argmax(diag)
    delta_max = diag[nu]
    if verbose:
        print("# Generating Cholesky decomposition of ERIs." % nchol_max)
        print("# max number of cholesky vectors = %d" % nchol_max)
        print("# iteration %5d: delta_max = %f" % (0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = np.searchsorted(dims, j)
    sl = np.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = np.zeros(nao * nao)
    # ERI[:,jl]
    eri_col = mol.intor(
        "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
    )
    cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
    chol_vecs[0] = np.copy(eri_col[:, :, cj, cl].reshape(nao * nao)) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = np.searchsorted(dims, j)
        sl = np.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor(
            "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
        )
        # Select correct ERI chunk from shell.
        cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
        Munu0 = eri_col[:, :, cj, cl].reshape(nao * nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = np.dot(chol_vecs[: nchol + 1, nu], chol_vecs[: nchol + 1, :])
        chol_vecs[nchol + 1] = (Munu0 - R) / (delta_max) ** 0.5
        nchol += 1
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print("# iteration %5d: delta_max = %13.8e: time = %13.8e" % info)

    if chol_len > nchol:
        raise ValueError(f"given cholesky vector length {chol_len} exceeded the \n"
                         f"decompostion {nchol} lower chol_len or higher the chol_cut")

    return chol_vecs[:chol_len]

def fix_len_generate_integrals(mol, hcore, X, chol_len, chol_cut=1e-6, verbose=False, DFbas=None):
    # Unpack SCF data.
    # Step 1. Rotate core Hamiltonian to orthogonal basis.
    if verbose:
        print(" # Transforming hcore and eri to ortho AO basis.")
    if len(X.shape) == 2:
        h1e = np.dot(X.T, np.dot(hcore, X))
    elif len(X.shape) == 3:
        h1e = np.dot(X[0].T, np.dot(hcore, X[0]))

    if DFbas is not None:
        chol_vecs = df.incore.cholesky_eri(mol, auxbasis=DFbas)
        chol_vecs = lib.unpack_tril(chol_vecs).reshape(chol_vecs.shape[0], -1)
    else:  # do cholesky
        # nbasis = h1e.shape[-1]
        # Step 2. Genrate Cholesky decomposed ERIs in non-orthogonal AO basis.
        if verbose:
            print(" # Performing modified Cholesky decomposition on ERI tensor.")
        chol_vecs = fix_len_chunked_cholesky(mol,chol_len,max_error=chol_cut,verbose=verbose)

    if verbose:
        print(" # Orthogonalising Cholesky vectors.")
    start = time.time()

    # Step 2.a Orthogonalise Cholesky vectors.
    if len(X.shape) == 2 and X.shape[0] != X.shape[1]:
        chol_vecs = ao2mo_chol_copy(chol_vecs, X)
    elif len(X.shape) == 2:
        ao2mo_chol(chol_vecs, X)
    elif len(X.shape) == 3:
        ao2mo_chol(chol_vecs, X[0])
    if verbose:
        print(" # Time to orthogonalise: %f" % (time.time() - start))
    enuc = mol.energy_nuc()
    # Step 3. (Optionally) freeze core / virtuals.
    nelec = mol.nelec
    return h1e, chol_vecs, nelec, enuc

def fix_len_chol_prep(
    mf_or_cc: Union[scf.uhf.UHF, scf.rhf.RHF, CCSD, UCCSD],
    chol_len,
    basis_coeff: Optional[np.ndarray] = None,
    norb_frozen: int = 0,
    chol_cut: float = 1e-7,
    mo_file = "mo_coeff.npz",
    chol_file = "FCIDUMP_chol"
):
    
    mf = mf_or_cc
    mol = mf.mol

    if basis_coeff is None:
        if isinstance(mf, scf.uhf.UHF):
            basis_coeff = mf.mo_coeff[0]
        else:
            basis_coeff = mf.mo_coeff
            
    print("# Calculating Cholesky integrals")

    h1e, chol, nelec, enuc, nbasis,_ = [None] * 6

    h1e, chol, nelec, enuc = \
        fix_len_generate_integrals(mol, mf.get_hcore(), basis_coeff, chol_len, chol_cut)
    nbasis = h1e.shape[-1]
    nelec = mol.nelec

    if norb_frozen > 0:
        assert norb_frozen * 2 < sum(
            nelec
        ), "Frozen orbitals exceed number of electrons"
        mc = mcscf.CASSCF(
            mf, mol.nao - norb_frozen, mol.nelectron - 2 * norb_frozen
        )
        nelec = mc.nelecas  # type: ignore
        mc.mo_coeff = basis_coeff  # type: ignore
        h1e, enuc = mc.get_h1eff()  # type: ignore
        chol = chol.reshape((-1, nbasis, nbasis))
        chol = chol[:, mc.ncore : mc.ncore + mc.ncas, mc.ncore : mc.ncore + mc.ncas]  # type: ignore

    print("# Finished calculating Cholesky integrals\n#")

    nbasis = h1e.shape[-1]
    print("# Size of the correlation space:")
    print(f"# Number of electrons: {nelec}")
    print(f"# Number of basis functions: {nbasis}")
    print(f"# Number of Cholesky vectors: {chol.shape[0]}\n#")
    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * np.einsum("nik,njk->ij", chol, chol, optimize="optimal")
    h1e_mod = h1e - v0
    chol = chol.reshape((chol.shape[0], -1))

    # write trial mo coefficients
    trial_coeffs = np.empty((2, nbasis, nbasis))
    overlap = mf.get_ovlp(mol)
    if isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)):
        uhfCoeffs = np.empty((nbasis, 2 * nbasis))
        if isinstance(mf, scf.uhf.UHF):
            q, r = np.linalg.qr(
                basis_coeff[:, norb_frozen:]
                .T.dot(overlap)
                .dot(mf.mo_coeff[0][:, norb_frozen:])
            )
            sgn = np.sign(r.diagonal())
            q = np.einsum("ij,j->ij", q, sgn)
            # q2 = basis_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[0][:, norb_frozen:])
            # print("max err a", np.max(abs(q-q2)))
            # q, _ = np.linalg.qr(
            #    basis_coeff[:, norb_frozen:]
            #    .T.dot(overlap)
            #    .dot(mf.mo_coeff[0][:, norb_frozen:])
            # )
            uhfCoeffs[:, :nbasis] = q
            q, r = np.linalg.qr(
                basis_coeff[:, norb_frozen:]
                .T.dot(overlap)
                .dot(mf.mo_coeff[1][:, norb_frozen:])
            )
            sgn = np.sign(r.diagonal())
            q = np.einsum("ij,j->ij", q, sgn)
            # q2 = basis_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[1][:, norb_frozen:])
            # print("max err b", np.max(abs(q-q2)))
            # import pdb
            # pdb.set_trace()
            # q, _ = np.linalg.qr(
            #     basis_coeff[:, norb_frozen:]
            #     .T.dot(overlap)
            #     .dot(mf.mo_coeff[1][:, norb_frozen:])
            # )
            uhfCoeffs[:, nbasis:] = q
        else:
            q, r = np.linalg.qr(
                basis_coeff[:, norb_frozen:]
                .T.dot(overlap)
                .dot(mf.mo_coeff[:, norb_frozen:])
            )
            sgn = np.sign(r.diagonal())
            q = np.einsum("ij,j->ij", q, sgn)
            uhfCoeffs[:, :nbasis] = q
            uhfCoeffs[:, nbasis:] = q

        trial_coeffs[0] = uhfCoeffs[:, :nbasis]
        trial_coeffs[1] = uhfCoeffs[:, nbasis:]
        # np.savetxt("uhf.txt", uhfCoeffs)
        np.savez(mo_file, mo_coeff=trial_coeffs)

    elif isinstance(mf, scf.rhf.RHF):
        q, _ = np.linalg.qr(
            basis_coeff[:, norb_frozen:]
            .T.dot(overlap)
            .dot(mf.mo_coeff[:, norb_frozen:])
        )
        trial_coeffs[0] = q
        trial_coeffs[1] = q
        np.savez(mo_file, mo_coeff=trial_coeffs)

    write_dqmc(
        h1e,
        h1e_mod,
        chol,
        sum(nelec),
        nbasis,
        enuc,
        ms=mol.spin,
        filename=chol_file,
        mo_coeffs=trial_coeffs,
    )

def pre_lno(options,mf,mo_coeff,lo_coeff,frozen=None,eris=None,chol_cut=1e-5,
            option_file='options.bin',mo_file='mo.npz',chol_file='FCIDUMP'):
    from functools import reduce
    from pyscf import ao2mo
    _fdot = np.dot
    fdot = lambda *args: reduce(_fdot, args)

    print("#\n# Preparing LNO-AFQMC calculation")
    mol = mf.mol
    # frozen = frzfrag2
    # choose the orbital basis
    # mo_coeff = orbfrag2
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
        
    orbfrzocc = mo_coeff[:,~maskact& maskocc]
    orbactocc = mo_coeff[:, maskact& maskocc]
    orbactvir = mo_coeff[:, maskact&~maskocc]
    orbfrzvir = mo_coeff[:,~maskact&~maskocc]
    _, nactocc, nactvir, _ = \
        [orb.shape[1] for orb in [orbfrzocc,orbactocc,orbactvir,orbfrzvir]]
    norb_act = (nactocc+nactvir)
    nelec_act = nactocc*2
    norb_frozen = frozen

    # lo_coeff = orbfragloc2
    s1e = mf.get_ovlp() if eris is None else eris.s1e
    prjlo = fdot(lo_coeff.T, s1e, orbactocc)
    options["prjlo"] = prjlo
    import pickle
    with open(option_file, 'wb') as f:
        pickle.dump(options, f)

    # calculate cholesky integrals
    print('# Generating Cholesky Integrals')
    nbasis = mf.mol.nao
    act_idx = [i for i in range(nbasis) if i not in norb_frozen]
    _, chol, _, _ = \
        pyscf_interface.generate_integrals(mol,mf.get_hcore(),mo_coeff[:,act_idx],chol_cut,DFbas=mf.with_df.auxmol.basis)
    # nbasis = h1e.shape[-1]
    # nelec = mol.nelec
    mc = mcscf.CASSCF(mf, norb_act, nelec_act) 
    mc.frozen = norb_frozen
    nelec = mc.nelecas
    mc.mo_coeff = mo_coeff
    h1e, enuc = mc.get_h1eff()

    # nbasis = mo_coeff.shape[-1]
    # act = [i for i in range(nbasis) if i not in norb_frozen]
    # e = ao2mo.kernel(mf.mol,mo_coeff[:,act],compact=False)
    # chol = pyscf_interface.modified_cholesky(e,max_error = chol_cut)

    print(f'# local active orbitals are {act_idx}') #yichi
    print(f'# local active space size {len(act_idx)}') #yichi
    # print(f'# loc_eris shape: {e.shape}') #yichi
    print(f'# chol shape: {chol.shape}') #yichi

    nbasis = h1e.shape[-1]
    print("# Finished calculating Cholesky integrals\n")
    print('# Size of the correlation space:')
    print(f'# Number of electrons: {nelec}')
    print(f'# Number of basis functions: {nbasis}')
    print(f'# Number of Cholesky vectors: {chol.shape[0]}\n')
    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * np.einsum('nik,njk->ij', chol, chol, optimize='optimal')
    h1e_mod = h1e - v0
    chol = chol.reshape((chol.shape[0], -1))

    # write mo coefficients
    trial_coeffs = np.empty((2, nbasis, nbasis))
    # overlap = mf.get_ovlp(mol)
    #q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[:, norb_frozen:]))
    q = np.eye(mol.nao- len(norb_frozen))
    trial_coeffs[0] = q
    trial_coeffs[1] = q
    np.savez(mo_file,mo_coeff=trial_coeffs)
    pyscf_interface.write_dqmc(h1e,h1e_mod,chol,sum(nelec),nbasis,enuc,ms=mol.spin,
                               filename=chol_file,mo_coeffs=trial_coeffs)
    
    return None


def mk_cs_frag(ifrag,mf1,mf2,frozen,options,
               chol_cut=1e-5,lno_thresh=1e-4,
               no_type='ie',lo_type="pm"):
    
    from ad_afqmc.lno.afqmc import LNOAFQMC
    mfcc1 = LNOAFQMC(mf1,thresh=lno_thresh,frozen=frozen)
    mfcc1.thresh_occ = lno_thresh*10
    mfcc1.thresh_vir = lno_thresh
    mfcc1.nwalk_per_proc = options["n_walkers"]
    mfcc1.chol_cut = chol_cut

    mfcc2 = LNOAFQMC(mf2,thresh=lno_thresh,frozen=frozen)
    mfcc2.thresh_occ = lno_thresh*10
    mfcc2.thresh_vir = lno_thresh
    mfcc2.nwalk_per_proc = options["n_walkers"]
    mfcc2.chol_cut = chol_cut

    eris1 = mfcc1.ao2mo()
    orbloc1 = mfcc1.get_lo(lo_type=lo_type)
    # frag_atmlist1=None
    frag_lolist1 = [[i] for i in range(orbloc1.shape[1])]
    frag_nonvlist1 = mfcc1.frag_nonvlist
    nfrag1 = len(frag_lolist1)
    if frag_nonvlist1 is None: frag_nonvlist1 = [[None,None]] * nfrag1

    eris2 = mfcc2.ao2mo()
    orbloc2 = mfcc2.get_lo(lo_type=lo_type)
    # frag_atmlist2=None
    frag_lolist2 = [[i] for i in range(orbloc2.shape[1])]
    frag_nonvlist2 = mfcc2.frag_nonvlist
    nfrag2 = len(frag_lolist2)
    if frag_nonvlist2 is None: frag_nonvlist2 = [[None,None]] * nfrag2

    if nfrag1 != nfrag2: 
        raise ValueError("number of fragments are different in two system!")

    from ad_afqmc.lno.base import lno
    # for ifrag in range(nfrag):
    frag_target_nocc1, frag_target_nvir1 = frag_nonvlist1[ifrag]
    fraglo1 = frag_lolist1[ifrag]
    # frag_res1 = [None] * nfrag1
    orbfragloc1 = orbloc1[:,fraglo1]
    frag_target_nocc2, frag_target_nvir2 = frag_nonvlist2[ifrag]
    fraglo2 = frag_lolist2[ifrag]
    # frag_res2 = [None] * nfrag2
    orbfragloc2 = orbloc2[:,fraglo2]

    # make fpno
    THRESH_INTERNAL = 1e-10
    thresh_pno1 = [mfcc1.thresh_occ, mfcc1.thresh_vir]
    frozen_mask1 = mfcc1.get_frozen_mask()
    thresh_pno2 = [mfcc2.thresh_occ, mfcc2.thresh_vir]
    frozen_mask2 = mfcc2.get_frozen_mask()
    
    frzfrag1, orbfrag1, can_orbfrag1 = lno.make_fpno1(mfcc1,eris1,orbfragloc1,no_type,
                                                      THRESH_INTERNAL,thresh_pno1,
                                                      frozen_mask=frozen_mask1,
                                                      frag_target_nocc=frag_target_nocc1,
                                                      frag_target_nvir=frag_target_nvir1,
                                                      canonicalize=False)
    
    frzfrag2, orbfrag2, can_orbfrag2 = lno.make_fpno1(mfcc2,eris2,orbfragloc2,no_type,
                                                      THRESH_INTERNAL, thresh_pno2,
                                                      frozen_mask=frozen_mask2,
                                                      frag_target_nocc=frag_target_nocc2,
                                                      frag_target_nvir=frag_target_nvir2,
                                                      canonicalize=False)
    #take the larger active space
    if len(frzfrag1) > len(frzfrag2):
        frzfrag = frzfrag2
    else:
        frzfrag = frzfrag1
    
    pre_lno(options,mf1,orbfrag1,orbfragloc1,frozen=frzfrag,eris=eris1,chol_cut=chol_cut,
            option_file='option1.bin',mo_file='mo1.npz',chol_file='chol1')
    pre_lno(options,mf2,orbfrag2,orbfragloc2,frozen=frzfrag,eris=eris2,chol_cut=chol_cut,
            option_file='option2.bin',mo_file='mo2.npz',chol_file='chol2')
        
    return None #can_orbfrag1,can_orbfrag2 for mp2

# sampler_eq = sampling.sampler(n_prop_steps=50, n_ene_blocks=5, n_sr_blocks=10)

def init_prop(ham_data, ham, prop, trial, wave_data, seed, MPI):
    comm = MPI.COMM_WORLD
    #size = comm.Get_size()
    rank = comm.Get_rank()
    #seed = options["seed"]
    #neql = options["n_eql"]
    init_walkers: Optional[Union[List, jax.Array]] = None
    trial_rdm1 = trial.get_rdm1(wave_data)
    if "rdm1" not in wave_data:
        wave_data["rdm1"] = trial_rdm1
    ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
    ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)
    prop_data = prop.init_prop_data(trial, wave_data, ham_data, init_walkers)
    prop_data["key"] = random.PRNGKey(seed + rank)
    prop_data["n_killed_walkers"] = 0

    comm.Barrier()
    if rank == 0:
        print(f"# initial energy: {prop_data['e_estimate']:.6f}")
    comm.Barrier()
    
    return prop_data, ham_data

# @partial(jit, static_argnums=(2,3))
def en_samples(prop_data,ham_data,prop,trial,wave_data):
    energy_samples = jnp.real(
        trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
    )
    energy_samples = jnp.where(
        jnp.abs(energy_samples - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
        prop_data["e_estimate"],
        energy_samples,
    )
    return energy_samples

# @partial(jit, static_argnums=(2))
def orb_en_samples(prop_data,ham_data,trial,wave_data):
    orb_en_samples = jnp.real(
        trial.calc_orbenergy(prop_data['walkers'],ham_data,wave_data,0)
        )
    # orbE_samples = jnp.where(jnp.abs(energy_samples - prop_data['pop_control_ene_shift']) 
    #                      > jnp.sqrt(2./prop.dt), prop_data['pop_control_ene_shift'],
    #                      orbE_samples)
    return orb_en_samples

# @partial(jit, static_argnums=(2,3))
def block_en_weight(prop_data,ham_data,prop,trial,wave_data):

    energy_samples = en_samples(prop_data,ham_data,prop,wave_data,trial)

    block_weight = jnp.sum(prop_data["weights"])
    block_energy = jnp.sum(energy_samples * prop_data["weights"]) / block_weight
    return block_energy, block_weight


@partial(jit, static_argnums=(3,4,6))
def field_block_scan(
        prop_data: dict,
        fields,
        ham_data: dict,
        prop: propagator,
        trial: wave_function,
        wave_data: dict,
        sampler: sampling.sampler,
        ) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
    """Block scan function for a given field"""
    # with open("options.pkl", "rb") as file:
    #     options = pickle.load(file)
    # if options["free_proj"]:
    #     # print("free projection propagation")
    #     _step_scan_wrapper = lambda x, y: sampler_eq._step_scan_free(
    #         x, y, ham_data, prop, trial, wave_data
    #     )
    #     prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
    #     # energy_samples = trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
    #     # # energy_samples = jnp.where(jnp.abs(energy_samples - ham_data['ene0']) > jnp.sqrt(2./propagator.dt), ham_data['ene0'],     energy_samples)
    #     # block_energy = jnp.sum(energy_samples * prop_data["overlaps"]) / jnp.sum(
    #     #     prop_data["overlaps"]
    #     # )
    #     # #block_weight = jnp.sum(prop_data["overlaps"])
    # else:
        # print("phaseless propagation")
    _step_scan_wrapper = lambda x, y: sampler._step_scan(
        x, y, ham_data, prop, trial, wave_data
    )
    prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
    prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
        prop_data["weights"]
    )
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

    block_energy,_ = block_en_weight(prop_data,ham_data,prop,wave_data,trial)
    prop_data["pop_control_ene_shift"] = (
        0.9 * prop_data["pop_control_ene_shift"] + 0.1 * block_energy
    )
    return prop_data

@partial(jit, static_argnums=(2))
def corr_otler_rm(prop_data1, prop_data2, z_thresh = 5):

    # problem for overlap as a measure
    # for large system which should be describe by many determinants
    # the overlaps with a single determinant trial generally
    # becomes very small

    #######################################################################
    # not efficient when running on cpus                                  #
    # this function is applied locally on each core                       #
    # this function is applied locally on each core                       #
    # a glbal outliner may not be a local outliner                        #
    # especially when having very few walkers on each core                #
    # which restrict me from defining this function in a more generalized #
    # way. i.e., z_score > z_thresh*std                                   #
    #######################################################################
    
    # olp1 = jnp.sqrt(prop_data1["overlaps"].real**2+prop_data1["overlaps"].imag**2)
    # olp2 = jnp.sqrt(prop_data2["overlaps"].real**2+prop_data2["overlaps"].imag**2)
    # olp = jnp.array(jnp.where(olp1 < olp2, olp1, olp2)) # take the smaller
    # olp_thresh = 0.02
    wt1 = jnp.array(prop_data1["weights"])
    wt2 = jnp.array(prop_data2["weights"])
    # wt = jnp.array(jnp.where(wt1 > wt2, wt1, wt2)) # take the larger
    wt1_mean = jnp.mean(wt1)
    wt1_std = jnp.std(wt1)
    wt1_z = jnp.abs(wt1-wt1_mean)/wt1_std
    wt2_mean = jnp.mean(wt2)
    wt2_std = jnp.std(wt2)
    wt2_z = jnp.abs(wt2-wt2_mean)/wt2_std
    wt_z = jnp.array(jnp.where(wt1_z > wt2_z, wt1_z, wt2_z)) # take the larger
    prop_data1["weights"] = jnp.array(jnp.where(wt_z > z_thresh, 0.0, wt1))
    prop_data2["weights"] = jnp.array(jnp.where(wt_z > z_thresh, 0.0, wt2))
    # prop_data1["weights"] = jnp.array(jnp.where(olp < olp_thresh, 0.0, prop_data1["weights"]))
    # prop_data2["weights"] = jnp.array(jnp.where(olp < olp_thresh, 0.0, prop_data2["weights"]))
    return prop_data1, prop_data2


@partial(jit, static_argnums=(2,3,7,8,10))
def cs_block_scan(
        prop_data1: dict,
        ham_data1: dict,
        prop1: propagator,
        trial1: wave_function,
        wave_data1: dict,
        prop_data2: dict,
        ham_data2: dict,
        prop2: propagator,
        trial2: wave_function,
        wave_data2: dict,
        sampler: sampling.sampler,
        ):
    '''correlated sampling of two blocks of walkers over the same field'''
    prop_data1["key"], subkey1 = random.split(prop_data1["key"])
    fields = random.normal(
        subkey1,
        shape=(
            sampler.n_prop_steps,
            prop1.n_walkers,
            ham_data1["chol"].shape[0],
        )
    )
    prop_data1 = field_block_scan(
        prop_data1,fields,ham_data1,prop1,trial1,wave_data1,sampler)
    prop_data2 = field_block_scan(
        prop_data2,fields,ham_data2,prop2,trial2,wave_data2,sampler)
    
    prop_data1, prop_data2 = corr_otler_rm(prop_data1, prop_data2)

    return prop_data1, prop_data2

# @partial(jit, static_argnums=(2,3,7,8,10))
# def ucs_block_scan(
#         prop_data1: dict,
#         ham_data1: dict,
#         prop1: propagator,
#         trial1: wave_function,
#         wave_data1: dict,
#         prop_data2: dict,
#         ham_data2: dict,
#         prop2: propagator,
#         trial2: wave_function,
#         wave_data2: dict,
#         sampler: sampling.sampler):
#     '''correlated sampling of two blocks over the same field'''
#     prop_data1["key"], subkey1 = random.split(prop_data1["key"])
#     fields1 = random.normal(
#         subkey1,
#         shape=(
#             sampler.n_prop_steps,
#             prop1.n_walkers,
#             ham_data1["chol"].shape[0],
#         )
#     )
#     prop_data1 = field_block_scan(prop_data1,fields1,ham_data1,prop1,trial1,wave_data1)

#     prop_data2["key"], subkey2 = random.split(prop_data2["key"])
#     fields2 = random.normal(
#         subkey2,
#         shape=(
#             sampler.n_prop_steps,
#             prop2.n_walkers,
#             ham_data2["chol"].shape[0],
#         )
#     )
#     prop_data2 = field_block_scan(prop_data2,fields2,ham_data2,prop2,trial2,wave_data2)

#     return prop_data1, prop_data2

@partial(jit, static_argnums=(0,3,4,8,9,11))
def cs_steps_scan(steps,
                  prop_data1,ham_data1,prop1,trial1,wave_data1,
                  prop_data2,ham_data2,prop2,trial2,wave_data2,
                  sampler,):

    cs_prop_data = (prop_data1,prop_data2)
    def cs_step(cs_prop_data,_):
        prop_data1,prop_data2= cs_prop_data
        prop_data1,prop_data2 = cs_block_scan(
            prop_data1,ham_data1,prop1,trial1,wave_data1,
            prop_data2,ham_data2,prop2,trial2,wave_data2,
            sampler)
        cs_prop_data = (prop_data1,prop_data2)
        loc_en_samples1 = en_samples(prop_data1,ham_data1,prop1,trial1,wave_data1)
        loc_en_samples2 = en_samples(prop_data2,ham_data2,prop2,trial2,wave_data2)
        loc_weight_sample1 = prop_data1["weights"]
        loc_weight1 = sum(loc_weight_sample1)
        loc_weight_sample2 = prop_data2["weights"]
        loc_weight2 = sum(loc_weight_sample2)
        loc_en_sample1 = loc_en_samples1*loc_weight_sample1
        loc_en_sample2 = loc_en_samples2*loc_weight_sample2
        loc_en1 = sum(loc_en_sample1) #not normalized
        loc_en2 = sum(loc_en_sample2) #not normalized
        return cs_prop_data, (loc_en1,loc_weight1,loc_en2,loc_weight2)

    cs_prop_data, (loc_en1,loc_weight1,loc_en2,loc_weight2) \
        = jax.lax.scan(cs_step,cs_prop_data,xs=None,length=steps)
    return cs_prop_data, (loc_en1,loc_weight1,loc_en2,loc_weight2)

@partial(jit, static_argnums=(0,3,4,8,9,11))
def lno_cs_steps_scan(steps,
                      prop_data1,ham_data1,prop1,trial1,wave_data1,
                      prop_data2,ham_data2,prop2,trial2,wave_data2,
                      sampler):

    cs_prop_data = (prop_data1,prop_data2)
    def lno_cs_step(cs_prop_data,_):
        prop_data1,prop_data2= cs_prop_data
        prop_data1,prop_data2 = cs_block_scan(
            prop_data1,ham_data1,prop1,trial1,wave_data1,
            prop_data2,ham_data2,prop2,trial2,wave_data2,
            sampler)
        cs_prop_data = (prop_data1,prop_data2)
        loc_en_sp1 = en_samples(prop_data1,ham_data1,prop1,trial1,wave_data1)
        loc_en_sp2 = en_samples(prop_data2,ham_data2,prop2,trial2,wave_data2)
        loc_orb_en_sp1 = orb_en_samples(prop_data1,ham_data1,trial1,wave_data1)
        loc_orb_en_sp2 = orb_en_samples(prop_data2,ham_data2,trial2,wave_data2)
        loc_wt_sp1 = prop_data1["weights"]
        loc_wt1 = sum(loc_wt_sp1)
        loc_wt_sp2 = prop_data2["weights"]
        loc_wt2 = sum(loc_wt_sp2)
        loc_en_sp1 = loc_en_sp1*loc_wt_sp1
        loc_en_sp2 = loc_en_sp2*loc_wt_sp2
        loc_orb_en_sp1 = loc_orb_en_sp1*loc_wt_sp1
        loc_orb_en_sp2 = loc_orb_en_sp2*loc_wt_sp2
        loc_en1 = sum(loc_en_sp1) #not normalized
        loc_en2 = sum(loc_en_sp2) #not normalized
        loc_orb_en1 = sum(loc_orb_en_sp1) #not normalized
        loc_orb_en2 = sum(loc_orb_en_sp2) #not normalized
        return cs_prop_data, (loc_en1,loc_orb_en1,loc_wt1,loc_en2,loc_orb_en2,loc_wt2)

    cs_prop_data, (loc_en1,loc_orb_en1,loc_wt1,loc_en2,loc_orb_en2,loc_wt2) \
        = jax.lax.scan(lno_cs_step,cs_prop_data,xs=None,length=steps)
    return cs_prop_data, (loc_en1,loc_orb_en1,loc_wt1,loc_en2,loc_orb_en2,loc_wt2)


# @partial(jit, static_argnums=(0,3,4,8,9))
# def ucs_steps_scan(steps,
#                   prop_data1,ham_data1,prop1,trial1,wave_data1,
#                   prop_data2,ham_data2,prop2,trial2,wave_data2
#                   ):

#     ucs_prop_data = (prop_data1,prop_data2)
#     def ucs_step(ucs_prop_data,_):
#         prop_data1,prop_data2= ucs_prop_data
#         prop_data1,prop_data2 = ucs_block_scan(prop_data1,ham_data1,prop1,trial1,wave_data1,
#                                                prop_data2,ham_data2,prop2,trial2,wave_data2)
#         ucs_prop_data = (prop_data1,prop_data2)
#         loc_en_samples1 = en_samples(prop_data1,ham_data1,prop1,trial1,wave_data1)
#         loc_en_samples2 = en_samples(prop_data2,ham_data2,prop2,trial2,wave_data2)
#         loc_weight_sample1 = prop_data1["weights"]
#         loc_weight1 = jnp.sum(loc_weight_sample1)
#         loc_weight_sample2 = prop_data2["weights"]
#         loc_weight2 = jnp.sum(loc_weight_sample2)
#         loc_en_sample1 = loc_en_samples1*loc_weight_sample1
#         loc_en_sample2 = loc_en_samples2*loc_weight_sample2
#         loc_en1 = sum(loc_en_sample1) #not normalized
#         loc_en2 = sum(loc_en_sample2) #not normalized
#         return ucs_prop_data, (loc_en1,loc_weight1,loc_en2,loc_weight2)

#     ucs_prop_data, (loc_en1,loc_weight1,loc_en2,loc_weight2) \
#         = jax.lax.scan(ucs_step,ucs_prop_data,xs=None,length=steps)
#     return ucs_prop_data, (loc_en1,loc_weight1,loc_en2,loc_weight2)

@partial(jit, static_argnums=(1,4,5,9,10,12,13))
def scan_seeds(seeds,eq_steps,
               prop_data1_init,ham_data1_init,prop1,trial1,wave_data1,
               prop_data2_init,ham_data2_init,prop2,trial2,wave_data2, 
               sampler,MPI):
    '''
    do a number of independent runs of given equilirium steps
    for a given array of seeds
    return local energy of system1, local weight of system1
    and the same for system2.
    the ensemble energy average for each system should be 
    loc_en/loc_weight
    '''

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    def _seed_cs(carry,seed):
        prop_data1_init["key"] = jax.random.PRNGKey(seed + rank)
        _,(loc_en1,loc_weight1,loc_en2,loc_weight2) \
            = cs_steps_scan(eq_steps,
                            prop_data1_init,ham_data1_init,prop1,trial1,wave_data1,
                            prop_data2_init,ham_data2_init,prop2,trial2,wave_data2,
                            sampler)
        
        return carry, (loc_en1,loc_weight1,loc_en2,loc_weight2)
    
    _, (loc_en1,loc_weight1,loc_en2,loc_weight2) = jax.lax.scan(_seed_cs, None, seeds)

    return loc_en1,loc_weight1,loc_en2,loc_weight2

@partial(jit, static_argnums=(1,4,5,9,10,12,13))
def lno_cs_seeds_scan(seeds,eq_steps,
                      prop_data1_init,ham_data1_init,prop1,trial1,wave_data1,
                      prop_data2_init,ham_data2_init,prop2,trial2,wave_data2,
                      sampler,MPI):
    '''
    do a number of independent runs of given equilirium steps
    for a given array of seeds
    return local energy of system1, local weight of system1
    and the same for system2.
    the ensemble energy average for each system should be 
    loc_en/loc_weight
    '''

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    def lno_cs_seed(carry,seed):
        prop_data1_init["key"] = jax.random.PRNGKey(seed + rank)
        _,(loc_en1,loc_orb_en1,loc_wt1,loc_en2,loc_orb_en2,loc_wt2) \
            = lno_cs_steps_scan(eq_steps,
                                prop_data1_init,ham_data1_init,prop1,trial1,wave_data1,
                                prop_data2_init,ham_data2_init,prop2,trial2,wave_data2,
                                sampler)
        
        return carry, (loc_en1,loc_orb_en1,loc_wt1,loc_en2,loc_orb_en2,loc_wt2)
    
    _, (loc_en1,loc_orb_en1,loc_wt1,loc_en2,loc_orb_en2,loc_wt2) \
        = jax.lax.scan(lno_cs_seed, None, seeds)

    return loc_en1,loc_orb_en1,loc_wt1,loc_en2,loc_orb_en2,loc_wt2

# def ucs_scan_seeds(seeds,eq_steps,
#                   prop_data1_init,ham_data1_init,prop1,trial1,wave_data1,
#                   prop_data2_init,ham_data2_init,prop2,trial2,wave_data2, 
#                   MPI):
#     '''
#     do a number of independent runs of given equilirium steps
#     for a given array of seeds
#     return local energy of system1, local weight of system1
#     and the same for system2.
#     the ensemble energy average for each system should be 
#     loc_en/loc_weight
#     '''

#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()

#     def _seed_ucs(carry,seed):
#         (seed1,seed2) = seed
#         prop_data1_init["key"] = jax.random.PRNGKey(seed1 + rank)
#         prop_data2_init["key"] = jax.random.PRNGKey(seed2 + rank)
#         _,(loc_en1,loc_weight1,loc_en2,loc_weight2) \
#             = ucs_steps_scan(eq_steps,
#                             prop_data1_init,ham_data1_init,prop1,trial1,wave_data1,
#                             prop_data2_init,ham_data2_init,prop2,trial2,wave_data2)
        
#         return carry, (loc_en1,loc_weight1,loc_en2,loc_weight2)
    
#     _, (loc_en1,loc_weight1,loc_en2,loc_weight2) = jax.lax.scan(_seed_ucs, None, seeds)

#     return loc_en1,loc_weight1,loc_en2,loc_weight2

def run_cs_afqmc(options=None,files=None,script=None,mpi_prefix=None, nproc=None):
    if options is None:
        options = {}
    if files is None:
        raise ValueError("files for correlated sampling not found!")
    with open("options.bin", "wb") as f:
        pickle.dump(options, f)
    with open("files.bin", "wb") as f:
        pickle.dump(files, f)
    if script is None:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        script = f"{dir_path}/run_cs.py"
    use_gpu = options["use_gpu"]
    if use_gpu:
        # config.afqmc_config["use_gpu"] = True
        gpu_flag = "--use_gpu"
    else: gpu_flag = ""
    if mpi_prefix is None:
        if use_gpu:
            mpi_prefix = ""
        else:
            mpi_prefix = "mpirun "
    if nproc is not None:
        mpi_prefix += f"-np {nproc} "
    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; {mpi_prefix} python {script} {gpu_flag} |tee cs_afqmc.out"
    )
    # try:
    #     ene_err = np.loadtxt("ene_err.txt")
    # except:
    #     print("AFQMC did not execute correctly.")
    #     ene_err = 0.0, 0.0
    return None


def data_analyze(rlx_data,prop_data,rhf_en=None,ccsd_en=None,ccsd_t_en=None,fci_en=None):
    
    from matplotlib import pyplot as plt

    rlx_en_diff = []
    rlx_en1 = []
    rlx_en2 = []
    lines = rlx_data.splitlines()
    for line in lines:
        if not line.startswith("#"): 
            columns = line.split()
            if len(columns) > 1:
                rlx_en1.append(columns[1])
            if len(columns) > 2:
                rlx_en2.append(columns[2])
            if len(columns) > 3:
                rlx_en_diff.append(columns[3])
                
    rlx_en1 = np.array(rlx_en1,dtype='float32')
    rlx_en2 = np.array(rlx_en2,dtype='float32')
    rlx_en_diff = np.array(rlx_en_diff,dtype='float32')

    prop_en1 = []
    prop_en1_err = []
    prop_en2 = []
    prop_en2_err = []
    prop_en_diff = []
    prop_en_diff_err = []
    lines = prop_data.splitlines()
    for line in lines:
        if not line.startswith("#"): 
            columns = line.split()
            if len(columns) > 1:
                prop_en1.append(columns[1])
            if len(columns) > 2:
                prop_en1_err.append(columns[2])
            if len(columns) > 3:
                prop_en2.append(columns[3])
            if len(columns) > 4:
                prop_en2_err.append(columns[4])
            if len(columns) > 5:
                prop_en_diff.append(columns[5])
            if len(columns) > 6:
                prop_en_diff_err.append(columns[6])

    prop_en1 = np.array(prop_en1,dtype='float32')
    prop_en1_err = np.array(prop_en1_err,dtype='float32')
    prop_en2 = np.array(prop_en2,dtype='float32')
    prop_en2_err = np.array(prop_en2_err,dtype='float32')
    prop_en_diff = np.array(prop_en_diff,dtype='float32')
    prop_en_diff_err = np.array(prop_en_diff_err,dtype='float32')

    rlx_steps = np.arange(len(rlx_en_diff))
    prop_steps = np.arange(len(rlx_en_diff),len(rlx_en_diff)+len(prop_en_diff))
    x_steps = np.linspace(0,max(prop_steps),100)
    
    if rhf_en is not None:
        rhf_en = [rhf_en]*100
        plt.plot(x_steps,rhf_en,label='rhf')
    if ccsd_en is not None:
        ccsd_en = [ccsd_en]*100
        plt.plot(x_steps,ccsd_en,label='ccsd')
    if ccsd_t_en is not None:
        ccsd_t_en = [ccsd_t_en]*100
        plt.plot(x_steps,ccsd_t_en,label='ccsd(t)')
    if fci_en is not None:
        fci_en = [fci_en]*100
        plt.plot(x_steps,fci_en,label='fci')
        
    plt.plot(rlx_steps,rlx_en_diff,'o',label='cs_relaxation')
    plt.errorbar(prop_steps,prop_en_diff,yerr=prop_en_diff_err, fmt='o', capsize=5,label='cs_prop')
    plt.xlabel('steps')
    plt.ylabel('energy difference')
    plt.title('correlated walkers energy differences')
    plt.legend()
    plt.show()

    plt.plot(rlx_steps,rlx_en1,'o',color='blue')
    plt.plot(rlx_steps,rlx_en2,'o',color='orange')
    plt.errorbar(prop_steps,prop_en1,yerr=prop_en1_err, fmt='o', color='blue', capsize=5,label='system 1')
    plt.errorbar(prop_steps,prop_en2,yerr=prop_en2_err, fmt='o', color='orange', capsize=5,label='system 2')
    plt.xlabel('steps')
    plt.ylabel('energy')
    plt.title('system1 and system2')
    plt.legend()
    plt.show()

    return None