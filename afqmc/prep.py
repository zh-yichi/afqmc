import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import pickle
# from typing import Optional, Union
import h5py
import numpy as np
import jax.numpy as jnp

# import opt_einsum as oe

# from pyscf import mcscf, scf
# from pyscf.cc.ccsd import CCSD
# from pyscf.cc.uccsd import UCCSD

from afqmc import hamiltonian, cholesky
from afqmc import propagation, sampling, fp_sampling
from afqmc.wavefunctions import wavefunctions_restricted
from afqmc.wavefunctions import wavefunctions_unrestricted
from functools import partial
print = partial(print, flush=True)

# def prep_afqmc(
#     mf_or_cc: Union[scf.rhf.RHF, scf.uhf.UHF, CCSD, UCCSD],
#     basis_coeff: Optional[np.ndarray] = None,
#     norb_frozen: int = 0,
#     chol_cut: float = 1e-5,
#     amp_file = "amplitudes.npz",
#     chol_file = "FCIDUMP_chol"
# ):

#     print("Preparing AFQMC calculation")

#     if isinstance(mf_or_cc, (CCSD, UCCSD)):
#         mf = mf_or_cc._scf
#         cc = mf_or_cc
#         if cc.frozen is not None:
#             norb_frozen = cc.frozen
#         if isinstance(cc, UCCSD):
#             # spin_type = 'unrestricted'
#             t1a = np.array(cc.t1[0])
#             t1b = np.array(cc.t1[1])
#             t2aa, t2ab, t2bb = cc.t2
#             t2aa = (t2aa - t2aa.transpose(0, 1, 3, 2)) / 2
#             t2bb = (t2bb - t2bb.transpose(0, 1, 3, 2)) / 2
#             t2aa = t2aa.transpose(0, 2, 1, 3)
#             t2bb = t2bb.transpose(0, 2, 1, 3)
#             t2ab = t2ab.transpose(0, 2, 1, 3)
#             np.savez(
#                 amp_file,
#                 t1a=t1a,
#                 t1b=t1b,
#                 t2aa=t2aa,
#                 t2ab=t2ab,
#                 t2bb=t2bb,
#             )
#         elif isinstance(cc, CCSD):
#             # spin_type = 'restricted'
#             t2 = cc.t2
#             t2 = t2.transpose(0, 2, 1, 3)
#             t1 = np.array(cc.t1)
#             np.savez(amp_file, t1=t1, t2=t2)
#     else:
#         mf = mf_or_cc

#     if isinstance(mf, scf.rhf.RHF):
#         spin_type = 'restricted'
#     elif isinstance(mf, scf.uhf.UHF):
#         spin_type = 'unrestricted'

#     mol = mf.mol
#     nao = mf.mol.nao

#     if basis_coeff is None:
#         basis_coeff = mf.mo_coeff

#     print("Calculating Cholesky integrals")
    
#     if getattr(mf, "with_df", None) is not None:
#         print('Find Density Fit Teonsers in MF object')
#         print('Integrals will be built by DF Tensors')
#         useDF = True
#     else:
#         useDF = False

#     if spin_type == 'restricted':

#         # mc = mcscf.CASSCF(
#         #     mf, nao - norb_frozen, mol.nelectron - 2 * norb_frozen
#         # )
#         # nelec = mc.nelecas
#         # mc.mo_coeff = basis_coeff
#         # h1e, enuc = mc.get_h1eff()
#         # _, chol, _, _ = \
#         #     pyscf_interface.generate_integrals(
#         #     mol, mf.get_hcore(), basis_coeff, chol_cut, DFbas=DFbas)

#         nbasis = nao - norb_frozen
#         nocc = int(np.count_nonzero(mf.mo_occ))
#         nelec = [nocc - norb_frozen, nocc - norb_frozen]
#         h1e, enuc = integral.h1e_ras(mf, basis_coeff, nbasis, norb_frozen, useDF)
#         chol_ao = cholesky.cholesky_by_mol(mol, max_error=chol_cut, cmax=10)
#         chol_ao = jnp.array(chol_ao.reshape((-1, nao, nao)))
#         chol = cholesky.cderi2mo_gpu(chol_ao, basis_coeff)
#         chol = cholesky.unpack_symmetric(chol, nao)
#         chol = chol[:, norb_frozen:, norb_frozen:]
#         # print("Finished calculating Cholesky integrals")
#         # print("Size of the correlation space:")
#         # print(f"Number of electrons: {nelec}")
#         # print(f"Number of basis functions: {nbasis}")
#         # print(f"Number of Cholesky vectors: {chol.shape[0]}")
#         v0 = 0.5 * oe.contract("gpr,gqr->pq", chol, chol, backend="jax")
#         h1e_mod = h1e - v0
#         chol = chol.reshape((chol.shape[0], -1))
            
#     elif spin_type == 'unrestricted':
#         # mc = mcscf.UCASSCF(
#         #     mf, nao - norb_frozen,
#         #     mol.nelectron - 2 * norb_frozen)
#         # nelec = mc.nelecas
#         # mc.mo_coeff = mf.mo_coeff
#         # h1e, enuc = mc.get_h1eff()
#         # nbasis = mc.ncas

#         # _, chol_a, _, _ = pyscf_interface.generate_integrals(
#         #     mol, mf.get_hcore(), mf.mo_coeff[0], chol_cut, DFbas=DFbas
#         # )

#         ncore = np.array([norb_frozen, norb_frozen], dtype = np.int32)
#         nocc = np.array([np.count_nonzero(mf.mo_occ[0]),
#                          np.count_nonzero(mf.mo_occ[1])],
#                          dtype = np.int32)
#         nelec = nocc - norb_frozen
#         ncas = nao - ncore
#         nbasis = ncas[0]
#         h1e, enuc = integral.h1e_uas(mf, basis_coeff, ncas, ncore, useDF)

#         chol_ao = cholesky.cholesky_by_mol(mol, max_error=chol_cut, cmax=10)
#         chol_ao = jnp.array(chol_ao.reshape((-1, nao, nao)))
#         chol_a = cholesky.cderi2mo_gpu(chol_ao, basis_coeff[0])
#         chol_b = cholesky.cderi2mo_gpu(chol_ao, basis_coeff[1])
#         chol_a = cholesky.unpack_symmetric(chol_a, nao)
#         chol_b = cholesky.unpack_symmetric(chol_b, nao)
#         print(f"Alpha Cholesky shape: {chol_a.shape} ")
#         print(f" Beta Cholesky shape: {chol_b.shape} ")

#         # nao = mf.mol.nao
#         # chol_a = chol_a.reshape((-1, nao, nao))
#         # s1e = mf.get_ovlp()
#         # a2b = mf.mo_coeff[1].T @ s1e @ mf.mo_coeff[0]
#         # chol_b = jnp.einsum('pr,grs,sq->gpq',a2b,chol_a,a2b.T)
#         # chol_b = chol_b.reshape((-1, nao, nao))
        
#         # froze orbitals
#         chol_a = chol_a[:, ncore[0]:, ncore[0]:]
#         chol_b = chol_b[:, ncore[1]:, ncore[1]:]
#         v0_a = 0.5 * oe.contract("gpr,gqr->pq", chol_a, chol_a, backend="jax")
#         v0_b = 0.5 * oe.contract("gpr,gqr->pq", chol_b, chol_b, backend="jax")
#         h1e = jnp.array(h1e)
#         h1e_mod = jnp.array(h1e - jnp.array([v0_a,v0_b]))
#         chol = jnp.array([chol_a.reshape(chol_a.shape[0], -1), chol_b.reshape(chol_b.shape[0], -1)])

#     print("Finished calculating Cholesky integrals")
#     print("Size of the correlation space:")
#     print(f"Number of electrons: {nelec}")
#     print(f"Number of basis functions: {nbasis}")
#     print(f"Number of Cholesky vectors: {chol.shape[-2]} {chol.shape}")

#     write_integral(
#         enuc,
#         h1e,
#         h1e_mod,
#         chol,
#         sum(nelec),
#         nbasis,
#         ms=mol.spin,
#         filename=chol_file,
#     )

# def write_integral(enuc, hcore, hcore_mod, chol,
#                    nelec, nmo, ms, 
#                    filename="FCIDUMP_chol",):
    
#     with h5py.File(filename, "w") as fh5:
#         fh5["header"] = np.array([nelec, nmo, ms, chol.shape[-1]])
#         fh5["hcore"] = hcore.flatten()
#         fh5["hcore_mod"] = hcore_mod.flatten()
#         fh5["chol"] = chol.flatten()
#         fh5["energy_core"] = enuc

# print = partial(print, flush=True)

def _prep_afqmc(options=None,
                option_file="options.bin",
                amp_file="amplitudes.npz",
                chol_file="FCIDUMP_chol"):
    
    if options is None:
        try:
            with open(option_file, "rb") as f:
                options = pickle.load(f)
        except:
            options = {}

    options["dt"] = options.get("dt", 0.01)
    options["n_exp_terms"] = options.get("n_exp_terms",6)
    options["n_walkers"] = options.get("n_walkers", 50)
    options["n_prop_steps"] = options.get("n_prop_steps", 50)
    options["n_blocks"] = options.get("n_blocks", 500)
    options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))
    options["n_eql"] = options.get("n_eql", 1)
    options["walker_type"] = options.get("walker_type", "rhf")
    options["save_walkers"] = options.get("save_walkers", False)
    options["trial"] = options.get("trial", None)
    options["free_projection"] = options.get("free_projection", False)
    options["n_batch"] = options.get("n_batch", 1)
    options["max_error"] = options.get("max_error", 1e-3)
    options["nchol_chunk"] = options.get("nchol_chunk", 100)
    options["max_memory"] = options.get("max_memory", 2000) # MB
    options["mix_precision"] = options.get("mix_precision", True)

    print("Load system from Integral File")

    with h5py.File(chol_file, "r") as fh5:
        [nelec, norb, ms] = fh5["header"]
        spin_type = fh5["spin_type"][()]
        h0 = jnp.array(fh5.get("energy_core"))
        h1 = jnp.array(fh5.get("hcore"))
        chol = jnp.array(fh5.get("chol"))
        h1_mod = jnp.array(fh5.get("hcore_mod"))
    
    if isinstance(spin_type, bytes):
        spin_type = spin_type.decode()

    assert spin_type in ["restricted", "unrestricted"]

    print(f"AFQMC Object Spin type: {spin_type}")

    if spin_type == 'restricted':
        h1 = jnp.array(h1).reshape(norb, norb)
        h1_mod = jnp.array(h1_mod).reshape(norb, norb)
        chol = jnp.array(chol).reshape(-1, norb, norb)

    elif spin_type == 'unrestricted':
        h1 = jnp.array(h1).reshape(2, norb, norb)
        h1_mod = jnp.array(h1_mod).reshape(2, norb, norb)
        chol = jnp.array(chol).reshape(2, -1, norb, norb)

    assert type(ms) is np.int64
    assert type(nelec) is np.int64
    assert type(norb) is np.int64

    ms, nelec, norb = int(ms), int(nelec), int(norb)
    nelec_sp = ((nelec + abs(ms)) // 2, (nelec - abs(ms)) // 2)

    ham = hamiltonian.hamiltonian(norb)
    ham_data = {}
    ham_data["h0"] = h0

    if spin_type == 'restricted':
        ham_data["h1"] = jnp.array([h1, h1])
        ham_data["h1_mod"] = jnp.array(h1_mod)
        nchol = chol.shape[0]
        ham_data["chol"] = jnp.array(chol.reshape(chol.shape[0], -1))
    elif spin_type == 'unrestricted':
        ham_data["h1"] = jnp.array(h1)
        ham_data["h1_mod"] = jnp.array(h1_mod)
        nchol = chol[0].shape[0]
        ham_data["chol"] = jnp.array([chol[0].reshape(chol[0].shape[0], -1),
                                      chol[1].reshape(chol[1].shape[0], -1)])

    # options["nchol_chunk"] = min(options["nchol_chunk"], nchol)
    options["nchol_chunk"] = cholesky.chunk_chol(chol, options["nchol_chunk"], 
                                                 options["max_memory"]/options["n_walkers"])

    wave_data = {}
    mo_coeff = jnp.array([np.eye(norb),np.eye(norb)])

    if spin_type == "restricted":
        if options["trial"] == "rhf":
            trial = wavefunctions_restricted.rhf(norb, nelec_sp, 
                                                 n_batch=options["n_batch"],
                                                 nchol_chunk=options["nchol_chunk"],
                                                 )
            wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0]]

        elif "cisd" in options["trial"]:
            try:
                amplitudes = np.load(amp_file)
                t1 = jnp.array(amplitudes["t1"])
                t2 = jnp.array(amplitudes["t2"])
                ci2 = t2 + jnp.einsum("ia,jb->iajb", t1, t1)
                trial_wave_data = {"ci1": t1, "ci2": ci2, 
                                "mo_coeff": mo_coeff[0][:, : nelec_sp[0]]}
                wave_data.update(trial_wave_data)
                trial = wavefunctions_restricted.cisd(norb, nelec_sp, 
                                                      n_batch=options["n_batch"]
                                                      )
                if "/" in options["trial"]:
                    guide_wave = wavefunctions_restricted.cisd(norb, nelec_sp, n_batch=options["n_batch"])
                    trial_wave = wavefunctions_restricted.rhf(norb, nelec_sp, n_batch=options["n_batch"])
                    trial = wavefunctions_restricted.mixed(guide_wave, trial_wave)
            except:
                raise ValueError("Trial specified as cisd, but amplitudes.npz not found.")

        elif options["trial"] == "cid":
            try:
                amplitudes = np.load(amp_file)
                t2 = jnp.array(amplitudes["t2"])
                trial_wave_data = {"ci2": t2, "mo_coeff": mo_coeff[0][:, : nelec_sp[0]]}
                wave_data.update(trial_wave_data)
                trial = wavefunctions_restricted.cid(norb, nelec_sp, n_batch=options["n_batch"])
            except:
                raise ValueError("Trial specified as cisd, but amplitudes.npz not found.")
            
        elif options["trial"] == "ptccsd":
            amplitudes = np.load(amp_file)
            t1 = jnp.array(amplitudes["t1"])
            t2 = jnp.array(amplitudes["t2"])
            trial_wave_data = {"t1": t1, "t2": t2}
            wave_data.update(trial_wave_data)
            wave_data["mo_coeff"] = mo_coeff[0][:,:nelec_sp[0]]
            trial = wavefunctions_restricted.ptccsd(norb, nelec_sp, n_batch=options["n_batch"])
            if "ad" in options["trial"]:
                trial = wavefunctions_restricted.ptccsd_ad(norb, nelec_sp, n_batch=options["n_batch"])
        
        elif options["trial"] == "ptccd":
            amplitudes = np.load(amp_file)
            t2 = jnp.array(amplitudes["t2"])
            trial_wave_data = {"t2": t2}
            wave_data.update(trial_wave_data)
            wave_data["mo_coeff"] = mo_coeff[0][:,:nelec_sp[0]]
            trial = wavefunctions_restricted.ptccd(norb, nelec_sp, n_batch=options["n_batch"])

        elif options["trial"] == "pt2ccsd":
            trial = wavefunctions_restricted.pt2ccsd(norb, nelec_sp, 
                                                     n_batch=options["n_batch"],
                                                     nchol_chunk=options["nchol_chunk"], 
                                                     mix_precision=options["mix_precision"],
                                                     )
            nocc = nelec_sp[0]
            amplitudes = np.load(amp_file)
            t1 = jnp.array(amplitudes["t1"])
            t2 = jnp.array(amplitudes["t2"])
            trial_wave_data = {"t1": t1, "t2": t2}
            wave_data.update(trial_wave_data)
            mo_t = trial.thouless_trans(t1)[:,:nocc]
            wave_data['mo_t'] = mo_t
            wave_data["mo_coeff"] = mo_coeff[0][:,:nelec_sp[0]]
            if "ad" in options["trial"]:
                trial = wavefunctions_restricted.pt2ccsd_ad(norb, nelec_sp, 
                                                            n_batch=options["n_batch"])
                rot_t2 = jnp.einsum('il,jk,lakb->iajb',
                                mo_t[:nocc,:nocc].T,mo_t[:nocc,:nocc].T,t2)
                wave_data['rot_t2'] = rot_t2

        elif "stoccsd" in options["trial"]:
            if "2" in options["trial"]:
                trial = wavefunctions_restricted.stoccsd2(
                    norb,
                    nelec_sp,
                    n_batch = options["n_batch"],
                    nslater = options['nslater']
                    )
                    
                sampler = sampling.sampler_stoccsd2(
                    n_prop_steps = options["n_prop_steps"],
                    n_blocks = options["n_blocks"],
                    n_chol = nchol,
                    )
            else:
                trial = wavefunctions_restricted.stoccsd(
                    norb,
                    nelec_sp,
                    n_batch = options["n_batch"],
                    nslater = options['nslater']
                    )
                    
                sampler = sampling.sampler_stoccsd(
                    n_prop_steps = options["n_prop_steps"],
                    n_blocks = options["n_blocks"],
                    n_chol = nchol,
                    )
            
            nocc = nelec_sp[0]
            amplitudes = np.load(amp_file)
            t1 = jnp.array(amplitudes["t1"])
            t2 = jnp.array(amplitudes["t2"])
            trial_wave_data = {"t1": t1, "t2": t2}
            wave_data.update(trial_wave_data)
            init_sd = jnp.eye(norb)[:,:nocc]
            mo_t = trial._thouless(init_sd, t1)
            wave_data['mo_t'] = mo_t
            wave_data['tau'] = trial.decompose_t2(t2)
            wave_data["mo_coeff"] = mo_coeff[0][:,:nocc]
    
    elif spin_type == "unrestricted":
        if options["trial"] == "uhf":
            trial = wavefunctions_unrestricted.uhf(norb, nelec_sp, 
                                                   n_batch=options["n_batch"])
            wave_data["mo_coeff"] = [
                mo_coeff[0][:, : nelec_sp[0]],
                mo_coeff[1][:, : nelec_sp[1]],
            ]

        elif options["trial"] == "ucisd":
            trial = wavefunctions_unrestricted.ucisd(
                    norb, nelec_sp, n_batch=options["n_batch"])
            nocc_a, nocc_b = trial.nelec[0], trial.nelec[1]
            try:
                amplitudes = np.load(amp_file)
                t1a = jnp.array(amplitudes["t1a"])
                t1b = jnp.array(amplitudes["t1b"])
                t2aa = jnp.array(amplitudes["t2aa"])
                t2ab = jnp.array(amplitudes["t2ab"])
                t2bb = jnp.array(amplitudes["t2bb"])
                ci2aa = t2aa + 2 * jnp.einsum("ia,jb->iajb", t1a, t1a)
                ci2ab = t2ab + jnp.einsum("ia,jb->iajb", t1a, t1b)
                ci2bb = t2bb + 2 * jnp.einsum("ia,jb->iajb", t1b, t1b)
                ci2aa = (ci2aa - ci2aa.transpose(0, 3, 2, 1)) / 2
                ci2bb = (ci2bb - ci2bb.transpose(0, 3, 2, 1)) / 2
                trial_wave_data = {
                    "ci1A": t1a,
                    "ci1B": t1b,
                    "ci2AA": ci2aa,
                    "ci2AB": ci2ab,
                    "ci2BB": ci2bb,
                    "mo_coeff": mo_coeff,
                }
                wave_data.update(trial_wave_data)
                mo = [mo_coeff[0][:,:nocc_a], mo_coeff[1][:,:nocc_b]]
                mo_t = trial._thouless(mo, [t1a, t1b])
                wave_data['mo_ta'] = mo_t[0]
                wave_data['mo_tb'] = mo_t[1]
                wave_data['tau'] = trial.decompose_t2([t2aa,t2ab,t2bb])
            except:
                raise ValueError("Trial specified as ucisd, but amplitudes.npz not found.")

        elif options["trial"] == "uptccsd":
            trial = wavefunctions_unrestricted.uptccsd(norb, nelec_sp, n_batch = options["n_batch"])
            noccA, noccB = trial.nelec[0], trial.nelec[1]
            wave_data["mo_coeff"] = [
                mo_coeff[0][:, : noccA],
                mo_coeff[1][:, : noccB],
            ]
            ham_data['h1_mod'] = h1_mod
            amplitudes = np.load(amp_file)
            t1a = jnp.array(amplitudes["t1a"])
            t1b = jnp.array(amplitudes["t1b"])
            t2aa = jnp.array(amplitudes["t2aa"])
            t2ab = jnp.array(amplitudes["t2ab"])
            t2bb = jnp.array(amplitudes["t2bb"])
            wave_data['t1a'] = t1a
            wave_data['t1b'] = t1b
            wave_data["t2aa"] = t2aa
            wave_data["t2bb"] = t2bb
            wave_data["t2ab"] = t2ab
            if "ad" in options["trial"]:
                trial = wavefunctions_unrestricted.uptccsd_ad(
                    norb, nelec_sp, n_batch=options["n_batch"])
                mo_a_A = wave_data['mo_coeff'][0]
                mo_b_B = wave_data['mo_coeff'][1]
                wave_data["rot_t1A"] = mo_a_A[:noccA,:noccA].T @ t1a
                wave_data["rot_t2AA"] = jnp.einsum('ik,jl,kalb->iajb',
                    mo_a_A[:noccA,:noccA].T,mo_a_A[:noccA,:noccA].T,t2aa)
                wave_data["rot_t1B"] = mo_b_B[:noccB,:noccB].T @ t1b
                wave_data["rot_t2BB"] = jnp.einsum('ik,jl,kalb->iajb',
                    mo_b_B[:noccB,:noccB].T,mo_b_B[:noccB,:noccB].T,t2bb)
                wave_data["rot_t2AB"] = jnp.einsum('ik,jl,kalb->iajb',
                    mo_a_A[:noccA,:noccA].T,mo_b_B[:noccB,:noccB].T,t2ab)

        elif "upt2ccsd" in options["trial"]:
            trial = wavefunctions_unrestricted.upt2ccsd(
                norb, nelec_sp, 
                n_batch=options["n_batch"], 
                nchol_chunk=options["nchol_chunk"],
                mix_precision=options["mix_precision"],
                )
            noccA, noccB = trial.nelec[0], trial.nelec[1]
            wave_data["mo_coeff"] = [
                mo_coeff[0][:, : noccA],
                mo_coeff[1][:, : noccB],
            ]
            ham_data['h1_mod'] = h1_mod
            amplitudes = np.load(amp_file)
            t1a = jnp.array(amplitudes["t1a"])
            t1b = jnp.array(amplitudes["t1b"])
            t2aa = jnp.array(amplitudes["t2aa"])
            t2ab = jnp.array(amplitudes["t2ab"])
            t2bb = jnp.array(amplitudes["t2bb"])
            mo_ta = trial.thouless_trans(t1a)[:,:noccA]
            mo_tb = trial.thouless_trans(t1b)[:,:noccB]
            wave_data['mo_ta'] = mo_ta
            wave_data['mo_tb'] = mo_tb
            wave_data["t2aa"] = t2aa
            wave_data["t2bb"] = t2bb
            wave_data["t2ab"] = t2ab
            wave_data['tau'] = trial.decompose_t2([t2aa,t2ab,t2bb])
            if "ad" in options["trial"]:
                trial = wavefunctions_unrestricted.upt2ccsd_ad(
                    norb, nelec_sp, n_batch=options["n_batch"])
                wave_data["rot_t2aa"] = jnp.einsum('ik,jl,kalb->iajb',
                    mo_ta[:noccA,:noccA].T,mo_ta[:noccA,:noccA].T,t2aa)
                wave_data["rot_t2bb"] = jnp.einsum('ik,jl,kalb->iajb',
                    mo_tb[:noccB,:noccB].T,mo_tb[:noccB,:noccB].T,t2bb)
                wave_data["rot_t2ab"] = jnp.einsum('ik,jl,kalb->iajb',
                    mo_ta[:noccA,:noccA].T,mo_tb[:noccB,:noccB].T,t2ab)
            if "eff" in options["trial"]:
                trial = wavefunctions_unrestricted.upt2ccsd_eff(
                    norb, nelec_sp, n_batch=options["n_batch"])
                wave_data["rot_t2aa"] = jnp.einsum('ik,jl,kalb->iajb',
                    mo_ta[:noccA,:noccA].T,mo_ta[:noccA,:noccA].T,t2aa)
                wave_data["rot_t2bb"] = jnp.einsum('ik,jl,kalb->iajb',
                    mo_tb[:noccB,:noccB].T,mo_tb[:noccB,:noccB].T,t2bb)
                wave_data["rot_t2ab"] = jnp.einsum('ik,jl,kalb->iajb',
                    mo_ta[:noccA,:noccA].T,mo_tb[:noccB,:noccB].T,t2ab)

        elif options["trial"] == "ustoccsd2":
            trial = wavefunctions_unrestricted.ustoccsd2(
                norb,
                nelec_sp,
                n_batch = options["n_batch"],
                nslater = options['nslater']
                )
            nocc_a, nocc_b = nelec_sp
            amplitudes = np.load(amp_file)
            t1a = jnp.array(amplitudes["t1a"])
            t1b = jnp.array(amplitudes["t1b"])
            t2aa = jnp.array(amplitudes["t2aa"])
            t2ab = jnp.array(amplitudes["t2ab"])
            t2bb = jnp.array(amplitudes["t2bb"])
            mo = [mo_coeff[0][:,:nocc_a], mo_coeff[1][:,:nocc_b]]
            mo_t = trial._thouless(mo, [t1a, t1b])
            wave_data['mo_ta'] = mo_t[0]
            wave_data['mo_tb'] = mo_t[1]
            wave_data["t2aa"] = t2aa
            wave_data["t2bb"] = t2bb
            wave_data["t2ab"] = t2ab
            wave_data['tau'] = trial.decompose_t2([t2aa,t2ab,t2bb])
            wave_data["mo_coeff"] = [mo_coeff[0][:, : nocc_a], mo_coeff[1][:, : nocc_b]]

            sampler = sampling.sampler_stoccsd2(
                n_prop_steps = options["n_prop_steps"],
                n_blocks = options["n_blocks"],
                n_chol = nchol,
                )
    

    if options["walker_type"] == "rhf":
        prop = propagation.propagator_restricted(
                options["dt"], 
                options["n_walkers"], 
                options["n_exp_terms"],
                options["n_batch"]
            )

    elif options["walker_type"] == "uhf":
        prop = propagation.propagator_unrestricted(
                options["dt"],
                options["n_walkers"],
                options["n_exp_terms"],
                options["n_batch"],
            )

    if  'pt' in options['trial'] and 'cc' in options['trial']:
        if 'pt2' in options['trial']:
            sampler = sampling.sampler_pt2(
                options["n_prop_steps"],
                options["n_blocks"],
                nchol,)
        else:
            sampler = sampling.sampler_pt(
                options["n_prop_steps"],
                options["n_blocks"],
                nchol,)
            
    elif 'stoccsd' in options['trial']:
        if '2' in options['trial']:
            sampler = sampling.sampler_stoccsd2(
                options["n_prop_steps"],
                options["n_blocks"],
                nchol,)
        else:
            sampler = sampling.sampler_stoccsd(
                options["n_prop_steps"],
                options["n_blocks"],
                nchol,)
            
    else:
        sampler = sampling.sampler(
            options["n_prop_steps"],
            options["n_blocks"],
            nchol,)

    
    if options["free_projection"]:
        if 'pt2' not in options["trial"]:
            sampler = fp_sampling.fp_sampler(
                    options["n_prop_steps"],
                    options["n_eql_blocks"],
                    options["n_trj"],
                    nchol,
                    )
        elif 'pt2' in options["trial"]:
            sampler = fp_sampling.fp_sampler_pt2(
                    options["n_prop_steps"],
                    options["n_eql_blocks"],
                    options["n_trj"],
                    nchol,
                    )

    print(f"Number of electrons: {nelec_sp} | 2xSpin: {ms}")
    print(f"Number of orbitals: {norb}| Number of Chol: {nchol}")

    for op in options:
        if options[op] is not None:
            print(f"{op}: {options[op]}")

    return ham_data, ham, prop, trial, wave_data, sampler, options