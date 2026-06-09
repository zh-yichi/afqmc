import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import h5py
import pickle

import numpy as np

import opt_einsum as oe

import jax
import jax.numpy as jnp
from jax import scipy as jsp
from jax import jit, lax, random, vmap

from afqmc import hamiltonian, cholesky, linalg_utils
from afqmc import propagation, sampling, fp_sampling
from afqmc.wavefunctions import wavefunctions_restricted
from afqmc.wavefunctions import wavefunctions_unrestricted

from functools import partial
print = partial(print, flush=True)

def replicate_walker(walker, nwalker):
    if isinstance(walker, jax.Array):
        walkers = jnp.array([walker] * nwalker, dtype=jnp.complex128)
    elif isinstance(walker, (tuple, list)):
        walkers_a = jnp.array([walker[0]] * nwalker, dtype=jnp.complex128)
        walkers_b = jnp.array([walker[1]] * nwalker, dtype=jnp.complex128)
        walkers = [walkers_a, walkers_b]
    return walkers

# def init_walkers_1rdm(
#         trial,
#         wave_data: dict, 
#         n_walkers: int, 
#         restricted: bool = False
#         ):
#     """Initialize walkers by rdm1 natural orbitals.

#     Args:
#         trial: trial wavefunction object
#         wave_data: The trial wave function data.
#         n_walkers: The number of walkers.
#         restricted: Whether the walkers should be restricted.

#     Returns:
#         walkers: The initial walkers.
#             If restricted, a single jax.Array of shape (nwalkers, norb, nelec[0]).
#             If unrestricted, a list of two jax.Arrays each of shape (nwalkers, norb, nelec[sigma]).
#     """
#     rdm1 = trial.get_rdm1(wave_data)
#     natorbs_up = jnp.linalg.eigh(rdm1[0])[1][:, ::-1][:, : trial.nelec[0]]
#     natorbs_dn = jnp.linalg.eigh(rdm1[1])[1][:, ::-1][:, : trial.nelec[1]]
    
#     if restricted:
#         if trial.nelec[0] == trial.nelec[1]:
#             det_overlap = np.linalg.det(
#                 natorbs_up[:, : trial.nelec[0]].T @ natorbs_dn[:, : trial.nelec[1]]
#             )
#             if (
#                 np.abs(det_overlap) > 1e-3
#             ):  # probably should scale this threshold with number of electrons
#                 return jnp.array([natorbs_up + 0.0j] * n_walkers)
#             else:
#                 overlaps = np.array(
#                     [
#                         natorbs_up[:, i].T @ natorbs_dn[:, i]
#                         for i in range(trial.nelec[0])
#                     ]
#                 )
#                 new_vecs = natorbs_up[:, : trial.nelec[0]] + np.einsum(
#                     "ij,j->ij", natorbs_dn[:, : trial.nelec[1]], np.sign(overlaps)
#                 )
#                 new_vecs = np.linalg.qr(new_vecs)[0]
#                 det_overlap = np.linalg.det(
#                     new_vecs.T @ natorbs_up[:, : trial.nelec[0]]
#                 ) * np.linalg.det(new_vecs.T @ natorbs_dn[:, : trial.nelec[1]])
#                 if np.abs(det_overlap) > 1e-3:
#                     return jnp.array([new_vecs + 0.0j] * n_walkers)
#                 else:
#                     raise ValueError(
#                         "Cannot find a set of RHF orbitals with good trial overlap."
#                     )
#         else:
#             # bring the dn orbital projection onto up space to the front
#             dn_proj = natorbs_up.T.conj() @ natorbs_dn
#             proj_orbs = jnp.linalg.qr(dn_proj, mode="complete")[0]
#             orbs = natorbs_up @ proj_orbs
#             return jnp.array([orbs + 0.0j] * n_walkers)
#     else:
#         return [
#             jnp.array([natorbs_up + 0.0j] * n_walkers),
#             jnp.array([natorbs_dn + 0.0j] * n_walkers),
#         ]

#### restricted ####
def decompose_rt2(t2, thresh=1e-8):
    # adapted from Yann

    # nO = self.nelec[0]
    # nV = self.norb - nO

    nocc, nvir, _, _ = t2.shape
    npair = nocc * nvir

    # assert t2.shape == (nO, nV, nO, nV)
    
    t2 = t2.reshape(npair, npair)
    e_val, e_vec = jnp.linalg.eigh(t2)

    # Keep only important modes
    mask = jnp.abs(e_val) > thresh
    e_val_trunc = e_val[mask]
    e_vec_trunc = e_vec[:, mask]

    tau = e_vec_trunc @ jnp.diag(jnp.sqrt(e_val_trunc + 0.0j))
    
    err = jnp.linalg.norm(t2 - tau @ tau.T)
    assert err < 10 * thresh

    # print(f'Throw {len(e_val)-len(e_val_trunc)} vectors in T2 deomposition')
    # print(f'cutoff = {thresh:.2e} | error = {err:.2e}')
    # print(f'number of T2 decomposition vectors {len(e_val_trunc)}')

    tau = tau.T.reshape(-1, nocc, nvir)

    return tau

@jit
def rthouless(init_slater, t):
    '''
    restricted thouless transformation
    |psi'> = exp(t_ia a+ i)|psi>
    use the block form of t, no need to apply full exp(t)
    equivalent to the function below
    '''
    # norb, nocc = self.norb, self.nelec[0]
    # nvir = norb - nocc
    nocc, nvir = t.shape
    norb = nocc + nvir
    assert init_slater.shape == (norb, nocc)
    t_full = jnp.eye(norb, dtype=jnp.complex128)
    exp_t = t_full.at[:nocc, nocc:].set(t)
    return exp_t.T @ init_slater

@jit
def rthouless_full(init_slater, t):
    '''
    restricted thouless transformation
    |psi'> = exp(t_ia a+ i)|psi>
    apply full exp(t)
    equivalent to the function above
    '''
    nocc, nvir = t.shape
    norb = nocc + nvir
    assert init_slater.shape == (norb, nocc)
    t_full = jnp.zeros((norb, norb), dtype=jnp.complex128)
    t_full = t_full.at[:nocc, nocc:].set(t)
    exp_t = jsp.linalg.expm(t_full)
    return exp_t.T @ init_slater

@partial(jit, static_argnames=("n_walkers"))
def get_rccsd_walkers(prop_data, wave_data, n_walkers):
    prop_data["key"], subkey = random.split(prop_data["key"])
    
    fieldy = random.normal(
        subkey,
        shape=(
            n_walkers,
            wave_data['tau'].shape[0],
        ),
    )
    # ytaus shape (nwalker, nocc, nvir)
    ytaus = oe.contract("wg,gia->wia", fieldy, wave_data['tau'], backend='jax')

    slaters = vmap(lambda y: rthouless(wave_data['mo_t'], y))(ytaus)

    # mo_t = wave_data['mo_t']

    # def scan_body(carry, ytau):
    #     # ytau_up, ytau_dn = ytau
    #     slater = rthouless(wave_data['mo_t'], ytau)
    #     return carry, slater

    # # scan iterates over leading axis (n_walkers) of (ytaus_up, ytaus_dn)
    # _, slaters = lax.scan(scan_body, None, ytaus)

    return slaters, prop_data


def decompose_ut2(t2, thresh=1e-8):
    # adapted from Yann
    # norb = trial.norb
    # nocca, noccb = trial.nelec
    # nvira, nvirb = (norb - nocca, norb - noccb)

    t2aa, t2ab, t2bb = t2
    nocca, nvira, noccb, nvirb = t2ab.shape
    # Number of excitation pairs
    npaira = nocca * nvira
    npairb = noccb * nvirb

    assert t2aa.shape == (nocca, nvira, nocca, nvira)
    # assert t2ab.shape == (nocca, nvira, noccb, nvirb)
    assert t2bb.shape == (noccb, nvirb, noccb, nvirb)

    # print('Decomposing Unrestricted T2 amplitudes')

    t2aa = t2aa.reshape(npaira, npaira)
    t2ab = t2ab.reshape(npaira, npairb)
    t2bb = t2bb.reshape(npairb, npairb)

    # Symmetric full t2 
    # [[ t2aa/2  t2ab   ]]
    # [[ t2ab^T  t2bb/2 ]]
    t2full = np.zeros((npaira + npairb, npaira + npairb))
    t2full[:npaira, :npaira] = 0.5 * t2aa
    t2full[npaira:, :npaira] = t2ab.T
    t2full[:npaira, npaira:] = t2ab
    t2full[npaira:, npaira:] = 0.5 * t2bb
    t2full = jnp.array(t2full)

    # t2 = LL^T
    e_val, e_vec = jnp.linalg.eigh(t2full)

    # Keep only important modes
    mask = jnp.abs(e_val) > thresh
    e_val_trunc = e_val[mask]
    e_vec_trunc = e_vec[:, mask]
    
    tau = e_vec_trunc @ jnp.diag(np.sqrt(e_val_trunc + 0.0j))
    err = jnp.linalg.norm(t2full - tau @ tau.T)
    assert err < 10 * thresh
    # print(f'Throw {len(e_val)-len(e_val_trunc)} vectors in T2 deomposition')
    # print(f'SVD cutoff = {thresh:.2e} | error = {err:.2e}')
    # print(f'number of T2 decomposition vectors {len(e_val_trunc)}')

    # alpha/beta operators for HS
    # Summation on the left to have a list of operators
    taua = tau.T[:,:npaira]
    taub = tau.T[:, npaira:]
    taua = taua.reshape(-1, nocca, nvira)
    taub = taub.reshape(-1, noccb, nvirb)

    return [taua, taub]

@jit
def uthouless(slater, tau):
    # calculate |psi'> = exp(t_ia a+ i)|psi>
    
    slater_a, slater_b = slater
    ta, tb = tau
    nocc_a, nvir_a = ta.shape
    nocc_b, nvir_b = tb.shape
    
    norb_a = nocc_a + nvir_a
    norb_b = nocc_b + nvir_b
    
    assert norb_a == norb_b
    norb = norb_a
    
    assert slater_a.shape == (norb, nocc_a)
    assert slater_b.shape == (norb, nocc_b)

    ta_full = jnp.eye(norb, dtype=jnp.complex128)
    tb_full = jnp.eye(norb, dtype=jnp.complex128)
    exp_ta = ta_full.at[:nocc_a, nocc_a:].set(ta)
    exp_tb = tb_full.at[:nocc_b, nocc_b:].set(tb)

    slater_ta = exp_ta.T @ slater_a
    slater_tb = exp_tb.T @ slater_b

    return [slater_ta, slater_tb]

def thouless(slater, tau):
    if isinstance(slater, jax.Array) and len(slater.shape) == 2:
        return rthouless(slater, tau)
    elif isinstance(slater, (tuple, list)) and isinstance(tau, (tuple, list)):
        return uthouless(slater, tau)

@partial(jit, static_argnames=("n_walkers"))
def get_uccsd_walkers(prop_data, wave_data, n_walkers):
    prop_data["key"], subkey = random.split(prop_data["key"])
    
    fieldy = random.normal(
        subkey,
        shape=(
            n_walkers,
            wave_data['tau'][0].shape[0],
        ),
    )
    # ytaus shape (nwalker, nocc, nvir)
    ytaus_up = oe.contract("wg,gia->wia", fieldy, wave_data['tau'][0], backend='jax')
    ytaus_dn = oe.contract("wg,gia->wia", fieldy, wave_data['tau'][1], backend='jax')

    mo_t = (wave_data["mo_ta"], wave_data["mo_tb"])
    
    slaters_up, slaters_dn = vmap(
        lambda yu, yd: uthouless(mo_t, (yu, yd)))(ytaus_up, ytaus_dn)

    # mo_t = [wave_data['mo_ta'], wave_data['mo_tb']]

    # def scan_body(carry, ytau):
    #     ytau_up, ytau_dn = ytau
    #     slater_up, slater_dn = uthouless(mo_t, [ytau_up, ytau_dn])
    #     return carry, (slater_up, slater_dn)

    # # scan iterates over leading axis (n_walkers) of (ytaus_up, ytaus_dn)
    # _, (slaters_up, slaters_dn) = lax.scan(scan_body, None, (ytaus_up, ytaus_dn),)

    return [slaters_up, slaters_dn], prop_data


def get_ccsd_walkers(prop_data, wave_data, n_walkers, walker_type):
    if walker_type == "rhf":
        if "tau" not in wave_data:
            wave_data["tau"] = decompose_rt2(wave_data["t2"])
        return get_rccsd_walkers(prop_data, wave_data, n_walkers)
    elif walker_type == "uhf":
        if "tau" not in wave_data:
            wave_data["tau"] = decompose_ut2([wave_data["t2aa"],
                                              wave_data["t2ab"],
                                              wave_data["t2bb"]])
        return get_uccsd_walkers(prop_data, wave_data, n_walkers)
    else:
        raise ValueError(f"unsupport CCSD initial walker_type: {walker_type}")


def init_hf_prop_data(
    trial,
    wave_data,
    ham_data,
    options
    ):

    print("\nInitalize QMC walkers by HF")
    prop_data = {}
    prop_data["n_killed_walkers"] = 0
    prop_data["key"] = random.PRNGKey(options["seed"])

    weights0 = jnp.ones(options["n_walkers"], dtype=jnp.float64)
    walkers0 = replicate_walker(wave_data["mo_coeff"], options["n_walkers"])
    overlaps0 = trial.calc_overlap(walkers0, wave_data)
    energies0 = trial.calc_energy(walkers0, ham_data, wave_data)
    energy0 = jnp.sum(overlaps0 * energies0) / jnp.sum(overlaps0)

    prop_data["walkers"] = walkers0
    prop_data["weights"] = weights0
    prop_data["overlaps"] = overlaps0
    prop_data["e_estimate"] = jnp.real(energy0)
    prop_data["pop_control_ene_shift"] = jnp.real(energy0)

    return prop_data

def init_ccsd_prop_data(
    trial,
    wave_data,
    ham_data,
    options
    ):

    print("\nInitalize QMC walkers by stochastic CCSD")
    prop_data = {}
    prop_data["n_killed_walkers"] = 0
    prop_data["key"] = random.PRNGKey(options["seed"])

    weights0 = jnp.ones(options["n_walkers"], dtype=jnp.float64)
    walkers0 = replicate_walker(wave_data["mo_coeff"], options["n_walkers"])
    overlaps0 = trial.calc_overlap(walkers0, wave_data)

    walkers1, prop_data = get_ccsd_walkers(
        prop_data, wave_data, options["n_walkers"], options["walker_type"]
    )
    overlaps1 = trial.calc_overlap(walkers1, wave_data)
    weights1 = jnp.real(weights0 * overlaps1 / overlaps0)

    prop_data["weights"] = weights1
    prop_data["walkers"] = walkers1
    prop_data["overlaps"] = overlaps1

    h0 = ham_data["h0"]
    t1s, t2s, e0s, e1s = trial.calc_energy_pt(prop_data["walkers"], ham_data, wave_data)

    wt = jnp.sum(weights1)
    t1 = jnp.sum(weights1 * t1s) / wt
    t2 = jnp.sum(weights1 * t2s) / wt
    e0 = jnp.sum(weights1 * e0s) / wt
    e1 = jnp.sum(weights1 * e1s) / wt

    energy = jnp.real(h0 + e0 / t1 + e1 / t1 - t2 * e0 / t1**2)

    prop_data["e_estimate"] = energy
    prop_data["pop_control_ene_shift"] = energy

    return prop_data

def init_afqmc(options=None,
               option_file="options.bin",
               amp_file="amplitudes.npz",
               chol_file="FCIDUMP_chol"):
    
    if options is None:
        try:
            with open(option_file, "rb") as f:
                options = pickle.load(f)
        except:
            options = {}

    options["dt"] = options.get("dt", 0.005)
    options["n_exp_terms"] = options.get("n_exp_terms",6)
    options["n_walkers"] = options.get("n_walkers", 50)
    options["n_prop_steps"] = options.get("n_prop_steps", 50)
    options["n_blocks"] = options.get("n_blocks", 500)
    options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))
    options["eql_time"] = options.get("eql_time", 20)
    options["walker_type"] = options.get("walker_type", "rhf")
    options["save_walkers"] = options.get("save_walkers", False)
    options["trial"] = options.get("trial", None)
    options["free_projection"] = options.get("free_projection", False)
    options["n_batch"] = options.get("n_batch", 1)
    options["max_error"] = options.get("max_error", 0.0)
    options["nchol_chunk"] = options.get("nchol_chunk", 100)
    options["max_memory"] = options.get("max_memory", 2000) # MB
    options["mix_precision"] = options.get("mix_precision", True)

    print("\nLoad system from Integral File")

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

    # print(f"AFQMC Object Spin type: {spin_type}")

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

    options["nchol_chunk"] = cholesky.chunk_chol(
        chol, options["nchol_chunk"], options["max_memory"]/options["n_walkers"])

    wave_data = {}
    mo_coeff = [jnp.eye(norb), jnp.eye(norb)]

    if spin_type == "restricted":
        wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0]]
        if options["trial"] == "rhf":
            trial = wavefunctions_restricted.rhf(norb, nelec_sp, 
                                                 n_batch=options["n_batch"],
                                                 nchol_chunk=options["nchol_chunk"],
                                                 )
            # wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0]]

        elif "cisd" in options["trial"]:
            try:
                amplitudes = np.load(amp_file)
                t1 = jnp.array(amplitudes["t1"])
                t2 = jnp.array(amplitudes["t2"])
                ci2 = t2 + jnp.einsum("ia,jb->iajb", t1, t1)
                trial_wave_data = {"ci1": t1, "ci2": ci2}
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
                trial_wave_data = {"ci2": t2}
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
            trial = wavefunctions_restricted.ptccsd(norb, nelec_sp, n_batch=options["n_batch"])
            if "ad" in options["trial"]:
                trial = wavefunctions_restricted.ptccsd_ad(norb, nelec_sp, n_batch=options["n_batch"])
        
        elif options["trial"] == "ptccd":
            amplitudes = np.load(amp_file)
            t2 = jnp.array(amplitudes["t2"])
            trial_wave_data = {"t2": t2}
            wave_data.update(trial_wave_data)
            trial = wavefunctions_restricted.ptccd(norb, nelec_sp, n_batch=options["n_batch"])

        elif "pt2ccsd" in options["trial"]:
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
            mo_t = thouless(wave_data['mo_coeff'], t1)
            wave_data['mo_t'] = mo_t #thouless(wave_data['mo_coeff'], t1)
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
            wave_data['mo_t'] = thouless(wave_data['mo_coeff'], t1)
            wave_data['tau'] = trial.decompose_t2(t2)
    
    elif spin_type == "unrestricted":
        wave_data["mo_coeff"] = [mo_coeff[0][:, : nelec_sp[0]],
                                 mo_coeff[1][:, : nelec_sp[1]],]

        if options["trial"] == "uhf":
            trial = wavefunctions_unrestricted.uhf(norb, nelec_sp, n_batch=options["n_batch"])

        elif options["trial"] == "ucisd":
            trial = wavefunctions_unrestricted.ucisd(norb, nelec_sp, n_batch=options["n_batch"])
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
                wave_data["ci1A"] = t1a
                wave_data["ci1B"] = t1b
                wave_data["ci2AA"] = ci2aa
                wave_data["ci2AB"] = ci2ab
                wave_data["ci2BB"] = ci2bb
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
            ham_data['h1_mod'] = h1_mod
            amplitudes = np.load(amp_file)
            t1a = jnp.array(amplitudes["t1a"])
            t1b = jnp.array(amplitudes["t1b"])
            t2aa = jnp.array(amplitudes["t2aa"])
            t2ab = jnp.array(amplitudes["t2ab"])
            t2bb = jnp.array(amplitudes["t2bb"])
            # mo_ta = trial.thouless_trans(t1a)[:,:noccA]
            # mo_tb = trial.thouless_trans(t1b)[:,:noccB]
            # wave_data['mo_ta'] = mo_ta
            # wave_data['mo_tb'] = mo_tb
            [mo_ta, mo_tb] = thouless(wave_data['mo_coeff'], [t1a, t1b])
            wave_data['mo_ta'] = mo_ta
            wave_data['mo_tb'] = mo_tb
            wave_data["t2aa"] = t2aa
            wave_data["t2bb"] = t2bb
            wave_data["t2ab"] = t2ab
            # wave_data['tau'] = trial.decompose_t2([t2aa,t2ab,t2bb])
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
            # mo = [mo_coeff[0][:,:nocc_a], mo_coeff[1][:,:nocc_b]]
            [mo_ta, mo_tb] = thouless(wave_data['mo_coeff'], [t1a, t1b])
            wave_data['mo_ta'] = mo_ta
            wave_data['mo_tb'] = mo_tb
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

    print("\nQMC System")
    print(f"Number of electrons: {nelec_sp}")
    print(f"Spin Multiplicity:   {ms}")
    print(f"Number of orbitals:  {norb}")
    print(f"Number of Chol:      {nchol}")

    print("\nQMC Parameters")
    for op in options:
        if options[op] is not None:
            print(f"{str(op):<20s}: {str(options[op]):>20s}")

    return ham_data, ham, prop, trial, wave_data, sampler, options


def print_start():
    banner = r"""
    ________                     _____                    
    ___  __ \___  __________________(_)_____________ _    
    __  /_/ /  / / /_  __ \_  __ \_  /__  __ \_  __ `/    
    _  _, _// /_/ /_  / / /  / / /  / _  / / /  /_/ /     
    /_/ |_| \__,_/ /_/ /_//_/ /_//_/  /_/ /_/_\__, /      
                                             /____/       
    _____________________________  ___________            
    ___    |__  ____/_  __ \__   |/  /_  ____/            
    __  /| |_  /_   _  / / /_  /|_/ /_  /                 
    _  ___ |  __/   / /_/ /_  /  / / / /___               
    /_/  |_/_/      \___\_\/_/  /_/  \____/               
"""
    print(banner)