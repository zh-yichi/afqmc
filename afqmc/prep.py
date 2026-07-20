import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import h5py
import pickle

import numpy as np

import opt_einsum as oe

import jax
import jax.numpy as jnp
from jax import scipy as jsp
from jax import random

from . import hamiltonian, cholesky, walker_tools, slater_tools, t2_tools
from . import propagation, sampling, fp_sampling
from .wavefunctions import wavefunctions_restricted
from .wavefunctions import wavefunctions_unrestricted

from functools import partial
print = partial(print, flush=True)

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


def load_cc_amplitude(wave_data=None, amp_file="amplitudes.npz"):
    if wave_data is None:
        wave_data = {}
    t1, t2 = t2_tools.read_cc_amps(amp_file)
    if isinstance(t2, jax.Array):
        wave_data["t1"] = t1
        wave_data["t2"] = t2
    elif isinstance(t2, tuple):
        wave_data["t1a"] = t1[0]
        wave_data["t1b"] = t1[1]
        wave_data["t2aa"] = t2[0]
        wave_data["t2ab"] = t2[1]
        wave_data["t2bb"] = t2[2]
    return wave_data

def load_ci_amplitude(wave_data=None, amp_file="amplitudes.npz"):
    if wave_data is None:
        wave_data = {}
    t1, t2 = t2_tools.read_cc_amps(amp_file)
    ci1, ci2 = t2_tools.cc2ci(t1, t2)
    if isinstance(ci2, jax.Array):
        wave_data["ci1"] = ci1
        wave_data["ci2"] = ci2
    elif isinstance(ci2, tuple):
        wave_data["ci1A"] = ci1[0]
        wave_data["ci1B"] = ci1[1]
        wave_data["ci2AA"] = ci2[0]
        wave_data["ci2AB"] = ci2[1]
        wave_data["ci2BB"] = ci2[2]
    return wave_data

def init_hf_prop_data(
    wave,
    wave_data,
    ham_data,
    options
    ):

    print("\nInitalize QMC walkers by HF")
    prop_data = {}
    prop_data["n_killed_walkers"] = 0
    prop_data["key"] = random.PRNGKey(options["seed"])

    weights0 = jnp.ones(options["n_walkers"], dtype=jnp.float64)
    walkers0 = walker_tools.replicate_walker(wave_data["mo_coeff"], options["n_walkers"])
    overlaps0 = wave.calc_overlap(walkers0, wave_data)
    energies0 = wave.calc_energy(walkers0, ham_data, wave_data)
    energy0 = jnp.sum(overlaps0 * energies0) / jnp.sum(overlaps0)

    prop_data["walkers"] = walkers0
    prop_data["weights"] = weights0
    prop_data["overlaps"] = overlaps0
    prop_data["e_estimate"] = jnp.real(energy0)
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

    return prop_data

def init_hf_prop_data_exp(
    wave,
    wave_data,
    ham_data,
    options
    ):

    print("\nInitalize QMC walkers by HF")
    prop_data = {}
    prop_data["n_killed_walkers"] = 0
    prop_data["key"] = random.PRNGKey(options["seed"])

    prop_data["weights"] = jnp.ones(options["n_walkers"], dtype=jnp.float64)
    walkers0 = walker_tools.replicate_walker(wave_data["mo_coeff"], options["n_walkers"])
    g_overlaps0 = wave.calc_overlap(walkers0, wave_data)
    t_overlaps0 = wave.calc_trial_overlap(walkers0, wave_data)
    weights0 = prop_data["weights"] * t_overlaps0 / g_overlaps0
    samples0 = wave.calc_energy(walkers0, ham_data, wave_data)
    weight0_mean, energy0_mean, energy0_err \
            = wave.calc_sample_energy(weights0, samples0, ham_data)

    prop_data["walkers"] = walkers0
    prop_data["overlaps"] = g_overlaps0
    prop_data["e_estimate"] = jnp.real(energy0_mean)
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

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
    walkers0 = walker_tools.replicate_walker(wave_data["mo_coeff"], options["n_walkers"])
    overlaps0 = trial.calc_overlap(walkers0, wave_data)

    walkers1, prop_data = walker_tools.get_ccsd_walkers(
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

def get_qmc_options(options=None, option_file="options.bin"):
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
    options["trial"] = options.get("trial", None)
    if "u" not in options["trial"]:
        options["walker_type"] = options.get("walker_type", "rhf")
    elif "u" in options["trial"]:
        options["walker_type"] = options.get("walker_type", "uhf")
    options["n_batch"] = options.get("n_batch", 1)
    options["max_error"] = options.get("max_error", 0.0)
    options["nchol_chunk"] = options.get("nchol_chunk", 100)
    options["max_memory"] = options.get("max_memory", 2000) # MB
    options["mix_precision"] = options.get("mix_precision", True)
    options["free_projection"] = options.get("free_projection", False)

    print("\nQMC Parameters")
    for op in options:
        if options[op] is not None:
            print(f"{str(op):<15s} - {str(options[op]):>10s}")

    return options

def load_chol(chol_file="FCIDUMP_chol"):
    with h5py.File(chol_file, "r") as fh5:
        [nelec, norb, ms] = fh5["header"]
        spin_type = fh5["spin_type"][()]
        h0 = jnp.array(fh5.get("energy_core"))
        h1 = jnp.array(fh5.get("hcore"))
        chol = jnp.array(fh5.get("chol"))
    
    if isinstance(spin_type, bytes):
        spin_type = spin_type.decode()

    assert spin_type in ["restricted", "unrestricted"]

    if spin_type == 'restricted':
        h1 = jnp.array(h1).reshape(norb, norb)
        chol = jnp.array(chol).reshape(-1, norb, norb)

    elif spin_type == 'unrestricted':
        h1 = (jnp.array(h1[0].reshape(norb, norb)),
              jnp.array(h1[1].reshape(norb, norb)))
        chol = (jnp.array(chol[0].reshape(-1, norb, norb)),
                jnp.array(chol[1].reshape(-1, norb, norb)))

    assert type(ms) is np.int64
    assert type(nelec) is np.int64
    assert type(norb) is np.int64

    ms, nelec, norb = int(ms), int(nelec), int(norb)
    nelec_sp = ((nelec + abs(ms)) // 2, (nelec - abs(ms)) // 2)

    return h0, h1, chol, ms, nelec_sp, norb, spin_type

def get_hamiltonian(h0, h1, chol, norb):

    ham = hamiltonian.hamiltonian(norb)
    ham_data = {}
    ham_data["h0"] = h0
    
    if isinstance(chol, (jax.Array, np.ndarray)):
        ham_data["h1"] = (jnp.array(h1), jnp.array(h1))
        nchol = chol.shape[0]
        ham_data["chol"] = jnp.array(chol.reshape(chol.shape[0], -1))
    
    elif isinstance(chol, (list, tuple)):
        ham_data["h1"] = (jnp.array(h1[0] + h1[0].T) / 2.0, 
                          jnp.array(h1[1] + h1[1].T) / 2.0)
        nchola = chol[0].shape[0]
        ncholb = chol[1].shape[0]
        assert nchola == ncholb, f"nchol mismatch: alpha={nchola}, beta={ncholb}"
        nchol = nchola
        ham_data["chol"] = (jnp.array(chol[0].reshape(nchol, -1)),
                            jnp.array(chol[1].reshape(nchol, -1)))
        
    return ham, ham_data, nchol

def get_wavefunction(spin_type, norb, nelec_sp, nchol_chunk, options, amp_file):
    wave_data = {}

    if spin_type == "restricted":
        wave_data["mo_coeff"] = jnp.eye(norb)[:, : nelec_sp[0]]
        if options["trial"] == "rhf":
            trial = wavefunctions_restricted.rhf(norb, nelec_sp, 
                                                 n_batch=options["n_batch"],
                                                 nchol_chunk=nchol_chunk,
                                                 )

        elif "cisd" in options["trial"]:
            wave_data = load_ci_amplitude(wave_data, amp_file)
            trial = wavefunctions_restricted.cisd(norb, nelec_sp, 
                                                    n_batch=options["n_batch"]
                                                    )

        elif options["trial"] == "cid":
            wave_data = load_ci_amplitude(wave_data, amp_file)
            trial = wavefunctions_restricted.cid(norb, nelec_sp, n_batch=options["n_batch"])
            
        elif options["trial"] == "ptccsd":
            wave_data = load_cc_amplitude(wave_data, amp_file)
            trial = wavefunctions_restricted.ptccsd(norb, nelec_sp, n_batch=options["n_batch"])
            if "ad" in options["trial"]:
                trial = wavefunctions_restricted.ptccsd_ad(norb, nelec_sp, n_batch=options["n_batch"])
        
        elif options["trial"] == "ptccd":
            wave_data = load_cc_amplitude(wave_data, amp_file)
            trial = wavefunctions_restricted.ptccd(norb, nelec_sp, n_batch=options["n_batch"])

        elif "pt2ccsd" in options["trial"]:
            wave_data = load_cc_amplitude(wave_data, amp_file)
            wave_data["mo_t"] = slater_tools.thouless(wave_data["mo_coeff"], wave_data["t1"])
            trial = wavefunctions_restricted.pt2ccsd(norb, nelec_sp, 
                                                     n_batch=options["n_batch"],
                                                     nchol_chunk=nchol_chunk, 
                                                     mix_precision=options["mix_precision"],
                                                     )
            if "ad" in options["trial"]:
                nocc = nelec_sp[0]
                rot_t2 = oe.contract('il,jk,lakb->iajb',
                                     wave_data["mo_t"][:nocc,:nocc].T,
                                     wave_data["mo_t"][:nocc,:nocc].T,
                                     wave_data["t2"], 
                                     backend='jax')
                wave_data['rot_t2'] = rot_t2
                trial = wavefunctions_restricted.pt2ccsd_ad(norb, nelec_sp, 
                                                            n_batch=options["n_batch"])

        elif "stoccsd" in options["trial"]:
            wave_data = load_cc_amplitude(wave_data, amp_file)
            wave_data['mo_t'] = slater_tools.thouless(wave_data['mo_coeff'], wave_data["t1"])
            wave_data['tau'] = t2_tools.decompose_t2(wave_data["t2"])

            if "2" in options["trial"]:
                trial = wavefunctions_restricted.stoccsd2(
                    norb,
                    nelec_sp,
                    n_batch = options["n_batch"],
                    nslater = options['nslater']
                    )
            else:
                trial = wavefunctions_restricted.stoccsd(
                    norb,
                    nelec_sp,
                    n_batch = options["n_batch"],
                    nslater = options['nslater']
                    )
                    
    
    elif spin_type == "unrestricted":
        nocc_a, nocc_b = nelec_sp
        wave_data["mo_coeff"] = (jnp.eye(norb)[:,:nocc_a],
                                 jnp.eye(norb)[:,:nocc_b])

        if options["trial"] == "uhf":
            trial = wavefunctions_unrestricted.uhf(norb, nelec_sp, n_batch=options["n_batch"])

        elif options["trial"] == "ucisd":
            wave_data = load_ci_amplitude(wave_data, amp_file)
            trial = wavefunctions_unrestricted.ucisd(norb, nelec_sp, n_batch=options["n_batch"])

        elif options["trial"] == "uptccsd":
            wave_data = load_cc_amplitude(wave_data, amp_file)
            trial = wavefunctions_unrestricted.uptccsd(norb, nelec_sp, n_batch = options["n_batch"])
            if "ad" in options["trial"]:
                trial = wavefunctions_unrestricted.uptccsd_ad(
                    norb, nelec_sp, n_batch=options["n_batch"])
                

                wave_data["rot_t1a"] = wave_data['mo_coeff'][0][:nocc_a,:nocc_a].T @ wave_data["t1a"]
                wave_data["rot_t1b"] = wave_data['mo_coeff'][1][:nocc_b,:nocc_b].T @ wave_data["t1b"]
                
                wave_data["rot_t2aa"] = oe.contract('ik,jl,kalb->iajb',
                                                    wave_data['mo_coeff'][0][:nocc_a,:nocc_a].T,
                                                    wave_data['mo_coeff'][1][:nocc_a,:nocc_a].T,
                                                    wave_data["t2aa"], 
                                                    backend='jax')
                wave_data["rot_t2ab"] = oe.contract('ik,jl,kalb->iajb',
                                                    wave_data['mo_coeff'][0][:nocc_a,:nocc_a].T,
                                                    wave_data['mo_coeff'][1][:nocc_b,:nocc_b].T,
                                                    wave_data["t2ab"], 
                                                    backend='jax')
                wave_data["rot_t2bb"] = oe.contract('ik,jl,kalb->iajb',
                                                    wave_data['mo_coeff'][0][:nocc_b,:nocc_b].T,
                                                    wave_data['mo_coeff'][1][:nocc_b,:nocc_b].T,
                                                    wave_data["t2bb"], 
                                                    backend='jax')

        elif "upt2ccsd" in options["trial"]:
            wave_data = load_cc_amplitude(wave_data, amp_file)
            (wave_data['mo_ta'], wave_data['mo_tb']) = slater_tools.thouless(
                wave_data['mo_coeff'], (wave_data["t1a"], wave_data["t1b"]))
            trial = wavefunctions_unrestricted.upt2ccsd(
                norb, nelec_sp, 
                n_batch=options["n_batch"], 
                nchol_chunk=nchol_chunk,
                mix_precision=options["mix_precision"],
                )
            
            if "bar" in options["trial"]:
                trial = wavefunctions_unrestricted.upt2ccsd_bar(
                    norb, nelec_sp, 
                    n_batch=options["n_batch"], 
                    nchol_chunk=nchol_chunk,
                    mix_precision=options["mix_precision"],
                    )
                wave_data['mo_ta'] = None
                wave_data['mo_tb'] = None
                t1a, t1b = wave_data["t1a"], wave_data["t1b"]
                t1a_full = np.zeros((norb, norb), dtype=np.float64)
                t1b_full = np.zeros((norb, norb), dtype=np.float64)
                t1a_full[:nocc_a, nocc_a:] = t1a
                t1b_full[:nocc_b, nocc_b:] = t1b
                wave_data['exp_t1a'] = jsp.linalg.expm(t1a_full)
                wave_data['exp_mt1a'] = jsp.linalg.expm(-t1a_full)
                wave_data['exp_t1b'] = jsp.linalg.expm(t1b_full)
                wave_data['exp_mt1b'] = jsp.linalg.expm(-t1b_full)
            
            elif "cisd" in options["trial"]:
                wave_data = load_cc_amplitude(wave_data, amp_file)
                wave_data = load_ci_amplitude(wave_data, amp_file)
                wave_data['mo_ta'] = None
                wave_data['mo_tb'] = None
                t1a, t1b = wave_data["t1a"], wave_data["t1b"]
                t1a_full = np.zeros((norb, norb), dtype=np.float64)
                t1b_full = np.zeros((norb, norb), dtype=np.float64)
                t1a_full[:nocc_a, nocc_a:] = t1a
                t1b_full[:nocc_b, nocc_b:] = t1b
                wave_data['exp_t1a'] = jsp.linalg.expm(t1a_full)
                wave_data['exp_mt1a'] = jsp.linalg.expm(-t1a_full)
                wave_data['exp_t1b'] = jsp.linalg.expm(t1b_full)
                wave_data['exp_mt1b'] = jsp.linalg.expm(-t1b_full)
                trial = wavefunctions_unrestricted.upt2ccsd_cisd(
                    norb, nelec_sp, 
                    n_batch=options["n_batch"], 
                    nchol_chunk=nchol_chunk,
                    mix_precision=options["mix_precision"],
                    )
                
            if "ad" in options["trial"]:
                trial = wavefunctions_unrestricted.upt2ccsd_ad(
                    norb, nelec_sp, n_batch=options["n_batch"])
                wave_data["rot_t2aa"] = oe.contract('ik,jl,kalb->iajb',
                                                    wave_data['mo_ta'][:nocc_a,:nocc_a].T,
                                                    wave_data['mo_ta'][:nocc_a,:nocc_a].T,
                                                    wave_data["t2aa"], 
                                                    backend='jax')
                wave_data["rot_t2ab"] = oe.contract('ik,jl,kalb->iajb',
                                                    wave_data['mo_ta'][:nocc_a,:nocc_a].T,
                                                    wave_data['mo_tb'][:nocc_b,:nocc_b].T,
                                                    wave_data["t2ab"], 
                                                    backend='jax')
                wave_data["rot_t2bb"] = oe.contract('ik,jl,kalb->iajb',
                                                    wave_data['mo_tb'][:nocc_b,:nocc_b].T,
                                                    wave_data['mo_tb'][:nocc_b,:nocc_b].T,
                                                    wave_data["t2bb"], 
                                                    backend='jax')
            if "eff" in options["trial"]:
                trial = wavefunctions_unrestricted.upt2ccsd_eff(
                    norb, nelec_sp, n_batch=options["n_batch"])
                wave_data["rot_t2aa"] = oe.contract('ik,jl,kalb->iajb',
                                                    wave_data['mo_ta'][:nocc_a,:nocc_a].T,
                                                    wave_data['mo_ta'][:nocc_a,:nocc_a].T,
                                                    wave_data["t2aa"], 
                                                    backend='jax')
                wave_data["rot_t2ab"] = oe.contract('ik,jl,kalb->iajb',
                                                    wave_data['mo_ta'][:nocc_a,:nocc_a].T,
                                                    wave_data['mo_tb'][:nocc_b,:nocc_b].T,
                                                    wave_data["t2ab"], 
                                                    backend='jax')
                wave_data["rot_t2bb"] = oe.contract('ik,jl,kalb->iajb',
                                                    wave_data['mo_tb'][:nocc_b,:nocc_b].T,
                                                    wave_data['mo_tb'][:nocc_b,:nocc_b].T,
                                                    wave_data["t2bb"], 
                                                    backend='jax')

        elif options["trial"] == "ustoccsd2":
            wave_data = load_cc_amplitude(wave_data, amp_file)
            (wave_data['mo_ta'], wave_data['mo_tb']) = slater_tools.thouless(
                wave_data['mo_coeff'], (wave_data["t1a"], wave_data["t1b"]))
            wave_data['tau'] = t2_tools.decompose_t2((wave_data["t2aa"],
                                                      wave_data["t2ab"],
                                                      wave_data["t2bb"]))
            trial = wavefunctions_unrestricted.ustoccsd2(
                norb,
                nelec_sp,
                n_batch = options["n_batch"],
                nslater = options['nslater']
                )
    
    return trial, wave_data

def get_propagator(options):
    
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
        
    return prop

def get_sampler(options, nchol):

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
    
    return sampler

def init_afqmc(options=None,
               option_file="options.bin",
               amp_file="amplitudes.npz",
               chol_file="FCIDUMP_chol"):
    
    options = get_qmc_options(options, option_file)

    print("\nLoad system from Integral File")


    h0, h1, chol, ms, nelec_sp, norb, spin_type = load_chol(chol_file)
    # print(h1)

    ham, ham_data, nchol = get_hamiltonian(h0, h1, chol, norb)

    nchol_chunk = cholesky.chunk_chol(
        chol, options["nchol_chunk"], options["max_memory"]/options["n_walkers"])
    
    trial, wave_data = get_wavefunction(
        spin_type, norb, nelec_sp, nchol_chunk, options, amp_file)

    # wave_data = {}

    # if spin_type == "restricted":
    #     wave_data["mo_coeff"] = jnp.eye(norb)[:, : nelec_sp[0]]
    #     if options["trial"] == "rhf":
    #         trial = wavefunctions_restricted.rhf(norb, nelec_sp, 
    #                                              n_batch=options["n_batch"],
    #                                              nchol_chunk=options["nchol_chunk"],
    #                                              )

    #     elif "cisd" in options["trial"]:
    #         wave_data = load_ci_amplitude(wave_data, amp_file)
    #         trial = wavefunctions_restricted.cisd(norb, nelec_sp, 
    #                                                 n_batch=options["n_batch"]
    #                                                 )

    #     elif options["trial"] == "cid":
    #         wave_data = load_ci_amplitude(wave_data, amp_file)
    #         trial = wavefunctions_restricted.cid(norb, nelec_sp, n_batch=options["n_batch"])
            
    #     elif options["trial"] == "ptccsd":
    #         wave_data = load_cc_amplitude(wave_data, amp_file)
    #         trial = wavefunctions_restricted.ptccsd(norb, nelec_sp, n_batch=options["n_batch"])
    #         if "ad" in options["trial"]:
    #             trial = wavefunctions_restricted.ptccsd_ad(norb, nelec_sp, n_batch=options["n_batch"])
        
    #     elif options["trial"] == "ptccd":
    #         wave_data = load_cc_amplitude(wave_data, amp_file)
    #         trial = wavefunctions_restricted.ptccd(norb, nelec_sp, n_batch=options["n_batch"])

    #     elif "pt2ccsd" in options["trial"]:
    #         wave_data = load_cc_amplitude(wave_data, amp_file)
    #         wave_data["mo_t"] = slater_tools.thouless(wave_data["mo_coeff"], wave_data["t1"])
    #         trial = wavefunctions_restricted.pt2ccsd(norb, nelec_sp, 
    #                                                  n_batch=options["n_batch"],
    #                                                  nchol_chunk=options["nchol_chunk"], 
    #                                                  mix_precision=options["mix_precision"],
    #                                                  )
    #         if "ad" in options["trial"]:
    #             nocc = nelec_sp[0]
    #             rot_t2 = oe.contract('il,jk,lakb->iajb',
    #                                  wave_data["mo_t"][:nocc,:nocc].T,
    #                                  wave_data["mo_t"][:nocc,:nocc].T,
    #                                  wave_data["t2"], 
    #                                  backend='jax')
    #             wave_data['rot_t2'] = rot_t2
    #             trial = wavefunctions_restricted.pt2ccsd_ad(norb, nelec_sp, 
    #                                                         n_batch=options["n_batch"])

    #     elif "stoccsd" in options["trial"]:
    #         wave_data = load_cc_amplitude(wave_data, amp_file)
    #         wave_data['mo_t'] = slater_tools.thouless(wave_data['mo_coeff'], wave_data["t1"])
    #         wave_data['tau'] = t2_tools.decompose_t2(wave_data["t2"])

    #         if "2" in options["trial"]:
    #             trial = wavefunctions_restricted.stoccsd2(
    #                 norb,
    #                 nelec_sp,
    #                 n_batch = options["n_batch"],
    #                 nslater = options['nslater']
    #                 )
    #         else:
    #             trial = wavefunctions_restricted.stoccsd(
    #                 norb,
    #                 nelec_sp,
    #                 n_batch = options["n_batch"],
    #                 nslater = options['nslater']
    #                 )
                    
    
    # elif spin_type == "unrestricted":
    #     nocc_a, nocc_b = nelec_sp
    #     wave_data["mo_coeff"] = (jnp.eye(norb)[:,:nocc_a],
    #                              jnp.eye(norb)[:,:nocc_b])

    #     if options["trial"] == "uhf":
    #         trial = wavefunctions_unrestricted.uhf(norb, nelec_sp, n_batch=options["n_batch"])

    #     elif options["trial"] == "ucisd":
    #         wave_data = load_ci_amplitude(wave_data, amp_file)
    #         trial = wavefunctions_unrestricted.ucisd(norb, nelec_sp, n_batch=options["n_batch"])

    #     elif options["trial"] == "uptccsd":
    #         wave_data = load_cc_amplitude(wave_data, amp_file)
    #         trial = wavefunctions_unrestricted.uptccsd(norb, nelec_sp, n_batch = options["n_batch"])
    #         if "ad" in options["trial"]:
    #             trial = wavefunctions_unrestricted.uptccsd_ad(
    #                 norb, nelec_sp, n_batch=options["n_batch"])
                

    #             wave_data["rot_t1a"] = wave_data['mo_coeff'][0][:nocc_a,:nocc_a].T @ wave_data["t1a"]
    #             wave_data["rot_t1b"] = wave_data['mo_coeff'][1][:nocc_b,:nocc_b].T @ wave_data["t1b"]
                
    #             wave_data["rot_t2aa"] = oe.contract('ik,jl,kalb->iajb',
    #                                                 wave_data['mo_coeff'][0][:nocc_a,:nocc_a].T,
    #                                                 wave_data['mo_coeff'][1][:nocc_a,:nocc_a].T,
    #                                                 wave_data["t2aa"], 
    #                                                 backend='jax')
    #             wave_data["rot_t2ab"] = oe.contract('ik,jl,kalb->iajb',
    #                                                 wave_data['mo_coeff'][0][:nocc_a,:nocc_a].T,
    #                                                 wave_data['mo_coeff'][1][:nocc_b,:nocc_b].T,
    #                                                 wave_data["t2ab"], 
    #                                                 backend='jax')
    #             wave_data["rot_t2bb"] = oe.contract('ik,jl,kalb->iajb',
    #                                                 wave_data['mo_coeff'][0][:nocc_b,:nocc_b].T,
    #                                                 wave_data['mo_coeff'][1][:nocc_b,:nocc_b].T,
    #                                                 wave_data["t2bb"], 
    #                                                 backend='jax')

    #     elif "upt2ccsd" in options["trial"]:
    #         wave_data = load_cc_amplitude(wave_data, amp_file)
    #         (wave_data['mo_ta'], wave_data['mo_tb']) = slater_tools.thouless(
    #             wave_data['mo_coeff'], (wave_data["t1a"], wave_data["t1b"]))
    #         trial = wavefunctions_unrestricted.upt2ccsd(
    #             norb, nelec_sp, 
    #             n_batch=options["n_batch"], 
    #             nchol_chunk=options["nchol_chunk"],
    #             mix_precision=options["mix_precision"],
    #             )
    #         if "bar" in options["trial"]:
    #             trial = wavefunctions_unrestricted.upt2ccsd_bar(
    #                 norb, nelec_sp, 
    #                 n_batch=options["n_batch"], 
    #                 nchol_chunk=options["nchol_chunk"],
    #                 mix_precision=options["mix_precision"],
    #                 )
    #             wave_data['mo_ta'] = None
    #             wave_data['mo_tb'] = None
    #             t1a, t1b = wave_data["t1a"], wave_data["t1b"]
    #             t1a_full = np.zeros((norb, norb), dtype=np.float64)
    #             t1b_full = np.zeros((norb, norb), dtype=np.float64)
    #             t1a_full[:nocc_a, nocc_a:] = t1a
    #             t1b_full[:nocc_b, nocc_b:] = t1b
    #             wave_data['exp_t1a'] = jsp.linalg.expm(t1a_full)
    #             wave_data['exp_mt1a'] = jsp.linalg.expm(-t1a_full)
    #             wave_data['exp_t1b'] = jsp.linalg.expm(t1b_full)
    #             wave_data['exp_mt1b'] = jsp.linalg.expm(-t1b_full)
    #         if "ad" in options["trial"]:
    #             trial = wavefunctions_unrestricted.upt2ccsd_ad(
    #                 norb, nelec_sp, n_batch=options["n_batch"])
    #             wave_data["rot_t2aa"] = oe.contract('ik,jl,kalb->iajb',
    #                                                 wave_data['mo_ta'][:nocc_a,:nocc_a].T,
    #                                                 wave_data['mo_ta'][:nocc_a,:nocc_a].T,
    #                                                 wave_data["t2aa"], 
    #                                                 backend='jax')
    #             wave_data["rot_t2ab"] = oe.contract('ik,jl,kalb->iajb',
    #                                                 wave_data['mo_ta'][:nocc_a,:nocc_a].T,
    #                                                 wave_data['mo_tb'][:nocc_b,:nocc_b].T,
    #                                                 wave_data["t2ab"], 
    #                                                 backend='jax')
    #             wave_data["rot_t2bb"] = oe.contract('ik,jl,kalb->iajb',
    #                                                 wave_data['mo_tb'][:nocc_b,:nocc_b].T,
    #                                                 wave_data['mo_tb'][:nocc_b,:nocc_b].T,
    #                                                 wave_data["t2bb"], 
    #                                                 backend='jax')
    #         if "eff" in options["trial"]:
    #             trial = wavefunctions_unrestricted.upt2ccsd_eff(
    #                 norb, nelec_sp, n_batch=options["n_batch"])
    #             wave_data["rot_t2aa"] = oe.contract('ik,jl,kalb->iajb',
    #                                                 wave_data['mo_ta'][:nocc_a,:nocc_a].T,
    #                                                 wave_data['mo_ta'][:nocc_a,:nocc_a].T,
    #                                                 wave_data["t2aa"], 
    #                                                 backend='jax')
    #             wave_data["rot_t2ab"] = oe.contract('ik,jl,kalb->iajb',
    #                                                 wave_data['mo_ta'][:nocc_a,:nocc_a].T,
    #                                                 wave_data['mo_tb'][:nocc_b,:nocc_b].T,
    #                                                 wave_data["t2ab"], 
    #                                                 backend='jax')
    #             wave_data["rot_t2bb"] = oe.contract('ik,jl,kalb->iajb',
    #                                                 wave_data['mo_tb'][:nocc_b,:nocc_b].T,
    #                                                 wave_data['mo_tb'][:nocc_b,:nocc_b].T,
    #                                                 wave_data["t2bb"], 
    #                                                 backend='jax')

    #     elif options["trial"] == "ustoccsd2":
    #         wave_data = load_cc_amplitude(wave_data, amp_file)
    #         (wave_data['mo_ta'], wave_data['mo_tb']) = slater_tools.thouless(
    #             wave_data['mo_coeff'], (wave_data["t1a"], wave_data["t1b"]))
    #         wave_data['tau'] = t2_tools.decompose_t2((wave_data["t2aa"],
    #                                                   wave_data["t2ab"],
    #                                                   wave_data["t2bb"]))
    #         trial = wavefunctions_unrestricted.ustoccsd2(
    #             norb,
    #             nelec_sp,
    #             n_batch = options["n_batch"],
    #             nslater = options['nslater']
    #             )
    

    # if options["walker_type"] == "rhf":
    #     prop = propagation.propagator_restricted(
    #             options["dt"], 
    #             options["n_walkers"], 
    #             options["n_exp_terms"],
    #             options["n_batch"]
    #         )

    # elif options["walker_type"] == "uhf":
    #     prop = propagation.propagator_unrestricted(
    #             options["dt"],
    #             options["n_walkers"],
    #             options["n_exp_terms"],
    #             options["n_batch"],
    #         )

    # if  'pt' in options['trial'] and 'cc' in options['trial']:
    #     if 'pt2' in options['trial']:
    #         sampler = sampling.sampler_pt2(
    #             options["n_prop_steps"],
    #             options["n_blocks"],
    #             nchol,)
    #     else:
    #         sampler = sampling.sampler_pt(
    #             options["n_prop_steps"],
    #             options["n_blocks"],
    #             nchol,)
            
    # elif 'stoccsd' in options['trial']:
    #     if '2' in options['trial']:
    #         sampler = sampling.sampler_stoccsd2(
    #             options["n_prop_steps"],
    #             options["n_blocks"],
    #             nchol,)
    #     else:
    #         sampler = sampling.sampler_stoccsd(
    #             options["n_prop_steps"],
    #             options["n_blocks"],
    #             nchol,)
            
    # else:
    #     sampler = sampling.sampler(
    #         options["n_prop_steps"],
    #         options["n_blocks"],
    #         nchol,)

    
    # if options["free_projection"]:
    #     if 'pt2' not in options["trial"]:
    #         sampler = fp_sampling.fp_sampler(
    #                 options["n_prop_steps"],
    #                 options["n_eql_blocks"],
    #                 options["n_trj"],
    #                 nchol,
    #                 )
    #     elif 'pt2' in options["trial"]:
    #         sampler = fp_sampling.fp_sampler_pt2(
    #                 options["n_prop_steps"],
    #                 options["n_eql_blocks"],
    #                 options["n_trj"],
    #                 nchol,
    #                 )

    prop = get_propagator(options)
    sampler = get_sampler(options, nchol)

    print("\nQMC System")
    print(f"Number of electrons: {nelec_sp}")
    print(f"Spin Multiplicity:   {ms}")
    print(f"Number of orbitals:  {norb}")
    print(f"Number of Chol:      {nchol}")

    return ham_data, ham, prop, trial, wave_data, sampler, options


def init_afqmc_exp(
        options=None,
        option_file="options.bin",
        amp_file="amplitudes.npz",
        chol_file="FCIDUMP_chol"
        ):
    from .wavefunctions import wfn_exp, rhf_wfn, uhf_wfn, rcisd_wfn, ucisd_wfn, rpt2ccsd_wfn, upt2ccsd_wfn
    
    options = get_qmc_options(options, option_file)

    print("\nLoad system from Integral File")

    with h5py.File(chol_file, "r") as fh5:
        [nelec, norb, ms] = fh5["header"]
        spin_type = fh5["spin_type"][()]
        h0 = jnp.array(fh5.get("energy_core"))
        h1 = jnp.array(fh5.get("hcore"))
        chol = jnp.array(fh5.get("chol"))
    
    if isinstance(spin_type, bytes):
        spin_type = spin_type.decode()

    assert spin_type in ["restricted", "unrestricted"]

    if spin_type == 'restricted':
        h1 = jnp.array(h1).reshape(norb, norb)
        chol = jnp.array(chol).reshape(-1, norb, norb)

    elif spin_type == 'unrestricted':
        h1 = jnp.array(h1).reshape(2, norb, norb)
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
        ham_data["h1"] = (jnp.array(h1), jnp.array(h1))
        nchol = chol.shape[0]
        ham_data["chol"] = jnp.array(chol.reshape(chol.shape[0], -1))
    elif spin_type == 'unrestricted':
        ham_data["h1"] = (jnp.array(h1[0] + h1[0].T) / 2, 
                          jnp.array(h1[1] + h1[0].T) / 2)
        nchol = chol[0].shape[0]
        ham_data["chol"] = (jnp.array(chol[0].reshape(chol[0].shape[0], -1)),
                            jnp.array(chol[1].reshape(chol[1].shape[0], -1)))

    options["nchol_chunk"] = cholesky.chunk_chol(
        chol, options["nchol_chunk"], options["max_memory"]/options["n_walkers"])

    print("\nQMC System")
    print(f"Number of electrons: {nelec_sp}")
    print(f"Spin Multiplicity:   {ms}")
    print(f"Number of orbitals:  {norb}")
    print(f"Number of Chol:      {nchol}")

    print("\nQMC Parameters")
    for op in options:
        if options[op] is not None:
            print(f"{str(op):<15s} - {str(options[op]):>10s}")

    wave_data = {}
    if "ci" in options["trial"] or "ci" in options["guide"]:
        wave_data = load_ci_amplitude(wave_data, amp_file)
    if "cc" in options["trial"] or "cc" in options["guide"]:
        wave_data = load_cc_amplitude(wave_data, amp_file)

    if spin_type == "restricted":
        wave_data["mo_coeff"] = jnp.eye(norb)[:, : nelec_sp[0]]
        wave_data["rdm1"] = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T] * 2)
        # guide
        if options["guide"] == "rhf":
            guide_overlap_fn = rhf_wfn.calc_overlap
            guide_force_bias_fn = rhf_wfn.calc_rot_force_bias
        if options["guide"] == "rcisd":
            guide_overlap_fn = rcisd_wfn.calc_overlap
            guide_force_bias_fn = rcisd_wfn.calc_force_bias

        # trial
        if options["trial"] == "rhf":
            trial_overlap_fn = rhf_wfn.calc_overlap
            trial_energy_fn = rhf_wfn.calc_rot_energy
            trial_intermediate_fn = rhf_wfn.calc_intermediate
            energy_formula_fn = rhf_wfn.energy_formula
        elif options["trial"] == "rcisd":
            trial_overlap_fn = rcisd_wfn.calc_overlap
            trial_energy_fn = rcisd_wfn.calc_energy
            trial_intermediate_fn = rcisd_wfn.calc_intermediate
            energy_formula_fn = rcisd_wfn.energy_formula
        elif options["trial"] == "rpt2ccsd":
            trial_overlap_fn = rpt2ccsd_wfn.calc_overlap
            trial_energy_fn = rpt2ccsd_wfn.calc_energy
            trial_intermediate_fn = rpt2ccsd_wfn.calc_intermediate
            energy_formula_fn = rpt2ccsd_wfn.energy_formula

    elif spin_type == "unrestricted":
        nocc_a, nocc_b = nelec_sp
        wave_data["mo_coeff"] = (jnp.eye(norb)[:,:nocc_a],
                                 jnp.eye(norb)[:,:nocc_b])
        wave_data["rdm1"] = (jnp.array([wave_data["mo_coeff"][0] @ wave_data["mo_coeff"][0].T]),
                             jnp.array([wave_data["mo_coeff"][1] @ wave_data["mo_coeff"][1].T]))
        # guide
        if options["guide"] == "uhf":
            guide_overlap_fn = uhf_wfn.calc_overlap
            guide_force_bias_fn = uhf_wfn.calc_force_bias
        if options["guide"] == "ucisd":
            guide_overlap_fn = ucisd_wfn.calc_overlap
            guide_force_bias_fn = ucisd_wfn.calc_force_bias

        # trial
        if options["trial"] == "uhf":
            trial_overlap_fn = uhf_wfn.calc_overlap
            trial_energy_fn = uhf_wfn.calc_rot_energy
            trial_intermediate_fn = uhf_wfn.calc_intermediate
            energy_formula_fn = uhf_wfn.energy_formula
        elif options["trial"] == "ucisd":
            trial_overlap_fn = ucisd_wfn.calc_overlap
            trial_energy_fn = ucisd_wfn.calc_energy
            trial_intermediate_fn = ucisd_wfn.calc_intermediate
            energy_formula_fn = ucisd_wfn.energy_formula
        elif options["trial"] == "upt2ccsd":
            trial_overlap_fn = upt2ccsd_wfn.calc_overlap
            trial_energy_fn = upt2ccsd_wfn.calc_energy
            trial_intermediate_fn = upt2ccsd_wfn.calc_intermediate
            energy_formula_fn = upt2ccsd_wfn.energy_formula
        elif options["trial"] == "upt2ccsd_bar":
            trial_overlap_fn = upt2ccsd_wfn.calc_overlap_bar
            trial_energy_fn = upt2ccsd_wfn.calc_energy_bar
            trial_intermediate_fn = upt2ccsd_wfn.calc_intermediate_bar
            energy_formula_fn = upt2ccsd_wfn.energy_formula

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

    wave = wfn_exp.wfn(    
        guide_overlap_fn=guide_overlap_fn,
        guide_force_bias_fn=guide_force_bias_fn,
        trial_overlap_fn=trial_overlap_fn,
        trial_energy_fn=trial_energy_fn,
        trial_intermediate_fn=trial_intermediate_fn,
        energy_formula_fn=energy_formula_fn,
        nelec=nelec_sp,
        norb=norb,
        nchol=nchol,
        nchol_chunk=options["nchol_chunk"],
        )

    sampler = sampling.sampler_exp(
        n_prop_steps=options["n_prop_steps"],
        n_blocks=options["n_blocks"],
        n_chol=nchol,
        )
    
    return ham, prop, wave, ham_data, wave_data, sampler, options