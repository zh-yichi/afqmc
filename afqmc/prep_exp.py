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

from . import hamiltonian, cholesky, walker_tools, t2_tools
from . import propagation, sampling_exp

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


def load_ci_amplitude(wave_data=None, amp_file="amplitudes.npz"):
    if wave_data is None:
        wave_data = {}
    t1, t2 = t2_tools.read_cc_amps(amp_file)
    ci1, ci2 = t2_tools.cc2ci(t1, t2)
    wave_data["ci1"] = ci1
    wave_data["ci2"] = ci2
    return wave_data

def load_cc_amplitude(wave_data=None, amp_file="amplitudes.npz"):
    if wave_data is None:
        wave_data = {}
    t1, t2 = t2_tools.read_cc_amps(amp_file)

    wave_data["t1"] = t1
    wave_data["t2"] = t2

    return wave_data

def init_prop_data(
    wave,
    wave_data,
    ham_data,
    options
    ):

    print(f"\nInitalize QMC walkers by {options["init_walkers"]}")
    prop_data = {}
    prop_data["n_killed_walkers"] = 0
    prop_data["key"] = random.PRNGKey(options["seed"])

    prop_data["weights"] = jnp.ones(options["n_walkers"], dtype=jnp.float64)

    if options["init_walkers"] == "hf":
        walkers0 = walker_tools.replicate_walker(wave_data["mo_coeff"], options["n_walkers"])
        g_olps0 = wave.calc_overlap(walkers0, wave_data)
    elif options["init_walkers"] == "stocc":
        # starting with different walkers need to multiply the overlap update factor
        walkers00 = walker_tools.replicate_walker(wave_data["mo_coeff"], options["n_walkers"])
        g_olps00 = wave.calc_overlap(walkers00, wave_data)
        walkers0, prop_data = walker_tools.get_ccsd_walkers(
            prop_data, wave_data, options["n_walkers"], options["walker_type"]
        )
        g_olps0 = wave.calc_overlap(walkers0, wave_data)
        prop_data["weights"] *= jnp.real(g_olps0 / g_olps00)

    t_olps0 = wave.calc_trial_overlap(walkers0, wave_data)
    weighps = prop_data["weights"] * t_olps0 / g_olps0
    t_enes = wave.calc_energy(walkers0, ham_data, wave_data)
    init_w, init_e, err \
            = wave.energy_formula(weighps, t_enes, ham_data)

    prop_data["walkers"] = walkers0
    prop_data["overlaps"] = g_olps0
    prop_data["e_estimate"] = init_e
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

    return prop_data, init_w, init_e

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
    options["n_blocks"] = options.get("n_blocks", 300)
    options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))
    options["eql_time"] = options.get("eql_time", 20)
    options["trial"] = options.get("trial", None)
    if "u" not in options["trial"]:
        options["walker_type"] = options.get("walker_type", "rhf")
    elif "u" in options["trial"]:
        options["walker_type"] = options.get("walker_type", "uhf")
    if "stocc" not in options["trial"]:
        options["n_slater"] = options.get("n_slater", 10)
    options["nwalker_batch"] = options.get("nwalker_batch", 1)
    options["max_error"] = options.get("max_error", 0.0)
    options["nchol_chunk"] = options.get("nchol_chunk", 100)
    options["max_memory"] = options.get("max_memory", 2000) # MB
    options["mix_precision"] = options.get("mix_precision", True)
    options["init_walkers"] = options.get("init_walkers", "hf")

    print("\nQMC Parameters")
    for op in options:
        val = options[op]
        if val is not None:
            val_str = f"{val:.2g}" if isinstance(val, float) else str(val)
            print(f"{str(op):<15s} - {val_str:>10s}")

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

    # ham = hamiltonian.hamiltonian(norb)
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
        
    return ham_data, nchol

def init_afqmc(
        options=None,
        option_file="options.bin",
        amp_file="amplitudes.npz",
        chol_file="FCIDUMP_chol"
        ):
    from .wavefunctions import (wfn_exp, 
                                rhf_wfn, uhf_wfn, 
                                rms_wfn, ums_wfn, 
                                rcisd_wfn, ucisd_wfn, 
                                rpt2ccsd_wfn, upt2ccsd_wfn,
                                rstoccsd_wfn, ustoccsd_wfn)
    
    options = get_qmc_options(options, option_file)

    h0, h1, chol, ms, nelec_sp, norb, spin_type = load_chol(chol_file)

    ham_data, nchol = get_hamiltonian(h0, h1, chol, norb)

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
            guide_overlap_fn = rhf_wfn.overlap
            guide_force_bias_fn = rhf_wfn.rot_force_bias
            guide_energy_fn = rhf_wfn.rot_energy
            guide_intermediate_fn = rhf_wfn.build_intermediate
        if options["guide"] == "rcisd":
            guide_overlap_fn = rcisd_wfn.overlap
            guide_force_bias_fn = rcisd_wfn.force_bias
            guide_energy_fn = rcisd_wfn.energy
            guide_intermediate_fn = rcisd_wfn.build_intermediate
        if options["guide"] == "rstoccsd":
            guide_overlap_fn = rstoccsd_wfn.overlap
            guide_force_bias_fn = rstoccsd_wfn.force_bias
            guide_energy_fn = rstoccsd_wfn.rot_energy
            guide_intermediate_fn = rstoccsd_wfn.build_intermediate
            wave_data['t2_thresh'] = options['t2_thresh']
            wave_data['n_slater'] = options['n_slater']
            wave_data['seed'] = options['seed'] + 1

        # trial
        if options["trial"] == "rhf":
            trial_overlap_fn = rhf_wfn.overlap
            trial_energy_fn = rhf_wfn.rot_energy
            trial_intermediate_fn = rhf_wfn.build_intermediate
            energy_formula_fn = rhf_wfn.energy_formula
        elif options["trial"] == "rcisd":
            trial_overlap_fn = rcisd_wfn.overlap
            trial_energy_fn = rcisd_wfn.energy
            trial_intermediate_fn = rcisd_wfn.build_intermediate
            energy_formula_fn = rcisd_wfn.energy_formula
        elif options["trial"] == "rpt2ccsd":
            trial_overlap_fn = rpt2ccsd_wfn.overlap
            trial_energy_fn = rpt2ccsd_wfn.energy
            trial_intermediate_fn = rpt2ccsd_wfn.build_intermediate
            energy_formula_fn = rpt2ccsd_wfn.energy_formula
        elif options["trial"] == "rstoccsd":
            trial_overlap_fn = rstoccsd_wfn.overlap
            trial_energy_fn = rstoccsd_wfn.energy
            trial_intermediate_fn = rstoccsd_wfn.build_intermediate
            energy_formula_fn = rstoccsd_wfn.energy_formula
            wave_data['t2_thresh'] = options['t2_thresh']
            wave_data['n_slater'] = options['n_slaters']
            wave_data['seed'] = options['seed'] + 1

    elif spin_type == "unrestricted":
        nocc_a, nocc_b = nelec_sp
        wave_data["mo_coeff"] = (jnp.eye(norb)[:,:nocc_a],
                                 jnp.eye(norb)[:,:nocc_b])
        wave_data["rdm1"] = (jnp.array([wave_data["mo_coeff"][0] @ wave_data["mo_coeff"][0].T]),
                             jnp.array([wave_data["mo_coeff"][1] @ wave_data["mo_coeff"][1].T]))
        # guide
        if options["guide"] == "uhf":
            guide_overlap_fn = uhf_wfn.overlap
            guide_force_bias_fn = uhf_wfn.force_bias
        if options["guide"] == "ucisd":
            guide_overlap_fn = ucisd_wfn.overlap
            guide_force_bias_fn = ucisd_wfn.force_bias

        # trial
        if options["trial"] == "uhf":
            trial_overlap_fn = uhf_wfn.overlap
            trial_energy_fn = uhf_wfn.rot_energy
            trial_intermediate_fn = uhf_wfn.build_intermediate
            energy_formula_fn = uhf_wfn.energy_formula
        elif options["trial"] == "ucisd":
            trial_overlap_fn = ucisd_wfn.overlap
            trial_energy_fn = ucisd_wfn.energy
            trial_intermediate_fn = ucisd_wfn.build_intermediate
            energy_formula_fn = ucisd_wfn.energy_formula
        elif options["trial"] == "upt2ccsd":
            trial_overlap_fn = upt2ccsd_wfn.overlap
            trial_energy_fn = upt2ccsd_wfn.energy
            trial_intermediate_fn = upt2ccsd_wfn.build_intermediate
            energy_formula_fn = upt2ccsd_wfn.energy_formula
        elif options["trial"] == "upt2ccsd_bar":
            trial_overlap_fn = upt2ccsd_wfn.overlap_bar
            trial_energy_fn = upt2ccsd_wfn.energy_bar
            trial_intermediate_fn = upt2ccsd_wfn.build_intermediate_bar
            energy_formula_fn = upt2ccsd_wfn.energy_formula

    wave = wfn_exp.wfn(
        guide_overlap_fn=guide_overlap_fn,
        guide_force_bias_fn=guide_force_bias_fn,
        guide_energy_fn=guide_energy_fn,
        guide_intermediate_fn=guide_intermediate_fn,
        trial_overlap_fn=trial_overlap_fn,
        trial_energy_fn=trial_energy_fn,
        energy_formula_fn=energy_formula_fn,
        trial_intermediate_fn=trial_intermediate_fn,
        nelec=nelec_sp,
        norb=norb,
        nchol=nchol,
        nchol_chunk=options["nchol_chunk"],
        nwalker_batch=options["nwalker_batch"],
        mix_precision=options["mix_precision"],
        )

    if options["walker_type"] == "rhf":
        prop = propagation.propagator_restricted(
                options["dt"], 
                options["n_walkers"], 
                options["n_exp_terms"],
                options["nwalker_batch"]
            )
    elif options["walker_type"] == "uhf":
        prop = propagation.propagator_unrestricted(
                options["dt"],
                options["n_walkers"],
                options["n_exp_terms"],
                options["nwalker_batch"],
            )
    if "stocc" in options["guide"]:
        sampler = sampling_exp.stocc_sampler(
                n_walkers=options["n_walkers"],
                n_prop_steps=options["n_prop_steps"],
                n_blocks=options["n_blocks"],
                n_chol=nchol,
                n_slater=options["n_slater"],
                )
        prop = propagation.propagator_restricted_stocc(
                options["dt"], 
                options["n_walkers"], 
                options["n_exp_terms"],
                options["nwalker_batch"],
                n_slater = options["n_slater"], 
            )
    else:
        sampler = sampling_exp.sampler(
            n_walkers=options["n_walkers"],
            n_prop_steps=options["n_prop_steps"],
            n_blocks=options["n_blocks"],
            n_chol=nchol,
            )
    
    return prop, wave, ham_data, wave_data, sampler, options