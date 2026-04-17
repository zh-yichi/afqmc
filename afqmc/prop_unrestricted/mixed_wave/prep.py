import pickle
import h5py
import numpy as np
import jax.numpy as jnp

from afqmc import hamiltonian
from afqmc.prop_unrestricted import propagation
from afqmc.prop_unrestricted.mixed_wave import sampling
from afqmc.prop_unrestricted.mixed_wave import wavefunctions_restricted
from afqmc.prop_unrestricted.mixed_wave import wavefunctions_unrestricted
from afqmc.prop_unrestricted.prep import prep_afqmc
from functools import partial
print = partial(print, flush=True)

prep_afqmc = prep_afqmc

def _prep_afqmc(options=None,
                option_file="options.bin",
                amp_file="amplitudes.npz",
                chol_file="FCIDUMP_chol"
                ):
    
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
    options["n_ene_blocks"] = options.get("n_ene_blocks", 50)
    options["n_sr_blocks"] = options.get("n_sr_blocks", 1)
    options["n_blocks"] = options.get("n_blocks", 50)
    options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))
    options["n_eql"] = options.get("n_eql", 1)
    options["walker_type"] = options.get("walker_type", "rhf")
    options["trial"] = options.get("trial", None)
    options["n_batch"] = options.get("n_batch", 1)
    options["max_error"] = options.get("max_error", 1e-3)
    options["nslater"] = options.get("nslater", 100)

    if 'u' not in options['trial']:
        spin_type = 'restricted'
    elif 'u' in options['trial']:
        spin_type = 'unrestricted'

    if spin_type == 'restricted':
        with h5py.File(chol_file, "r") as fh5:
            [nelec, nmo, ms, nchol] = fh5["header"]
            h0 = jnp.array(fh5.get("energy_core"))
            h1 = jnp.array(fh5.get("hcore")).reshape(nmo, nmo)
            chol = jnp.array(fh5.get("chol")).reshape(-1, nmo, nmo)
            h1_mod = jnp.array(fh5.get("hcore_mod")).reshape(nmo, nmo)
    elif spin_type == 'unrestricted':
        with h5py.File(chol_file, "r") as fh5:
            [nelec, nmo, ms, nchol] = fh5["header"]
            h0 = jnp.array(fh5.get("energy_core"))
            h1 = jnp.array(fh5.get("hcore")).reshape(2, nmo, nmo)
            chol = jnp.array(fh5.get("chol")).reshape(2, -1, nmo, nmo)
            h1_mod = jnp.array(fh5.get("hcore_mod")).reshape(2, nmo, nmo)

    assert type(ms) is np.int64
    assert type(nelec) is np.int64
    assert type(nmo) is np.int64
    assert type(nchol) is np.int64
    ms, nelec, nmo, nchol = int(ms), int(nelec), int(nmo), int(nchol)
    nelec_sp = ((nelec + abs(ms)) // 2, (nelec - abs(ms)) // 2)
    norb = nmo

    ham = hamiltonian.hamiltonian(nmo)
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

    wave_data = {}
    mo_coeff = jnp.array([np.eye(norb), np.eye(norb)])

    if spin_type == "restricted":
        if "stoccsd" in options["trial"]:
            if "2" in options["trial"]:
                trial = wavefunctions_restricted.stoccsd2(
                    norb,
                    nelec_sp,
                    n_batch = options["n_batch"],
                    nslater = options['nslater']
                    )
                    
                sampler = sampling.sampler_stoccsd2(
                    n_prop_steps = options["n_prop_steps"],
                    n_sr_blocks = options["n_sr_blocks"],
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
                    n_sr_blocks = options["n_sr_blocks"],
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
        if options["trial"] == "ustoccsd2":
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
                n_sr_blocks = options["n_sr_blocks"],
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
            options["n_batch"]
        )

    print(f"nelec: {nelec_sp}")
    print(f"norb: {norb}")
    print(f"nchol: {nchol}")
    for op in options:
        if options[op] is not None:
            print(f"{op}: {options[op]}")

    return ham_data, ham, prop, trial, wave_data, sampler, options