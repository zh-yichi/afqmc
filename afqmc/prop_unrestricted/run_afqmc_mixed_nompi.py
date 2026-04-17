import time
import argparse
import numpy as np
from jax import random
from jax import numpy as jnp
from functools import partial
from ad_afqmc import config, stat_utils
from ad_afqmc.prop_unrestricted import prep, sampling

init_time = time.time()
print = partial(print, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    if args.use_gpu:
        config.afqmc_config["use_gpu"] = True

config.setup_jax()
ham_data, ham, prop, trial, wave_data, sampler, options = (prep._prep_afqmc())
init_walkers = None
trial_rdm1 = trial.get_rdm1(wave_data)
if "rdm1" not in wave_data:
    wave_data["rdm1"] = trial_rdm1
ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)

prop_data = prop.init_prop_data(trial, wave_data, ham_data, init_walkers)
if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError(
        "Initial overlaps are zero. Pass walkers with non-zero overlap."
    )

prop_data["key"] = random.PRNGKey(options["seed"])
wave_data['xtau'] = trial.get_xtau(wave_data, prop_data)
prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0

olp, e1, e2 = trial.calc_energy_mixed(prop_data['walkers'], ham_data, wave_data)
olp = jnp.real(olp)
e1 = jnp.real(e1)
e2 = jnp.real(e2)

e1_init = jnp.array(jnp.sum(e1) / prop.n_walkers)
e2_init = jnp.array(jnp.sum(e2) / prop.n_walkers)
prop_data["e_estimate"] = e1
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

print(f'# Propagating with {options["n_walkers"]} walkers')
print("# Equilibration sweeps:")
print("# AtomTime \t Energy_G \t Energy_T \t Energy_T_cr \t WallTime")
print(f"   {0:.2f} \t {e1_init:.6f} \t {e2_init:.6f} \t {time.time() - init_time:.2f}")

sampler_eq = sampling.sampler_mixed(
    n_prop_steps=50, 
    n_ene_blocks=5, 
    n_sr_blocks=10, 
    n_chol = sampler.n_chol
    )

block_time = prop.dt * sampler_eq.n_prop_steps * sampler_eq.n_ene_blocks * sampler_eq.n_sr_blocks

for n in range(options["n_eql"]):
    prop_data, (wt, wp, wp_cr, e1, e2, e2_cr) =\
        sampler_eq.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)
    # e1 is by the guide and e2 is by the trial
    # wp = oT/oG
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e1
    print(f"  {(n+1)*block_time:.2f} \t "
          f"  {e1:.6f} \t {e2:.6f} \t {e2_cr:.6f} \t"
          f"  {time.time() - init_time:.2f}")


print("# Sampling sweeps:")
print("# nBlock \t <Guide|H|AF> \t Error \t \t <Trial|H|AF> \t Error \t \t <Trial|H|AF>_cr \t Error \t  Walltime")

wt_sp = np.zeros(sampler.n_blocks,dtype="float64")
wp_sp = np.zeros(sampler.n_blocks,dtype="float64")
wpcr_sp = np.zeros(sampler.n_blocks,dtype="float64")
e1_sp = np.zeros(sampler.n_blocks,dtype="float64")
e2_sp = np.zeros(sampler.n_blocks,dtype="float64")
e2cr_sp = np.zeros(sampler.n_blocks,dtype="float64")
    
for n in range(sampler.n_blocks):
    prop_data, (wt, wp, wpcr, e1, e2, e2cr) =\
        sampler.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)

    wt_sp[n] = wt
    wp_sp[n] = wp
    wpcr_sp[n] = wpcr
    e1_sp[n] = e1
    e2_sp[n] = e2
    e2cr_sp[n] = e2cr

    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e1

    if (n+1) % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:

        wt = np.sum(wt_sp[:n+1])
        wp = np.sum(wp_sp[:n+1])
        wpcr = np.sum(wpcr_sp[:n+1])
        e1 = np.sum(wt_sp[:n+1] * e1_sp[:n+1]) / wt
        e2 = np.sum(wp_sp[:n+1] * e2_sp[:n+1]) / wp
        e2cr = np.sum(wpcr_sp[:n+1] * e2_sp[:n+1]) / wpcr

        e1_err = np.sqrt(np.sum(wt_sp[:n+1] * (e1_sp[:n+1] - e1)**2) / wt / (n+1))
        e2_err = np.sqrt(np.sum(wp_sp[:n+1] * (e2_sp[:n+1] - e2)**2) / wp / (n+1))
        e2cr_err = np.sqrt(np.sum(wpcr_sp[:n+1] * (e2cr_sp[:n+1] - e2cr)**2) / wpcr / (n+1))

        print(f"  {n:4d} \t \t "
              f"  {e1:.6f} \t {e1_err:.6f} \t"
              f"  {e2:.6f} \t {e2_err:.6f} \t"
              f"  {e2cr:.6f} \t {e2cr_err:.6f} \t"
              f"  {time.time() - init_time:.2f}")
        
        if e1_err < options["max_error"] and n > 20:
                break

nsamples = n + 1
wt_sp = wt_sp[:nsamples]
wp_sp = wp_sp[:nsamples]
wpcr_sp = wpcr_sp[:nsamples]
e1_sp = e1_sp[:nsamples]
e2_sp = e2_sp[:nsamples]
e2cr_sp = e2cr_sp[:nsamples]

samples_clean, idx \
    = stat_utils.reject_outliers(
        np.stack((wt_sp, 
                  wp_sp, 
                  wpcr_sp, 
                  e1_sp, 
                  e2_sp,
                  e2cr_sp
                  )).T, 3
                  )

print(f"# Number of outliers in post: {nsamples - samples_clean.shape[0]}")
    
wt_sp = samples_clean[:, 0]
wp_sp = samples_clean[:, 1]
wpcr_sp = samples_clean[:, 2]
e1_sp = samples_clean[:, 3]
e2_sp = samples_clean[:, 4]
e2cr_sp = samples_clean[:, 5]

wt = np.sum(wt_sp)
wp = np.sum(wp_sp)
wpcr = np.sum(wpcr_sp)
e1 = np.sum(wt_sp * e1_sp) / wt
e2 = np.sum(wp_sp * e2_sp) / wp
e2cr = np.sum(wpcr_sp * e2cr_sp) / wpcr

e1_err = np.sqrt(np.sum(wt_sp * (e1_sp - e1)**2) / wt / samples_clean.shape[0])
e2_err = np.sqrt(np.sum(wp_sp * (e2_sp - e2)**2) / wp / samples_clean.shape[0])
e2cr_err = np.sqrt(np.sum(wpcr_sp * (e2cr_sp - e2cr)**2) / wpcr / samples_clean.shape[0])

print(f"# Final Results:")
print(f"# AFQMC Energy Guide HF: {e1:.6f} +/- {e1_err:.6f}")
print(f"# AFQMC Energy Trial CISD: {e2:.6f} +/- {e2_err:.6f}")
print(f"# AFQMC Energy Trial CISD + stoCCSD Correction: {e2cr:.6f} +/- {e2cr_err:.6f}")

print(f"# total run time: {time.time() - init_time:.2f}")
