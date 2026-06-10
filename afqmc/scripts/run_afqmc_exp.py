import time
import numpy as np
# from jax import random
from jax import numpy as jnp

from afqmc import config
from afqmc import prep, sampling

from functools import partial
print = partial(print, flush=True)

init_time = time.time()

prep.print_start()
config.setup_jax()

ham, prop, wave, ham_data, wave_data, sampler, options = (prep.init_afqmc_exp())
ham_data = wave.build_trial_intermediate(ham_data, wave_data)
ham_data = ham.build_propagation_intermediates(ham_data, prop, wave, wave_data)
prop_data = prep.init_hf_prop_data(wave, wave_data, ham_data, options)
guide_olps = wave.calc_overlap(prop_data["walkers"], wave_data)
trial_olps = wave.calc_trial_overlap(prop_data["walkers"], wave_data)
wt_init = jnp.sum(prop_data["weights"])
wp_init = jnp.sum(prop_data["weights"] * trial_olps / guide_olps)
e_init = prop_data["e_estimate"]

print(wave_data["rdm1"])

print("\nEquilibration")
print(f"{'1/T':>5s}  {'weight':>10s}  {'weightp':>10s}  "
      f"{'energy':>10s}  {'realTime':>8s}")
print(f"{0.:5.2f}  {wt_init:10.5f}  {wp_init.real:10.5f}  "
      f"{e_init:10.5f}  {time.time() - init_time:8.2f}")

block_time = prop.dt * sampler.n_prop_steps
neql_block = int(-(-options["eql_time"] // block_time))

for n in range(1,neql_block+1):
    prop_data, (wt, wp, e) \
        = sampler.sample_energy(prop, wave, prop_data, ham_data, wave_data)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e.real

    if (n+1) % (min(max(neql_block // 10, 1), 20)) == 0 and n > 0:
        print(f"{(n+1)*block_time:5.2f}  "
              f"{wt:10.5f}  {wp.real:10.5f}  {e.real:10.5f}  "
              f"{time.time() - init_time:8.2f}")

print("\nSampling")
print(f"{'N':>4s}  {'killW':>5s}  "
      f"{'weight':>10s}  {'weightp':>10s}"
      f"{'energy':>10s}  {'error':>8s}  "
      f"{'runTime':>8s}")

wt_sp = np.zeros(sampler.n_blocks,dtype="float64")
wp_sp = np.zeros(sampler.n_blocks,dtype="complex128")
en_sp = np.zeros(sampler.n_blocks,dtype="complex128")
n_killed = np.zeros(sampler.n_blocks,dtype="int32")

for n in range(sampler.n_blocks):
    prop_data, (wt, wp, en) \
        = sampler.sample_energy(prop, wave, prop_data, ham_data, wave_data)
    wt_sp[n] = wt
    wp_sp[n] = wp
    en_sp[n] = en
    n_killed[n] = prop_data["n_killed_walkers"]
    prop_data["n_killed_walkers"] = 0
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e.real

    if (n+1) % (min(max(sampler.n_blocks // 10, 1), 20)) == 0 and n > 0:
        weight = np.mean(wt_sp[:n+1])
        weighp = np.mean(wp_sp[:n+1])
        energy = np.mean(wp_sp[:n+1] * en_sp[:n+1]) / weighp
        err = sampling.blocking_analysis(wp_sp[:n+1], en_sp[:n+1], min_nblocks=20, final=False)
        tot_kw = np.sum(n_killed)
        print(f"{n+1:4d}  {tot_kw:5d}  "
              f"{weight:10.5f}  {weighp.real:10.5f}"
              f"{energy.real:10.5f}  {err:8.5f}  "
              f"{time.time() - init_time:8.2f}")
        
        if err < 0.75 * options["max_error"] and n > 100:
            break

print("\nPost Propagation Process")
nsamples = n + 1
print(f'total number of samples {nsamples}')
wp_sp = wp_sp[:nsamples]
en_sp = en_sp[:nsamples]

mask = sampling.filter_outliers(en_sp, zeta=30)
print(f"Removed {nsamples-sum(mask)} Outliers")
print(f"Outliers AFQMC Energy {en_sp[~mask]}")
wp_sp = wp_sp[mask]
en_sp = en_sp[mask]

print("\nBlocking Analysis")
final_energy = (np.sum(wp_sp * en_sp) / np.sum(wp_sp)).real
final_err = sampling.blocking_analysis(wp_sp, en_sp, min_nblocks=20, final=True)
print(f"Final AFQMC: {final_energy:.5f} +/- {final_err:.5f}")
print(f"total run time: {time.time() - init_time:.2f}")
print(f"AFQMC Sampling Finished\n")