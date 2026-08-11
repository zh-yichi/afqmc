import time
import numpy as np
# from jax import random
from jax import numpy as jnp

from afqmc import config
from afqmc import prep_exp, sampling_exp

from functools import partial
print = partial(print, flush=True)

init_time = time.time()

prep_exp.print_start()
config.setup_jax()

prop, wave, ham_data, wave_data, sampler, options = prep_exp.init_afqmc()
ham_data, wave_data = wave.build_intermediate(ham_data, wave_data)
ham_data = prop._build_propagation_intermediates(ham_data, wave, wave_data)
prop_data, init_w, init_e = prep_exp.init_prop_data(wave, wave_data, ham_data, options)

print(f'propagator:   {prop}')
print(f'wavefunction: {wave}')

block_time = prop.dt * sampler.n_prop_steps
neql_block = int(-(-options["eql_time"] // block_time))
print_every = (min(max(neql_block // 10, 1), 20))
print(f"block time = {block_time:.2f} | print every {print_every} blocks")

print("\nEquilibration")
print(f"{'1/T':>5s}  {'nodes':>5s}  {'weight':>10s}  "
      f"{'energy':>12s}  {'error':>8s}  "
      f"{'Walltime':>10s}")
print(f"{0.:5.2f}  {0:5d}  {init_w.real:10.5f}  "
      f"{init_e.real:12.5f}  {0.:8.5f}  "
      f"{time.time() - init_time:10.2f}")

weight_list, sample_list, nodes = [], [], 0

for n in range(1, neql_block + 1):
      prop_data, (weight, sample, _) \
            = sampler.block_sample(prop, wave, prop_data, ham_data, wave_data)

      weight_list.append(weight)
      sample_list.append(sample)
      weights = jnp.stack(weight_list)
      samples = jnp.stack(sample_list)
      nodes += prop_data["n_killed_walkers"]
      prop_data["n_killed_walkers"] = 0

      if n % print_every == 0:
            weight_mean, energy_mean, energy_err \
                  = wave.energy_formula(
                        weights[-print_every:], samples[-print_every:], ham_data)

            print(f"{n*block_time:5.2f}  {nodes:5d}  {weight_mean.real:10.5f}  "
                  f"{energy_mean.real:12.5f}  {energy_err.real:8.5f}  "
                  f"{time.time() - init_time:10.2f}")

            prop_data["e_estimate"] = 0.8 * prop_data["e_estimate"] + 0.2 * energy_mean.real
            prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
            nodes = 0

print_every = (min(max(sampler.n_blocks // 10, 1), 20))
print(f"\nSampling | print every {print_every}")
weight_list, sample_list, nodes = [], [], 0

print(f"Target (raw) 0.75 x max_error = {0.75 * options['max_error']:.5f}")
print(f"{'blocks':>6s}  {'nodes':>5s}  "
      f"{'weight':>10s}  {'Enegry':>12s}  {'error':>8s}  "
      f"{'Walltime':>10s}")

for n in range(1,sampler.n_blocks+1):
      prop_data, (weight, sample, _) \
            = sampler.block_sample(prop, wave, prop_data, ham_data, wave_data)

      weight_list.append(weight)
      sample_list.append(sample)
      weights = jnp.stack(weight_list)
      samples = jnp.stack(sample_list)
      nodes += prop_data["n_killed_walkers"]
      prop_data["n_killed_walkers"] = 0

      if n % print_every == 0:
            weight_mean, energy_mean, energy_err \
                  = wave.energy_formula(
                        weights, samples, ham_data)
  
            prop_data["e_estimate"] = 0.8 * prop_data["e_estimate"] + 0.2 * energy_mean.real
            print(f"{n:6d}  {nodes:5d}  "
                  f"{weight_mean.real:10.5f}  {energy_mean.real:12.5f}  {energy_err.real:8.5f}  "
                  f"{time.time() - init_time:10.2f}")
            
            if energy_err < 0.75 * options["max_error"] and n > 120:
                  break

print("\nPost Propagation Process")
print(f'total number of samples {len(weights)}')
mask = sampling_exp.filter_outliers(samples, zeta=30)
print(f"Removed {sum(~mask)} Outliers")
print(f"Outliers AFQMC Samples {samples[~mask]}")
weights = weights[mask]
samples = samples[mask]

print("\nBlocking Analysis")
weight, energy, err = sampling_exp.blocking(weights, samples, min_nblocks=20, final=True)

runtime = time.time() - init_time
h, rem = divmod(runtime, 3600)
m, s = divmod(rem, 60)
runtime_str = f"{int(h):d}h {int(m):02d}m {s:05.2f}s" if h else \
              f"{int(m):d}m {s:05.2f}s" if m else f"{s:.2f}s"

print("\n" + "=" * 50)
print("  AFQMC Result")
print("-" * 50)
print(f"  {'Average weight':<16s}{weight.real:>24.5f}")
print(f"  {'Energy (Ha)':<16s}{energy.real:>16.5f} +/- {err.real:<.5f}")
print(f"  {'Run time':<16s}{runtime_str:>24s}")
print("=" * 50)
print("\nAFQMC Sampling Finished\n")