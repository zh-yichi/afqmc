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

print("\nEquilibration")
print(f"{'1/T':>5s}  {'weight':>10s}  "
      f"{'energy':>12s}  {'Time':>10s}")
print(f"{0.:5.2f}  {init_w:10.5f}  {init_e:12.5f}  "
      f"{time.time() - init_time:10.2f}")

block_time = prop.dt * sampler.n_prop_steps
neql_block = int(-(-options["eql_time"] // block_time))

for n in range(1,neql_block+1):
    prop_data, (weight, sample, sample_err) \
        = sampler.block_sample(prop, wave, prop_data, ham_data, wave_data)
    weight_mean, energy_mean, energy_err \
            = wave.energy_formula(jnp.array([weight]), sample, ham_data)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * energy_mean

    if (n+1) % (min(max(neql_block // 10, 1), 20)) == 0 and n > 0:
        print(f"{(n+1)*block_time:5.2f}  "
              f"{weight_mean:10.5f}  {energy_mean:12.5f}  "
              f"{time.time() - init_time:10.2f}")

# print("\nSampling")
# print(f"{'N':>4s}  {'nodes':>5s}  "
#       f"{'weight':>8s}  {'energy':>12s}  {'error':>8s}  "
#       f"{'Time':>10s}")

# weight_list, sample_list = [], []
# nodes = 0

# for n in range(sampler.n_blocks):
#     prop_data, (weight, sample) \
#         = sampler.sample_energy(prop, wave, prop_data, ham_data, wave_data)

#     weight_list.append(weight)
#     sample_list.append(sample)
#     weights = jnp.stack(weight_list)
#     samples = jnp.stack(sample_list)
    
#     nodes += prop_data["n_killed_walkers"]
#     prop_data["n_killed_walkers"] = 0

#     if (n+1) % (min(max(sampler.n_blocks // 10, 1), 20)) == 0 and n > 0:
#         # weight_mean, energy_mean, energy_err \
#             # = wave.calc_sample_energy(weights, samples, ham_data)

#         weight_mean, energy_mean, energy_err = sampling.blocking(weights, samples, final=False)
#         prop_data["e_estimate"] = 0.8 * prop_data["e_estimate"] + 0.2 * energy_mean
#         print(f"{n+1:4d}  {nodes:5d}  {weight_mean.real:8.5f}  "
#               f"{energy_mean.real:12.5f}  {energy_err:8.5f}  "
#               f"{time.time() - init_time:10.2f}")
        
#         if energy_err < 0.75 * options["max_error"] and n > 120:
#             break

# print("\nPost Propagation Process")
# nsamples = len(weights)
# print(f'total number of samples {nsamples}')

# mask = sampling.filter_outliers(samples, zeta=30)
# print(f"Removed {nsamples-sum(mask)} Outliers")
# print(f"Outliers AFQMC Energy {samples[~mask]}")
# weights = weights[mask]
# samples = samples[mask]

# print("\nBlocking Analysis")
# final_energy = (np.sum(wt_sp * en_sp) / np.sum(wt_sp)).real
# final_err = sampling.blocking_analysis(wt_sp, en_sp, min_nblocks=20, final=True)
# print(f"Final AFQMC: {final_energy:.5f} +/- {final_err:.5f}")
print(f"total run time: {time.time() - init_time:.2f}")
print(f"AFQMC Sampling Finished\n")