import time
import numpy as np
from jax import random
from jax import numpy as jnp

from afqmc import config
from afqmc import prep, sampling

from functools import partial
print = partial(print, flush=True)

init_time = time.time()

prep.print_start()
config.setup_jax()

ham_data, ham, prop, trial, wave_data, sampler, options = (prep.init_afqmc())


wave_data["rdm1"] = trial.get_rdm1(wave_data)
ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)

prop_data = prep.init_hf_prop_data(trial, wave_data, ham_data, options)

w_init = jnp.sum(prop_data["weights"])
e_init = prop_data["e_estimate"]
w_init = jnp.sum(prop_data["weights"])

print(wave_data["rdm1"])

print("\nEquilibration")

print(f"{'inv_T':>5s}  {'weight':>10s}  {'energy':>10s}  {'runTime':>8s}")
print(f"{0.:5.2f}  {w_init:10.5}  {e_init:10.5f}  {time.time() - init_time:8.2f}")

sampler_eq = sampling.sampler(
    n_prop_steps=50,
    n_chol = sampler.n_chol
    )

block_time = prop.dt * sampler_eq.n_prop_steps
neql_block = int(-(-options["eql_time"] // block_time))

for n in range(1,neql_block+1):
    prop_data, (wt, e) \
        = sampler_eq.block_sample(prop_data, ham_data, prop, trial, wave_data)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e

    if (n+1) % (min(max(neql_block // 10, 1), 20)) == 0 and n > 0:
        print(f"{(n+1)*block_time:5.2f}  {wt:10.5f}  {e:10.5f}  {time.time() - init_time:8.2f}")
        print(f"{(n+1)*block_time:5.2f}  {wt:10.5f}  {e:10.5f}  {time.time() - init_time:8.2f}")

print("\nSampling")
print(f"{'N':>4s}  {'killW':>5s}  {'weight':>10s}  "
      f"{'energy':>10s}  {'error':>8s}  {'runTime':>10s}")

wt_sp = np.zeros(sampler.n_blocks,dtype="float64")
e_sp = np.zeros(sampler.n_blocks,dtype="float64")
n_killed = np.zeros(sampler.n_blocks,dtype="int32")

for n in range(sampler.n_blocks):
    prop_data, (wt, e) \
        = sampler.block_sample(prop_data, ham_data, prop, trial, wave_data)
    
    wt_sp[n] = wt
    e_sp[n] = e
    n_killed[n] = prop_data["n_killed_walkers"]
    prop_data["n_killed_walkers"] = 0
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e

    if (n+1) % (min(max(sampler.n_blocks // 10, 1), 20)) == 0 and n > 0:
        weight = np.mean(wt_sp[:n+1])
        energy = np.mean(wt_sp[:n+1] * e_sp[:n+1]) / weight
        err = sampling.blocking_analysis(wt_sp[:n+1], e_sp[:n+1], min_nblocks=20, final=False)
        tot_kw = np.sum(n_killed)
        print(f"{n+1:4d}  {tot_kw:5d}  {weight:10.5f}  "
              f"{energy:10.5f}  {err:8.5f}  {time.time() - init_time:10.2f}")
        
        if err < 0.75 * options["max_error"] and n > 100:
            break

print("\nPost Propagation Process")
nsamples = n + 1
print(f'total number of samples {nsamples}')
wt_sp = wt_sp[:nsamples]
e_sp = e_sp[:nsamples]

# energy = np.sum(wt_sp * e_sp) / np.sum(wt_sp)
# err = sampler.blocking_analysis(wt_sp, e_sp, min_nblocks=20,final=False)

# print(f"Raw AFQMC: {energy:.6f} +/- {err:.6f}")

# print("\nRemove Outliers")
# def filter_outliers(e_sp, zeta=30):

#     median = np.median(e_sp)
#     mad = 1.4826 * np.median(np.abs(e_sp - median))
#     bound = zeta * mad
#     mask = np.abs(e_sp - median) < bound
#     print(f"removing energies outside zeta > {zeta}")
#     print(f"Energy bound [{median-bound:.6f}, {median+bound:.6f}]")
    
#     return mask

mask = sampling.filter_outliers(e_sp, zeta=30)

wt_sp = wt_sp[mask]
nsample_clean = len(wt_sp)
print(f"Removed {nsamples-nsample_clean} Outliers")
print(f"Outliers AFQMC Energy {e_sp[~mask]}")
e_sp = e_sp[mask]

print("\nBlocking Analysis")
energy = np.sum(wt_sp * e_sp) / np.sum(wt_sp)
err = sampling.blocking_analysis(wt_sp, e_sp, min_nblocks=20, final=True)

print(f"Final AFQMC: {energy:.6f} +/- {err:.6f}")

print(f"total run time: {time.time() - init_time:.2f}")
print(f"\nAFQMC Sampling Finished\n")