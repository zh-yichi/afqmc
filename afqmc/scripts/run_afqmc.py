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

print("\nEquilibration")

block_time = prop.dt * options["n_prop_steps"]
neql_block = int(-(-options["eql_time"] // block_time))

print(f"{'1/T':>5s}  {'weight':>10s}  {'energy':>10s}  {'error':>8s}  {'runTime':>6s}")
print(f"{0.:5.2f}  {w_init:10.5f}  {e_init:10.5f}  {0:8.5f}  {time.time() - init_time:6.2f}")

sampler_eq = sampling.sampler(
    n_prop_steps = options["n_prop_steps"],
    n_chol = sampler.n_chol,
    n_blocks = neql_block,
    )

for n in range(1,neql_block+1):
    prop_data, (wt, en, err) \
        = sampler_eq.block_sample(prop_data, ham_data, prop, trial, wave_data)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * en

    if (n+1) % (min(max(neql_block // 10, 1), 20)) == 0 and n > 0:
        print(f"{(n+1)*block_time:5.2f}  {wt:10.5f}  {en:10.5f}  {err:8.5f}  {time.time() - init_time:6.2f}")

print("\n")
print(f"Sampling Target (raw) 0.6 x max_error = {0.6 * options['max_error']:.5f}")
print(f"{'N':>4s}  {'nodes':>5s}  {'weight':>10s}  "
      f"{'energy':>10s}  {'error':>8s}  {'runTime':>10s}")

wt_sp = np.zeros(sampler.n_blocks,dtype="float64")
e_sp = np.zeros(sampler.n_blocks,dtype="float64")
nodes = np.zeros(sampler.n_blocks,dtype="int32")

for n in range(sampler.n_blocks):
    prop_data, (wt, e, _) \
        = sampler.block_sample(prop_data, ham_data, prop, trial, wave_data)
    
    wt_sp[n] = wt
    e_sp[n] = e
    nodes[n] = prop_data["n_killed_walkers"]
    prop_data["n_killed_walkers"] = 0
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e

    if (n+1) % (min(max(sampler.n_blocks // 10, 1), 20)) == 0 and n > 0:
        weight, energy, err = sampling.blocking(wt_sp[:n+1], e_sp[:n+1], min_nblocks=20, final=False)
        print(f"{n+1:4d}  {sum(nodes):5d}  {weight:10.5f}  "
              f"{energy:10.5f}  {err:8.5f}  {time.time() - init_time:10.2f}")
        
        if err < 0.6 * options["max_error"] and n > 120:
            break

print("\nPost Propagation Process")
nsamples = n + 1
print(f'total number of samples {nsamples}')
wt_sp = wt_sp[:nsamples]
e_sp = e_sp[:nsamples]

mask = sampling.filter_outliers(e_sp, zeta=30)

wt_sp = wt_sp[mask]
nsample_clean = len(wt_sp)
print(f"Removed {nsamples-nsample_clean} Outliers")
print(f"Outliers AFQMC Energy {e_sp[~mask]}")
e_sp = e_sp[mask]

print("\nBlocking Analysis")
weight, energy, err = sampling.blocking(wt_sp, e_sp, min_nblocks=20, final=True)

runtime = time.time() - init_time
h, rem = divmod(runtime, 3600)
m, s = divmod(rem, 60)
runtime_str = f"{int(h):d}h {int(m):02d}m {s:05.2f}s" if h else \
              f"{int(m):d}m {s:05.2f}s" if m else f"{s:.2f}s"

print("\n" + "=" * 50)
print("  AFQMC Result")
print("-" * 50)
print(f"  {'Average weight':<16s}{weight:>24.5f}")
print(f"  {'Energy (Ha)':<16s}{energy:>16.5f} +/- {err:<.5f}")
print(f"  {'Run time':<16s}{runtime_str:>24s}")
print("=" * 50)
print("\nAFQMC Sampling Finished\n")