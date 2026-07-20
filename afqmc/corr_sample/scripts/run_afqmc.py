import time
import numpy as np
from jax import numpy as jnp

from afqmc import config
from afqmc.corr_sample import prep
from afqmc.corr_sample import sampling as csp
from afqmc import sampling as sp

from functools import partial
print = partial(print, flush=True)

init_time = time.time()

prep.print_start()
config.setup_jax()

(prop, sampler, options, 
 trial1, ham_data1, wave_data1, 
 trial2, ham_data2, wave_data2) = (prep.init_afqmc())

wave_data1["rdm1"] = trial1.get_rdm1(wave_data1)
wave_data2["rdm1"] = trial2.get_rdm1(wave_data2)

ham_data1 = trial1._build_measurement_intermediates(ham_data1, wave_data1)
ham_data2 = trial2._build_measurement_intermediates(ham_data2, wave_data2)

ham_data1 = prop._build_propagation_intermediates(ham_data1, trial1, wave_data1)
ham_data2 = prop._build_propagation_intermediates(ham_data2, trial2, wave_data2)

prop_data1 = prep.init_hf_prop_data(trial1, wave_data1, ham_data1, options)
prop_data2 = prep.init_hf_prop_data(trial2, wave_data2, ham_data2, options)

print("\nEquilibration")

print(
    f"{'1/T':>5s}  "
    f"{'weight1':>10s}  {'Energy1':>12s}  "
    f"{'weight2':>10s}  {'Energy2':>12s}  "
    f"{'dE12':>10s}  {'runTime':>8s}"
    )

print(
    f"{0.:5.2f}  "
    f"{jnp.sum(prop_data1["weights"]):10.5f}  {prop_data1["e_estimate"]:12.5f}  "
    f"{jnp.sum(prop_data2["weights"]):10.5f}  {prop_data2["e_estimate"]:12.5f}  "
    f"{prop_data1["e_estimate"]-prop_data2["e_estimate"]:10.5f}  {time.time() - init_time:8.2f}"
    )

eql_prop_steps = 50
block_time = prop.dt * eql_prop_steps
neql_block = int(-(-options["eql_time"] // block_time))

sampler_eq = csp.sampler(
    n_prop_steps = 50,
    n_blocks = neql_block,
    n_walkers = sampler.n_walkers,
    n_chol = sampler.n_chol,
    )

for n in range(1,neql_block+1):
    (prop_data1, prop_data2), (wt1, en1, wt2, en2) \
        = sampler_eq.block_sample(prop, 
                                  trial1, prop_data1, ham_data1, wave_data1,
                                  trial2, prop_data2, ham_data2, wave_data2)
    
    prop_data1["e_estimate"] = 0.9 * prop_data1["e_estimate"] + 0.1 * en1
    prop_data2["e_estimate"] = 0.9 * prop_data2["e_estimate"] + 0.1 * en2

    if (n+1) % (min(max(neql_block // 10, 1), 20)) == 0 and n > 0:
        print(
            f"{(n+1)*block_time:5.2f}  "
            f"{wt1:10.5f}  {en1:12.5f}  "
            f"{wt2:10.5f}  {en2:12.5f}  "
            f"{en1-en2:10.5f}  {time.time()-init_time:8.2f}"
            )

print("\nSampling")

print(f"{'N':>4s}  "
      f"{'node1':>6s}  {'weight1':>10s}  {'Energy1':>12s}  {'error':>8s}  "
      f"{'node2':>6s}  {'weight2':>10s}  {'Energy2':>12s}  {'error':>8s}  "
      f"{'dE12':>10s}  {'error':>8s} {'runTime':>10s}")

nodes1 = np.zeros(sampler.n_blocks,dtype="int32")
nodes2 = np.zeros(sampler.n_blocks,dtype="int32")
wt1_sp = np.zeros(sampler.n_blocks,dtype="float64")
en1_sp = np.zeros(sampler.n_blocks,dtype="float64")
wt2_sp = np.zeros(sampler.n_blocks,dtype="float64")
en2_sp = np.zeros(sampler.n_blocks,dtype="float64")
de12_sp = np.zeros(sampler.n_blocks,dtype="float64")

for n in range(sampler.n_blocks):
    (prop_data1, prop_data2), (wt1, en1, wt2, en2) \
        = sampler.block_sample(prop, 
                               trial1, prop_data1, ham_data1, wave_data1,
                               trial2, prop_data2, ham_data2, wave_data2)
    
    wt1_sp[n] = wt1
    en1_sp[n] = en1
    wt2_sp[n] = wt2
    en2_sp[n] = en2
    de12_sp[n] = en1 - en2
    nodes1[n] = prop_data1["n_killed_walkers"]
    nodes2[n] = prop_data2["n_killed_walkers"]
    prop_data1["n_killed_walkers"] = 0
    prop_data2["n_killed_walkers"] = 0
    prop_data1["e_estimate"] = 0.9 * prop_data1["e_estimate"] + 0.1 * en1
    prop_data2["e_estimate"] = 0.9 * prop_data2["e_estimate"] + 0.1 * en2

    if (n+1) % (min(max(sampler.n_blocks // 10, 1), 20)) == 0 and n > 0:
        # weight1 = np.mean(wt1_sp[:n+1])
        # weight2 = np.mean(wt2_sp[:n+1])
        # energy1 = np.mean(wt1_sp[:n+1] * en1_sp[:n+1]) / weight1
        # energy2 = np.mean(wt2_sp[:n+1] * en2_sp[:n+1]) / weight2
        # de12 = energy1 - energy2
        weight1, energy1, err1 = sp.blocking(wt1_sp[:n+1], en1_sp[:n+1], min_nblocks=20, final=False)
        weight2, energy2, err2 = sp.blocking(wt2_sp[:n+1], en2_sp[:n+1], min_nblocks=20, final=False)
        de12, cs_err = csp.blocking(wt1_sp[:n+1], en1_sp[:n+1], wt2_sp[:n+1], en2_sp[:n+1], 
                                    min_nblocks=20, final=False)
        print(f"{n+1:4d}  "
              f"{nodes1[n]:6d}  {weight1:10.5f}  {energy1:12.5f}  {err1:8.5f}  "
              f"{nodes2[n]:6d}  {weight2:10.5f}  {energy2:12.5f}  {err2:8.5f}  "
              f"{de12:10.5f}  {cs_err:8.5f}  {time.time() - init_time:10.2f}")
        
        if cs_err < 6 * options["max_error"] and n > 120:
            break

print("\nPost Propagation Process")

# --- valid-sample count: keep the pairing, so use a common cutoff ---
nsamples1 = np.count_nonzero(wt1_sp)
nsamples2 = np.count_nonzero(wt2_sp)
if nsamples1 != nsamples2:
    print(f"Warning: sample1 has {nsamples1} filled slots, sample2 has {nsamples2}; "
          f"truncating both to {min(nsamples1, nsamples2)} to keep pairs aligned")
nsamples = min(nsamples1, nsamples2)
print(f'total number of paired samples {nsamples}')

wt1_sp = wt1_sp[:nsamples]
en1_sp  = en1_sp[:nsamples]
wt2_sp = wt2_sp[:nsamples]
en2_sp  = en2_sp[:nsamples]

# --- correlated outlier removal: drop pair i if EITHER member is an outlier ---
mask1 = sp.filter_outliers(en1_sp, zeta=30)
mask2 = sp.filter_outliers(en2_sp, zeta=30)
mask = mask1 & mask2                      # keep only pairs clean in BOTH sets

nsample_clean = np.count_nonzero(mask)
print(f"Removed {nsamples - nsample_clean} paired Outliers")
print(f"  ({np.count_nonzero(~mask1)} flagged in sample1, "
      f"{np.count_nonzero(~mask2)} in sample2)")
print(f"Outlier AFQMC Energy sample1 {en1_sp[~mask]}")
print(f"Outlier AFQMC Energy sample2 {en2_sp[~mask]}")

wt1_sp = wt1_sp[mask]; en1_sp = en1_sp[mask]
wt2_sp = wt2_sp[mask]; en2_sp = en2_sp[mask]

print("\nBlocking Analysis")

weight1, energy1, err1 = sp.blocking(wt1_sp, en1_sp, min_nblocks=20, final=True)
weight2, energy2, err2 = sp.blocking(wt2_sp, en2_sp, min_nblocks=20, final=True)
de12, cs_err = csp.blocking(wt1_sp, en1_sp, wt2_sp, en2_sp, min_nblocks=20, final=True)

print(f"Final AFQMC E1:    {energy1:.5f} +/- {err1:.5f}")
print(f"Final AFQMC E2:    {energy2:.5f} +/- {err2:.5f}")
print(f"Final AFQMC dE:    {de12:.5f} +/- {cs_err:.5f}")
print(f"total run time: {time.time() - init_time:.2f}")
print(f"\nAFQMC Sampling Finished\n")