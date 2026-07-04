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
    f"{'weight1':>10s}  {'Eguide1':>12s}  "
    f"{'weight2':>10s}  {'Eguide2':>12s}  "
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
    (prop_data1, prop_data2), (wt1, eg1, wt2, eg2) \
        = sampler_eq.block_sample(prop, 
                                  trial1, prop_data1, ham_data1, wave_data1,
                                  trial2, prop_data2, ham_data2, wave_data2)
    
    prop_data1["e_estimate"] = 0.9 * prop_data1["e_estimate"] + 0.1 * eg1
    prop_data2["e_estimate"] = 0.9 * prop_data2["e_estimate"] + 0.1 * eg2

    if (n+1) % (min(max(neql_block // 10, 1), 20)) == 0 and n > 0:
        print(
            f"{(n+1)*block_time:5.2f}  "
            f"{wt1:10.5f}  {eg1:12.5f}  "
            f"{wt2:10.5f}  {eg2:12.5f}  "
            f"{eg1-eg2:10.5f}  {time.time()-init_time:8.2f}"
            )

print("\nSampling")

print(f"{'N':>4s}  "
      f"{'node1':>6s}  {'weight1':>10s}  "
    #   f"{'Eguide1':>12s}  {'error':>8s}  "
      f"{'Ept1':>12s}  {'error':>8s}  "
      f"{'node2':>6s}  {'weight2':>10s}  "
    #   f"{'Eguide2':>12s}  {'error':>8s}  "
      f"{'Ept2':>12s}  {'error':>8s}  "
      f"{'dEpt12':>12s}  {'error':>8s}  {'runTime':>10s}")

nodes1 = np.zeros(sampler.n_blocks,dtype="int32")
wt_sp1 = np.zeros(sampler.n_blocks,dtype="float64")
eg_sp1 = np.zeros(sampler.n_blocks,dtype="float64")
t1_sp1 = np.zeros(sampler.n_blocks,dtype="complex128")
t2_sp1 = np.zeros(sampler.n_blocks,dtype="complex128")
e0_sp1 = np.zeros(sampler.n_blocks,dtype="complex128")
e1_sp1 = np.zeros(sampler.n_blocks,dtype="complex128")

nodes2 = np.zeros(sampler.n_blocks,dtype="int32")
wt_sp2 = np.zeros(sampler.n_blocks,dtype="float64")
eg_sp2 = np.zeros(sampler.n_blocks,dtype="float64")
t1_sp2 = np.zeros(sampler.n_blocks,dtype="complex128")
t2_sp2 = np.zeros(sampler.n_blocks,dtype="complex128")
e0_sp2 = np.zeros(sampler.n_blocks,dtype="complex128")
e1_sp2 = np.zeros(sampler.n_blocks,dtype="complex128")

# de12_sp = np.zeros(sampler.n_blocks,dtype="float64")

for n in range(sampler.n_blocks):
    (prop_data1, prop_data2), (wt1, eg1, t11, t21, e01, e11, 
                               wt2, eg2, t12, t22, e02, e12) \
        = sampler.block_sample(prop, 
                               trial1, prop_data1, ham_data1, wave_data1,
                               trial2, prop_data2, ham_data2, wave_data2)
    
    wt_sp1[n] = wt1
    eg_sp1[n] = eg1
    t1_sp1[n] = t11
    t2_sp1[n] = t21
    e0_sp1[n] = e01
    e1_sp1[n] = e11

    wt_sp2[n] = wt2
    eg_sp2[n] = eg2
    t1_sp2[n] = t12
    t2_sp2[n] = t22
    e0_sp2[n] = e02
    e1_sp2[n] = e12

    # de12_sp[n] = eg1 - eg2
    nodes1[n] = prop_data1["n_killed_walkers"]
    nodes2[n] = prop_data2["n_killed_walkers"]
    prop_data1["n_killed_walkers"] = 0
    prop_data2["n_killed_walkers"] = 0

    prop_data1["e_estimate"] = 0.9 * prop_data1["e_estimate"] + 0.1 * eg1
    prop_data2["e_estimate"] = 0.9 * prop_data2["e_estimate"] + 0.1 * eg2

    if (n+1) % (min(max(sampler.n_blocks // 10, 1), 20)) == 0 and n > 0:
        weight1 = np.mean(wt_sp1[:n+1])
        weight2 = np.mean(wt_sp2[:n+1])

        guide1 , eg_err1 = sp.blocking(wt_sp1[:n+1], eg_sp1[:n+1])
        guide2 , eg_err2 = sp.blocking(wt_sp2[:n+1], eg_sp2[:n+1])
        
        ept1, ept_err1 = sp.pt2blocking(
            ham_data1["h0"], wt_sp1[:n+1], t1_sp1[:n+1], t2_sp1[:n+1], e0_sp1[:n+1], e1_sp1[:n+1]
            )
        ept2, ept_err2 = sp.pt2blocking(
            ham_data2["h0"], wt_sp2[:n+1], t1_sp2[:n+1], t2_sp2[:n+1], e0_sp2[:n+1], e1_sp2[:n+1]
            )
        
        dept12, dept12_err = csp.pt2blocking(
            ham_data1["h0"], wt_sp1[:n+1], t1_sp1[:n+1], t2_sp1[:n+1], e0_sp1[:n+1], e1_sp1[:n+1],
            ham_data2["h0"], wt_sp2[:n+1], t1_sp2[:n+1], t2_sp2[:n+1], e0_sp2[:n+1], e1_sp2[:n+1]
            )
        
        print(f"{n+1:4d}  "
              f"{np.sum(nodes1[:n+1]):6d}  {weight1:10.5f}  "
            #   f"{guide1:12.5f}  {eg_err1:8.5f}  "
              f"{ept1:12.5f}  {ept_err1:8.5f}  "
              f"{np.sum(nodes2[:n+1]):6d}  {weight2:10.5f}  "
            #   f"{guide2:12.5f}  {eg_err2:8.5f}  "
              f"{ept2:12.5f}  {ept_err2:8.5f}  "
              f"{dept12:12.5f}  {dept12_err:8.5f}  "
              f"{time.time() - init_time:10.2f}")
        
        if dept12_err < 6 * options["max_error"] and n > 120:
            break

print("\nPost Propagation Process")

# --- valid-sample count: keep the pairing, use a common cutoff ---
nsamples1 = np.count_nonzero(wt_sp1)
nsamples2 = np.count_nonzero(wt_sp2)
if nsamples1 != nsamples2:
    print(f"Warning: sample1 has {nsamples1} filled slots, sample2 has {nsamples2}; "
          f"truncating both to {min(nsamples1, nsamples2)} to keep pairs aligned")
nsamples = min(nsamples1, nsamples2)
print(f"total number of paired samples {nsamples}")

# truncate EVERY array of BOTH sets to the common length (keeps pairs aligned)
wt_sp1 = wt_sp1[:nsamples]; eg_sp1 = eg_sp1[:nsamples]
t1_sp1 = t1_sp1[:nsamples]; t2_sp1 = t2_sp1[:nsamples]
e0_sp1 = e0_sp1[:nsamples]; e1_sp1 = e1_sp1[:nsamples]

wt_sp2 = wt_sp2[:nsamples]; eg_sp2 = eg_sp2[:nsamples]
t1_sp2 = t1_sp2[:nsamples]; t2_sp2 = t2_sp2[:nsamples]
e0_sp2 = e0_sp2[:nsamples]; e1_sp2 = e1_sp2[:nsamples]

# --- correlated outlier removal on the guiding energy ---
# drop pair i if EITHER set flags it as an outlier
mask1 = sp.filter_outliers(eg_sp1, zeta=30)
mask2 = sp.filter_outliers(eg_sp2, zeta=30)
mask = mask1 & mask2

nsample_clean = np.count_nonzero(mask)
print(f"Removed {nsamples - nsample_clean} paired Outliers")
print(f"  ({np.count_nonzero(~mask1)} flagged in sample1, "
      f"{np.count_nonzero(~mask2)} in sample2)")
print(f"Outlier Eguide sample1 {eg_sp1[~mask]}")
print(f"Outlier Eguide sample2 {eg_sp2[~mask]}")

# apply the SAME mask to every array of both sets
wt_sp1 = wt_sp1[mask]; eg_sp1 = eg_sp1[mask]
t1_sp1 = t1_sp1[mask]; t2_sp1 = t2_sp1[mask]
e0_sp1 = e0_sp1[mask]; e1_sp1 = e1_sp1[mask]

wt_sp2 = wt_sp2[mask]; eg_sp2 = eg_sp2[mask]
t1_sp2 = t1_sp2[mask]; t2_sp2 = t2_sp2[mask]
e0_sp2 = e0_sp2[mask]; e1_sp2 = e1_sp2[mask]

print("\nBlocking Analysis")

# --- guiding energies (per set), final blocking ---
print("\n[Guide] sample 1")
guide1, eg_err1 = sp.blocking(wt_sp1, eg_sp1, min_nblocks=20, final=True)
print("\n[Guide] sample 2")
guide2, eg_err2 = sp.blocking(wt_sp2, eg_sp2, min_nblocks=20, final=True)

# --- PT2 energies (per set); h0 is the LAST data argument ---
print("\n[PT2] sample 1")
ept1, ept_err1 = sp.pt2blocking(
    ham_data1["h0"], wt_sp1, t1_sp1, t2_sp1, e0_sp1, e1_sp1,
    min_nblocks=20, final=True)
print("\n[PT2] sample 2")
ept2, ept_err2 = sp.pt2blocking(
    ham_data2["h0"], wt_sp2, t1_sp2, t2_sp2, e0_sp2, e1_sp2,
    min_nblocks=20, final=True)

# --- correlated differences ---
print("\n[Guide difference] correlated")
dguide12, dguide12_err = csp.blocking(
    wt_sp1, eg_sp1, wt_sp2, eg_sp2, min_nblocks=20, final=True)

print("\n[PT2 difference] correlated")
dept12, dept12_err = csp.pt2blocking(
    ham_data1["h0"], wt_sp1, t1_sp1, t2_sp1, e0_sp1, e1_sp1,
    ham_data2["h0"], wt_sp2, t1_sp2, t2_sp2, e0_sp2, e1_sp2,
    min_nblocks=20, final=True)

print("\n" + "=" * 60)
print(f"Final Eguide 1:   {guide1:12.5f} +/- {eg_err1:.5f}")
print(f"Final Eguide 2:   {guide2:12.5f} +/- {eg_err2:.5f}")
print(f"Final Eguide dE:  {dguide12:12.5f} +/- {dguide12_err:.5f}")
print(f"Final EPT2 1:     {ept1:12.5f} +/- {ept_err1:.5f}")
print(f"Final EPT2 2:     {ept2:12.5f} +/- {ept_err2:.5f}")
print(f"Final EPT2 dE:    {dept12:12.5f} +/- {dept12_err:.5f}")
print("=" * 60)
print(f"total run time: {time.time() - init_time:.2f}")
print(f"\nAFQMC Sampling Finished\n")