from functools import partial
print = partial(print, flush=True)

print("\nCFS-AFQMC Started")

from afqmc import config
config.setup_jax()

import time
import numpy as np
from jax import numpy as jnp

from afqmc import sampling as sp
from afqmc.lno_afqmc import cfs_sampling as cfs
from afqmc.lno_afqmc import sampling as lsp
from afqmc.corr_sample import sampling as csp
from afqmc.lno_afqmc import prep

init_time = time.time()

(prop, sampler, options, 
 trial1, ham_data1, wave_data1, 
 trial2, ham_data2, wave_data2) = prep.init_cfs_afqmc()

wave_data1["rdm1"] = trial1.get_rdm1(wave_data1)
wave_data2["rdm1"] = trial2.get_rdm1(wave_data2)
    
ham_data1 = trial1._build_measurement_intermediates(ham_data1, wave_data1)
ham_data2 = trial2._build_measurement_intermediates(ham_data2, wave_data2)
ham_data1 = prop._build_propagation_intermediates(ham_data1, trial1, wave_data1)
ham_data2 = prop._build_propagation_intermediates(ham_data2, trial2, wave_data2)

prop_data1 = prep.init_hf_prop_data(trial1, wave_data1, ham_data1, options)
prop_data2 = prep.init_hf_prop_data(trial2, wave_data2, ham_data2, options)

def get_ept2orb(trial, prop_data, ham_data, wave_data):
    
    eg_sp, t1_sp, t2frg_sp, e0frg_sp, e1frg_sp, e0_sp \
        = trial.calc_ept2_frag(prop_data['walkers'], ham_data, wave_data)
    
    wt_sp = prop_data["weights"]
    wp_sp = wt_sp * t1_sp
    
    wt     = jnp.sum(wt_sp)
    eg     = jnp.sum(wt_sp * eg_sp) / wt

    wp     = jnp.sum(wp_sp)
    t2frg  = jnp.sum(wp_sp * t2frg_sp) / wp
    e0frg  = jnp.sum(wp_sp * e0frg_sp) / wp
    e1frg  = jnp.sum(wp_sp * e1frg_sp) / wp
    e0     = jnp.sum(wp_sp * e0_sp) / wp

    ept2_frg = jnp.real(e0frg + e1frg - t2frg * e0)

    return eg.real, ept2_frg

_, ept2orb1 = get_ept2orb(trial1, prop_data1, ham_data1, wave_data1)
_, ept2orb2 = get_ept2orb(trial2, prop_data2, ham_data2, wave_data2)

init_w1 = np.sum(prop_data1["weights"])
init_w2 = np.sum(prop_data2["weights"])

init_e1 = prop_data1["e_estimate"]
init_e2 = prop_data2["e_estimate"]

print("\nEquilibration")

print("system1 system2")

print(f"Initial Fragment energy: {ept2orb1:.6f}  {ept2orb2:.6f}")

print(f"{'inv_T':>5s}  "
      f"{'node1':>5s}  {'weight1':>10s}  {'energy1':>12s}  "
      f"{'node2':>5s}  {'weight2':>10s}  {'energy2':>12s}  "
      f"{'DE12':>10s}  {'runTime':>8s}")

print(f"{0.:5.2f}  "
      f"{0:5d}  {init_w1:10.5f}  {init_e1:12.5f}  "
      f"{0:5d}  {init_w2:10.5f}  {init_e2:12.5f}  "
      f"{init_e1 - init_e2:10.5f}  {time.time()-init_time:8.2f}")

block_time = prop.dt * options["n_prop_steps"]
neql_block = int(-(-options["eql_time"] // block_time))

sampler_eq = csp.sampler(
    n_walkers = options["n_walkers"],
    n_blocks = neql_block,
    n_prop_steps=50,
    n_chol = sampler.n_chol
    )

for n in range(sampler_eq.n_blocks):
    (prop_data1, prop_data2), (wt1, eg1, wt2, eg2) \
        = sampler_eq.block_sample(prop,
                                  trial1, prop_data1, ham_data1, wave_data1,
                                  trial2, prop_data2, ham_data2, wave_data2)
    
    node1 = prop_data1["n_killed_walkers"]
    node2 = prop_data2["n_killed_walkers"]
    
    prop_data1["n_killed_walkers"] = 0
    prop_data2["n_killed_walkers"] = 0

    if (n+1) % (min(max(neql_block // 10, 1), 20)) == 0 and n > 0:
        print(
            f"{(n+1)*block_time:5.2f}  "
            f"{node1:5d}  {wt1:10.5f}  {eg1:12.5f}  "
            f"{node2:5d}  {wt2:10.5f}  {eg2:12.5f}  "
            f"{eg1-eg2:10.5f}  {time.time()-init_time:8.2f}"
            )

print("\nSampling")

print(f"{'N':>4s}  "
      f"{'node1':>5s}  {'weight1':>8s}  {'Eguide1':>12s}  {'error':>8s}  {'Efrag1':>8s}  {'error':>8s}  "
      f"{'node2':>5s}  {'weight2':>8s}  {'Eguide2':>12s}  {'error':>8s}  {'Efrag2':>8s}  {'error':>8s}  "
      f"{'dEfrag12':>8s}  {'error':>8s}  {'runTime':>10s}")

wt_sp1 = np.zeros(sampler.n_blocks,dtype="float64")
eg_sp1 = np.zeros(sampler.n_blocks,dtype="float64")
t1_sp1 = np.zeros(sampler.n_blocks,dtype="complex128")
e0orb_sp1 = np.zeros(sampler.n_blocks,dtype="complex128")
e1orb_sp1 = np.zeros(sampler.n_blocks,dtype="complex128")
t2orb_sp1 = np.zeros(sampler.n_blocks,dtype="complex128")
e0bar_sp1 = np.zeros(sampler.n_blocks,dtype="complex128")

wt_sp2 = np.zeros(sampler.n_blocks,dtype="float64")
eg_sp2 = np.zeros(sampler.n_blocks,dtype="float64")
t1_sp2 = np.zeros(sampler.n_blocks,dtype="complex128")
e0orb_sp2 = np.zeros(sampler.n_blocks,dtype="complex128")
e1orb_sp2 = np.zeros(sampler.n_blocks,dtype="complex128")
t2orb_sp2 = np.zeros(sampler.n_blocks,dtype="complex128")
e0bar_sp2 = np.zeros(sampler.n_blocks,dtype="complex128")

node1 = 0
node2 = 0

for n in range(sampler.n_blocks):
    (prop_data1, prop_data2), (wt1, eg1, t11, t2orb1, e0orb1, e1orb1, e0bar1,
                               wt2, eg2, t12, t2orb2, e0orb2, e1orb2, e0bar2) \
                                = sampler.block_sample(prop,
                                    trial1, prop_data1, ham_data1, wave_data1,
                                    trial2, prop_data2, ham_data2, wave_data2)
    
    wt_sp1[n] = wt1
    eg_sp1[n] = eg1
    t1_sp1[n] = t11
    t2orb_sp1[n] = t2orb1
    e0orb_sp1[n] = e0orb1
    e1orb_sp1[n] = e1orb1
    e0bar_sp1[n] = e0bar1

    wt_sp2[n] = wt2
    eg_sp2[n] = eg2
    t1_sp2[n] = t12
    t2orb_sp2[n] = t2orb2
    e0orb_sp2[n] = e0orb2
    e1orb_sp2[n] = e1orb2
    e0bar_sp2[n] = e0bar2
    
    node1 += prop_data1["n_killed_walkers"]
    node2 += prop_data2["n_killed_walkers"]
    prop_data1["n_killed_walkers"] = 0
    prop_data2["n_killed_walkers"] = 0

    if (n+1) % (min(max(sampler.n_blocks // 10, 1), 20)) == 0 and n > 0:
        
        weight1, guide1 , eg_err1 = sp.blocking(wt_sp1[:n+1], eg_sp1[:n+1])
        weight2, guide2 , eg_err2 = sp.blocking(wt_sp2[:n+1], eg_sp2[:n+1])      

        eorb1, eorb_err1 \
            = lsp.pt2orbblocking(
                wt_sp1[:n+1], t1_sp1[:n+1], t2orb_sp1[:n+1], 
                e0orb_sp1[:n+1], e1orb_sp1[:n+1], e0bar_sp1[:n+1], 
                final=False)
        
        eorb2, eorb_err2 \
            = lsp.pt2orbblocking(
                wt_sp2[:n+1], t1_sp2[:n+1], t2orb_sp2[:n+1], 
                e0orb_sp2[:n+1], e1orb_sp2[:n+1], e0bar_sp2[:n+1], 
                final=False)
        
        d12, d12_err = cfs.pt2orbblocking(
            wt_sp1[:n+1], t1_sp1[:n+1], t2orb_sp1[:n+1], e0orb_sp1[:n+1], e1orb_sp1[:n+1], e0bar_sp1[:n+1],
            wt_sp2[:n+1], t1_sp2[:n+1], t2orb_sp2[:n+1], e0orb_sp2[:n+1], e1orb_sp2[:n+1], e0bar_sp2[:n+1],
            final=False,)
        
        print(f"{n+1:4d}  "
              f"{node1:5d}  {weight1:8.4f}  "
              f"{guide1:12.5f}  {eg_err1:8.5f}  "
              f"{eorb1:8.5f}  {eorb_err1:8.5f}  "
              f"{node2:5d}  {weight2:8.4f}  "
              f"{guide2:12.5f}  {eg_err2:8.5f}  "
              f"{eorb2:8.5f}  {eorb_err2:8.5f}  "
              f"{d12:8.5f}  {d12_err:8.5f}  "
              f"{time.time() - init_time:10.2f}")
        
        if d12_err < 0.6 * options["max_error"] and n > 120:
            break

print("\nPost Propagation")
nsamples1 = np.count_nonzero(wt_sp1)
nsamples2 = np.count_nonzero(wt_sp2)
if nsamples1 != nsamples2:
    print(f"Warning: sample1 has {nsamples1} filled slots, sample2 has {nsamples2}; "
          f"truncating both to {min(nsamples1, nsamples2)} to keep pairs aligned")
nsamples = min(nsamples1, nsamples2)
print(f"total number of paired samples {nsamples}")

wt_sp1 = wt_sp1[:nsamples]
eg_sp1 = eg_sp1[:nsamples]
e0orb_sp1 = e0orb_sp1[:nsamples]
e1orb_sp1 = e1orb_sp1[:nsamples]
t2orb_sp1 = t2orb_sp1[:nsamples]
e0bar_sp1 = e0bar_sp1[:nsamples]
t1_sp1 = t1_sp1[:nsamples]

wt_sp2 = wt_sp2[:nsamples]
eg_sp2 = eg_sp2[:nsamples]
e0orb_sp2 = e0orb_sp2[:nsamples]
e1orb_sp2 = e1orb_sp2[:nsamples]
t2orb_sp2 = t2orb_sp2[:nsamples]
e0bar_sp2 = e0bar_sp2[:nsamples]
t1_sp2 = t1_sp2[:nsamples]

ept2orb_sp1 = (e0orb_sp1/t1_sp1+ e1orb_sp1/t1_sp1 - t2orb_sp1*e0bar_sp1/t1_sp1**2).real
ept2orb_sp2 = (e0orb_sp2/t1_sp2+ e1orb_sp2/t1_sp2 - t2orb_sp2*e0bar_sp2/t1_sp2**2).real

mask1 = sp.filter_outliers(ept2orb_sp1, zeta=30)
mask2 = sp.filter_outliers(ept2orb_sp2, zeta=30)
mask = mask1 & mask2

print(f"Removed {np.sum(~mask)} paired Outliers")
print(f"  ({np.count_nonzero(~mask1)} flagged in sample1, "
      f"{np.count_nonzero(~mask2)} in sample2)")
print(f"Outlier Eguide sample1 {eg_sp1[~mask]}")
print(f"Outlier Eguide sample2 {eg_sp2[~mask]}")

# apply the SAME mask to every array of both sets
wt_sp1 = wt_sp1[mask]; eg_sp1 = eg_sp1[mask]
t1_sp1 = t1_sp1[mask]; t2orb_sp1 = t2orb_sp1[mask]
e0orb_sp1 = e0orb_sp1[mask]; e1orb_sp1 = e1orb_sp1[mask]
e0bar_sp1 = e0bar_sp1[mask]

wt_sp2 = wt_sp2[mask]; eg_sp2 = eg_sp2[mask]
t1_sp2 = t1_sp2[mask]; t2orb_sp2 = t2orb_sp2[mask]
e0orb_sp2 = e0orb_sp2[mask]; e1orb_sp2 = e1orb_sp2[mask]
e0bar_sp2 = e0bar_sp2[mask]
print("\nBlocking Analysis")

energy1, plateau_err1 = lsp.pt2orbblocking(
    wt_sp1, t1_sp1, t2orb_sp1, e0orb_sp1, e1orb_sp1, e0bar_sp1, final=True)

energy2, plateau_err2 = lsp.pt2orbblocking(
        wt_sp2, t1_sp2, t2orb_sp2, e0orb_sp2, e1orb_sp2, e0bar_sp2, final=True)

d12, d12_err = cfs.pt2orbblocking(
    wt_sp1, t1_sp1, t2orb_sp1, e0orb_sp1, e1orb_sp1, e0bar_sp1,
    wt_sp2, t1_sp2, t2orb_sp2, e0orb_sp2, e1orb_sp2, e0bar_sp2, 
    final=True)

# naive (uncorrelated) difference error, shown for reference
d12_naive_err = np.sqrt(plateau_err1**2 + plateau_err2**2)

width = 60
print()
print("=" * width)
print(f"{'Correlated Fragment Sampling  |  Final Result':^{width}}")
print("=" * width)
print(f"  {'':<20}{'Energy (Ha)':>16}{'Error':>16}")
print("-" * width)
print(f"  {'System 1':<20}{energy1:>16.6f}{plateau_err1:>16.6f}")
print(f"  {'System 2':<20}{energy2:>16.6f}{plateau_err2:>16.6f}")
print("-" * width)
print(f"  {'Difference (1-2)':<20}{d12:>16.6f}{d12_err:>16.6f}")
print(f"  {'  naive combined':<20}{'':>16}{d12_naive_err:>16.6f}")
print("=" * width)
print(f"  {'total run time':<20}{time.time() - init_time:>13.2f} s")
print("=" * width)
print("\nAFQMC Sampling Finished\n")
