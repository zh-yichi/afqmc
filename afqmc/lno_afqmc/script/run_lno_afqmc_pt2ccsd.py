from functools import partial
print = partial(print, flush=True)

print("\nLNO-AFQMC Started")

from afqmc import config
config.setup_jax()

import time
import numpy as np
from jax import numpy as jnp

from afqmc import sampling as sp
from afqmc.lno_afqmc import sampling as lsp
from afqmc.lno_afqmc import prep

init_time = time.time()

ham_data, prop, trial, wave_data, sampler, options = (prep.init_afqmc())

wave_data["rdm1"] = trial.get_rdm1(wave_data)    
ham_data = trial._build_measurement_intermediates(ham_data, wave_data)
ham_data = prop._build_propagation_intermediates(ham_data, trial, wave_data)
prop_data = prep.init_hf_prop_data(trial, wave_data, ham_data, options)

def get_ept2orb(trial, prop_data, ham_data, wave_data):
    
    wt_sp = prop_data["weights"]
    eg_sp, t1_sp, e0orb_sp, e1orb_sp, t2orb_sp, e0bar_sp \
        = trial.calc_eorb_pt2(prop_data['walkers'], ham_data, wave_data)

    wt     = jnp.sum(wt_sp)
    eg     = jnp.sum(wt_sp * eg_sp) / wt
    t1     = jnp.sum(wt_sp * t1_sp) / wt
    e0orb  = jnp.sum(wt_sp * e0orb_sp) / wt
    e1orb  = jnp.sum(wt_sp * e1orb_sp) / wt
    t2orb  = jnp.sum(wt_sp * t2orb_sp) / wt
    e0bar  = jnp.sum(wt_sp * e0bar_sp) / wt

    ept2_orb = jnp.real(e0orb/t1 + e1orb/t1 - t2orb*e0bar/t1**2)

    return eg.real, ept2_orb

_, ept2orb = get_ept2orb(trial, prop_data, ham_data, wave_data)

print("\nEquilibration")
print(f"Initial Orbital energy: {ept2orb:.5f}")
print(f"{'inv_T':>5s}  {'nodes':>5s}  {'weight':>10s}  {'energy':>10s}  {'error':>8s}  "
      f"{'runTime':>8s}")

print(f"{0.:5.2f}  {prop_data["n_killed_walkers"]:5d}  {np.sum(prop_data["weights"]):10.5f}  "
      f"{prop_data["e_estimate"]:10.5f}  {0.:8.5f}  {time.time()-init_time:8.2f}")

block_time = prop.dt * options["n_prop_steps"]
neql_block = int(-(-options["eql_time"] // block_time))

sampler_eq = sp.sampler(
    n_prop_steps = options["n_prop_steps"],
    n_blocks = neql_block,
    n_chol = sampler.n_chol,
    )

for n in range(1, neql_block+1):
    prop_data, (wt, eg, err) \
        = sampler_eq.block_sample(prop_data, ham_data, prop, trial, wave_data)
    
    prop_data["n_killed_walkers"] = 0

    if (n+1) % (min(max(neql_block // 10, 1), 20)) == 0 and n > 0:
        nodes = prop_data["n_killed_walkers"]
        print(f"{(n+1)*block_time:5.2f}  {nodes:5d}  {wt:10.5f}  {eg:10.5f}  {err:8.5f}  "
              f"{time.time() - init_time:8.2f}")

print("\nSampling Blocks")

print(f"Target Final Error ~ {options['max_error']:.6f}")
print(f"{'N':>4s}  {'nodes':>5s}  {'weight':>10s}"
      f"{'E(Guide)':>12s}  {'Error':>8s}  "
      f"{'E(Orb)':>10s}  {'Error':>8s}  "
      f"{'Time':>8s}")

wt_sp = np.zeros(sampler.n_blocks,dtype="float64")
eg_sp = np.zeros(sampler.n_blocks,dtype="float64")
t1_sp = np.zeros(sampler.n_blocks,dtype="complex128")
t2orb_sp = np.zeros(sampler.n_blocks,dtype="complex128")
e0orb_sp = np.zeros(sampler.n_blocks,dtype="complex128")
e1orb_sp = np.zeros(sampler.n_blocks,dtype="complex128")
e0bar_sp = np.zeros(sampler.n_blocks,dtype="complex128")

nodes = 0

for n in range(sampler.n_blocks):
    prop_data, (wt, eg, t1, t2orb, e0orb, e1orb, t2orb, e0bar) = \
        sampler.block_sample(prop_data, ham_data, prop, trial, wave_data)
    
    wt_sp[n] = wt
    eg_sp[n] = eg
    t1_sp[n] = t1
    t2orb_sp[n] = t2orb
    e0orb_sp[n] = e0orb
    e1orb_sp[n] = e1orb
    e0bar_sp[n] = e0bar
    
    nodes += prop_data["n_killed_walkers"]
    prop_data["n_killed_walkers"] = 0

    if (n+1) % (min(max(sampler.n_blocks // 10, 1), 20)) == 0 and n > 0:      
        weight, guide, guide_err = sp.blocking(wt_sp[:n+1], eg_sp[:n+1])
        eorb, eorb_err = lsp.pt2orbblocking(
            wt_sp[:n+1], t1_sp[:n+1], t2orb_sp[:n+1], 
            e0orb_sp[:n+1], e1orb_sp[:n+1], e0bar_sp[:n+1])
                
        print(f"{n+1:4d}  {nodes:5d}  {wt:10.5f}  "
              f"{guide:12.5f}  {guide_err:8.5f}  "
              f"{eorb:8.5f}  {eorb_err:8.5f}  "
              f"{time.time() - init_time:8.2f}")

        if eorb_err < 0.6 * options["max_error"] and n > 120:
            break

print("\nPost Propagation")
nsamples = np.count_nonzero(wt_sp)
print(f'Total number of samples {nsamples}')
wt_sp = wt_sp[:nsamples]
eg_sp = eg_sp[:nsamples]
t1_sp = t1_sp[:nsamples]
t2orb_sp = t2orb_sp[:nsamples]
e0orb_sp = e0orb_sp[:nsamples]
e1orb_sp = e1orb_sp[:nsamples]
e0bar_sp = e0bar_sp[:nsamples]

print("Remove Outliers")
ept2orb_sp = (e0orb_sp/t1_sp + e1orb_sp/t1_sp - t2orb_sp*e0bar_sp/t1_sp**2).real
mask = sp.filter_outliers(ept2orb_sp, zeta=30)
print(f"Removed {np.sum(~mask)} Outliers")
print(f"Outliers Energy {ept2orb_sp[~mask]}")

wt_sp = wt_sp[mask]
t1_sp = t1_sp[mask]
t2orb_sp = t2orb_sp[mask]
e0orb_sp = e0orb_sp[mask]
e1orb_sp = e1orb_sp[mask]
e0bar_sp = e0bar_sp[mask]

energy, plateau_value = lsp.pt2orbblocking(
    wt_sp, t1_sp, t2orb_sp, e0orb_sp, e1orb_sp, e0bar_sp, final=True)

print(f"Final AFQMC/pt2CCSD Orbital Energy: {energy:.5f} +/- {plateau_value:.5f}")
print(f"Total run time: {time.time() - init_time:.2f}")
print("\nAFQMC Sampling Finished\n")
