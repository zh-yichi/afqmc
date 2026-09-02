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

print(f"Trial:   {trial}")
print(f"Sampler: {sampler}")

wave_data["rdm1"] = trial.get_rdm1(wave_data)    
ham_data = trial._build_measurement_intermediates(ham_data, wave_data)
ham_data = prop._build_propagation_intermediates(ham_data, trial, wave_data)
prop_data = prep.init_hf_prop_data(trial, wave_data, ham_data, options)

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

_, ept2orb = get_ept2orb(trial, prop_data, ham_data, wave_data)

print("\nEquilibration")
print(f"Initial Orbital energy: {ept2orb:.5f}")
print(f"{'inv_T':>5s}  {'nodes':>5s}  {'weight':>10s}  {'energy':>10s}  {'error':>8s}  "
      f"{'runTime':>8s}")

print(f"{0.:5.2f}  {prop_data['n_killed_walkers']:5d}  {np.sum(prop_data['weights']):10.5f}  "
      f"{prop_data['e_estimate']:10.5f}  {0.:8.5f}  {time.time()-init_time:8.2f}")

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
print(f"{'N':>4s}  {'nodes':>5s}  {'weight':>10s}  {'E(Guide)':>12s}  {'Error':>8s}  "
      f"{'weightp':>10s}  {'E_frag':>8s}  {'Error':>8s}  {'Time':>8s}")

wt_sp = np.zeros(sampler.n_blocks,dtype="float64")
eg_sp = np.zeros(sampler.n_blocks,dtype="float64")

wp_sp    = np.zeros(sampler.n_blocks,dtype="complex128")
t2frg_sp = np.zeros(sampler.n_blocks,dtype="complex128")
e0frg_sp = np.zeros(sampler.n_blocks,dtype="complex128")
e1frg_sp = np.zeros(sampler.n_blocks,dtype="complex128")
e0_sp    = np.zeros(sampler.n_blocks,dtype="complex128")

nodes = 0

for n in range(sampler.n_blocks):
    prop_data, (wt, eg, wp, t2frg, e0frg, e1frg, e0) = \
        sampler.block_sample(prop_data, ham_data, prop, trial, wave_data)
    
    wt_sp[n] = wt
    eg_sp[n] = eg

    wp_sp[n] = wp
    t2frg_sp[n] = t2frg
    e0frg_sp[n] = e0frg
    e1frg_sp[n] = e1frg
    e0_sp[n]    = e0
    
    nodes += prop_data["n_killed_walkers"]
    prop_data["n_killed_walkers"] = 0

    if (n+1) % (min(max(sampler.n_blocks // 10, 1), 20)) == 0 and n > 0:      
        weight, guide, guide_err = sp.blocking(wt_sp[:n+1], eg_sp[:n+1])
        weighp, efrg, efrg_err = lsp.ept2frg_blocking(
            wp_sp[:n+1], t2frg_sp[:n+1], e0frg_sp[:n+1], e1frg_sp[:n+1], e0_sp[:n+1],)
                
        print(f"{n+1:4d}  {nodes:5d}  {wt:10.5f}  {guide:12.5f}  {guide_err:8.5f}  "
              f"{weighp.real:10.5f}  {efrg.real:8.5f}  {efrg_err:8.5f}  "
              f"{time.time() - init_time:8.2f}")
        
        prop_data["e_estimate"] = 0.8 * prop_data["e_estimate"] + 0.2 * eg.real

        if efrg_err < 0.75 * options["max_error"] and n > 120:
            break

print("\nPost Propagation")
nsamples = np.count_nonzero(wt_sp)
print(f'Total number of samples {nsamples}')
wt_sp = wt_sp[:nsamples]
eg_sp = eg_sp[:nsamples]

wp_sp    = wp_sp[:nsamples]
t2frg_sp = t2frg_sp[:nsamples]
e0frg_sp = e0frg_sp[:nsamples]
e1frg_sp = e1frg_sp[:nsamples]
e0_sp = e0_sp[:nsamples]

print("Remove Outliers")
ept2frg_sp = (e0frg_sp + e1frg_sp - t2frg_sp * e0_sp).real
mask = sp.filter_outliers(ept2frg_sp, zeta=30)
print(f"Removed {np.sum(~mask)} Outliers")
print(f"Outliers Energy {ept2frg_sp[~mask]}")

wp_sp    = wp_sp[mask]
t2frg_sp = t2frg_sp[mask]
e0frg_sp = e0frg_sp[mask]
e1frg_sp = e1frg_sp[mask]
e0_sp    = e0_sp[mask]

weight, eguide, guide_err = sp.blocking(wt_sp, eg_sp,  final=True)
weighp, efrag, efrag_err = lsp.ept2frg_blocking(wp_sp, t2frg_sp, e0frg_sp, e1frg_sp, e0_sp, final=True)

print(f"Final AFQMC/HF Guiding Energy:      {eguide:.4f} +/- {guide_err:.4f}")
print(f"Final AFQMC/pt2CCSD Orbital Energy: {efrag:.5f} +/- {efrag_err:.5f}")
print(f"<t1> = weightp/weight = {jnp.real(weighp/weight):.5f}")
print(f"Total run time: {time.time() - init_time:.2f}")
print("\nAFQMC Sampling Finished\n")
