import time

import numpy as np

from afqmc import config, prep, sampling

from functools import partial

print = partial(print, flush=True)
init_time = time.time()

prep.print_start()
config.setup_jax()

ham_data, ham, prop, trial, wave_data, sampler, options = (prep.init_afqmc())

if "rdm1" not in wave_data:
    wave_data["rdm1"] = trial.get_rdm1(wave_data)
ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)
h0 = ham_data['h0']

prop_data = prep.init_hf_prop_data(trial, wave_data, ham_data, options)

init_e = prop_data["e_estimate"]
init_w = np.sum(prop_data["weights"])

print("\nEquilibration")

print(f"{'1/T':>5s}  "
      f"{'nodes':>5s}  {'weight':>12s}  "
      f"{'energy':>12s}  {'runTime':>8s}")
print(f"{0.:5.2f}  "
      f"{0:5d}  {init_w:12.5f}  "
      f"{init_e:12.5f}  {time.time() - init_time:8.2f}")

block_time = prop.dt * options["n_prop_steps"]
neql_block = int(-(-options["eql_time"] // block_time))

sampler_eq = sampling.sampler(
    n_prop_steps = options["n_prop_steps"],
    n_chol = sampler.n_chol,
    n_blocks = neql_block,
    )

for n in range(sampler_eq.n_blocks):
    prop_data, (wt, e, _ ) \
        = sampler_eq.block_sample(prop_data, ham_data, prop, trial, wave_data)
    nodes = prop_data["n_killed_walkers"]

    if (n+1) % (min(max(neql_block // 10, 1), 20)) == 0 and n > 0:
        print(f"{(n+1)*block_time:5.2f}  "
              f"{nodes:5d}  {wt:12.5f}  "
              f"{e:12.5f}  {time.time() - init_time:8.2f}")
        # prop_data = prop.stochastic_reconfiguration_local(prop_data)
        # prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["n_killed_walkers"] = 0

print("\nSampling")
print(f"Target (raw) 0.6 x max_error = {0.75 * options['max_error']:.5f}")
print(f"{'blocks':>6s}  {'nodes':>5s}  "
      f"{'weight':>10s}  {'E_Guide':>12s}  {'error':>8s}  "
      f"{'weightp':>10s}  {'E_Trial':>12s}  {'error':>8s}  "
    #   f"{'energy0':>10s}  {'error':>8s}  "
      f"{'Walltime':>10s}")

wt_sp = np.zeros(sampler.n_blocks, dtype="float64")
eg_sp = np.zeros(sampler.n_blocks, dtype="float64")
wp_sp = np.zeros(sampler.n_blocks, dtype="complex128")
t2_sp = np.zeros(sampler.n_blocks, dtype="complex128")
e0_sp = np.zeros(sampler.n_blocks, dtype="complex128")
e1_sp = np.zeros(sampler.n_blocks, dtype="complex128")

nodes = 0
for n in range(sampler.n_blocks):
    prop_data, (wt, eg, wp, t2, e0, e1) =\
        sampler.block_sample(prop_data, ham_data, prop, trial, wave_data)
    
    wt_sp[n] = wt
    eg_sp[n] = eg
    wp_sp[n] = wp
    t2_sp[n] = t2
    e0_sp[n] = e0
    e1_sp[n] = e1
    
    nodes += prop_data["n_killed_walkers"]
    prop_data["n_killed_walkers"] = 0

    if (n+1) % (min(max(sampler.n_blocks // 10, 1), 20)) == 0 and n > 0:
        
        weight, eguide, eg_err = sampling.blocking(wt_sp[:n+1], eg_sp[:n+1], final=False)

        # _, e0_mean, e0_err = sampling.blocking(wt_sp[:n+1], h0+e0_sp[:n+1], final=False)

        weighp, ept2, ept2_err \
            = sampling.pt2blocking(h0, wp_sp[:n+1], t2_sp[:n+1], e0_sp[:n+1], e1_sp[:n+1], final=False)
        
        print(f"{n+1:6d}  {nodes:5d}  "
              f"{weight:10.5f}  {eguide:12.5f}  {eg_err:8.5f}  "
              f"{weighp.real:10.5f}  {ept2:12.5f}  {ept2_err:8.5f}  "
            #   f"{e0_mean:10.5f}  {e0_err:8.5f}  "
              f"{time.time() - init_time:10.2f}")
        
        prop_data["e_estimate"] = 0.8 * prop_data["e_estimate"] + 0.2 * eguide.real
        # prop_data = prop.stochastic_reconfiguration_local(prop_data)
        # prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        
        if ept2_err < 0.75 * options["max_error"] and n > 120:
            break

print("\n ***** Post Propagation Process ***** ")

nsamples = np.count_nonzero(wt_sp)
print(f'Total number of samples {nsamples}')

wt_sp = np.array(wt_sp[:nsamples])
eg_sp = np.array(eg_sp[:nsamples])
wp_sp = np.array(wp_sp[:nsamples])
t2_sp = np.array(t2_sp[:nsamples])
e0_sp = np.array(e0_sp[:nsamples])
e1_sp = np.array(e1_sp[:nsamples])

print("\nRemove AFQMC/Guide Outliers")
eg_mask = np.array(sampling.filter_outliers(eg_sp, zeta=30))
wt_sp = np.array(wt_sp[eg_mask])
eg_sp = np.array(eg_sp[eg_mask])
print(f"Removed {np.sum(~eg_mask)} Outliers")
print(f"Outliers Energy: {eg_sp[~eg_mask]}")

print("\nRemove AFQMC/pt2CCSD Outliers")
ept2_sp = (h0 + e0_sp + e1_sp - t2_sp * e0_sp).real
ept2mask = np.array(sampling.filter_outliers(ept2_sp, zeta=30))
wp_sp = np.array(wp_sp[ept2mask])
t2_sp = np.array(t2_sp[ept2mask])
e0_sp = np.array(e0_sp[ept2mask])
e1_sp = np.array(e1_sp[ept2mask])
print(f"Removed {np.sum(~ept2mask)} Outliers")
print(f"Outliers Energy {ept2_sp[~ept2mask]}")

print("\nBlocking Analysis")

print("\nAFQMC/Guide")
wt, eguide, eg_err = sampling.blocking(wt_sp, eg_sp, final=True)

print("\nAFQMC/pt2CCSD:")
wp, ept2, ept2_err = sampling.pt2blocking(h0, wp_sp, t2_sp, e0_sp, e1_sp, final=True)

runtime = time.time() - init_time
h, rem = divmod(runtime, 3600)
m, s = divmod(rem, 60)
runtime_str = f"{int(h):d}h {int(m):02d}m {s:05.2f}s" if h else \
              f"{int(m):d}m {s:05.2f}s" if m else f"{s:.2f}s"

print("\n" + "=" * 60)
print("  AFQMC/pt2CCSD Result")
print("-" * 60)
print(f"  {'[Guide] Average Weight':<26s}{wt.real:>24.5f}")
print(f"  {'[Guide] Energy (Ha)':<26s}{eguide:>16.5f} +/- {eg_err:<.5f}")
print(f"  {'[Trial] Average Weight':<26s}{wp.real:>24.5f}")
print(f"  {'[Trial] Energy (Ha)':<26s}{ept2:>16.5f} +/- {ept2_err:<.5f}")
print(f"  {'Run Time':<26s}{runtime_str:>24s}")
print("=" * 60)
print("\n ***** AFQMC Sampling Finished ***** \n")
