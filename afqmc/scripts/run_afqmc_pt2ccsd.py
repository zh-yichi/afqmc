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
      f"{'weight':>12s}  {'nodes':>5s}  "
      f"{'energy':>12s}  {'runTime':>8s}")
print(f"{0.:5.2f}  "
      f"{init_w:12.5f}  {0:5d}  "
      f"{init_e:12.5f}  {time.time() - init_time:8.2f}")

block_time = prop.dt * options["n_prop_steps"]
neql_block = int(-(-options["eql_time"] // block_time))

sampler_eq = sampling.sampler(
    n_blocks = neql_block,
    n_prop_steps = 50, 
    n_chol = sampler.n_chol
    )

for n in range(sampler_eq.n_blocks):
    prop_data, (wt, e, _ ) \
        = sampler_eq.block_sample(prop_data, ham_data, prop, trial, wave_data)
    prop_data["n_killed_walkers"] = 0

    if (n+1) % (min(max(neql_block // 10, 1), 20)) == 0 and n > 0:
        nodes = prop_data["n_killed_walkers"]
        print(f"{(n+1)*block_time:5.2f}  "
              f"{wt:12.5f}  {nodes:5d}  "
              f"{e:12.5f}  {time.time() - init_time:8.2f}")

print("\nSampling")
print(f"Target (raw) 0.6 x max_error = {0.75 * options['max_error']:.5f}")
print(f"{'blocks':>6s}  "
      f"{'weight':>12s}  {'nodes':>5s}  "
      f"{'E_Guide':>12s}  {'error':>8s}  "
      f"{'E_Trial':>12s}  {'error':>8s}  "
    #   f"{'olp_T/G':>10s}  {'error':>8s}  "
      f"{'Walltime':>10s}")

wt_sp = np.zeros(sampler.n_blocks, dtype="float64")
eg_sp = np.zeros(sampler.n_blocks, dtype="float64")
t1_sp = np.zeros(sampler.n_blocks, dtype="complex128")
t2_sp = np.zeros(sampler.n_blocks, dtype="complex128")
e0_sp = np.zeros(sampler.n_blocks, dtype="complex128")
e1_sp = np.zeros(sampler.n_blocks, dtype="complex128")
nodes = 0

for n in range(sampler.n_blocks):
    prop_data, (wt, eg, t1, t2, e0, e1) =\
        sampler.block_sample(prop_data, ham_data, prop, trial, wave_data)
    
    wt_sp[n] = wt
    eg_sp[n] = eg
    t1_sp[n] = t1
    t2_sp[n] = t2
    e0_sp[n] = e0
    e1_sp[n] = e1
    nodes += prop_data["n_killed_walkers"]
    prop_data["n_killed_walkers"] = 0

    if (n+1) % (min(max(sampler.n_blocks // 10, 1), 20)) == 0 and n > 0:
        _, eguide, eg_err = sampling.blocking(wt_sp[:n+1], eg_sp[:n+1], final=False)
        weight, ept2, ept2_err = sampling.pt2blocking(
            h0, wt_sp[:n+1], t1_sp[:n+1], t2_sp[:n+1], e0_sp[:n+1], e1_sp[:n+1], final=False)
        # _, otg, otg_err = sampling.blocking(wt_sp[:n+1], t1_sp[:n+1].real, final=False)
        
        print(f"{n+1:6d}  "
              f"{weight:12.5f}  {nodes:5d}  "
              f"{eguide:12.5f}  {eg_err:8.5f}  "
              f"{ept2:12.5f}  {ept2_err:8.5f}  "
            #   f"{otg.real:10.6f}  {otg_err.real:8.5f}"
              f"{time.time() - init_time:10.2f}")
        
        prop_data["e_estimate"] = 0.8 * prop_data["e_estimate"] + 0.2 * eg.real
        
        if ept2_err < 0.75 * options["max_error"] and n > 120:
            break

print("\nPost Propagation Process")
nsamples = n + 1
print(f'Total number of samples {nsamples}')
wt_sp = wt_sp[:nsamples]
eg_sp = eg_sp[:nsamples]
t1_sp = t1_sp[:nsamples]
t2_sp = t2_sp[:nsamples]
e0_sp = e0_sp[:nsamples]
e1_sp = e1_sp[:nsamples]

print("\nRemove Outliers")
ept2_sp = (h0 + e0_sp/t1_sp + e1_sp/t1_sp - t2_sp*e0_sp/t1_sp**2).real
mask = sampling.filter_outliers(ept2_sp, zeta=30)

wt_sp = wt_sp[mask]
t1_sp = t1_sp[mask]
t2_sp = t2_sp[mask]
e0_sp = e0_sp[mask]
e1_sp = e1_sp[mask]

print(f"Removed {np.sum(~mask)} Outliers")
print(f"Outliers Energy {ept2_sp[~mask]}")

print("\nBlocking Analysis")

print("\nOverlap Ratio:")
_, t1, t1_err = sampling.blocking(wt_sp, t1_sp.real, final=True)

print("\nAFQMC/pt2CCSD:")
weight, energy, err = sampling.pt2blocking(h0, wt_sp, t1_sp, t2_sp, e0_sp, e1_sp, final=True)

runtime = time.time() - init_time
h, rem = divmod(runtime, 3600)
m, s = divmod(rem, 60)
runtime_str = f"{int(h):d}h {int(m):02d}m {s:05.2f}s" if h else \
              f"{int(m):d}m {s:05.2f}s" if m else f"{s:.2f}s"

print("\n" + "=" * 50)
print("  AFQMC/pt2CCSD Result")
print("-" * 50)
print(f"  {'Average weight':<16s}{weight:>24.5f}")
print(f"  {'Energy (Ha)':<16s}{energy:>16.5f} +/- {err:<.5f}")
print(f"  {'Run time':<16s}{runtime_str:>24s}")
print("=" * 50)
print("\nAFQMC Sampling Finished\n")

# print(f"\n Final AFQMC/pt2CCSD overlap ratio: {t1.real:.5f} +/- {t1_err:.5f}")
# print(f"Final AFQMC/pt2CCSD energy: {ept2:.5f} +/- {ept2_err:.5f} weight = {weight:.5f}")
# print(f"Total run time: {time.time() - init_time:.2f}")
# print(f"AFQMC Sampling Finished\n")
