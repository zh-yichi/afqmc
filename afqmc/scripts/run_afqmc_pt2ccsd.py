import time

import numpy as np
# from jax import random
# from jax import numpy as jnp

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

sampler_eq = sampling.sampler(
    n_prop_steps=50, 
    n_chol = sampler.n_chol
    )

block_time = prop.dt * sampler_eq.n_prop_steps
neql_block = int(-(-options["eql_time"] // block_time))

for n in range(1, neql_block+1):
    prop_data, (wt, e) \
        = sampler_eq.block_sample(prop_data, ham_data, prop, trial, wave_data)

    if (n+1) % (min(max(neql_block // 10, 1), 20)) == 0 and n > 0:
        nodes = prop_data["n_killed_walkers"]
        print(f"{(n+1)*block_time:5.2f}  "
              f"{wt:12.5f}  {nodes:5d}  "
              f"{e:12.5f}  {time.time() - init_time:8.2f}")

print("\nSampling")
print(f"Target (raw) 0.6 x max_error = {0.6 * options["max_error"]:.5f})")
print(f"{'blocks':>6s}  "
      f"{'weight':>12s}  {'nodes':>5s}  "
      f"{'E_Guide':>12s}  {'error':>8s}  "
      f"{'E_Trial':>12s}  {'error':>8s}  "
      f"{'olp_T/G':>10s}  {'error':>8s}  "
      f"{'Walltime':>10s}")

wt_sp = np.zeros(sampler.n_blocks, dtype="float64")
eg_sp = np.zeros(sampler.n_blocks, dtype="float64")
t1_sp = np.zeros(sampler.n_blocks, dtype="complex128")
t2_sp = np.zeros(sampler.n_blocks, dtype="complex128")
e0_sp = np.zeros(sampler.n_blocks, dtype="complex128")
e1_sp = np.zeros(sampler.n_blocks, dtype="complex128")
ept_sp = np.zeros(sampler.n_blocks, dtype="float64")
# n_killed = np.zeros(sampler.n_blocks,dtype="int32")

for n in range(sampler.n_blocks):
    prop_data, (wt, eg, t1, t2, e0, e1) =\
        sampler.block_sample(prop_data, ham_data, prop, trial, wave_data)
    
    wt_sp[n] = wt
    eg_sp[n] = eg
    t1_sp[n] = t1
    t2_sp[n] = t2
    e0_sp[n] = e0
    e1_sp[n] = e1
    nodes = prop_data["n_killed_walkers"]

    ept = (h0 + 1/t1*e0 + 1/t1*e1 - 1/t1**2 * t2 * e0).real
    ept_sp[n] = ept

    if (n+1) % (min(max(sampler.n_blocks // 10, 1), 20)) == 0 and n > 0:
        weight = np.mean(wt_sp[:n+1])
        eg, eg_err = sampling.blocking(wt_sp[:n+1], eg_sp[:n+1], final=False)
        ept, ept_err = sampling.pt2blocking(
            h0, wt_sp[:n+1], t1_sp[:n+1], t2_sp[:n+1], e0_sp[:n+1], e1_sp[:n+1], final=False)
        otg, otg_err = sampling.blocking(wt_sp[:n+1], t1_sp[:n+1].real, final=False)
        prop_data["e_estimate"] = 0.8 * prop_data["e_estimate"] + 0.2 * eg
        
        print(f"{n+1:6d}  "
              f"{weight:12.5f}  {nodes:5d}  "
              f"{eg:12.5f}  {eg_err:8.5f}  "
              f"{ept:12.5f}  {ept_err:8.5f}  "
              f"{otg.real:10.6f}  {otg_err.real:8.5f}"
              f"{time.time() - init_time:10.2f}")
        
        if ept_err < 0.6 * options["max_error"] and n > 120:
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
ept_sp = ept_sp[:nsamples]

# weight = np.mean(wt_sp)
# eg = np.mean(wt_sp * eg_sp) / weight
# t1 = np.mean(wt_sp * t1_sp) / weight
# t2 = np.mean(wt_sp * t2_sp) / weight
# e0 = np.mean(wt_sp * e0_sp) / weight
# e1 = np.mean(wt_sp * e1_sp) / weight

# eg_err = sampling.blocking(wt_sp, eg_sp, min_nblocks=40, final=False)
# t1_err = sampling.blocking(wt_sp, t1_sp.real, min_nblocks=40, final=False)

# ept = (h0 + 1/t1 * e0 + 1/t1 * e1 - 1/t1**2 * t2 * e0).real
# # dE = (pE/pt1,pE/pt2,pE/pe0,pE/pe1)
# # dE = np.array([-1/t1**2 * (e0+e1) + 2/t1**3 * t2 * e0,
# #                -1/t1**2 * e0,
# #                 1/t1 - 1/t1**2 * t2,
# #                 1/t1])
# # cov_te0e1 = np.cov([t1_sp, t2_sp, e0_sp, e1_sp])
# # ept_cov_err = (np.sqrt(dE @ cov_te0e1 @ dE) / np.sqrt(len(wt_sp))).real

# ept_sp_err = np.std(ept_sp) / np.sqrt(nsamples)
# t1_err = np.sqrt(np.sum(wt_sp * (t1_sp - t1)**2) / wt / nsamples).real

# print(f"Raw AFQMC/HF (Guide) energy:           {eg:.5f} ± {eg_err:.5f}")
# print(f"Raw Trial/Guide overlap ratio:          {t1.real:.5f} ± {t1_err:.5f}")
# print(f"Raw AFQMC/pt2CCSD energy (covariance): {ept:.5f} ± {ept_cov_err:.5f}")
# print(f"Raw AFQMC/pt2CCSD energy (dir sample): {ept:.5f} ± {ept_sp_err:.5f}")

print("\nRemove Outliers")
mask = sampling.filter_outliers(ept_sp, zeta=30)

wt_clean = wt_sp[mask]
eg_clean = eg_sp[mask]
t1_clean = t1_sp[mask]
t2_clean = t2_sp[mask]
e0_clean = e0_sp[mask]
e1_clean = e1_sp[mask]
ept_clean = ept_sp[mask]

nclean = len(wt_clean)
print(f"Removed {nsamples-nclean} outliers with energies {ept_sp[~mask]}")

print("\nBlocking Analysis")
print("\nGuide:")
eg, eg_err = sampling.blocking(wt_clean, eg_clean.real, final=True)
print("\nOverlap Ratio:")
t1, t1_err = sampling.blocking(wt_clean, t1_clean.real, final=True)
print("\nPT2:")
ept, ept_err = sampling.pt2blocking(h0, wt_clean, t1_clean, t2_clean, e0_clean, e1_clean, final=True)

print(f"\nFinal AFQMC/HF (Guide) energy:           {eg:.5f} ± {eg_err:.5f}")
print(f"Final AFQMC/pt2CCSD overlap ratio: {t1.real:.5f} ± {t1_err:.5f}")
print(f"Final AFQMC/pt2CCSD energy: {ept:.5f} ± {ept_err:.5f}")
print(f"Total run time: {time.time() - init_time:.2f}")
print(f"AFQMC Sampling Finished\n")
