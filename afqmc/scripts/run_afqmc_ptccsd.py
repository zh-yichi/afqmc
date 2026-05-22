import time
import numpy as np
from functools import partial
from afqmc import config, prep, sampling

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

prop_data = prep.init_ccsd_prop_data(
    wave_data, ham_data, prop, trial,
    options["n_walkers"], options["walker_type"], options["seed"],
)

init_e = prop_data["e_estimate"]
init_w = np.sum(prop_data["weights"])

print("\nEquilibration")
print(f"{'inv_T':>5s}  "
      f"{'weight':>12s}  {'killW':>5s}  "
      f"{'energy':>12s}  {'runTime':>8s}")
print(f"{0.:5.2f}  "
      f"{init_w:12.6f}  {0:5d}  "
      f"{init_e:12.6f}  {time.time() - init_time:8.2f}")

sampler_eq = sampling.sampler(
    n_prop_steps = 50, 
    n_chol = sampler.n_chol
    )

block_time = prop.dt * sampler_eq.n_prop_steps

for n in range(1,options["n_eql"]+1):
    prop_data, (wt, e) \
        = sampler_eq.block_sample(prop_data, ham_data, prop, trial, wave_data)

    if (n+1) % (min(max(options["n_eql"] // 10, 1), 20)) == 0 and n > 0:
        nkill = prop_data["n_killed_walkers"]
        print(f"{(n+1)*block_time:5.2f}  "
              f"{wt:12.6f}  {nkill:5d}  "
              f"{e:12.6f}  {time.time() - init_time:8.2f}")

print("\nSampling")

print(f"{'blocks':>6s}  "
      f"{'weight':>12s}  {'killW':>5s}  "
      f"{'E_Guide':>12s}  {'error':>8s}  "
      f"{'energy':>12s}  {'error':>8s}  "
      f"{'Walltime':>8s}")

wt_sp = np.zeros(sampler.n_blocks, dtype="float64")
t_sp = np.zeros(sampler.n_blocks, dtype="complex128") # dtype="float64") # dtype="complex128")
e0_sp = np.zeros(sampler.n_blocks, dtype="float64") #dtype="float64") # dtype="complex128")
et_sp = np.zeros(sampler.n_blocks, dtype="complex128") #dtype="float64") # dtype="complex128")
ept_sp = np.zeros(sampler.n_blocks, dtype="float64")
n_killed = np.zeros(sampler.n_blocks,dtype="int32")
    
for n in range(sampler.n_blocks):
    prop_data, (wt, t, e0, et) =\
        sampler.block_sample(prop_data, ham_data, prop, trial, wave_data)
    
    wt_sp[n] = wt
    t_sp[n] = t
    e0_sp[n] = e0.real
    et_sp[n] = et
    n_killed[n] = prop_data["n_killed_walkers"]

    ept = (e0 + et - t * (e0 - h0)).real
    ept_sp[n] = ept

    if (n+1) % (min(max(sampler.n_blocks // 10, 1), 20)) == 0 and n > 0:
        weight = np.mean(wt_sp[:n+1])
        t = np.mean(wt_sp[:n+1] * t_sp[:n+1]) / weight
        e0 = np.mean(wt_sp[:n+1] * e0_sp[:n+1]) / weight
        et = np.mean(wt_sp[:n+1] * et_sp[:n+1]) / weight

        e0_err = sampler.blocking_analysis(wt_sp[:n+1], e0_sp[:n+1], min_nblocks=20, final=False)
        
        ept = (e0 + et- t*(e0-h0)).real
        # covariant error (pE/pt,pE/pe0,pE/pet)
        dE = np.array([-e0+h0,1-t,1])
        cov_te0et = np.cov([t_sp[:n+1], e0_sp[:n+1], et_sp[:n+1]])
        ept_err = (np.sqrt(dE @ cov_te0et @ dE) / np.sqrt((n+1))).real
        
        tot_kw = np.sum(n_killed)
        prop_data["e_estimate"] = 0.8 * prop_data["e_estimate"] + 0.2 * e0
        
        print(f"{n+1:6d}  "
              f"{weight:12.6f}  {tot_kw:5d}  "
              f"{e0:12.6f}  {e0_err:8.6f}  "
              f"{ept:12.6f}  {ept_err:8.6f}  "
              f"{time.time() - init_time:8.2f}")
        if ept_err < 0.75 * options["max_error"] and n > 100:
            break

print("\nPost Propagation Process")
nsamples = np.count_nonzero(wt_sp)
print(f'Total number of samples {nsamples}')
wt_sp = wt_sp[:nsamples]
t_sp = t_sp[:nsamples]
e0_sp = e0_sp[:nsamples]
et_sp = et_sp[:nsamples]
ept_sp = ept_sp[:nsamples]

wt = np.sum(wt_sp)
t = np.sum(wt_sp * t_sp) / wt
e0 = np.sum(wt_sp * e0_sp) / wt
et = np.sum(wt_sp * et_sp) / wt

ept = (e0 + et - t*(e0-h0)).real

dE = np.array([-e0+h0, 1-t, 1])
cov_te0et = np.cov([t_sp, e0_sp, et_sp])
ept_cov_err = (np.sqrt(dE @ cov_te0et @ dE)/np.sqrt(nsamples)).real
ept_sp_err = np.std(ept_sp) / np.sqrt(nsamples)

print(f"Raw AFQMC/ptCCSD energy (covariance): {ept:.6f} +/- {ept_cov_err:.6f}")
print(f"Raw AFQMC/ptCCSD energy (dir sample): {ept:.6f} +/- {ept_sp_err:.6f}")

print("\nRemove Outliers")

def filter_outliers(ept_sp, zeta=10):

    median = np.median(ept_sp)
    mad = 1.4826 * np.median(np.abs(ept_sp - median))
    bound = zeta * mad
    mask = np.abs(ept_sp - median) < bound
    
    print(f"Outlier energy bound [{median-bound:.6f}, {median+bound:.6f}]")
    
    return mask

mask = filter_outliers(ept_sp, zeta=20)

wt_clean = wt_sp[mask]
nclean = len(wt_clean)

t_clean = t_sp[mask]
e0_clean = e0_sp[mask]
et_clean = et_sp[mask]
ept_clean = ept_sp[mask]

print(f"Removed {nsamples-nclean} outliers with energies {ept_sp[~mask]}")

wt = np.sum(wt_clean)
t = np.sum(wt_clean * t_clean) / wt
e0 = np.sum(wt_clean * e0_clean) / wt
et = np.sum(wt_clean * et_clean) / wt

ept = (e0 + et - t*(e0-h0)).real

dE = np.array([-e0+h0, 1-t, 1])
cov_te0et = np.cov([t_clean, e0_clean, et_clean])
ept_cov_err = (np.sqrt(dE @ cov_te0et @ dE)/np.sqrt(nclean)).real

ept_sp_err = np.std(ept_clean) / np.sqrt(nclean)

print(f"Clean AFQMC/ptCCSD energy (covariance): {ept:.6f} +/- {ept_cov_err:.6f}")
print(f"Clean AFQMC/ptCCSD energy (dir sample): {ept:.6f} +/- {ept_sp_err:.6f}")

print("\nBlocking Analysis")

plateau_value = sampler.ptblocking_analysis(
    wt_clean, 
    t_clean, 
    e0_clean, 
    et_clean,
    h0,
    min_nblocks=20
    )

print(f"Final AFQMC/pt2CCSD energy: {ept:.6f} ± {plateau_value:.6f}")
print(f"Total run time: {time.time() - init_time:.2f}")

print(f"\nAFQMC Sampling Finished\n")

