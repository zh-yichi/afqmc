import time
import argparse
import numpy as np
from jax import random
from jax import numpy as jnp
from ad_afqmc import config
from ad_afqmc.prop_unrestricted.mixed_wave import prep, sampling
from functools import partial
print = partial(print, flush=True)

init_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    if args.use_gpu:
        config.afqmc_config["use_gpu"] = True

config.setup_jax()

ham_data, ham, prop, trial, wave_data, sampler, options = (prep._prep_afqmc())

print(f'Sampler: {sampler}')
print(f'Trial: {trial}')

init_walkers = None
trial_rdm1 = trial.get_rdm1(wave_data)
if "rdm1" not in wave_data:
    wave_data["rdm1"] = trial_rdm1
ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)

prop_data = prop.init_prop_data(trial, wave_data, ham_data, init_walkers)

if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError(
        "Initial overlaps are zero. Pass walkers with non-zero overlap."
    )

prop_data["key"] = random.PRNGKey(options["seed"])
xtaus, prop_data = trial.get_xtaus(prop_data, wave_data, prop)

prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
ehf_init = jnp.real(prop_data["e_estimate"])

olp_cc, ene_cc = trial.calc_energy_stoccsd(prop_data["walkers"], xtaus, ham_data, wave_data)
olp_hf = prop_data["overlaps"]
wt_hf = prop_data["weights"]
ecc_init = jnp.real(jnp.sum(wt_hf*olp_cc/olp_hf*ene_cc)/jnp.sum(wt_hf*olp_cc/olp_hf))

ehf = jnp.real(trial.calc_energy(prop_data["walkers"], ham_data, wave_data))[0]

print(f'initial AFQMC/stoCCSD Energy is {ecc_init:.6f}')
print(f'Propagating with {options["n_walkers"]} walkers')
print(f"---------------- Equilibration sweeps -----------------")
print(f"{'inv_T':>5s}  {'energy':>10s}  {'runTime':>8s}")
now = time.time() - init_time
print(f"{0.:5.2f}  {ehf:10.6f}  {now:8.2f}")

sampler_eq = sampling.sampler(
    n_prop_steps = 50, 
    n_ene_blocks = 1, 
    n_sr_blocks = 50, 
    n_chol = sampler.n_chol
    )

block_time = prop.dt * sampler_eq.n_prop_steps * sampler_eq.n_ene_blocks * sampler_eq.n_sr_blocks

for n in range(1,options["n_eql"]+1):
    prop_data, (wt, e) = \
        sampler_eq.propagate_phaseless(
            prop_data, ham_data, prop, trial, wave_data)
    
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e 
    now = time.time() - init_time
    print(f"{n*block_time:5.2f}  {e:10.6f}  {now:8.2f}")

print("----------------- Sampling sweeps --------------------")
print(f"{'NB':>5s}  "
      f"{'energy/hf':>10s}  {'error':>8s}  "
      f"{'energy/cc':>10s}  {'error':>8s}  "
      f"{'overlapT/G':>10s}  {'error':>8s}  "
      f"{'runTime':>8s}")

whf_sp = np.zeros(sampler.n_blocks,dtype="float64")
ehf_sp = np.zeros(sampler.n_blocks,dtype="float64")
num_sp = np.zeros(sampler.n_blocks,dtype="complex128")
den_sp = np.zeros(sampler.n_blocks,dtype="complex128")

for n in range(sampler.n_blocks):
    prop_data, (whf, ehf, num, den) \
        = sampler.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)

    whf_sp[n] = whf
    ehf_sp[n] = ehf
    num_sp[n] = num
    den_sp[n] = den
    ecc_estimate = jnp.real(num / den)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * ecc_estimate

    if (n+1) % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:
        weight = np.sum(whf_sp[:n+1])
        whf_ehf = np.sum(whf_sp[:n+1] * ehf_sp[:n+1])
        whf_num = np.sum(whf_sp[:n+1] * num_sp[:n+1])
        whf_den = np.sum(whf_sp[:n+1] * den_sp[:n+1])

        ehf_avg = np.real(whf_ehf / weight)
        ehf_err = np.sqrt(np.sum(whf_sp[:n+1] * (ehf_sp[:n+1]-ehf_avg)**2) / weight) / np.sqrt(n)

        ecc_avg = np.real(whf_num / whf_den)
        ecc_sp = np.real((whf_sp[:n+1] * num_sp[:n+1]) / (whf_sp[:n+1] * den_sp[:n+1]))
        ecc_err = np.real(np.std(ecc_sp) / np.sqrt(n))
        
        den_avg = np.real(whf_den / weight)
        den_err = np.sqrt(np.sum(whf_sp[:n+1] * (den_sp[:n+1] - den_avg)**2) / weight / n).real
        # print(f" {n+1:4d}  {ehf_avg}  {ehf_err}  {ecc_avg}  {ecc_err}  {time.time() - init_time:.2f}")
        now = time.time() - init_time
        print(f"{n+1:5d}  "
              f"{ehf_avg:10.6f}  {ehf_err:8.6f}  "
              f"{ecc_avg:10.6f}  {ecc_err:8.6f}  "
              f"{den_avg:10.6f}  {den_err:8.6f}  "
              f"{now:8.2f}")

print('------------- Post Propagation -----------------')

def filter_outliers(wts, nums, dens, zeta=10):
    energies = (nums / dens).real
    median = np.median(energies)
    mad = 1.4826 * np.median(np.abs(energies - median))
    bound = zeta * mad
    mask = np.abs(energies - median) < bound
    
    wts_filtered = wts[mask]
    nums_filtered = nums[mask]
    dens_filtered = dens[mask]
    
    n_removed = len(wts) - len(wts_filtered)
    print(f"Outlier energy bound [{median-bound:.6f}, {median+bound:.6f}]")
    print(f"Removed {n_removed} outliers with energies {energies[~mask]}")
    
    return wts_filtered, nums_filtered, dens_filtered

whf_clean, num_clean, den_clean = filter_outliers(whf_sp, num_sp, den_sp, zeta=10)

nclean = len(whf_clean)
whf = np.sum(whf_clean)
whf_num = np.sum(whf_clean * num_clean)
whf_den = np.sum(whf_clean * den_clean)

ecc_clean = np.real(whf_num / whf_den)
block_errs = sampler.blk_average(whf_clean, num_clean, den_clean, printE=True)

den_avg = np.real(whf_den / whf)
den_err = np.sqrt(np.sum(whf_clean * (den_clean - den_avg)**2) / whf / nclean).real
print(f'Clean AFQMC/stoCCSD Overlap Ratio: {den_avg:.6f} +/- {den_err:.6f}')

from scipy.optimize import curve_fit
block_sizes = np.arange(1, len(block_errs) + 1)

# Model: error(x) = A - B * exp(-x / tau) find A
def model(x, a, b, tau):
    return a - b * np.exp(-x / tau)

p0 = [block_errs[-1], block_errs[-1] - block_errs[0], 5.0]
popt, pcov = curve_fit(model, block_sizes, block_errs, p0=p0, maxfev=10000)
plateau_value = popt[0]
perr = np.sqrt(np.diag(pcov))

print(f"Plateau error estimate: {plateau_value:.6f} ± {perr[0]:.6f}")
print(f"Decay constant (tau):   {popt[2]:.2f} blocks")

# find the block size where error reaches 95% of plateau
convergence_block = -popt[2] * np.log(0.05)
print(f"~95% of plateau reached at block_size ≈ {convergence_block:.0f}")
if convergence_block > nclean or convergence_block < 0:
    print(f"Plateau not reached within sampled blocks, use max error")
    plateau_value = block_errs.max()
print(f'Clean AFQMC/stoCCSD Energy: {ecc_clean:.6f} +/- {plateau_value:.6f}')
print(f"total run time: {time.time() - init_time:.2f}")
