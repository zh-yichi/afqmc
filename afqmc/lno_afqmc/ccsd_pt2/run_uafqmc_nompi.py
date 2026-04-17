import time
import argparse
import numpy as np
import jax
from jax import random
from jax import numpy as jnp
from functools import partial
from ad_afqmc import config
from ad_afqmc.lno_afqmc import sampling
from ad_afqmc.lno_afqmc import ulno_afqmc

init_time = time.time()
print = partial(print, flush=True)
print(f'------------------- AFQMC Sampling Started -------------------')

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    if args.use_gpu:
        config.afqmc_config["use_gpu"] = True

config.setup_jax()

ham_data, prop, trial, wave_data, sampler, options = (ulno_afqmc._prep_afqmc())

print(f"Trial: {trial}")
print(f"Sampler: {sampler}")

print(f"norb: {trial.norb}")
print(f"nelec: {trial.nelec}")
print(f"nchol: {sampler.n_chol}")

for op in options:
    if options[op] is not None:
        print(f"{op}: {options[op]}")

### initialize propagation
init_walkers = None
trial_rdm1 = trial.get_rdm1(wave_data)
if "rdm1" not in wave_data:
    wave_data["rdm1"] = trial_rdm1
ham_data = trial._build_measurement_intermediates(ham_data, wave_data)
ham_data = prop._build_propagation_intermediates(ham_data, trial, wave_data)

prop_data = prop.init_prop_data(trial, wave_data, ham_data, init_walkers)
if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError(
        "Initial overlaps are zero. Pass walkers with non-zero overlap."
    )
prop_data["key"] = random.PRNGKey(options["seed"])
prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
e_init = prop_data["e_estimate"]

e0, t1olp, eorb, t2eorb, t2orb, e0bar \
    = trial.calc_eorb_pt2(prop_data['walkers'], ham_data, wave_data)

e0 = jnp.real(e0)[0]
t1olp = jnp.real(t1olp)[0]
eorb = jnp.real(eorb)[0]
t2eorb = jnp.real(t2eorb)[0]
t2orb = jnp.real(t2orb)[0]
e0bar = jnp.real(e0bar)[0]
eorb_pt = jnp.real(eorb/t1olp + t2eorb/t1olp - t2orb*e0bar/t1olp**2)

print(f'Propagating with {options["n_walkers"]} walkers')
print(f"Initial Orbital energy: {eorb_pt:.6f}")
print("----------------------- Equilibration -----------------------")
print(f"{'Inv_T':>5s}  {'E(Guide)':>12s}  {'runTime':>8s}")
print(f"{0.:5.2f}  {e0:12.6f}  {time.time() - init_time:8.2f}")

sampler_eq = sampling.sampler_eq(
    n_prop_steps=50,
    n_ene_blocks=1,
    n_sr_blocks=50,
    n_chol = sampler.n_chol
    )

block_time = prop.dt * sampler_eq.n_prop_steps * sampler_eq.n_ene_blocks * sampler_eq.n_sr_blocks

for n in range(options["n_eql"]):
    prop_data, (wt, e0) = \
        sampler_eq.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)

    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e0
    print(f"{(n+1)*block_time:5.2f}  {e0:12.6f}  {time.time() - init_time:8.2f} ")

print("--------------------- Sampling Blocks -----------------------")
print(f"Target Final Error ~ {options['max_error']:.6f}")
print(f"{'N':>4s}  "
      f"{'E(Guide)':>12s}  {'Error':>8s}  "
      f"{'E(Orb)':>10s}  {'Error':>8s}  "
      f"{'Kill_WK':>7s}  {'Time':>8s}")

wt_sp = np.zeros(sampler.n_blocks,dtype="float64")
e0_sp = np.zeros(sampler.n_blocks,dtype="float64")
eorb_sp = np.zeros(sampler.n_blocks,dtype="complex128") # "float64")
t2eorb_sp = np.zeros(sampler.n_blocks,dtype="complex128") # "float64")
t2orb_sp = np.zeros(sampler.n_blocks,dtype="complex128") # "float64")
e0bar_sp = np.zeros(sampler.n_blocks,dtype="complex128") # "float64")
t1olp_sp = np.zeros(sampler.n_blocks,dtype="complex128") # "float64")
ept_sp = np.zeros(sampler.n_blocks,dtype="float64")
n_killed = 0

for n in range(sampler.n_blocks):
    prop_data, (wt, e0, eorb, t2eorb, t2orb, e0bar, t1olp) = \
        sampler.block_sample_sr(prop_data, ham_data, prop, trial, wave_data)
    
    wt_sp[n] = wt
    e0_sp[n] = e0
    eorb_sp[n] = eorb
    t2eorb_sp[n] = t2eorb
    t2orb_sp[n] = t2orb
    e0bar_sp[n] = e0bar
    t1olp_sp[n] = t1olp
    n_killed += prop_data["n_killed_walkers"]

    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e0
    
    ept_sp[n] = (eorb/t1olp + t2eorb/t1olp - t2orb*e0bar/t1olp**2).real

    # if (n+1) % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:
    if (n+1) % (min(max(sampler.n_blocks // 10, 1), 20)) == 0 and n > 0:          
        wt = np.sum(wt_sp[:n+1])
        e0 = np.sum(wt_sp[:n+1] * e0_sp[:n+1]) / wt
        eorb = np.sum(wt_sp[:n+1] * eorb_sp[:n+1]) / wt
        t2eorb = np.sum(wt_sp[:n+1] * t2eorb_sp[:n+1]) / wt
        t2orb = np.sum(wt_sp[:n+1] * t2orb_sp[:n+1]) / wt
        e0bar = np.sum(wt_sp[:n+1] * e0bar_sp[:n+1]) / wt
        t1olp = np.sum(wt_sp[:n+1] * t1olp_sp[:n+1]) / wt
        
        e0_err = (np.sqrt(np.sum(wt_sp[:n+1] * (e0_sp[:n+1] - e0)**2) / wt / (n+1))).real

        eorb_pt = (eorb/t1olp + t2eorb/t1olp - t2orb*e0bar/t1olp**2).real
        # (p_eorb,p_t2eorb,p_t2orb,p_t2orb,p_t1olp)
        dE = np.array([
            1/t1olp,
            1/t1olp,
            -e0bar/t1olp**2,
            -t2orb/t1olp**2,
            -eorb/t1olp**2 - t2eorb/t1olp**2 + 2*t2orb*e0bar/t1olp**3
            ])
        cov = np.cov([
            eorb_sp[:n+1],
            t2eorb_sp[:n+1],
            t2orb_sp[:n+1],
            e0bar_sp[:n+1], 
            t1olp_sp[:n+1]
            ])
        
        eorb_pt_err = ((np.sqrt(dE @ cov @ dE)) / np.sqrt(n+1)).real
        
        print(f"{n+1:4d}  "
              f"{e0:12.6f}  {e0_err:8.6f}  "
              f"{eorb_pt:10.6f}  {eorb_pt_err:8.6f}  "
              f"{n_killed:7d}  {time.time() - init_time:8.2f}")

        if eorb_pt_err < 0.75 * options["max_error"] and n > 100:
            break

print('---------------------- Post Propagation ---------------------')
nsamples = n + 1
print(f'Total number of samples {nsamples}')

wt_sp = wt_sp[:nsamples]
e0_sp = e0_sp[:nsamples]
eorb_sp = eorb_sp[:nsamples]
t2eorb_sp = t2eorb_sp[:nsamples]
t2orb_sp = t2orb_sp[:nsamples]
e0bar_sp = e0bar_sp[:nsamples]
t1olp_sp = t1olp_sp[:nsamples]
ept_sp = ept_sp[:nsamples]

wt = np.sum(wt_sp)
e0 = np.sum(wt_sp * e0_sp) / wt
eorb = np.sum(wt_sp * eorb_sp) / wt
t2eorb = np.sum(wt_sp * t2eorb_sp) / wt
t2orb = np.sum(wt_sp * t2orb_sp) / wt
e0bar = np.sum(wt_sp * e0bar_sp) / wt
t1olp = np.sum(wt_sp * t1olp_sp) / wt

e0_err = np.sqrt(np.sum(wt_sp * (e0_sp - e0)**2) / wt / nsamples)

eorb_pt = (eorb/t1olp + t2eorb/t1olp - t2orb*e0bar/t1olp**2).real
# (p_eorb,p_t2eorb,p_t2orb,p_t2orb,p_t1olp)
dE = np.array([1/t1olp,
               1/t1olp,
               -e0bar/t1olp**2,
               -t2orb/t1olp**2,
               -eorb/t1olp**2 - t2eorb/t1olp**2 + 2*t2orb*e0bar/t1olp**3
               ])
cov = np.cov([
    eorb_sp,
    t2eorb_sp,
    t2orb_sp,
    e0bar_sp,
    t1olp_sp,
    ])

eorb_pt_cov_err = (np.sqrt(dE @ cov @ dE) / np.sqrt(nsamples)).real
eorb_pt_sp_err = (np.std(ept_sp, ddof=1) / np.sqrt(nsamples)).real

print(f"Raw AFQMC/HF Energy: {e0:.6f} +/- {e0_err:.6f}")
print(f"Raw AFQMC/pt2CCSD Orbital Energy (covariance): {eorb_pt:.6f} +/- {eorb_pt_cov_err:.6f}")
print(f"Raw AFQMC/pt2CCSD Orbital Energy (direct obs): {eorb_pt:.6f} +/- {eorb_pt_sp_err:.6f}")

print("--------------------- Clean Observation ---------------------")
def filter_outliers(ept_sp, zeta=10):

    median = np.median(ept_sp)
    mad = 1.4826 * np.median(np.abs(ept_sp - median))
    bound = zeta * mad
    mask = np.abs(ept_sp - median) < bound
    
    print(f"Energy bound [{median-bound:.6f}, {median+bound:.6f}]")
    
    return mask

mask = filter_outliers(ept_sp, zeta=20)

wt_sp = wt_sp[mask]

nsample_clean = len(wt_sp)
print(f"Removed {nsamples-nsample_clean} Outliers")
print(f"Outliers Orbital Energy {ept_sp[~mask]}")

eorb_sp = eorb_sp[mask]
t2eorb_sp = t2eorb_sp[mask]
t2orb_sp = t2orb_sp[mask]
e0bar_sp = e0bar_sp[mask]
t1olp_sp = t1olp_sp[mask]
ept_sp = ept_sp[mask]

wt = np.sum(wt_sp)
eorb = np.sum(wt_sp * eorb_sp) / wt
t2eorb = np.sum(wt_sp * t2eorb_sp) / wt
t2orb = np.sum(wt_sp * t2orb_sp) / wt
e0bar = np.sum(wt_sp * e0bar_sp) / wt
t1olp = np.sum(wt_sp * t1olp_sp) / wt

eorb_pt = (eorb/t1olp + t2eorb/t1olp - t2orb*e0bar/t1olp**2).real
# (p_eorb,p_t2eorb,p_t2orb,p_t2orb,p_t1olp)
dE = np.array([1/t1olp,
               1/t1olp,
               -e0bar/t1olp**2,
               -t2orb/t1olp**2,
               -eorb/t1olp**2 - t2eorb/t1olp**2 + 2*t2orb*e0bar/t1olp**3,
               ])
cov = np.cov([
    eorb_sp,
    t2eorb_sp,
    t2orb_sp,
    e0bar_sp,
    t1olp_sp,
    ])

eorb_pt_cov_err = (np.sqrt(dE @ cov @ dE) / np.sqrt(nsample_clean)).real
eorb_pt_sp_err = (np.std(ept_sp, ddof=1) / np.sqrt(nsample_clean)).real

print(f"Clean AFQMC/pt2CCSD Orbital Energy (covariance): {eorb_pt:.6f} +/- {eorb_pt_cov_err:.6f}")
print(f"Clean AFQMC/pt2CCSD Orbital Energy (direct obs): {eorb_pt:.6f} +/- {eorb_pt_sp_err:.6f}")

print("--------------------- Blocking Analysis ---------------------")
# nsample = len(wt_sp)
# min_nblocks = 20
# max_size = nsample // min_nblocks
# if max_size < 10:
#     min_nblocks = max(nsample // 10, 3)
#     max_size = nsample // min_nblocks
#     print(f"Warning: small dataset, relaxed min_nblocks to {min_nblocks}")

# max_size = nsample // 4
# block_errs = np.zeros(max_size)
# print(f"{'B':>4s}  {'NB':>4s}  {'NS':>4s}  {'Energy':>10s}  {'Error':>8s}")
# for i, block_size in enumerate(range(1,max_size+1)):
#     n_blocks = nsample // block_size

#     wt_truncated = wt_sp[:n_blocks * block_size]
#     eorb_truncated = eorb_sp[:n_blocks * block_size]
#     t2eorb_truncated = t2eorb_sp[:n_blocks * block_size]
#     t2orb_truncated = t2orb_sp[:n_blocks * block_size]
#     e0bar_truncated = e0bar_sp[:n_blocks * block_size]
#     t1olp_truncated = t1olp_sp[:n_blocks * block_size]

#     wt_eorb = wt_truncated * eorb_truncated
#     wt_t2eorb = wt_truncated * t2eorb_truncated
#     wt_t2orb = wt_truncated * t2orb_truncated
#     wt_e0bar = wt_truncated * e0bar_truncated
#     wt_t1olp = wt_truncated * t1olp_truncated

#     wt = wt_truncated.reshape(n_blocks, block_size)
#     wt_eorb = wt_eorb.reshape(n_blocks, block_size)
#     wt_t2eorb = wt_t2eorb.reshape(n_blocks, block_size)
#     wt_t2orb = wt_t2orb.reshape(n_blocks, block_size)
#     wt_e0bar = wt_e0bar.reshape(n_blocks, block_size)
#     wt_t1olp = wt_t1olp.reshape(n_blocks, block_size)

#     block_eorb = np.sum(wt_eorb, axis=1)     # / block_wt
#     block_t2eorb = np.sum(wt_t2eorb, axis=1) # / block_wt
#     block_t2orb = np.sum(wt_t2orb, axis=1)   # / block_wt
#     block_e0bar = np.sum(wt_e0bar, axis=1)   # / block_wt
#     block_t1olp = np.sum(wt_t1olp, axis=1)   # / block_wt

#     block_energy = (block_eorb/block_t1olp + block_t2eorb/block_t1olp 
#                     - (block_t2orb*block_e0bar)/block_t1olp**2).real
#     block_mean = np.mean(block_energy)
#     block_error = np.std(block_energy, ddof=1) / np.sqrt(n_blocks)
#     print(f'{block_size:4d}  {n_blocks:4d}  {block_size*n_blocks:4d}  {block_mean:10.6f}  {block_error:8.6f}')
#     block_errs[i] = block_error

# blocked = False

# for i, err in enumerate(block_errs):
#     if i > 1 and np.abs((err - block_errs[i-1]) / err) < 0.05:
#         blocked = True
#         break

# if not blocked:
#     err = (block_errs).max()

def blocking_analysis(wt_sp, eorb_sp, t2eorb_sp, t2orb_sp, e0bar_sp, t1olp_sp, min_nblocks=20):
    nsample = len(wt_sp)

    max_size = nsample // min_nblocks
    if max_size < 10:
        min_nblocks = max(nsample // 10, 3)
        max_size = nsample // min_nblocks
        print(f"Warning: small dataset, relaxed min_nblocks to {min_nblocks}")

    block_sizes = np.arange(1, max_size + 1)
    block_errs = np.zeros(max_size)
    block_err_errs = np.zeros(max_size)

    print(f"nsample = {nsample}, max_block_size = {max_size}, min_nblocks = {min_nblocks}")
    print(f"{'Blk_SZ':>6s}  {'NBlk':>5s}  {'NSmp':>5s}  {'Energy':>10s}  {'Error':>8s}  {'dError':>8s}")

    for i, block_size in enumerate(block_sizes):
        n_blocks = nsample // block_size
        sl = slice(0, n_blocks * block_size)

        wt_eorb  = (wt_sp[sl] * eorb_sp[sl]).reshape(n_blocks, block_size)
        wt_t2eorb = (wt_sp[sl] * t2eorb_sp[sl]).reshape(n_blocks, block_size)
        wt_t2orb = (wt_sp[sl] * t2orb_sp[sl]).reshape(n_blocks, block_size)
        wt_e0bar = (wt_sp[sl] * e0bar_sp[sl]).reshape(n_blocks, block_size)
        wt_t1olp = (wt_sp[sl] * t1olp_sp[sl]).reshape(n_blocks, block_size)

        block_eorb  = np.sum(wt_eorb, axis=1)
        block_t2eorb = np.sum(wt_t2eorb, axis=1)
        block_t2orb = np.sum(wt_t2orb, axis=1)
        block_e0bar = np.sum(wt_e0bar, axis=1)
        block_t1olp = np.sum(wt_t1olp, axis=1)

        block_energy = (block_eorb / block_t1olp + block_t2eorb / block_t1olp
                        - (block_t2orb * block_e0bar) / block_t1olp**2).real

        block_mean = np.mean(block_energy)
        block_error = np.std(block_energy, ddof=1) / np.sqrt(n_blocks)
        err_of_err = block_error / np.sqrt(2.0 * (n_blocks - 1))

        # block_means[i] = block_mean
        block_errs[i] = block_error
        block_err_errs[i] = err_of_err

        print(f'{block_size:6d}  {n_blocks:5d}  {block_size*n_blocks:5d}  '
              f'{block_mean:10.6f}  {block_error:8.6f}  {err_of_err:8.6f}')

    # Weighted fit: error(B) = a - b * exp(-B / tau)
    from scipy.optimize import curve_fit

    def model(x, a, b, tau):
        return a - b * np.exp(-x / tau)

    p0 = [block_errs.max(), block_errs.max() - block_errs[0], 5.0]
    try:
        popt, pcov = curve_fit(model, block_sizes, block_errs,
                               sigma=block_err_errs, absolute_sigma=True,
                               p0=p0, maxfev=10000)
        plateau_value = popt[0]
        plateau_uncertainty = np.sqrt(pcov[0, 0])
        tau = popt[2]
        ratio = 0.01 * popt[0] / popt[1]
        if ratio > 0:
            plateau_block_size = int(np.ceil(-popt[2] * np.log(ratio)))
        else:
            plateau_block_size = 1
        # plateau_block_size = min(plateau_block_size, max_size)
        print(f"Fit: plateau = {plateau_value:.6f} ± {plateau_uncertainty:.6f}")
        print(f"     autocorrelation length ~ {tau:.1f} blocks")
        print(f"     plateau reached at block size ~ {plateau_block_size}")
        if plateau_block_size > max_size:
            print(f"     !!!Failed to reach plateau in blocking")
            print(f"     Return max block error")
            idx_max = np.argmax(block_errs)
            plateau_value = block_errs[idx_max]
    except RuntimeError as e:
        print(f"\nFit failed: {e}")
        idx_max = np.argmax(block_errs)
        plateau_value = block_errs[idx_max]
        plateau_uncertainty = block_err_errs[idx_max]
        plateau_block_size = block_sizes[idx_max]
        popt, pcov = None, None
        print(f"Fallback max error: {plateau_value:.6f} +/- {plateau_uncertainty:.6f}")
        print(f"     plateau at block size ~ {plateau_block_size}")

    return plateau_value

plateau_value = blocking_analysis(wt_sp, eorb_sp, t2eorb_sp, t2orb_sp, e0bar_sp, t1olp_sp, min_nblocks=20)

print(f"Clean AFQMC/pt2CCSD Orbital Energy (blocking): {eorb_pt:.6f} +/- {plateau_value:.6f}")
print(f"total run time: {time.time() - init_time:.2f}")
print(f'------------------ AFQMC Sampling Finished -------------------')
