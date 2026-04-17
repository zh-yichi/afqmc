import time
import argparse
import numpy as np
from jax import random
from jax import numpy as jnp
from afqmc import config
from afqmc import prep, sampling
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

print(f'------------------- AFQMC Sampling Started -------------------')

ham_data, ham, prop, trial, wave_data, sampler, options = (prep._prep_afqmc())

init_walkers = None
trial_rdm1 = trial.get_rdm1(wave_data)
if "rdm1" not in wave_data:
    wave_data["rdm1"] = trial_rdm1
ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)

prop_data = prop.init_prop_data(trial, wave_data, ham_data, init_walkers)
if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError("Initial overlaps are zero. Pass walkers with non-zero overlap.")
prop_data["key"] = random.PRNGKey(options["seed"])

prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
e_init = prop_data["e_estimate"]

print(f'Propagating with {options["n_walkers"]} walkers')
print("----------------------- Equilibration -----------------------")
print(f"{'inv_T':>5s}  {'energy':>10s}  {'runTime':>8s}")
print(f"{0.:5.2f}  {e_init:10.6f}  {time.time() - init_time:8.2f}")

sampler_eq = sampling.sampler(
    n_prop_steps=50, 
    n_ene_blocks=1, 
    n_sr_blocks=1, 
    n_chol = sampler.n_chol
    )
block_time = prop.dt * sampler_eq.n_prop_steps * sampler_eq.n_ene_blocks * sampler_eq.n_sr_blocks

for n in range(1,options["n_eql"]+1):
    prop_data, (wt, e) \
        = sampler_eq.block_sample(prop_data, ham_data, prop, trial, wave_data)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e

    if (n+1) % (min(max(options["n_eql"] // 10, 1), 20)) == 0 and n > 0:
        print(f"{(n+1)*block_time:5.2f}  {e:10.6f}  {time.time() - init_time:8.2f}")

print("--------------------- Sampling Blocks -----------------------")
print(f"{'N':>4s}  {'weight':>12s}  {'killW':>5s}  "
      f"{'energy':>12s}  {'error':>8s}  {'runTime':>8s}")

wt_sp = np.zeros(sampler.n_blocks,dtype="float64")
e_sp = np.zeros(sampler.n_blocks,dtype="float64")
n_killed = np.zeros(sampler.n_blocks,dtype="int32")

for n in range(sampler.n_blocks):
    prop_data, (wt, e) \
        = sampler.block_sample(prop_data, ham_data, prop, trial, wave_data)
    
    wt_sp[n] = wt
    e_sp[n] = e
    n_killed[n] = prop_data["n_killed_walkers"]
    prop_data["n_killed_walkers"] = 0
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e

    if (n+1) % (min(max(sampler.n_blocks // 10, 1), 20)) == 0 and n > 0:
        weight = np.mean(wt_sp[:n+1])
        energy = np.mean(wt_sp[:n+1] * e_sp[:n+1]) / weight
        err = sampler.blocking_analysis(wt_sp[:n+1], e_sp[:n+1], min_nblocks=20, final=False)
        tot_kw = np.sum(n_killed)
        print(f"{n+1:4d}  {weight:12.6f}  {tot_kw:5d}  "
              f"{energy:12.6f}  {err:8.6f}  {time.time() - init_time:8.2f}")
        
        if err < 0.75 * options["max_error"] and n > 100:
            break

print('---------------------- Post Propagation ---------------------')
nsamples = n + 1
print(f'total number of samples {nsamples}')
wt_sp = wt_sp[:nsamples]
e_sp = e_sp[:nsamples]

energy = np.sum(wt_sp * e_sp) / np.sum(wt_sp)
err = sampler.blocking_analysis(wt_sp, e_sp, min_nblocks=20,final=False)

print(f"Raw AFQMC: {energy:.6f} +/- {err:.6f}")

print("--------------------- Clean Observation ---------------------")
def filter_outliers(e_sp, zeta=30):

    median = np.median(e_sp)
    mad = 1.4826 * np.median(np.abs(e_sp - median))
    bound = zeta * mad
    mask = np.abs(e_sp - median) < bound
    print(f"removing energies outside zeta > {zeta}")
    print(f"Energy bound [{median-bound:.6f}, {median+bound:.6f}]")
    
    return mask

mask = filter_outliers(e_sp, zeta=30)

wt_sp = wt_sp[mask]
nsample_clean = len(wt_sp)
print(f"Removed {nsamples-nsample_clean} Outliers")
print(f"Outliers AFQMC Energy {e_sp[~mask]}")
e_sp = e_sp[mask]

energy = np.sum(wt_sp * e_sp) / np.sum(wt_sp)
err = sampler.blocking_analysis(wt_sp, e_sp, min_nblocks=20, final=True)

print(f"Final AFQMC: {energy:.6f} +/- {err:.6f}")

print(f"total run time: {time.time() - init_time:.2f}")
print(f'------------------ AFQMC Sampling Finished -------------------')