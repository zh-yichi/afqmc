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

txt_width = 130
print(f"{' AFQMC Sampling Started ':-^{txt_width}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    if args.use_gpu:
        config.afqmc_config["use_gpu"] = True

config.setup_jax()

ham_data, ham, prop, trial, wave_data, sampler, options = (prep._prep_afqmc())

print(f"Sampler is {sampler}")
print(f"Propagator is {prop}")
print(f"Trial is {trial}")

trial_rdm1 = trial.get_rdm1(wave_data)
if "rdm1" not in wave_data:
    wave_data["rdm1"] = trial_rdm1

ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)
prop_data = prop.init_prop_data(trial, wave_data, ham_data, init_walkers = None)

init_e = prop_data["e_estimate"]
init_w = np.sum(prop_data["weights"])

if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError("Initial overlaps are zero. Pass walkers with non-zero overlap.")

prop_data["key"] = random.PRNGKey(options["seed"])
xtaus, prop_data = trial.get_xtaus(prop_data, wave_data, prop)

prop_data["n_killed_walkers"] = 0
oci, eci = trial.calc_energy_cid(prop_data["walkers"], ham_data, wave_data)
numcr, dencr = trial.calc_correction(prop_data["walkers"], xtaus, ham_data, wave_data)

eci_init = jnp.real(eci)[0]
ecc_init = jnp.real((oci*eci + numcr) / (oci + dencr))[0]
print(f'Initial AFQMC/CISD Energy is {eci_init:.6f}')
print(f'Initial AFQMC/stoCCSD Energy is {ecc_init:.6f}')


print(f"{' Equilibration ':-^{txt_width}}")
print(f"{'inv_T':>5s}  "
      f"{'weight':>12s}  {'killW':>5s}  "
      f"{'energy':>12s}  {'runTime':>8s}")
print(f"{0.:5.2f}  "
      f"{init_w:12.6f}  {0:5d}  "
      f"{init_e:12.6f}  {time.time() - init_time:8.2f}")

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

    if (n+1) % (min(max(options["n_eql"] // 10, 1), 20)) == 0 and n > 0:
        nkill = prop_data["n_killed_walkers"]
        print(f"{(n+1)*block_time:5.2f}  "
              f"{wt:12.6f}  {nkill:5d}  "
              f"{e:12.6f}  {time.time() - init_time:8.2f}")

print(f"{' Sampling Blocks ':-^{txt_width}}")
# print("------------------- Sampling sweeps ---------------------")
print(f"{'N':>5s}  "
      f"{'energy/HF':>10s}  {'error':>8s}  "
      f"{'energy/CI':>10s}  {'error':>8s}  "
      f"{'energy/CC':>10s}  {'error':>8s}  "
      f"{'o[CI/HF]':>10s}  {'error':>8s}  "
      f"{'o[CC/HF]':>10s}  {'error':>8s}  "
      f"{'runTime':>8s}")

whf_sp = np.zeros(sampler.n_blocks, dtype="float64")
ehf_sp = np.zeros(sampler.n_blocks,dtype="float64")
numci_sp = np.zeros(sampler.n_blocks, dtype="complex128")
denci_sp = np.zeros(sampler.n_blocks, dtype="complex128")
numcr_sp = np.zeros(sampler.n_blocks, dtype="complex128")
dencr_sp = np.zeros(sampler.n_blocks, dtype="complex128")
ecc_sp = np.zeros(sampler.n_blocks,dtype="float64")

for n in range(sampler.n_blocks):
    prop_data, (whf, ehf, numci, denci, numcr, dencr) \
        = sampler.block_sample(prop_data, ham_data, prop, trial, wave_data)

    whf_sp[n] = whf
    ehf_sp[n] = ehf
    numci_sp[n] = numci
    denci_sp[n] = denci
    numcr_sp[n] = numcr
    dencr_sp[n] = dencr

    ecc_estimate = jnp.real((numci + numcr) / (denci + dencr))
    ecc_sp[n] = ecc_estimate

    if (n+1) % (min(max(sampler.n_blocks // 10, 1), 20)) == 0 and n > 0:

        weight = np.sum(whf_sp[:n+1])

        whf_ehf = whf_sp[:n+1] * ehf_sp[:n+1]
        whf_numci = whf_sp[:n+1] * numci_sp[:n+1]
        whf_denci = whf_sp[:n+1] * denci_sp[:n+1]
        whf_numcr = whf_sp[:n+1] * numcr_sp[:n+1]
        whf_dencr = whf_sp[:n+1] * dencr_sp[:n+1]

        numci_avg = np.sum(whf_numci) / weight
        denci_avg = np.sum(whf_denci) / weight
        numcr_avg = np.sum(whf_numcr) / weight
        dencr_avg = np.sum(whf_dencr) / weight

        # denci_avg = denci_avg.real
        # denci_err = np.sqrt(np.sum(whf_sp[:n+1] * (denci_sp[:n+1] - denci_avg)**2) / weight / n).real
        denci_err = sampler.blocking_analysis(whf_sp[:n+1], (denci_sp[:n+1]).real, min_nblocks=40, final=False)

        dencc_avg = (denci_avg + dencr_avg).real
        # dencc_err = np.sqrt(np.sum(whf_sp[:n+1] * (denci_sp[:n+1] + dencr_sp[:n+1] - dencc_avg)**2) / weight / n).real
        dencc_err = sampler.blocking_analysis(whf_sp[:n+1], (denci_sp[:n+1] + dencr_sp[:n+1]).real, min_nblocks=40, final=False)

        ehf_avg = np.sum(whf_ehf) / weight        
        # ehf_err = np.sqrt(np.sum(whf_sp[:n+1] * (ehf_sp[:n+1]-ehf_avg)**2) / weight) / np.sqrt(n)
        ehf_err = sampler.blocking_analysis(whf_sp[:n+1], (ehf_sp[:n+1]).real, min_nblocks=40, final=False)

        eci_avg = (numci_avg / denci_avg).real
        ecis = (whf_numci / whf_denci).real
        eci_err = np.std(ecis) / np.sqrt(n)

        ecc_avg = ((numci_avg + numcr_avg) / (denci_avg + dencr_avg)).real
        eccs = ((whf_numci + whf_numcr) / (whf_denci + whf_dencr)).real
        ecc_err = np.std(eccs) / np.sqrt(n)

        prop_data["pop_control_ene_shift"] = 0.8 * prop_data["pop_control_ene_shift"] + 0.2 * ehf_avg

        print(f"{n+1:5d}  "
              f"{ehf_avg:10.6f}  {ehf_err:8.6f}  "
              f"{eci_avg:10.6f}  {eci_err:8.6f}  "
              f"{ecc_avg:10.6f}  {ecc_err:8.6f}  "
              f"{denci_avg.real:10.6f}  {denci_err:8.6f}  "
              f"{dencc_avg:10.6f}  {dencc_err:8.6f}  "
              f"{time.time() - init_time:8.2f}")
        
        if ecc_err < 0.75 * options["max_error"] and n > 100:
            break

print(f"{' Post Propagation ':-^{txt_width}}")
nsamples = n + 1
whf_sp = whf_sp[:nsamples]
ehf_sp = ehf_sp[:nsamples]
numci_sp = numci_sp[:nsamples]
denci_sp = denci_sp[:nsamples]
numcr_sp = numcr_sp[:nsamples]
dencr_sp = dencr_sp[:nsamples]
ecc_sp = ecc_sp[:nsamples]

whf = np.sum(whf_sp)
ehf_avg = np.sum(whf_sp * ehf_sp) / whf
numci_avg = np.sum(whf_sp * numci_sp) / whf
denci_avg = np.sum(whf_sp * denci_sp) / whf
numcr_avg = np.sum(whf_sp * numcr_sp) / whf
dencr_avg = np.sum(whf_sp * dencr_sp) / whf

occ_avg = (denci_avg + dencr_avg).real
ecc_avg = ((numci_avg + numcr_avg) / (denci_avg + dencr_avg)).real

occ_err = sampler.blocking_analysis(whf_sp, (denci_sp + dencr_sp).real, min_nblocks=40, final=False)
ehf_err = sampler.blocking_analysis(whf_sp, ehf_sp, min_nblocks=40, final=False)
ecc_err = sampler.sto_blocking_analysis(
    whf_sp, 
    (numci_sp + numcr_sp), 
    (denci_sp + dencr_sp), 
    min_nblocks=20, 
    final=False,
    )

print(f"Raw O[CC/HF]:                           {occ_avg.real:.6f} ± {occ_err:.6f}")
print(f"Raw AFQMC/HF (Guide) energy:           {ehf_avg:.6f} ± {ehf_err:.6f}")
print(f"Raw AFQMC/stoDiffCCSD energy:          {ecc_avg:.6f} ± {ecc_err:.6f}")

print(f"{' Clean Obeservation ':-^{txt_width}}")
mask = sampler.filter_outliers(ecc_sp, zeta=30)

whf_sp = whf_sp[mask]
numci_sp = numci_sp[mask]
denci_sp = denci_sp[mask]
numcr_sp = numcr_sp[mask]
dencr_sp = dencr_sp[mask]

print(f"Removed {sum(~mask)} outliers ")

whf = np.sum(whf_sp)
ehf_avg = np.sum(whf_sp * ehf_sp) / whf
numci_avg = np.sum(whf_sp * numci_sp) / whf
denci_avg = np.sum(whf_sp * denci_sp) / whf
numcr_avg = np.sum(whf_sp * numcr_sp) / whf
dencr_avg = np.sum(whf_sp * dencr_sp) / whf

occ_avg = (denci_avg + dencr_avg).real
ecc_avg = ((numci_avg + numcr_avg) / (denci_avg + dencr_avg)).real

occ_err = sampler.blocking_analysis(whf_sp, (denci_sp + dencr_sp).real, min_nblocks=40, final=True)
ehf_err = sampler.blocking_analysis(whf_sp, ehf_sp, min_nblocks=40, final=True)
ecc_err = sampler.sto_blocking_analysis(
    whf_sp, 
    (numci_sp + numcr_sp), 
    (denci_sp + dencr_sp), 
    min_nblocks=20, 
    final=True,
    )

print(f"Final O[CC/HF]:                           {occ_avg.real:.6f} ± {occ_err:.6f}")
print(f"Final AFQMC/HF (Guide) energy:           {ehf_avg:.6f} ± {ehf_err:.6f}")
print(f"Final AFQMC/stoDiffCCSD energy:          {ecc_avg:.6f} ± {ecc_err:.6f}")

print(f"Total run time: {time.time() - init_time:.2f}")
print(f"{' AFQMC Sampling Finished ':-^{txt_width}}")
