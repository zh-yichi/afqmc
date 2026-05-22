import numpy as np
from jax import random, jit
from jax import numpy as jnp

from afqmc import config, prep

import time
from functools import partial

print = partial(print, flush=True)

def error_estimate(h0, wt_trj, t1_trj, t2_trj, e0_trj, e1_trj):
    neql, ntrj = wt_trj.shape
    wt_mean = np.mean(wt_trj, axis=1)
    t1_mean = np.mean(wt_trj * t1_trj, axis=1) / wt_mean
    t2_mean = np.mean(wt_trj * t2_trj, axis=1) / wt_mean
    e0_mean = np.mean(wt_trj * e0_trj, axis=1) / wt_mean
    e1_mean = np.mean(wt_trj * e1_trj, axis=1) / wt_mean

    ept_trj = np.real(h0 + e0_trj / t1_trj + e1_trj / t1_trj - t2_trj * e0_trj / t1_trj**2) # (time, sample)
    ept_mean = np.real(h0 + e0_mean / t1_mean + e1_mean / t1_mean - t2_mean * e0_mean / t1_mean**2) # (time,)

    if ntrj == 1:
        ept_err = None
        
    elif ntrj > 1:
        ept_err  = np.real(np.std(ept_trj, axis=1, ddof=1)) / np.sqrt(ntrj) # (time,)

    return ept_mean, ept_err

config.setup_jax()

print = partial(print, flush=True)

ham_data, ham, prop, trial, wave_data, sampler, options = (prep.init_afqmc())

print(f"Trial is {trial}")
print(f"Propagator is {prop}")
print(f"Sampler is {sampler}")

init_time = time.time()

### initialize propagation
trial_rdm1 = trial.get_rdm1(wave_data)
if "rdm1" not in wave_data:
    wave_data["rdm1"] = trial_rdm1

ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)

seeds = random.randint(random.PRNGKey(options["seed"]), 
                       shape=(sampler.n_trj,),
                       minval=0, 
                       maxval=100*sampler.n_trj)

@partial(jit, static_argnames=("prop", "trial"))
def init_prop_data(wave_data, ham_data, prop, trial, seed):
    prop_data = {}
    prop_data["weights"] = jnp.ones(prop.n_walkers)
    prop_data["key"] = random.PRNGKey(seed)
    init_walkers, prop_data = trial.get_ccsd_walkers(prop_data, wave_data, prop)
    prop_data["walkers"] = init_walkers
    h0 = ham_data["h0"]
    t1_init, t2_init, e0_init, e1_init = trial.calc_energy_pt(prop_data["walkers"], ham_data, wave_data)
    e_init = np.real(h0 + e0_init/t1_init + e1_init/t1_init - t2_init * e0_init / t1_init**2)[0]
    prop_data["e_estimate"] = e_init
    prop_data["pop_control_ene_shift"] = e_init
    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

    return prop_data, (t1_init[0], t2_init[0], e0_init[0], e1_init[0])

# prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
blk_time = prop.dt * sampler.n_prop_steps

# h0 = ham_data["h0"]
# t1_init, t2_init, e0_init, e1_init = trial.calc_energy_pt(prop_data["walkers"], ham_data, wave_data)
# e_init = np.real(h0 + e0_init/t1_init + e1_init/t1_init - t2_init * e0_init / t1_init**2)[0]
# prop_data["e_estimate"] = e_init

# shape (inverse_T, trajectories)
wt_trj = np.zeros((sampler.n_eql_blocks+1, sampler.n_trj), dtype="complex128")
t1_trj = np.zeros((sampler.n_eql_blocks+1, sampler.n_trj), dtype="complex128")
t2_trj = np.zeros((sampler.n_eql_blocks+1, sampler.n_trj), dtype="complex128")
e0_trj = np.zeros((sampler.n_eql_blocks+1, sampler.n_trj), dtype="complex128")
e1_trj = np.zeros((sampler.n_eql_blocks+1, sampler.n_trj), dtype="complex128")

print(f"Propagating with {options['n_walkers']} walkers")

for i in range(sampler.n_trj):
    # prop_data["key"] = random.PRNGKey(seeds[i])
    prop_data, (t1_init, t2_init, e0_init, e1_init) \
        = init_prop_data(wave_data, ham_data, prop, trial, seeds[i])
    e_init = prop_data["e_estimate"]
    w_init = np.sum(prop_data["weights"])
    
    _, (blk_wt, blk_t1, blk_t2, blk_e0, blk_e1) \
        = sampler.scan_eql_blocks(prop_data, ham_data, prop, trial, wave_data)

    wt_trj[0:,i] = w_init
    t1_trj[0:,i] = t1_init
    t2_trj[0:,i] = t2_init
    e0_trj[0:,i] = e0_init
    e1_trj[0:,i] = e1_init
    wt_trj[1:, i] = np.asarray(blk_wt, dtype="complex128")
    t1_trj[1:, i] = np.asarray(blk_t1, dtype="complex128")
    t2_trj[1:, i] = np.asarray(blk_t2, dtype="complex128")
    e0_trj[1:, i] = np.asarray(blk_e0, dtype="complex128")
    e1_trj[1:, i] = np.asarray(blk_e1, dtype="complex128")

    e_mean, e_err = error_estimate(ham_data["h0"],
                                   wt_trj[:,:(i + 1)],
                                   t1_trj[:,:(i + 1)],
                                   t2_trj[:,:(i + 1)], 
                                   e0_trj[:,:(i + 1)], 
                                   e1_trj[:,:(i + 1)])
    if i == 0:
        print(f"Free Projection AFQMC trajector {i+1}/{sampler.n_trj} | seed = {seeds[i]}")
        print(f"{'Inv_T':>6s}  {'Energy':>10s}  {'Error':>8s}  {'Walltime':>8s}")
        # print(f"{0.:6.2f}  {e_init:10.5f}  {0.:8.5f}  {time.time() - init_time:8.2f}")
        for nb in range(len(e_mean)):
            print(f"{(nb)*blk_time:6.2f}  {e_mean[nb]:10.5f}  {'N/A':>8s}  {time.time() - init_time:8.2f} ")
    
    elif (i+1) % (min(max(sampler.n_trj // 10, 1), 20)) == 0 and i > 0:
        print(f"Free Projection AFQMC trajector {i+1}/{sampler.n_trj} | seed = {seeds[i]}")
        print(f"{'Inv_T':>6s}  {'Energy':>10s}  {'Error':>8s}  {'Walltime':>8s}")
        # print(f"{0.:6.2f}  {e_init:10.5f}  {0.:8.5f}  {time.time() - init_time:8.2f}")
        for nb in range(len(e_mean)):
            print(f"{(nb)*blk_time:6.2f}  {e_mean[nb]:10.5f}  {e_err[nb]:8.5f}  {time.time() - init_time:8.2f} ")

# ── Model: E(beta) = E_inf + A * exp(-beta / beta_c) ──────────────────────
from scipy.optimize import curve_fit

beta = blk_time * np.arange(1, sampler.n_eql_blocks+1)
energy = e_mean[1:]
denergy = e_err[1:]
# print(beta)
# print(energy)
# print(denergy)

def exp_plateau(tau, E_inf, A, beta_c):
    return E_inf + A * np.exp(-tau / beta_c)

# Initial guesses: E_inf ~ last E, A ~ E(0)-E_inf, beta_c ~ 1/0.3 ≈ 3.3
p0 = [e_mean[-1], e_mean[0] - e_mean[-1], 1.0]
popt, pcov = curve_fit(exp_plateau, beta, energy, p0=p0,
                       sigma=denergy, absolute_sigma=True,
                       maxfev=10000)

E_inf, A, beta_c = popt
perr = np.sqrt(np.diag(pcov))
dE_inf, dA, dbeta_c = perr

# ── Report ────────────────────────────────────────────────────────────
print("=" * 80)
print("  Exponential-Energy Decaying Fit:  E(beta) = E_inf + A exp(-beta/beta_c) ")
print("=" * 80)
print(f"  E_inf   = {E_inf:.6f} ± {dE_inf:.6f} a.u. ")
print(f"  A       = {A:.6f} ± {dA:.6f} a.u. ")
print(f"  beta_c  = {beta_c:.4f} ± {dbeta_c:.4f} a.u. (cooled to about 37% initial gap")
print(f"  System considered fully cooled at about beta = {6*beta_c:.4f} a.u. (6*beta_c) ")
# if 5/gamma < beta[-1]:
#     print("  System cooled: the exponential transient has died out ")
# else:
#     print("  !!!System NOT convincingly cooled — consider longer propagation. ")
# print(f"  Ground-state Energy estimate:  E_inf = {E_inf:.6f} ± {dE_inf:.6f}")
print("=" * 80)
print()

np.savez('./traject_pt2.npz', 
         time = beta, 
         wt = wt_trj,
         t1 = t1_trj,
         t2 = t2_trj,
         e0 = e0_trj, 
         e1 = e1_trj,
         en = e_mean)