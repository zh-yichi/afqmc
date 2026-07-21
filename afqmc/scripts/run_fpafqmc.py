import numpy as np
from jax import random, jit
from jax import numpy as jnp

from afqmc import config, prep, walker_tools

import time
from functools import partial

print = partial(print, flush=True)

def error_estimate(w_trj, e_trj):

    neql, ntrj = w_trj.shape # (time, sample)
    w_mean = np.mean(w_trj, axis=1)
    e_mean = np.real(np.mean(w_trj * e_trj, axis=1) / w_mean)

    if ntrj == 1:
        e_err = None
    elif ntrj > 1:
        e_err  = np.real(np.std(e_trj, axis=1, ddof=1)) / np.sqrt(ntrj) # (time,)

    return e_mean, e_err

config.setup_jax()

# @partial(jit, static_argnames=("trial", "n_walkers", "walker_type", "seeds"))
def init_ccsd_prop_data(
    trial,
    wave_data,
    n_walkers,
    walker_type,
    seeds,
    ):

    print("\nInitalize QMC walkers by stochastic CCSD")
    prop_data = {}
    prop_data["n_killed_walkers"] = 0
    prop_data["key"] = random.PRNGKey(seeds)

    weights0 = jnp.ones(n_walkers, dtype=jnp.float64)
    walkers0 = walker_tools.replicate_walker(wave_data["mo_coeff"], n_walkers)
    overlaps0 = trial.calc_overlap(walkers0, wave_data)

    walkers1, prop_data = walker_tools.get_ccsd_walkers(
        prop_data, wave_data, n_walkers, walker_type)

    overlaps1 = trial.calc_overlap(walkers1, wave_data)
    weights1 = jnp.real(weights0 * overlaps1 / overlaps0)
    energy = trial.calc_energy(walkers1, wave_data)

    prop_data["weights"] = weights1
    prop_data["walkers"] = walkers1
    prop_data["overlaps"] = overlaps1

    prop_data["e_estimate"] = energy
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

    return prop_data

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

prop_data = prop.init_prop_data(trial, wave_data, ham_data, init_walkers = None)

# if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
#     raise ValueError(
#         "Initial overlaps are zero. Pass walkers with non-zero overlap."
#     )

seeds = random.randint(random.PRNGKey(options["seed"]), 
                       shape=(sampler.n_trj,),
                       minval=0, 
                       maxval=100*sampler.n_trj)

# @partial(jit, static_argnames=("prop", "trial"))
# def init_prop_data(wave_data, ham_data, prop, trial, seed):
#     prop_data = {}
#     prop_data["weights"] = jnp.ones(prop.n_walkers)
#     prop_data["key"] = random.PRNGKey(seed)
#     init_walkers, prop_data = trial.get_ccsd_walkers(prop_data, wave_data, prop)
#     prop_data["walkers"] = init_walkers
#     energy_samples = jnp.real(trial.calc_energy(prop_data["walkers"], ham_data, wave_data))
#     e_estimate = jnp.array(jnp.sum(energy_samples) / prop.n_walkers)
#     prop_data["e_estimate"] = e_estimate
#     prop_data["pop_control_ene_shift"] = e_estimate
#     prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

#     return prop_data

prop_data["key"] = random.PRNGKey(options["seed"])
# prop_data = init_prop_data(wave_data, ham_data, prop, trial, options["seed"])
# e_init = prop_data["e_estimate"]

blk_time = prop.dt * sampler.n_prop_steps

# shape (inverse_T+1, trajectories)
w_trj = np.zeros((sampler.n_eql_blocks+1, sampler.n_trj), dtype="complex128")
e_trj = np.zeros((sampler.n_eql_blocks+1, sampler.n_trj), dtype="complex128")
# e_init = prop_data["e_estimate"]
print(f"Propagating with {options['n_walkers']} walkers")

for i in range(sampler.n_trj):
    prop_data = init_ccsd_prop_data(
        trial, wave_data, options["n_walkers"], options["walker_type"], seeds[i])
    e_init = prop_data["e_estimate"]
    w_init = np.sum(prop_data["weights"])

    prop_data, (blk_w, blk_e) \
        = sampler.scan_eql_blocks(prop_data, ham_data, prop, trial, wave_data)

    # blk_w = np.array([blk_w], dtype="complex128")
    # blk_e = np.array([blk_e], dtype="complex128")
    
    # w_trj[:,i] = blk_w
    # e_trj[:,i] = blk_e

    w_trj[0,  i] = w_init
    e_trj[0,  i] = e_init
    w_trj[1:, i] = np.asarray(blk_w, dtype="complex128")
    e_trj[1:, i] = np.asarray(blk_e, dtype="complex128")
    
    e_mean, e_err = error_estimate(w_trj[:,:(i + 1)], e_trj[:,:(i + 1)])
    
    if i == 0:
        print(f"Free Projection AFQMC trajector {i+1}/{sampler.n_trj} | key = {prop_data["key"]}")
        print(f"{'Inv_T':>6s}  {'Energy':>10s}  {'Error':>8s}  {'Walltime':>8s}")
        # print(f"{0.:6.2f}  {e_init:10.5f}  {0.:8.5f}  {time.time() - init_time:8.2f}")
        for nb in range(len(e_mean)):
            print(f"{(nb)*blk_time:6.2f}  {e_mean[nb]:10.5f}  {'N/A':>8s}  {time.time() - init_time:8.2f} ")

    elif (i+1) % (min(max(sampler.n_trj // 10, 1), 20)) == 0 and i > 0:
        print(f"Free Projection AFQMC trajector {i+1}/{sampler.n_trj} | key = {prop_data["key"]}")
        print(f"{'Inv_T':>6s}  {'Energy':>10s}  {'Error':>8s}  {'Walltime':>8s}")
        # print(f"{0.:6.2f}  {e_init:10.5f}  {0.:8.5f}  {time.time() - init_time:8.2f}")
        for nb in range(len(e_mean)):
            print(f"{(nb)*blk_time:6.2f}  {e_mean[nb]:10.5f}  {e_err[nb]:8.5f}  {time.time() - init_time:8.2f} ")

# # ── Model: E(beta) = E_inf + A * exp(-gamma * beta) ──────────────────────
# from scipy.optimize import curve_fit
# beta = blk_time * np.arange(1, sampler.n_eql_blocks+1)

# def exp_plateau(tau, E_inf, A, gamma):
#     return E_inf + A * np.exp(-gamma * tau)

# # Initial guesses: E_inf ~ last E, A ~ E(0)-E_inf, gamma ~ 0.3
# p0 = [e_mean[-1], e_mean[0]-e_mean[-1], 0.3]

# popt, pcov = curve_fit(exp_plateau, beta, e_mean[1:], p0=p0,
#                        sigma=e_err[1:], absolute_sigma=True,
#                        maxfev=10000)

# E_inf, A, gamma = popt
# perr = np.sqrt(np.diag(pcov))
# dE_inf, dA, dgamma = perr

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

# Initial guesses: E_inf ~ last E, A ~ E(0)-E_inf, beta_c ~ 1
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

np.savez('./traject.npz', 
            time = beta, 
            weights = w_trj, 
            energies = e_trj)