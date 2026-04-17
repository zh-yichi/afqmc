from functools import partial
from jax import random, lax, vmap
import numpy as np
from jax import numpy as jnp
from ad_afqmc import config
from ad_afqmc.prop_unrestricted import prep
import time
import argparse
from ad_afqmc import config

# def ratio_estimator_cov(num, den):
#     """
#     Estimate R = <N>/<D> and its standard error using the covariance (delta) method.
    
#     Parameters
#     ----------
#     num : array, numerator samples (o_i * E_i)
#     den : array, denominator samples (o_i)
    
#     Returns
#     -------
#     R : ratio estimate
#     sigma_R : standard error of R
#     """
#     n = num.shape
#     R = num.mean() / den.mean()

#     var_N = np.var(num, ddof=1)
#     var_D = np.var(den, ddof=1)
#     cov_ND = np.cov(num, den, ddof=1)[0, 1]

#     # Delta method: Var(R) = (1/<D>^2) [Var(N) + R^2 Var(D) - 2R Cov(N,D)] / n
#     var_R = (var_N + R**2 * var_D - 2 * R * cov_ND) / (den.mean()**2 * n)

#     return R, np.sqrt(var_R)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    if args.use_gpu:
        config.afqmc_config["use_gpu"] = True

config.setup_jax()

print = partial(print, flush=True)

ham_data, ham, prop, trial, wave_data, sampler, options = (prep._prep_afqmc())

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

if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError(
        "Initial overlaps are zero. Pass walkers with non-zero overlap."
    )

seeds = random.randint(random.PRNGKey(options["seed"]), 
                       shape=(sampler.n_trj,),
                       minval=0, 
                       maxval=100*sampler.n_trj)
prop_data["key"] = random.PRNGKey(options["seed"])
prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
blk_time = prop.dt * sampler.n_prop_steps

# shape (inverse_T, trajectories)
glb_blk_w = np.zeros((sampler.n_eql_blocks, sampler.n_trj), dtype="complex128")
glb_blk_e = np.zeros((sampler.n_eql_blocks, sampler.n_trj), dtype="complex128")
e_init = prop_data["e_estimate"]

# def scan_trjs(prop_data, _):
#     # prop_data["key"] = keysss
#     prop_data["key"], subkey = random.split(prop_data["key"])
#     _, (blk_w, blk_e) = sampler.scan_eql_blocks(
#         prop_data, ham_data, prop, trial, wave_data
#     )
#     return prop_data, (blk_w, blk_e)

# n_collect = sampler.n_trj // 10
# nscan = sampler.n_trj // n_collect
# for i in range(n_collect):

#     prop_data, (blk_w_chunk, blk_e_chunk) = lax.scan(scan_trjs, prop_data, None, length=nscan)

#     # blk_w_chunk, blk_e_chunk have shape (10, ...)
#     glb_blk_w[:, i * 10 : (i + 1) * 10] = np.array(blk_w_chunk, dtype="complex128").T
#     glb_blk_e[:, i * 10 : (i + 1) * 10] = np.array(blk_e_chunk, dtype="complex128").T

#     # Print progress / intermediate results every chunk
#     n_done = (i + 1) * 10

for i in range(sampler.n_trj):
    prop_data["key"] = random.PRNGKey(seeds[i])
    print(f"Propagating with {options['n_walkers']} walkers")
    print(f"Free Projection AFQMC trajector {i+1}/{sampler.n_trj}")
    print(f"{'Inv_T':>6s}  {'Energy':>10s}  {'Error':>8s}  {'Walltime':>8s}")
    print(f"{0.:6.2f}  {e_init:10.5f}  {0.:8.5f}  {time.time() - init_time:8.2f}")
    
    _, (blk_w, blk_e) \
        = sampler.scan_eql_blocks(prop_data, ham_data, prop, trial, wave_data)

    blk_w = np.array([blk_w], dtype="complex128")
    blk_e = np.array([blk_e], dtype="complex128")

    glb_blk_w[:,i] = blk_w
    glb_blk_e[:,i] = blk_e
    
    e_mean = np.real(
        np.sum(glb_blk_w[:,:(i + 1)] * glb_blk_e[:,:(i + 1)], axis=1
               ) / np.sum(glb_blk_w[:,:(i + 1)], axis=1))
    if i > 0:
        de_mean = np.real(
            np.sqrt(
            np.sum(glb_blk_w[:,:(i + 1)] * (glb_blk_e[:,:(i + 1)]-e_mean[:, None])**2, axis=1
                    ) / np.sum(glb_blk_w[:,:(i + 1)], axis=1)) / np.sqrt((i + 1)))
        # e_mean, de_mean = ratio_estimator_cov(, den)

        for nb in range(sampler.n_eql_blocks):
            print(f"{(nb+1)*blk_time:6.2f}  {e_mean[nb]:10.5f}  {de_mean[nb]:8.5f}  {time.time() - init_time:8.2f} ")
    elif i == 0:
        for nb in range(sampler.n_eql_blocks):
            print(f"{(nb+1)*blk_time:6.2f}  {e_mean[nb]:10.5f}  {'N/A':>8s}  {time.time() - init_time:8.2f} ")

# ── Model: E(beta) = E_inf + A * exp(-gamma * beta) ──────────────────────
from scipy.optimize import curve_fit
beta = blk_time * np.arange(1, sampler.n_eql_blocks+1)

def exp_plateau(tau, E_inf, A, gamma):
    return E_inf + A * np.exp(-gamma * tau)

# Initial guesses: E_inf ~ last E, A ~ E(0)-E_inf, gamma ~ 0.3
p0 = [e_mean[-1], e_mean[0]-e_mean[-1], 0.3]

popt, pcov = curve_fit(exp_plateau, beta, e_mean, p0=p0,
                       sigma=de_mean, absolute_sigma=True,
                       maxfev=10000)

E_inf, A, gamma = popt
perr = np.sqrt(np.diag(pcov))
dE_inf, dA, dgamma = perr

# ── Report ────────────────────────────────────────────────────────────
print("=" * 80)
print("  Exponential-Energy Decaying Fit:  E(beta) = E_inf + A exp(-Gamma*beta) ")
print("=" * 80)
print(f"  E_inf   = {E_inf:.6f} ± {dE_inf:.6f} a.u. ")
print(f"  A       = {A:.6f} ± {dA:.6f} a.u. ")
print(f"  Gamma   = {gamma:.4f} ± {dgamma:.4f} a.u. ")
print(f"  System cooled to about 37% initial Energy gap at beta = {1/gamma:.4f} a.u. (1/Gamma) ")
print(f"  System considered fully cooled at about beta = {5/gamma:.4f} a.u. (5/Gamma) ")
print()
if 5/gamma < beta[-1]:
    print("  System cooled: the exponential transient has died out ")
else:
    print("  !!!System NOT convincingly cooled — consider longer propagation. ")
print(f"  Ground-state Energy estimate:  E_inf = {E_inf:.6f} ± {dE_inf:.6f}")
print("=" * 80)

np.savez('./traject.npz', 
            time = beta, 
            weights = glb_blk_w, 
            energies = glb_blk_e)