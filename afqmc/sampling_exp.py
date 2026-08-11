from dataclasses import dataclass
from functools import partial
import jax.numpy as jnp
import jax
from jax import jit, lax, random
import numpy as np
from scipy.optimize import curve_fit

from . import cc_tools

def filter_outliers(samples, zeta=20):

    median = np.median(samples)
    mad = 1.4826 * np.median(np.abs(samples - median))
    bound = zeta * mad
    mask = np.abs(samples - median) < bound
    print(f"Remove samples outside Zeta > {zeta}")
    print(f"Outlier bound [{median-bound:.5f}, {median+bound:.5f}]")
    
    return mask

@jit
def weighted_average(weights, samples):
    # weights: (nsample,)
    # samples: (nsample, nterm)
    weights = jnp.atleast_1d(weights)          # tolerate a 0-d / scalar weight
    nsample = len(weights)                     # static under jit
    samples = samples.reshape(nsample, -1)     # handle the single-term case

    w_sum  = jnp.sum(weights)
    sample_mean = jnp.sum(weights[:, None] * samples, axis=0) / w_sum

    if nsample == 1:
        # A single sample gives no estimate of the estimator's spread,
        # so the standard error is undefined, not zero.
        sample_err = jnp.full_like(sample_mean.real, jnp.nan)
        return w_sum, jnp.squeeze(sample_mean), jnp.squeeze(sample_err)

    # Kish effective sample size: (sum w)^2 / sum(w^2)
    n_eff = w_sum**2 / jnp.sum(weights**2)

    # weighted (biased) variance per term
    deviations = samples - sample_mean
    sample_var = jnp.sum(weights[:, None] * jnp.abs(deviations)**2, axis=0) / w_sum

    # standard error of the weighted mean via effective sample size
    sample_err = jnp.sqrt(sample_var / n_eff)

    return w_sum, jnp.squeeze(sample_mean), jnp.squeeze(sample_err)

def blocking(wt_sp, en_sp, min_nblocks=20, final=False):

    nsample = len(wt_sp)
    weight = np.mean(wt_sp) 
    energy = (np.mean(wt_sp * en_sp) / weight).real

    if not final:
            W = np.sum(wt_sp)                              # total weight (not the mean)
            dev = en_sp.real - energy                      # energy is the real estimator
            var_mean = np.sum(wt_sp**2 * dev**2) / W**2    # scale-invariant SE^2
            var_mean *= nsample / (nsample - 1)            # small-sample correction
            return weight, energy, np.sqrt(var_mean).real

    # ---------------- full blocking analysis (final=True) ----------------
    max_size = nsample // min_nblocks
    if max_size < 10:
        min_nblocks = max(nsample // 10, 3)
        max_size = nsample // min_nblocks
        print(f"Warning: small dataset, relaxed min_nblocks to {min_nblocks}")

    block_sizes = np.arange(1, max_size + 1)
    block_vars = np.zeros(max_size)
    block_var_errs = np.zeros(max_size)
    block_means = np.zeros(max_size)

    print(f"nsample = {nsample}, max_block_size = {max_size}, min_nblocks = {min_nblocks}")
    print(f"{'B':>4s}  {'NB':>4s}  {'NS':>4s}  {'Observable':>10s}  {'Error':>8s}  {'dError':>8s}")

    for i, block_size in enumerate(block_sizes):
        n_blocks = nsample // block_size
        sl = slice(0, n_blocks * block_size)
        wt = (wt_sp[sl]).reshape(n_blocks, block_size)
        wt_en = (wt_sp[sl] * en_sp[sl]).reshape(n_blocks, block_size)
        block_weight = np.sum(wt, axis=1)
        block_energy = (np.sum(wt_en, axis=1) / block_weight).real
        block_mean = np.mean(block_energy)
        block_var = np.var(block_energy, ddof=1) / n_blocks  # variance of the mean
        block_error = np.sqrt(block_var)
        var_of_var = block_var * np.sqrt(2.0 / (n_blocks - 1))
        err_of_err = block_error / np.sqrt(2.0 * (n_blocks - 1))
        block_means[i] = block_mean
        block_vars[i] = block_var
        block_var_errs[i] = var_of_var
        print(f'{block_size:4d}  {n_blocks:4d}  {block_size*n_blocks:4d}  '
              f'{block_mean:10.5f}  {block_error:8.5f}  {err_of_err:8.5f}')

    def model(x, a, b, tau):
        return a - b * np.exp(-x / tau)
    p0 = [block_vars.max(), block_vars.max() - block_vars[0], 5.0]
    try:
        popt, pcov = curve_fit(model, block_sizes, block_vars,
                               sigma=block_var_errs, absolute_sigma=True,
                               p0=p0, maxfev=10000)
        plateau_var = popt[0]
        plateau_var_unc = np.sqrt(pcov[0, 0])
        plateau_err = np.sqrt(plateau_var)
        plateau_uncertainty = plateau_var_unc / (2.0 * plateau_err)
        tau = popt[2]
        ratio = 0.01 * popt[0] / popt[1]
        if ratio > 0:
            plateau_block_size = int(np.ceil(-popt[2] * np.log(ratio)))
        else:
            plateau_block_size = 1
        print(f"Fit (variance): plateau_var = {plateau_var:.3e} ± {plateau_var_unc:.3e}")
        print(f"Fit (error):    plateau = {plateau_err:.5f} ± {plateau_uncertainty:.5f}")
        print(f"     autocorrelation length ~ {tau:.1f} blocks")
        print(f"     plateau reached at block size ~ {plateau_block_size}")
        if tau > max_size or tau < 0:
            print(f"     !!!Failed to reach plateau in blocking")
            print(f"     Return max block error")
            plateau_err = np.sqrt(block_vars.max())
    except RuntimeError as e:
        print(f"\nFit failed: {e}")
        plateau_err = np.sqrt(block_vars.max())
        print(f"Fallback max error: {plateau_err:.5f}")

    return weight, energy, plateau_err

def pt2blocking(
        h0,
        wp_sp,
        t2_sp,
        e0_sp,
        e1_sp,
        min_nblocks=20,
        final=False,
        ):
    """Blocking analysis for the PT2 total-energy estimator

        ept2 = h0 + <e0> + <e1> - <t2>*<e0>

    where every average is taken with the *absorbed* weight
    wp = wt*t1, i.e. <x> = sum(wp*x)/sum(wp).

    In terms of weighted aggregates (E0 = sum(wp*e0), WP = sum(wp), etc.):

        E = h0 + E0/WP + E1/WP - T2*E0/WP**2

    Differences from the previous (t1-explicit) `pt2blocking`:
      - t1 is folded into the weight, so the normalizer is the plain
        total weight WP = sum(wp) rather than the aggregate sum(w*t1);
      - equivalently, the old `t1_sp` observable is replaced by the
        constant 1, so its per-sample contribution to the aggregates is
        just wp. The cross term still reuses e0, so E0 enters E twice.

    Returns (weight, energy, error) where weight = mean(wp_sp).
    """
    nsample = len(wp_sp)
    weight = np.mean(wp_sp)
    t2 = np.mean(wp_sp * t2_sp) / weight
    e0 = np.mean(wp_sp * e0_sp) / weight
    e1 = np.mean(wp_sp * e1_sp) / weight
    energy = (h0 + e0 + e1 - t2*e0).real

    if not final:
        # No blocking: weight-aware naive error of the nonlinear estimator
        #   E = h0 + E0/WP + E1/WP - T2*E0/WP**2
        # with E0=sum(wp*e0), E1=sum(wp*e1), T2=sum(wp*t2), WP=sum(wp).
        # Treat each sample as an independent unit and propagate its
        # contribution to each aggregate through a first-order (delta-method)
        # linearization -> per-sample influence, then variance of the mean.
        w  = wp_sp
        E0 = np.sum(w * e0_sp)
        E1 = np.sum(w * e1_sp)
        T2 = np.sum(w * t2_sp)
        WP = np.sum(w)                   # normalizer, replaces old T1

        # partials of E w.r.t. each aggregate (E0 enters twice)
        dfdE0 = 1.0 / WP - T2 / WP**2
        dfdE1 = 1.0 / WP
        dfdT2 = -E0 / WP**2
        dfdWP = -E0 / WP**2 - E1 / WP**2 + 2.0 * T2 * E0 / WP**3

        # per-sample influence on E (these sum to ~0 since E is scale-free).
        # The weight's own contribution to WP is just w (observable == 1).
        infl = (dfdE0 * (w * e0_sp)
                + dfdE1 * (w * e1_sp)
                + dfdT2 * (w * t2_sp)
                + dfdWP * w).real
        var_mean = np.sum(infl**2) * nsample / (nsample - 1)
        return weight, energy, np.sqrt(var_mean)

    # ---------------- full blocking analysis (final=True) ----------------
    max_size = nsample // min_nblocks
    if max_size < 10:
        min_nblocks = max(nsample // 10, 3)
        max_size = nsample // min_nblocks
        print(f"Warning: small dataset, relaxed min_nblocks to {min_nblocks}")
    block_sizes = np.arange(1, max_size + 1)
    block_vars = np.zeros(max_size)
    block_var_errs = np.zeros(max_size)
    block_means = np.zeros(max_size)
    print(f"nsample = {nsample}, max_block_size = {max_size}, min_nblocks = {min_nblocks}")
    print(f"{'Blk_SZ':>6s}  {'NBlk':>5s}  {'NSmp':>5s}  {'Energy':>10s}  {'Error':>8s}  {'dError':>8s}")
    for i, block_size in enumerate(block_sizes):
        n_blocks = nsample // block_size
        sl = slice(0, n_blocks * block_size)
        wp_blk = (wp_sp[sl]).reshape(n_blocks, block_size)
        wp_t2  = (wp_sp[sl] * t2_sp[sl]).reshape(n_blocks, block_size)
        wp_e0  = (wp_sp[sl] * e0_sp[sl]).reshape(n_blocks, block_size)
        wp_e1  = (wp_sp[sl] * e1_sp[sl]).reshape(n_blocks, block_size)
        block_wp = np.sum(wp_blk, axis=1)        # per-block normalizer
        block_t2 = np.sum(wp_t2,  axis=1)
        block_e0 = np.sum(wp_e0,  axis=1)
        block_e1 = np.sum(wp_e1,  axis=1)
        block_energy = (h0 + block_e0/block_wp + block_e1/block_wp
                        - (block_t2 * block_e0) / block_wp**2).real
        block_mean = np.mean(block_energy)
        block_var = np.var(block_energy, ddof=1) / n_blocks  # variance of the mean
        block_error = np.sqrt(block_var)
        # Uncertainty on variance: var / sqrt((n_blocks - 1) / 2)
        var_of_var = block_var * np.sqrt(2.0 / (n_blocks - 1))
        err_of_err = block_error / np.sqrt(2.0 * (n_blocks - 1))
        block_means[i] = block_mean
        block_vars[i] = block_var
        block_var_errs[i] = var_of_var
        print(f'{block_size:6d}  {n_blocks:5d}  {block_size*n_blocks:5d}  '
            f'{block_mean:10.5f}  {block_error:8.5f}  {err_of_err:8.5f}')

    def model(x, a, b, tau):
        return a - b * np.exp(-x / tau)
    p0 = [block_vars.max(), block_vars.max() - block_vars[0], 5.0]
    try:
        popt, pcov = curve_fit(model, block_sizes, block_vars,
                            sigma=block_var_errs, absolute_sigma=True,
                            p0=p0, maxfev=10000)
        plateau_var = popt[0]
        plateau_var_unc = np.sqrt(pcov[0, 0])
        plateau_err = np.sqrt(plateau_var)
        # Error propagation: d(sqrt(v)) = dv / (2 sqrt(v))
        plateau_uncertainty = plateau_var_unc / (2.0 * plateau_err)
        tau = popt[2]
        ratio = 0.01 * popt[0] / popt[1]
        if ratio > 0:
            plateau_block_size = int(np.ceil(-popt[2] * np.log(ratio)))
        else:
            plateau_block_size = 1
        print(f"Fit (variance): plateau_var = {plateau_var:.5e} ± {plateau_var_unc:.5e}")
        print(f"Fit (error):    plateau_err = {plateau_err:.5f} ± {plateau_uncertainty:.5f}")
        print(f"     autocorrelation length ~ {tau:.1f} blocks")
        print(f"     plateau reached at block size ~ {plateau_block_size}")
        if tau > max_size or tau < 0:
            print(f"     !!!Failed to reach plateau in blocking")
            print(f"     Return max block error")
            plateau_err = np.sqrt(block_vars.max())
    except RuntimeError as e:
        print(f"\nFit failed: {e}")
        plateau_err = np.sqrt(block_vars.max())
        print(f"Fallback max error: {plateau_err:.5f}")
    return weight, energy, plateau_err


@dataclass
class sampler:
    n_walkers: int
    n_prop_steps: int
    n_blocks: int
    n_chol: int

    @partial(jit, static_argnums=(0, 1, 2))
    def prop_nstep(self, prop, wave, prop_data, ham_data, wave_data):
        """Phaseless propagation scan function over steps."""
        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                self.n_prop_steps,
                prop.n_walkers,
                self.n_chol,
            ),
        )

        def scan_fn(carry, field):
            # field has shape (n_walkers, n_chol)
            prop_data = carry
            prop_data = prop.propagate(wave, ham_data, prop_data, field, wave_data)
            return prop_data, None

        prop_data, _ = lax.scan(scan_fn, prop_data, fields)

        prop_data["n_killed_walkers"] = (
                prop_data["weights"].size - jnp.count_nonzero(prop_data["weights"])
            )

        prop_data = prop.orthonormalize_walkers(prop_data)

        return prop_data

    @partial(jit, static_argnums=(0,1,2))
    def block_sample(
        self,
        prop,
        wave,
        prop_data,
        ham_data,
        wave_data,
        ):
        """Block scan function. Propagation and energy calculation."""
        prop_data = self.prop_nstep(prop, wave, prop_data, ham_data, wave_data)
        
        guide_olps = wave.calc_overlap(prop_data["walkers"], wave_data)
        guide_enes = wave.calc_guide_energy(prop_data["walkers"], ham_data, wave_data)

        outlier = jnp.abs(guide_enes - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt) # 20 Ha for dt = 0.005
        prop_data["weights"] = jnp.where(outlier, 0.0, prop_data["weights"])

        trial_olps = wave.calc_trial_overlap(prop_data["walkers"], wave_data)
        trial_enes = wave.calc_energy(prop_data["walkers"], ham_data, wave_data)

        olp_ratio = trial_olps / guide_olps
        weighps = prop_data["weights"] * olp_ratio

        wp, sample_mean, sample_err = weighted_average(weighps, trial_enes)

        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = wave.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        prop_data["n_killed_walkers"] += prop_data["weights"].size \
            - jnp.count_nonzero(prop_data["weights"])

        return prop_data, (wp, sample_mean, sample_err)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class stocc_sampler(sampler):
    n_walkers: int
    n_prop_steps: int
    n_blocks: int
    n_chol: int
    n_slater: int

    @partial(jit, static_argnums=(0, 1, 2))
    def prop_nstep(self, prop, wave, prop_data, ham_data, wave_data):
        """Phaseless propagation scan function over steps."""
        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                self.n_prop_steps,
                self.n_walkers,
                self.n_chol,
            ),
        )

        def scan_fn(carry, field):
            prop_data, wave_data = carry
            prop_data, wave_data = prop.propagate(wave, ham_data, prop_data, field, wave_data)
            return (prop_data, wave_data), None

        (prop_data, wave_data), _ = lax.scan(scan_fn, (prop_data, wave_data), fields)

        prop_data["n_killed_walkers"] \
            = prop_data["weights"].size - jnp.count_nonzero(prop_data["weights"])

        prop_data = prop.orthonormalize_walkers(prop_data)

        return prop_data, wave_data

    @partial(jit, static_argnums=(0,1,2))
    def block_sample(
        self,
        prop,
        wave,
        prop_data,
        ham_data,
        wave_data,
        ):
        """Block scan function. Propagation and energy calculation."""
        prop_data, wave_data = self.prop_nstep(prop, wave, prop_data, ham_data, wave_data)
        
        guide_olps = wave.calc_overlap(prop_data["walkers"], wave_data)
        guide_enes = wave.calc_guide_energy(prop_data["walkers"], ham_data, wave_data)

        outlier = jnp.abs(guide_enes - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt) # 20 Ha for dt = 0.005
        prop_data["weights"] = jnp.where(outlier, 0.0, prop_data["weights"])

        trial_olps = wave.calc_trial_overlap(prop_data["walkers"], wave_data)
        trial_enes = wave.calc_energy(prop_data["walkers"], ham_data, wave_data)

        olp_ratio = trial_olps / guide_olps
        weighps = prop_data["weights"] * olp_ratio

        wp, sample_mean, sample_err = weighted_average(weighps, trial_enes)

        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = wave.calc_overlap(prop_data["walkers"], wave_data)

        return prop_data, (wp, sample_mean, sample_err)


    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


