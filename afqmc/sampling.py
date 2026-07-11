from dataclasses import dataclass
from functools import partial
import jax.numpy as jnp
import jax
from jax import jit, lax, random
import numpy as np
from scipy.optimize import curve_fit

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
    nsample = len(weights)
    samples = samples.reshape(nsample, -1)   # handle the single-term case

    w_sum = jnp.sum(weights)
    sample_mean = jnp.sum(weights[:, None] * samples, axis=0) / w_sum

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

# def pt2blocking(
#         h0,
#         wt_sp,
#         t1_sp,
#         t2_sp,
#         e0_sp,
#         e1_sp,
#         min_nblocks=20,
#         final=False,
#         ):

#     nsample = len(wt_sp)
#     weight = np.mean(wt_sp)
#     t1 = np.mean(wt_sp * t1_sp) / weight
#     t2 = np.mean(wt_sp * t2_sp) / weight
#     e0 = np.mean(wt_sp * e0_sp) / weight
#     e1 = np.mean(wt_sp * e1_sp) / weight
#     energy = (h0 + e0/t1 + e1/t1 - t2*e0/t1**2).real

#     if not final:
#         # No blocking: weight-aware naive error of the nonlinear estimator
#         #   E = h0 + E0/T1 + E1/T1 - T2*E0/T1**2
#         # with E0=sum(w*e0), E1=sum(w*e1), T1=sum(w*t1), T2=sum(w*t2).
#         # Treat each sample as an independent unit and propagate its
#         # contribution to each aggregate through a first-order (delta-method)
#         # linearization -> per-sample influence, then variance of the mean.
#         w  = wt_sp
#         E0 = np.sum(w * e0_sp)
#         E1 = np.sum(w * e1_sp)
#         T1 = np.sum(w * t1_sp)
#         T2 = np.sum(w * t2_sp)

#         # partials of E w.r.t. each aggregate
#         dfdE0 = 1.0 / T1 - T2 / T1**2
#         dfdE1 = 1.0 / T1
#         dfdT1 = -E0 / T1**2 - E1 / T1**2 + 2.0 * T2 * E0 / T1**3
#         dfdT2 = -E0 / T1**2

#         # per-sample influence on E (these sum to ~0 since E is scale-free)
#         infl = (dfdE0 * (w * e0_sp)
#                 + dfdE1 * (w * e1_sp)
#                 + dfdT1 * (w * t1_sp)
#                 + dfdT2 * (w * t2_sp)).real
#         var_mean = np.sum(infl**2) * nsample / (nsample - 1)
#         return weight, energy, np.sqrt(var_mean)

#     # ---------------- full blocking analysis (final=True) ----------------
#     max_size = nsample // min_nblocks
#     if max_size < 10:
#         min_nblocks = max(nsample // 10, 3)
#         max_size = nsample // min_nblocks
#         print(f"Warning: small dataset, relaxed min_nblocks to {min_nblocks}")
#     block_sizes = np.arange(1, max_size + 1)
#     block_vars = np.zeros(max_size)
#     block_var_errs = np.zeros(max_size)
#     block_means = np.zeros(max_size)
#     print(f"nsample = {nsample}, max_block_size = {max_size}, min_nblocks = {min_nblocks}")
#     print(f"{'Blk_SZ':>6s}  {'NBlk':>5s}  {'NSmp':>5s}  {'Energy':>10s}  {'Error':>8s}  {'dError':>8s}")
#     for i, block_size in enumerate(block_sizes):
#         n_blocks = nsample // block_size
#         sl = slice(0, n_blocks * block_size)
#         wt = (wt_sp[sl]).reshape(n_blocks, block_size)
#         wt_t1 = (wt_sp[sl] * t1_sp[sl]).reshape(n_blocks, block_size)
#         wt_t2 = (wt_sp[sl] * t2_sp[sl]).reshape(n_blocks, block_size)
#         wt_e0 = (wt_sp[sl] * e0_sp[sl]).reshape(n_blocks, block_size)
#         wt_e1 = (wt_sp[sl] * e1_sp[sl]).reshape(n_blocks, block_size)
#         block_t1 = np.sum(wt_t1, axis=1)
#         block_t2 = np.sum(wt_t2, axis=1)
#         block_e0 = np.sum(wt_e0, axis=1)
#         block_e1 = np.sum(wt_e1, axis=1)
#         block_energy = (h0 + block_e0/block_t1 + block_e1/block_t1
#                         - (block_t2 * block_e0) / block_t1**2).real
#         block_mean = np.mean(block_energy)
#         block_var = np.var(block_energy, ddof=1) / n_blocks  # variance of the mean
#         block_error = np.sqrt(block_var)
#         # Uncertainty on variance: var / sqrt((n_blocks - 1) / 2)
#         var_of_var = block_var * np.sqrt(2.0 / (n_blocks - 1))
#         err_of_err = block_error / np.sqrt(2.0 * (n_blocks - 1))
#         block_means[i] = block_mean
#         block_vars[i] = block_var
#         block_var_errs[i] = var_of_var
#         print(f'{block_size:6d}  {n_blocks:5d}  {block_size*n_blocks:5d}  '
#             f'{block_mean:10.5f}  {block_error:8.5f}  {err_of_err:8.5f}')

#     def model(x, a, b, tau):
#         return a - b * np.exp(-x / tau)
#     p0 = [block_vars.max(), block_vars.max() - block_vars[0], 5.0]
#     try:
#         popt, pcov = curve_fit(model, block_sizes, block_vars,
#                             sigma=block_var_errs, absolute_sigma=True,
#                             p0=p0, maxfev=10000)
#         plateau_var = popt[0]
#         plateau_var_unc = np.sqrt(pcov[0, 0])
#         plateau_err = np.sqrt(plateau_var)
#         # Error propagation: d(sqrt(v)) = dv / (2 sqrt(v))
#         plateau_uncertainty = plateau_var_unc / (2.0 * plateau_err)
#         tau = popt[2]
#         ratio = 0.01 * popt[0] / popt[1]
#         if ratio > 0:
#             plateau_block_size = int(np.ceil(-popt[2] * np.log(ratio)))
#         else:
#             plateau_block_size = 1
#         print(f"Fit (variance): plateau_var = {plateau_var:.5e} ± {plateau_var_unc:.5e}")
#         print(f"Fit (error):    plateau_err = {plateau_err:.5f} ± {plateau_uncertainty:.5f}")
#         print(f"     autocorrelation length ~ {tau:.1f} blocks")
#         print(f"     plateau reached at block size ~ {plateau_block_size}")
#         if tau > max_size or tau < 0:
#             print(f"     !!!Failed to reach plateau in blocking")
#             print(f"     Return max block error")
#             plateau_err = np.sqrt(block_vars.max())
#     except RuntimeError as e:
#         print(f"\nFit failed: {e}")
#         plateau_err = np.sqrt(block_vars.max())
#         print(f"Fallback max error: {plateau_err:.5f}")
#     return weight, energy, plateau_err

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
    n_prop_steps: int
    n_blocks: int
    n_chol: int

    @partial(jit, static_argnums=(0, 1, 2))
    def prop_nstep(self, prop, trial, prop_data, ham_data, wave_data):
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
            prop_data = prop.propagate(trial, ham_data, prop_data, field, wave_data)
            return prop_data, None

        prop_data, _ = lax.scan(scan_fn, prop_data, fields)

        prop_data["n_killed_walkers"] = (
                prop_data["weights"].size - jnp.count_nonzero(prop_data["weights"])
            )

        prop_data = prop.orthonormalize_walkers(prop_data)

        return prop_data

    # @partial(jit, static_argnums=(0, 4, 5))
    # def _step_scan(
    #     self,
    #     prop_data,
    #     fields,
    #     ham_data,
    #     prop,
    #     trial,
    #     wave_data,
    #     ):
    #     """Phaseless propagation scan function over steps."""
    #     prop_data = prop.propagate(trial, ham_data, prop_data, fields, wave_data)
    #     return prop_data, fields

    @partial(jit, static_argnums=(0, 3, 4))
    def block_sample(
        self,
        prop_data,
        ham_data,
        prop,
        trial,
        wave_data,
        ):
        """Block scan function. Propagation and energy calculation."""
        # prop_data["key"], subkey = random.split(prop_data["key"])
        # fields = random.normal(
        #     subkey,
        #     shape=(
        #         self.n_prop_steps,
        #         prop.n_walkers,
        #         self.n_chol,
        #     ),
        # )
        # _step_scan_wrapper = lambda x, y: self._step_scan(
        #     x, y, ham_data, prop, trial, wave_data
        # )
        # prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
        # prop_data["n_killed_walkers"] = prop_data["weights"].size - jnp.count_nonzero(prop_data["weights"])

        prop_data = self.prop_nstep(prop, trial, prop_data, ham_data, wave_data)

        energies = jnp.real(trial.calc_energy(prop_data["walkers"], ham_data, wave_data))
        outlier = jnp.abs(energies - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt) # 20 Ha for dt = 0.005
        prop_data["weights"] = jnp.where(outlier, 0.0, prop_data["weights"])
                
        wts = prop_data["weights"]

        wt, en, err = weighted_average(wts, energies)

        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * en
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(prop_data["weights"])

        return prop_data, (wt, en, err)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))
    

@dataclass
class sampler_exp(sampler):
    """An experimental energy sampler for general Guide/Trial combination"""

    @partial(jit, static_argnums=(0,1,2))
    def sample_energy(
        self,
        prop,
        trial,
        prop_data,
        ham_data,
        wave_data,
        ):
        """Block scan function. Propagation and energy calculation."""
        prop_data = self.prop_nstep(prop, trial, prop_data, ham_data, wave_data)
        
        guide_olps = trial.calc_overlap(prop_data["walkers"], wave_data)
        trial_olps = trial.calc_trial_overlap(prop_data["walkers"], wave_data)
        prop_data["overlaps"] = guide_olps

        olp_ratio = trial_olps / guide_olps
        wps = prop_data["weights"] * olp_ratio
        samples = trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
        wp_mean, sample_mean, sample_err = weighted_average(wps, samples)
        tot_wp = jnp.sum(wps)

        # outlier = jnp.abs(energies - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt) # 20 Ha for dt = 0.005
        # weights = jnp.where(outlier, 0.0, prop_data["weights"])

        # blk_wt = jnp.sum(weights)
        # blk_wt = jnp.sum(weights)
        # blk_et = jnp.sum(weights * energies) / blk_wp

        # prop_data["pop_control_ene_shift"] = 0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_eg
        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

        return prop_data, (tot_wp, sample_mean)
    
    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class sampler_pt(sampler):

    @partial(jit, static_argnums=(0,3,4))
    def block_sample(
        self,
        prop_data,
        ham_data,
        prop,
        trial,
        wave_data,
        ):
        """Block scan function. Propagation and energy calculation."""
        prop_data = self.prop_nstep(prop, trial, prop_data, ham_data, wave_data)

        t, e0, e1 = trial.calc_energy_pt(prop_data["walkers"], ham_data, wave_data)

        outlier = jnp.abs(e0 - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt) # 20 Ha for dt = 0.005
        prop_data["weights"] = jnp.where(outlier, 0.0, prop_data["weights"])

        blk_wt = jnp.sum(prop_data["weights"])
        blk_t = jnp.sum(prop_data["weights"] * t) / blk_wt
        blk_e0 = jnp.sum(prop_data["weights"] * e0) / blk_wt
        blk_e1 = jnp.sum(prop_data["weights"] * e1) / blk_wt

        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["n_killed_walkers"] = prop_data["weights"].size - jnp.count_nonzero(prop_data["weights"])

        return prop_data, (blk_wt, blk_t, blk_e0, blk_e1)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))

@dataclass
class sampler_pt2(sampler):
        
    @partial(jit, static_argnums=(0,3,4))
    def block_sample(
        self,
        prop_data,
        ham_data,
        prop,
        trial,
        wave_data,
        ):
        """Block scan function. Propagation and energy calculation."""
        
        prop_data = self.prop_nstep(prop, trial, prop_data, ham_data, wave_data)

        eg_sp = jnp.real(trial.calc_energy(prop_data["walkers"], ham_data, wave_data))
        
        outlier = jnp.abs(eg_sp - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt) # 20 Ha for dt = 0.005
        # weights = jnp.where(outlier, 0.0, prop_data["weights"])
        prop_data["weights"] = jnp.where(outlier, 0.0, prop_data["weights"])

        t1_sp, t2_sp, e0_sp, e1_sp = trial.calc_energy_pt(prop_data["walkers"], ham_data, wave_data)

        wts = prop_data["weights"]
        wps = wts * t1_sp

        wt = jnp.sum(wts)
        eg = jnp.sum(wts * eg_sp) / wt

        wp = jnp.sum(wps)
        t2 = jnp.sum(wps * t2_sp) / wp
        e0 = jnp.sum(wps * e0_sp) / wp
        e1 = jnp.sum(wps * e1_sp) / wp

        prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * eg.real
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(prop_data["weights"])

        return prop_data, (wt, eg, wp, t2, e0, e1)
    
    # @partial(jit, static_argnums=(0,3,4))
    # def sample_energy(
    #     self,
    #     prop_data,
    #     ham_data,
    #     prop,
    #     trial,
    #     wave_data,
    #     ):
    #     """Block scan function. Propagation and energy calculation."""
    #     prop_data["key"], subkey = random.split(prop_data["key"])
    #     fields = random.normal(
    #         subkey,
    #         shape=(
    #             self.n_prop_steps,
    #             prop.n_walkers,
    #             self.n_chol,
    #         ),
    #     )
    #     _step_scan_wrapper = lambda x, y: self._step_scan(
    #         x, y, ham_data, prop, trial, wave_data
    #     )
    #     prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
    #     prop_data = prop.orthonormalize_walkers(prop_data)

    #     t1, t2, e0, e1 = trial.calc_energy_pt(prop_data["walkers"], ham_data, wave_data)

    #     blk_wt = jnp.sum(prop_data["weights"])
    #     blk_t1 = jnp.sum(prop_data["weights"] * t1) / blk_wt
    #     blk_t2 = jnp.sum(prop_data["weights"] * t2) / blk_wt
    #     blk_e0 = jnp.sum(prop_data["weights"] * e0) / blk_wt
    #     blk_e1 = jnp.sum(prop_data["weights"] * e1) / blk_wt

    #     prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
    #     prop_data = prop.stochastic_reconfiguration_local(prop_data)
    #     prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
    #     prop_data["n_killed_walkers"] = prop_data["weights"].size - jnp.count_nonzero(prop_data["weights"])

    #     return prop_data, (blk_wt, blk_t1, blk_t2, blk_e0, blk_e1)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class sampler_stoccsd(sampler):

    @partial(jit, static_argnums=(0,3,4))
    def block_sample(self, prop_data, ham_data, prop, trial, wave_data):
        """Block scan function. Propagation and energy calculation."""

        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                self.n_prop_steps,
                prop.n_walkers,
                self.n_chol,
            ),
        )
        _step_scan_wrapper = lambda x, y: self._step_scan(
            x, y, ham_data, prop, trial, wave_data
        )
        prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
        prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
            prop_data["weights"]
        )

        # raondom fields_x for T2 decomposition
        xtaus, prop_data = trial.get_xtaus(prop_data, wave_data, prop)
        prop_data = prop.orthonormalize_walkers(prop_data)
        # prop_data = prop.stochastic_reconfiguration_local(prop_data)

        olp_hf = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["overlaps"] = olp_hf
        ene_hf = jnp.real(trial.calc_energy(prop_data["walkers"], ham_data, wave_data))
        olp_cc, ene_cc = trial.calc_energy_stoccsd(prop_data["walkers"], xtaus, ham_data, wave_data)
        wt_hf = prop_data["weights"]

        blk_wt = jnp.sum(wt_hf)
        blk_ehf = jnp.sum(wt_hf * ene_hf) / blk_wt
        blk_num = jnp.sum(wt_hf * olp_cc / olp_hf * ene_cc) / blk_wt
        blk_den = jnp.sum(wt_hf * olp_cc / olp_hf) / blk_wt

        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

        return prop_data, (blk_wt, blk_ehf, blk_num, blk_den)

    
    def sto_blocking_analysis(self, wt_sp, num_sp, den_sp, min_nblocks=20, final=False,):
        
        nsample = len(wt_sp)
        max_size = nsample // min_nblocks
        if max_size < 10:
            min_nblocks = max(nsample // 10, 3)
            max_size = nsample // min_nblocks
            if final:
                print(f"Warning: small dataset, relaxed min_nblocks to {min_nblocks}")
        block_sizes = np.arange(1, max_size + 1)
        block_vars = np.zeros(max_size)
        block_var_errs = np.zeros(max_size)
        block_means = np.zeros(max_size)
        if final:
            print(f"nsample = {nsample}, max_block_size = {max_size}, min_nblocks = {min_nblocks}")
            print(f"{'B':>4s}  {'NB':>4s}  {'NS':>4s}  {'Energy':>12s}  {'Error':>8s}  {'dError':>8s}")
        for i, block_size in enumerate(block_sizes):
            n_blocks = nsample // block_size
            sl = slice(0, n_blocks * block_size)
            wt = (wt_sp[sl]).reshape(n_blocks, block_size)
            wt_num = (wt_sp[sl] * num_sp[sl]).reshape(n_blocks, block_size)
            wt_den = (wt_sp[sl] * den_sp[sl]).reshape(n_blocks, block_size)
            block_weight = np.sum(wt, axis=1)
            block_num = np.sum(wt_num, axis=1) / block_weight
            block_den = np.sum(wt_den, axis=1) / block_weight
            block_energy = (block_num / block_den).real
            block_mean = np.mean(block_energy)
            block_var = np.var(block_energy, ddof=1) / n_blocks  # variance of the mean
            block_error = np.sqrt(block_var)
            var_of_var = block_var * np.sqrt(2.0 / (n_blocks - 1))
            err_of_err = block_error / np.sqrt(2.0 * (n_blocks - 1))
            block_means[i] = block_mean
            block_vars[i] = block_var
            block_var_errs[i] = var_of_var
            if final:
                print(f'{block_size:4d}  {n_blocks:4d}  {block_size*n_blocks:4d}  '
                      f'{block_mean:12.6f}  {block_error:8.6f}  {err_of_err:8.6f}')
        
        if final:
            from scipy.optimize import curve_fit
            def model(x, a, b, tau):
                return a - b * np.exp(-x / tau)
            p0 = [block_vars.max(), block_vars.max() - block_vars[0], 5.0]
            try:
                popt, pcov = curve_fit(model, block_sizes, block_vars,
                                    sigma=block_var_errs, absolute_sigma=True,
                                    p0=p0, maxfev=10000)
                plateau_var = popt[0]
                plateau_var_unc = np.sqrt(pcov[0, 0])
                plateau_value = np.sqrt(plateau_var)
                plateau_uncertainty = plateau_var_unc / (2.0 * plateau_value)
                tau = popt[2]
                ratio = 0.01 * popt[0] / popt[1]
                if ratio > 0:
                    plateau_block_size = int(np.ceil(-popt[2] * np.log(ratio)))
                else:
                    plateau_block_size = 1
                print(f"Fit (variance): plateau_var = {plateau_var:.3e} ± {plateau_var_unc:.3e}")
                print(f"Fit (error):    plateau = {plateau_value:.6f} ± {plateau_uncertainty:.6f}")
                print(f"     autocorrelation length ~ {tau:.1f} blocks")
                print(f"     plateau reached at block size ~ {plateau_block_size}")
                if plateau_block_size > max_size:
                    print(f"     !!!Failed to reach plateau in blocking")
                    print(f"     Return max block error")
                    plateau_value = np.sqrt(block_vars.max())
            except RuntimeError as e:
                print(f"\nFit failed: {e}")
                plateau_value = np.sqrt(block_vars.max())
                print(f"Fallback max error: {plateau_value:.6f}")
        
        else: 
            plateau_value = np.sqrt(block_vars.max())
        
        return plateau_value

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class sampler_stoccsd2(sampler_stoccsd):

    @partial(jit, static_argnums=(0,3,4))
    def block_sample(
        self,
        prop_data,
        ham_data,
        prop,
        trial,
        wave_data,
        ):
        """Block scan function. Propagation and energy calculation."""

        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                self.n_prop_steps,
                prop.n_walkers,
                self.n_chol,
            ),
        )
        _step_scan_wrapper = lambda x, y: self._step_scan(
            x, y, ham_data, prop, trial, wave_data
        )
        prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
        prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
            prop_data["weights"]
        )

        # raondom fields_x for T2 decomposition
        xtaus, prop_data = trial.get_xtaus(prop_data, wave_data, prop)

        prop_data = prop.orthonormalize_walkers(prop_data)
        
        ene_hf = jnp.real(trial.calc_energy(prop_data["walkers"], ham_data, wave_data))
        outlier = jnp.abs(ene_hf - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt) # 20 Ha for dt = 0.005
        ene_hf = jnp.where(outlier, prop_data["e_estimate"], ene_hf)
        prop_data["weights"] = jnp.where(outlier, 0.0, prop_data["weights"])

        olp_hf = trial.calc_overlap(prop_data["walkers"], wave_data)
        olp_ci, ene_ci = trial.calc_energy_cid(prop_data["walkers"], ham_data, wave_data)
        num_cr, den_cr = trial.calc_correction(prop_data["walkers"], xtaus, ham_data, wave_data)

        num_ci = olp_ci * ene_ci / olp_hf
        den_ci = olp_ci / olp_hf
        num_cr = num_cr / olp_hf
        den_cr = den_cr / olp_hf

        whf = prop_data["weights"]

        blk_whf = jnp.sum(whf)
        blk_ehf = jnp.sum(whf * ene_hf) / blk_whf
        blk_num_ci = jnp.sum(whf * num_ci) / blk_whf
        blk_den_ci = jnp.sum(whf * den_ci) / blk_whf
        blk_num_cr = jnp.sum(whf * num_cr) / blk_whf
        blk_den_cr = jnp.sum(whf * den_cr) / blk_whf

        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["pop_control_ene_shift"] = 0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_ehf
        prop_data["n_killed_walkers"] = prop_data["weights"].size - jnp.count_nonzero(prop_data["weights"])

        return prop_data, (blk_whf, blk_ehf, blk_num_ci, blk_den_ci, blk_num_cr, blk_den_cr)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))