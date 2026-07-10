from dataclasses import dataclass
from functools import partial
# from typing import Tuple
# import jax
import jax.numpy as jnp
from jax import jit, lax, random
# from afqmc.propagation import propagator
# from afqmc.sampling import sampler

import numpy as np
from scipy.optimize import curve_fit

# def pt2orbblocking(
#         wt_sp,
#         t1_sp,
#         t2orb_sp,
#         e0orb_sp,
#         e1orb_sp,
#         e0bar_sp,
#         min_nblocks=20,
#         final=False,
#         ):
#     """Blocking analysis for the PT2 orbital-energy estimator

#         <ept2_orb> = <e0orb>/<t1> + <e1orb>/<t1> - <t2orb>*<e0bar>/<t1>^2

#     In terms of weighted aggregates (E0orb = sum(w*e0orb), etc.):

#         E = E0orb/T1 + E1orb/T1 - T2orb*E0bar/T1**2

#     Differences from the total-energy `pt2blocking`:
#       - no h0 term,
#       - the cross term pairs two *distinct* quantities (t2orb, e0bar)
#         rather than reusing e0, so five sampled quantities feed E.
#     """
#     nsample = len(wt_sp)
#     wt = np.sum(wt_sp)
#     t1    = np.sum(wt_sp * t1_sp)    / wt
#     t2orb = np.sum(wt_sp * t2orb_sp) / wt
#     e0orb = np.sum(wt_sp * e0orb_sp) / wt
#     e1orb = np.sum(wt_sp * e1orb_sp) / wt
#     e0bar = np.sum(wt_sp * e0bar_sp) / wt
#     energy = (e0orb/t1 + e1orb/t1 - t2orb*e0bar/t1**2).real

#     if not final:
#         # No blocking: weight-aware naive error of the nonlinear estimator
#         #   E = E0orb/T1 + E1orb/T1 - T2orb*E0bar/T1**2
#         # with E0orb=sum(w*e0orb), E1orb=sum(w*e1orb), E0bar=sum(w*e0bar),
#         #      T1=sum(w*t1), T2orb=sum(w*t2orb).
#         # Treat each sample as an independent unit and propagate its
#         # contribution to each aggregate through a first-order (delta-method)
#         # linearization -> per-sample influence, then variance of the mean.
#         w     = wt_sp
#         E0orb = np.sum(w * e0orb_sp)
#         E1orb = np.sum(w * e1orb_sp)
#         E0bar = np.sum(w * e0bar_sp)
#         T1    = np.sum(w * t1_sp)
#         T2orb = np.sum(w * t2orb_sp)

#         # partials of E w.r.t. each aggregate
#         dfdE0orb = 1.0 / T1
#         dfdE1orb = 1.0 / T1
#         dfdT2orb = -E0bar / T1**2
#         dfdE0bar = -T2orb / T1**2
#         dfdT1    = (-E0orb / T1**2 - E1orb / T1**2
#                     + 2.0 * T2orb * E0bar / T1**3)

#         # per-sample influence on E (these sum to ~0 since E is scale-free)
#         infl = (dfdE0orb * (w * e0orb_sp)
#                 + dfdE1orb * (w * e1orb_sp)
#                 + dfdT2orb * (w * t2orb_sp)
#                 + dfdE0bar * (w * e0bar_sp)
#                 + dfdT1    * (w * t1_sp)).real
#         var_mean = np.sum(infl**2) * nsample / (nsample - 1)
#         return energy, np.sqrt(var_mean)

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
#         wt_t1    = (wt_sp[sl] * t1_sp[sl]).reshape(n_blocks, block_size)
#         wt_t2orb = (wt_sp[sl] * t2orb_sp[sl]).reshape(n_blocks, block_size)
#         wt_e0orb = (wt_sp[sl] * e0orb_sp[sl]).reshape(n_blocks, block_size)
#         wt_e1orb = (wt_sp[sl] * e1orb_sp[sl]).reshape(n_blocks, block_size)
#         wt_e0bar = (wt_sp[sl] * e0bar_sp[sl]).reshape(n_blocks, block_size)
#         block_t1    = np.sum(wt_t1, axis=1)
#         block_t2orb = np.sum(wt_t2orb, axis=1)
#         block_e0orb = np.sum(wt_e0orb, axis=1)
#         block_e1orb = np.sum(wt_e1orb, axis=1)
#         block_e0bar = np.sum(wt_e0bar, axis=1)
#         block_energy = (block_e0orb/block_t1 + block_e1orb/block_t1
#                         - (block_t2orb * block_e0bar) / block_t1**2).real
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
#         plateau_value = np.sqrt(plateau_var)
#         # Error propagation: d(sqrt(v)) = dv / (2 sqrt(v))
#         plateau_uncertainty = plateau_var_unc / (2.0 * plateau_value)
#         tau = popt[2]
#         ratio = 0.01 * popt[0] / popt[1]
#         if ratio > 0:
#             plateau_block_size = int(np.ceil(-popt[2] * np.log(ratio)))
#         else:
#             plateau_block_size = 1
#         print(f"Fit (variance): plateau_var = {plateau_var:.5e} ± {plateau_var_unc:.5e}")
#         print(f"Fit (error):    plateau = {plateau_value:.5f} ± {plateau_uncertainty:.5f}")
#         print(f"     autocorrelation length ~ {tau:.1f} blocks")
#         print(f"     plateau reached at block size ~ {plateau_block_size}")
#         if tau > max_size or tau < 0:
#             print(f"     !!!Failed to reach plateau in blocking")
#             print(f"     Return max block error")
#             plateau_value = np.sqrt(block_vars.max())
#     except RuntimeError as e:
#         print(f"\nFit failed: {e}")
#         plateau_value = np.sqrt(block_vars.max())
#         print(f"Fallback max error: {plateau_value:.5f}")
#     return energy, plateau_value

def ept2frg_blocking(
        wp_sp,
        t2frg_sp,
        e0frg_sp,
        e1frg_sp,
        e0bar_sp,
        min_nblocks=20,
        final=False,
        ):
    """Blocking analysis for the PT2 fragment-energy estimator

        <ept2_frg> = <e0frg> + <e1frg> - <t2frg>*<e0bar>

    where every average is now taken with the *absorbed* weight
    wp = wt*t1, i.e. <x> = sum(wp*x)/sum(wp).

    In terms of weighted aggregates (E0frg = sum(wp*e0frg),
    WP = sum(wp), etc.):

        E = E0frg/WP + E1frg/WP - T2frg*E0bar/WP**2

    Differences from the previous (t1-explicit) `pt2frgblocking`:
      - t1 is folded into the weight, so the normalizer is the plain
        total weight WP = sum(wp) rather than the aggregate sum(w*t1);
      - equivalently, the old `t1_sp` observable is replaced by the
        constant 1, so its per-sample contribution to the aggregates is
        just wp. Four sampled observables (e0frg, e1frg, t2frg, e0bar)
        plus the weight itself feed E.

    Returns
    -------
    energy : float
        The estimator value.
    error : float
        Naive (final=False) or plateau (final=True) stochastic error.
    wp_mean : float
        Mean of the absorbed per-sample weights, mean(wp_sp).
    """
    nsample = len(wp_sp)
    weighp  = np.mean(wp_sp)
    t2frg = np.mean(wp_sp * t2frg_sp) / weighp
    e0frg = np.mean(wp_sp * e0frg_sp) / weighp
    e1frg = np.mean(wp_sp * e1frg_sp) / weighp
    e0bar = np.mean(wp_sp * e0bar_sp) / weighp
    energy = (e0frg + e1frg - t2frg*e0bar).real

    if not final:
        # No blocking: weight-aware naive error of the nonlinear estimator
        #   E = E0frg/WP + E1frg/WP - T2frg*E0bar/WP**2
        # with E0frg=sum(wp*e0frg), E1frg=sum(wp*e1frg),
        #      E0bar=sum(wp*e0bar), T2frg=sum(wp*t2frg), WP=sum(wp).
        # Treat each sample as an independent unit and propagate its
        # contribution to each aggregate through a first-order (delta-method)
        # linearization -> per-sample influence, then variance of the mean.
        w     = wp_sp
        E0frg = np.sum(w * e0frg_sp)
        E1frg = np.sum(w * e1frg_sp)
        E0bar = np.sum(w * e0bar_sp)
        T2frg = np.sum(w * t2frg_sp)
        WP    = np.sum(w)                 # normalizer, replaces old T1

        # partials of E w.r.t. each aggregate
        dfdE0frg = 1.0 / WP
        dfdE1frg = 1.0 / WP
        dfdT2frg = -E0bar / WP**2
        dfdE0bar = -T2frg / WP**2
        dfdWP    = (-E0frg / WP**2 - E1frg / WP**2
                    + 2.0 * T2frg * E0bar / WP**3)

        # per-sample influence on E (these sum to ~0 since E is scale-free).
        # The weight's own contribution to WP is just w (observable == 1).
        infl = (dfdE0frg * (w * e0frg_sp)
                + dfdE1frg * (w * e1frg_sp)
                + dfdT2frg * (w * t2frg_sp)
                + dfdE0bar * (w * e0bar_sp)
                + dfdWP    * w).real
        var_mean = np.sum(infl**2) * nsample / (nsample - 1)
        return weighp, energy, np.sqrt(var_mean)

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
        wp_blk   = (wp_sp[sl]).reshape(n_blocks, block_size)
        wp_t2frg = (wp_sp[sl] * t2frg_sp[sl]).reshape(n_blocks, block_size)
        wp_e0frg = (wp_sp[sl] * e0frg_sp[sl]).reshape(n_blocks, block_size)
        wp_e1frg = (wp_sp[sl] * e1frg_sp[sl]).reshape(n_blocks, block_size)
        wp_e0bar = (wp_sp[sl] * e0bar_sp[sl]).reshape(n_blocks, block_size)
        block_wp    = np.sum(wp_blk,   axis=1)   # per-block normalizer
        block_t2frg = np.sum(wp_t2frg, axis=1)
        block_e0frg = np.sum(wp_e0frg, axis=1)
        block_e1frg = np.sum(wp_e1frg, axis=1)
        block_e0bar = np.sum(wp_e0bar, axis=1)
        block_energy = (block_e0frg/block_wp + block_e1frg/block_wp
                        - (block_t2frg * block_e0bar) / block_wp**2).real
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
    return weighp, energy, plateau_err

@dataclass
class sampler:
    n_prop_steps: int = 50
    n_blocks: int = 500
    n_chol: int = 0

    @partial(jit, static_argnums=(0, 4, 5))
    def _step_scan(
        self,
        prop_data,
        fields,
        ham_data,
        prop,
        trial,
        wave_data,
        ):
        """Phaseless propagation scan function over steps."""
        prop_data = prop.propagate(trial, ham_data, prop_data, fields, wave_data)
        return prop_data, fields

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
        prop_data = prop.orthonormalize_walkers(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

        e0 = jnp.real(trial.calc_energy(prop_data["walkers"], ham_data, wave_data))

        e0 = jnp.real(e0)
        outlier = jnp.abs(e0 - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt) # 20 Ha for dt = 0.005
        e0 = jnp.where(outlier, prop_data["e_estimate"], e0)
        weights = jnp.where(outlier, 0.0, prop_data["weights"]) # outliers don't contribute

        eorb = trial.calc_orb_energy(prop_data["walkers"], ham_data, wave_data)

        blk_wt = jnp.sum(weights)
        blk_e = jnp.sum(e0 * weights) / blk_wt
        blk_eo = jnp.sum(eorb * weights) / blk_wt

        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * blk_e.real
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        
        return prop_data, (blk_wt, blk_e, blk_eo)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))
    

@dataclass
class sampler_pt(sampler):

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
        prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(prop_data["weights"])
        prop_data = prop.orthonormalize_walkers(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

        eorb, teorb, torb, e0 \
            = trial.calc_eorb_pt(prop_data["walkers"], ham_data, wave_data)
        
        e0 = jnp.real(e0)
        outlier = jnp.abs(e0 - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt) # 20 Ha for dt = 0.005
        e0 = jnp.where(outlier, prop_data["e_estimate"], e0)
        weights = jnp.where(outlier, 0.0, prop_data["weights"])

        blk_wt = jnp.sum(weights)
        blk_eorb = jnp.sum(eorb * weights) / blk_wt
        blk_teorb = jnp.sum(teorb * weights) / blk_wt
        blk_torb = jnp.sum(torb * weights) / blk_wt
        blk_e0 = jnp.sum(e0 * weights) / blk_wt

        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * blk_e0.real
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        
        return prop_data, (blk_wt, blk_eorb, blk_teorb, blk_torb, blk_e0)

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
        prop_data = prop.orthonormalize_walkers(prop_data)

        eg_sp, t1_sp, t2frg_sp, e0frg_sp, e1frg_sp, e0_sp \
            = trial.calc_ept2_frag(prop_data["walkers"],ham_data,wave_data)
        
        eg_sp = jnp.real(eg_sp)
        
        outlier = jnp.abs(eg_sp - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt) # 20 Ha for dt = 0.005
        wts = jnp.where(outlier, 0.0, prop_data["weights"])
        prop_data["weights"] = jnp.where(outlier, 0.0, prop_data["weights"])
        
        prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(prop_data["weights"])
        
        wts = prop_data["weights"]
        wps = wts * t1_sp

        wt    = jnp.sum(wts)
        eg    = jnp.sum(wts * eg_sp) / wt

        wp    = jnp.sum(wps)
        t2frg = jnp.sum(wps * t2frg_sp) / wp
        e0frg = jnp.sum(wps * e0frg_sp) / wp
        e1frg = jnp.sum(wps * e1frg_sp) / wp
        e0    = jnp.sum(wps * e0_sp) / wp
    
        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        # prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * eg.real
        prop_data["pop_control_ene_shift"] = 0.9 * prop_data["pop_control_ene_shift"] + 0.1 * eg.real

        return prop_data, (wt, eg, wp, t2frg, e0frg, e1frg, e0)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))
