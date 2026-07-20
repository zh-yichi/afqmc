from dataclasses import dataclass
from functools import partial
import jax.numpy as jnp
from jax import jit, lax, random

from afqmc.corr_sample import sr as csr

import numpy as np
from scipy.optimize import curve_fit

def blocking(wt_sp1, en_sp1, wt_sp2, en_sp2, min_nblocks=20, final=False):

    nsample = len(wt_sp1)
    assert len(en_sp1) == nsample == len(wt_sp2) == len(en_sp2), \
        "All four arrays must have the same length (pairwise-correlated samples)."
    
    w1 = wt_sp1.real; e1 = en_sp1.real
    w2 = wt_sp2.real; e2 = en_sp2.real
    W1 = np.sum(w1);  W2 = np.sum(w2)
    Ew1 = np.sum(w1 * e1) / W1
    Ew2 = np.sum(w2 * e2) / W2
    denergy = Ew1 - Ew2

    if not final:
        # No blocking: weight-aware naive error of the DIFFERENCE <E1> - <E2>.
        # Each pair i is treated as an independent unit, but the pairing is kept,
        # so the within-pair cross-correlation is retained.
        w1 = wt_sp1.real; e1 = en_sp1.real
        w2 = wt_sp2.real; e2 = en_sp2.real
        W1 = np.sum(w1);  W2 = np.sum(w2)
        Ew1 = np.sum(w1 * e1) / W1
        Ew2 = np.sum(w2 * e2) / W2
        # linearized per-sample contribution (score) of each weighted mean
        s1 = w1 * (e1 - Ew1) / W1
        s2 = w2 * (e2 - Ew2) / W2
        d = s1 - s2                          # per-pair contribution to the difference
        var_mean = np.sum(d**2)              # = Var(E1) + Var(E2) - 2 Cov(E1, E2)
        var_mean *= nsample / (nsample - 1)  # small-sample correction
        return denergy, np.sqrt(var_mean)

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
    print(f"{'B':>4s}  {'NB':>4s}  {'NS':>4s}  {'Delta_E':>10s}  {'Error':>8s}  {'dError':>8s}")

    for i, block_size in enumerate(block_sizes):
        n_blocks = nsample // block_size
        sl = slice(0, n_blocks * block_size)

        # SAME slicing/reshape for both sets -> pairwise correlation preserved
        wt1    = (wt_sp1[sl]).reshape(n_blocks, block_size)
        wt_en1 = (wt_sp1[sl] * en_sp1[sl]).reshape(n_blocks, block_size)
        wt2    = (wt_sp2[sl]).reshape(n_blocks, block_size)
        wt_en2 = (wt_sp2[sl] * en_sp2[sl]).reshape(n_blocks, block_size)

        block_energy1 = (np.sum(wt_en1, axis=1) / np.sum(wt1, axis=1)).real
        block_energy2 = (np.sum(wt_en2, axis=1) / np.sum(wt2, axis=1)).real

        # observable is the per-block energy DIFFERENCE
        block_diff = block_energy1 - block_energy2

        block_mean = np.mean(block_diff)
        block_var = np.var(block_diff, ddof=1) / n_blocks   # variance of the mean
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
        plateau_value = np.sqrt(plateau_var)
        plateau_uncertainty = plateau_var_unc / (2.0 * plateau_value)
        tau = popt[2]
        ratio = 0.01 * popt[0] / popt[1]
        if ratio > 0:
            plateau_block_size = int(np.ceil(-popt[2] * np.log(ratio)))
        else:
            plateau_block_size = 1
        print(f"Fit (variance): plateau_var = {plateau_var:.3e} ± {plateau_var_unc:.3e}")
        print(f"Fit (error):    plateau = {plateau_value:.5f} ± {plateau_uncertainty:.5f}")
        print(f"     autocorrelation length ~ {tau:.1f} blocks")
        print(f"     plateau reached at block size ~ {plateau_block_size}")
        if plateau_block_size > max_size:
            print(f"     !!!Failed to reach plateau in blocking")
            print(f"     Return max block error")
            plateau_value = np.sqrt(block_vars.max())
    except RuntimeError as e:
        print(f"\nFit failed: {e}")
        plateau_value = np.sqrt(block_vars.max())
        print(f"Fallback max error: {plateau_value:.5f}")

    return denergy, plateau_value


def pt2blocking(
        h01, wt_sp1, t1_sp1, t2_sp1, e0_sp1, e1_sp1,
        h02, wt_sp2, t1_sp2, t2_sp2, e0_sp2, e1_sp2,
        min_nblocks=20,
        final=False,
        ):

    nsample = len(wt_sp1)
    assert len(wt_sp2) == nsample, \
        "both sets must have the same (paired) length"

    # ---- whole-sample PT2 energies and their difference (this is returned) ----
    def _pt2energy(h0, w, t1, t2, e0, e1):
        W  = np.sum(w)
        T1 = np.sum(w * t1) / W
        T2 = np.sum(w * t2) / W
        E0 = np.sum(w * e0) / W
        E1 = np.sum(w * e1) / W
        return (h0 + E0/T1 + E1/T1 - T2*E0/T1**2).real

    energy1 = _pt2energy(h01, wt_sp1, t1_sp1, t2_sp1, e0_sp1, e1_sp1)
    energy2 = _pt2energy(h02, wt_sp2, t1_sp2, t2_sp2, e0_sp2, e1_sp2)
    denergy = energy1 - energy2

    if not final:
        # No blocking: weight-aware naive error of the DIFFERENCE E1 - E2.
        # Linearize each nonlinear estimator (delta method) into per-sample
        # influences, subtract PAIRWISE (same order kept), variance of the mean.
        # -> automatically Var(E1) + Var(E2) - 2 Cov(E1, E2).
        def _influence(w, t1, t2, e0, e1):
            E0 = np.sum(w * e0); E1 = np.sum(w * e1)
            T1 = np.sum(w * t1); T2 = np.sum(w * t2)
            dfdE0 = 1.0/T1 - T2/T1**2
            dfdE1 = 1.0/T1
            dfdT1 = -E0/T1**2 - E1/T1**2 + 2.0*T2*E0/T1**3
            dfdT2 = -E0/T1**2
            return (dfdE0*(w*e0) + dfdE1*(w*e1)
                    + dfdT1*(w*t1) + dfdT2*(w*t2)).real

        infl1 = _influence(wt_sp1, t1_sp1, t2_sp1, e0_sp1, e1_sp1)
        infl2 = _influence(wt_sp2, t1_sp2, t2_sp2, e0_sp2, e1_sp2)
        d = infl1 - infl2                    # per-pair contribution to E1 - E2
        var_mean = np.sum(d**2) * nsample / (nsample - 1)
        return denergy, np.sqrt(var_mean)

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
    print(f"{'Blk_SZ':>6s}  {'NBlk':>5s}  {'NSmp':>5s}  {'dEnergy':>10s}  {'Error':>8s}  {'dError':>8s}")

    def _block_energy(h0, w_sp, t1_sp, t2_sp, e0_sp, e1_sp, sl, n_blocks, block_size):
        b_t1 = np.sum((w_sp[sl] * t1_sp[sl]).reshape(n_blocks, block_size), axis=1)
        b_t2 = np.sum((w_sp[sl] * t2_sp[sl]).reshape(n_blocks, block_size), axis=1)
        b_e0 = np.sum((w_sp[sl] * e0_sp[sl]).reshape(n_blocks, block_size), axis=1)
        b_e1 = np.sum((w_sp[sl] * e1_sp[sl]).reshape(n_blocks, block_size), axis=1)
        return (h0 + b_e0/b_t1 + b_e1/b_t1 - (b_t2 * b_e0)/b_t1**2).real

    for i, block_size in enumerate(block_sizes):
        n_blocks = nsample // block_size
        sl = slice(0, n_blocks * block_size)
        # SAME slice/reshape for both sets -> pairwise correlation preserved
        block_energy1 = _block_energy(h01, wt_sp1, t1_sp1, t2_sp1, e0_sp1, e1_sp1,
                                      sl, n_blocks, block_size)
        block_energy2 = _block_energy(h02, wt_sp2, t1_sp2, t2_sp2, e0_sp2, e1_sp2,
                                      sl, n_blocks, block_size)
        block_diff = block_energy1 - block_energy2   # per-block DIFFERENCE

        block_mean = np.mean(block_diff)
        block_var = np.var(block_diff, ddof=1) / n_blocks  # variance of the mean
        block_error = np.sqrt(block_var)
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
        plateau_value = np.sqrt(plateau_var)
        plateau_uncertainty = plateau_var_unc / (2.0 * plateau_value)
        tau = popt[2]
        ratio = 0.01 * popt[0] / popt[1]
        if ratio > 0:
            plateau_block_size = int(np.ceil(-popt[2] * np.log(ratio)))
        else:
            plateau_block_size = 1
        print(f"Fit (variance): plateau_var = {plateau_var:.5e} ± {plateau_var_unc:.5e}")
        print(f"Fit (error):    plateau = {plateau_value:.5f} ± {plateau_uncertainty:.5f}")
        print(f"     autocorrelation length ~ {tau:.1f} blocks")
        print(f"     plateau reached at block size ~ {plateau_block_size}")
        if plateau_block_size > max_size:
            print(f"     !!!Failed to reach plateau in blocking")
            print(f"     Return max block error")
            plateau_value = np.sqrt(block_vars.max())
    except RuntimeError as e:
        print(f"\nFit failed: {e}")
        plateau_value = np.sqrt(block_vars.max())
        print(f"Fallback max error: {plateau_value:.5f}")

    return denergy, plateau_value


@dataclass
class sampler:
    n_prop_steps: int
    n_blocks: int
    n_walkers: int
    n_chol: int

    @partial(jit, static_argnums=(0, 1, 2, 6))
    def prop_step(self, prop,
                  trial1, prop_data1, ham_data1, wave_data1,
                  trial2, prop_data2, ham_data2, wave_data2,
                  ):
        """Phaseless propagation scan function over steps (correlated fields)."""
        # Draw one shared set of fields. Use a single key source (prop_data1's).
        prop_data1["key"], subkey = random.split(prop_data1["key"])
        fields = random.normal(
            subkey,
            shape=(
                self.n_prop_steps,
                self.n_walkers,
                self.n_chol,
            ),
        )

        def scan_fn(carry, field):
            # field has shape (n_walkers, n_chol) — shared by both sets
            prop_data1, prop_data2 = carry
            prop_data1 = prop.propagate(trial1, ham_data1, prop_data1, field, wave_data1)
            prop_data2 = prop.propagate(trial2, ham_data2, prop_data2, field, wave_data2)
            return (prop_data1, prop_data2), None

        (prop_data1, prop_data2), _ = lax.scan(scan_fn, (prop_data1, prop_data2), fields)

        prop_data1["n_killed_walkers"] = (
                prop_data1["weights"].size - jnp.count_nonzero(prop_data1["weights"])
            )
        prop_data2["n_killed_walkers"] = (
                prop_data2["weights"].size - jnp.count_nonzero(prop_data2["weights"])
            )

        prop_data1 = prop.orthonormalize_walkers(prop_data1)
        prop_data2 = prop.orthonormalize_walkers(prop_data2)

        return prop_data1, prop_data2

    @partial(jit, static_argnums=(0, 1 ,2, 6))
    def block_sample(self, prop,
                     trial1, prop_data1, ham_data1, wave_data1,
                     trial2, prop_data2, ham_data2, wave_data2,
                     ):
        """Block scan function. Propagation and energy calculation."""
        prop_data1, prop_data2 = self.prop_step(
            prop, 
            trial1, prop_data1, ham_data1, wave_data1,
            trial2, prop_data2, ham_data2, wave_data2
            )

        en_sp1 = jnp.real(trial1.calc_energy(prop_data1["walkers"], ham_data1, wave_data1))
        en_sp2 = jnp.real(trial2.calc_energy(prop_data2["walkers"], ham_data2, wave_data2))

        # rm extreme outliers
        outlier1 = jnp.abs(en_sp1 - prop_data1["e_estimate"]) > jnp.sqrt(2.0 / prop.dt)
        outlier2 = jnp.abs(en_sp2 - prop_data2["e_estimate"]) > jnp.sqrt(2.0 / prop.dt)
        outlier = outlier1 | outlier2 # union
        prop_data1["weights"] = jnp.where(outlier, 0.0, prop_data1["weights"])
        prop_data2["weights"] = jnp.where(outlier, 0.0, prop_data2["weights"])

        wt_sp1 = prop_data1["weights"]
        wt_sp2 = prop_data2["weights"]

        wt1 = jnp.sum(wt_sp1)
        wt2 = jnp.sum(wt_sp2)
        en1 = jnp.sum(wt_sp1 * en_sp1) / wt1
        en2 = jnp.sum(wt_sp2 * en_sp2) / wt2

        prop_data1, prop_data2 = csr.stochastic_reconfiguration(prop_data1, prop_data2)
        prop_data1["overlaps"] = trial1.calc_overlap(prop_data1["walkers"], wave_data1)
        prop_data2["overlaps"] = trial2.calc_overlap(prop_data2["walkers"], wave_data2)
        prop_data1["e_estimate"] = 0.9 * prop_data1["e_estimate"] + 0.1 * en1
        prop_data2["e_estimate"] = 0.9 * prop_data2["e_estimate"] + 0.1 * en2
        prop_data1["pop_control_ene_shift"] = prop_data1["e_estimate"]
        prop_data2["pop_control_ene_shift"] = prop_data2["e_estimate"]

        return (prop_data1, prop_data2), (wt1, en1, wt2, en2)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))
    

@dataclass
class sampler_pt2(sampler):

    @partial(jit, static_argnums=(0, 1 ,2, 6))
    def block_sample(self, prop,
                     trial1, prop_data1, ham_data1, wave_data1,
                     trial2, prop_data2, ham_data2, wave_data2,
                     ):
        """Block scan function. Propagation and energy calculation."""
        prop_data1, prop_data2 = self.prop_step(
            prop, 
            trial1, prop_data1, ham_data1, wave_data1,
            trial2, prop_data2, ham_data2, wave_data2
            )

        eg_sp1 = jnp.real(trial1.calc_energy(prop_data1["walkers"], ham_data1, wave_data1))
        eg_sp2 = jnp.real(trial2.calc_energy(prop_data2["walkers"], ham_data2, wave_data2))

        # rm extreme outliers
        outlier1 = jnp.abs(eg_sp1 - prop_data1["e_estimate"]) > jnp.sqrt(2.0 / prop.dt)
        outlier2 = jnp.abs(eg_sp2 - prop_data2["e_estimate"]) > jnp.sqrt(2.0 / prop.dt)
        outlier = outlier1 | outlier2 # union
        wt_sp1 = jnp.where(outlier, 0.0, prop_data1["weights"])
        wt_sp2 = jnp.where(outlier, 0.0, prop_data2["weights"])

        wt1 = jnp.sum(wt_sp1)
        wt2 = jnp.sum(wt_sp2)
        eg1 = jnp.sum(wt_sp1 * eg_sp1) / wt1
        eg2 = jnp.sum(wt_sp2 * eg_sp2) / wt2

        t1_sp1, t2_sp1, e0_sp1, e1_sp1 \
            = trial1.calc_energy_pt(prop_data1["walkers"], ham_data1, wave_data1)
        t1_sp2, t2_sp2, e0_sp2, e1_sp2 \
            = trial2.calc_energy_pt(prop_data2["walkers"], ham_data2, wave_data2)

        t11 = jnp.sum(wt_sp1 * t1_sp1) / wt1
        t21 = jnp.sum(wt_sp1 * t2_sp1) / wt1
        e01 = jnp.sum(wt_sp1 * e0_sp1) / wt1
        e11 = jnp.sum(wt_sp1 * e1_sp1) / wt1

        t12 = jnp.sum(wt_sp2 * t1_sp2) / wt2
        t22 = jnp.sum(wt_sp2 * t2_sp2) / wt2
        e02 = jnp.sum(wt_sp2 * e0_sp2) / wt2
        e12 = jnp.sum(wt_sp2 * e1_sp2) / wt2

        prop_data1, prop_data2 = csr.stochastic_reconfiguration(prop_data1, prop_data2)
        prop_data1["overlaps"] = trial1.calc_overlap(prop_data1["walkers"], wave_data1)
        prop_data2["overlaps"] = trial2.calc_overlap(prop_data2["walkers"], wave_data2)
        prop_data1["e_estimate"] = 0.9 * prop_data1["e_estimate"] + 0.1 * eg1
        prop_data2["e_estimate"] = 0.9 * prop_data2["e_estimate"] + 0.1 * eg2
        prop_data1["pop_control_ene_shift"] = prop_data1["e_estimate"]
        prop_data2["pop_control_ene_shift"] = prop_data2["e_estimate"]

        return (prop_data1, prop_data2), (wt1, eg1, t11, t21, e01, e11, 
                                          wt2, eg2, t12, t22, e02, e12)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))