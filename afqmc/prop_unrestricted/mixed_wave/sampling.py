import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, lax, random
from typing import Tuple
from functools import partial
from dataclasses import dataclass
from afqmc.prop_unrestricted.sampling import sampler
from afqmc.prop_unrestricted.propagation import propagator

sampler = sampler

@dataclass
class sampler_mixed(sampler):
    n_prop_steps: int = 50
    n_ene_blocks: int = 50
    n_sr_blocks: int = 1
    n_blocks: int = 50
    n_chol: int = 0

    @partial(jit, static_argnums=(0,3,4))
    def _block_scan(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
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
        og = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["overlaps"] = og
        otg, eg, et = trial.calc_energy_mixed(prop_data["walkers"],ham_data,wave_data)
        de, do = trial.calc_stoccsd_cr(prop_data["walkers"], ham_data, wave_data)

        ot = otg * og
        ot_cr = ot + do
        et_cr = (et * ot + de) / ot_cr

        otg = jnp.real(otg)
        eg = jnp.real(eg)
        et = jnp.real(et)
        ot_cr = jnp.real(ot_cr)
        et_cr = jnp.real(et_cr)

        eg = jnp.where(
            jnp.abs(eg - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
            prop_data["e_estimate"],
            eg,
        )
        
        wt = prop_data["weights"]
        wp = wt * otg
        wp_cr = wt * ot_cr / og

        blk_wt = jnp.sum(wt)
        blk_wp = jnp.sum(wp)
        blk_wp_cr = jnp.sum(wp_cr)
        blk_eg = jnp.sum(eg * wt) / blk_wt
        blk_et = jnp.sum(et * wp) / blk_wp
        blk_et_cr = jnp.sum(et_cr * wp_cr) / blk_wp_cr

        prop_data["pop_control_ene_shift"] = (
                0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_eg
            )

        return prop_data, (blk_wt, blk_wp, blk_wp_cr, blk_eg, blk_et, blk_et_cr)

    @partial(jit, static_argnums=(0,3,4))
    def _sr_block_scan(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
            
        def _block_scan_wrapper(x,_):
            return self._block_scan(x,ham_data,prop,trial,wave_data)
        
        prop_data, (blk_wt, blk_wp, blk_wp_cr, blk_eg, blk_et, blk_et_cr) \
            = lax.scan(
            _block_scan_wrapper, prop_data, None, length = self.n_ene_blocks
        )
        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        return prop_data, (blk_wt, blk_wp, blk_wp_cr, blk_eg, blk_et, blk_et_cr)

    @partial(jit, static_argnums=(0,3,4))
    def propagate_phaseless(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[jax.Array, dict]:
        def _sr_block_scan_wrapper(x,_):
            return self._sr_block_scan(x, ham_data, prop, trial, wave_data)

        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["n_killed_walkers"] = 0
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        prop_data, (blk_wt, blk_wp, blk_wp_cr, blk_eg, blk_et, blk_et_cr)\
            = lax.scan(
            _sr_block_scan_wrapper, prop_data, None, length = self.n_sr_blocks
        )
        prop_data["n_killed_walkers"] /= (
            self.n_sr_blocks * self.n_ene_blocks * prop.n_walkers
        )

        wt = jnp.sum(blk_wt)
        wp = jnp.sum(blk_wp)
        wp_cr = jnp.sum(blk_wp_cr)
        eg = jnp.sum(blk_eg * blk_wt) / wt
        et = jnp.sum(blk_et * blk_wp) / wp
        et_cr = jnp.sum(blk_et_cr * blk_wp_cr) / wp_cr

        return prop_data, (wt, wp, wp_cr, eg, et, et_cr)
    
    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class sampler_stoccsd(sampler):
    n_prop_steps: int = 50
    n_sr_blocks: int = 1
    n_blocks: int = 50
    n_chol: int = 0

    @partial(jit, static_argnums=(0,3,4))
    def _block_froze(self, prop_data, ham_data, prop, trial, wave_data):
        """Block scan function. Frozen walkers, no propagation, only energy calculation."""

        # raondom fields_x for T2 decomposition
        xtaus, prop_data = trial.get_xtaus(prop_data, wave_data, prop)
        olp_hf = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["overlaps"] = olp_hf
        ene_hf = jnp.real(trial.calc_energy(prop_data["walkers"], ham_data, wave_data))
        olp_cc, ene_cc = trial.calc_energy_stoccsd(prop_data["walkers"], xtaus, ham_data, wave_data)
        wt_hf = prop_data["weights"]

        weight = jnp.sum(wt_hf)
        ehf_avg = jnp.sum(wt_hf * ene_hf) / weight
        ecc_avg = jnp.sum(wt_hf * olp_cc / olp_hf * ene_cc) / weight
        ecc_abs = jnp.sum(wt_hf * jnp.abs(olp_cc / olp_hf) * ene_cc) / weight
        occ_avg = jnp.sum(wt_hf * olp_cc / olp_hf) / weight
        occ_abs = jnp.sum(wt_hf * jnp.abs(olp_cc / olp_hf)) / weight

        return prop_data, (weight, ehf_avg, ecc_avg, ecc_abs, occ_avg, occ_abs)

    @partial(jit, static_argnums=(0,3,4))
    def _block_same(self, prop_data, ham_data, prop, trial, wave_data):
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
        xtaus, prop_data = trial.get_same_xtaus(prop_data, wave_data)
        prop_data = prop.orthonormalize_walkers(prop_data)
        # prop_data = prop.stochastic_reconfiguration_local(prop_data)

        olp_hf = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["overlaps"] = olp_hf
        ene_hf = jnp.real(trial.calc_energy(prop_data["walkers"], ham_data, wave_data))
        olp_cc, ene_cc = trial.calc_energy_same_stoccsd(prop_data["walkers"], xtaus, ham_data, wave_data)
        wt_hf = prop_data["weights"]

        weight = jnp.sum(wt_hf)
        ehf_avg = jnp.sum(wt_hf * ene_hf) / weight
        ecc_avg = jnp.sum(wt_hf * olp_cc / olp_hf * ene_cc) / weight
        occ_avg = jnp.sum(wt_hf * olp_cc / olp_hf) / weight
        occ_abs = jnp.sum(wt_hf * jnp.abs(olp_cc / olp_hf)) / weight

        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

        return prop_data, (weight, ehf_avg, ecc_avg, occ_avg, occ_abs)
    

    @partial(jit, static_argnums=(0,3,4))
    def _block_get(self, prop_data, ham_data, prop, trial, wave_data):
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
        # xtaus, prop_data = trial.get_same_xtaus(prop_data, wave_data, prop)
        prop_data = prop.orthonormalize_walkers(prop_data)
        # prop_data = prop.stochastic_reconfiguration_local(prop_data)

        olp_hf = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["overlaps"] = olp_hf
        ene_hf = jnp.real(trial.calc_energy(prop_data["walkers"], ham_data, wave_data))
        olp_cc, ene_cc = trial.calc_energy_get_stoccsd(prop_data["walkers"], ham_data, wave_data, prop_data)
        wt_hf = prop_data["weights"]

        weight = jnp.sum(wt_hf)
        ehf_avg = jnp.sum(wt_hf * ene_hf) / weight
        ecc_avg = jnp.sum(wt_hf * olp_cc / olp_hf * ene_cc) / weight
        occ_avg = jnp.sum(wt_hf * olp_cc / olp_hf) / weight
        occ_abs = jnp.sum(wt_hf * jnp.abs(olp_cc / olp_hf)) / weight

        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

        return prop_data, (weight, ehf_avg, ecc_avg, occ_avg, occ_abs)


    @partial(jit, static_argnums=(0,3,4))
    def _block(self, prop_data, ham_data, prop, trial, wave_data):
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
    
    @partial(jit, static_argnums=(0,3,4))
    def propagate_phaseless(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[jax.Array, dict]:
        def _scan_blocks(x,_):
            return self._block(x, ham_data, prop, trial, wave_data)

        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["n_killed_walkers"] = 0
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        prop_data, (blk_wt, blk_ehf, blk_num, blk_den) \
            = lax.scan(
            _scan_blocks, prop_data, None, length = self.n_sr_blocks
        )
        prop_data["n_killed_walkers"] /= (
            self.n_sr_blocks * prop.n_walkers
        )
        
        wt = jnp.sum(blk_wt)
        ehf = jnp.sum(blk_wt * blk_ehf) / wt
        num = jnp.sum(blk_wt * blk_num) / wt
        den = jnp.sum(blk_wt * blk_den) / wt

        return prop_data, (wt, ehf, num, den)
    
    def blk_average(self, wt_sp, num_sp, den_sp, max_size=None, printE=False):
        n_total = len(wt_sp)
        if max_size is None:
            max_size = n_total // 10
        err = np.zeros(max_size)
        if printE:
            print(f"{'Blk_SZ':>6s}  {'NBlk':>6s}  {'NSmp':>6s}  {'Energy':>10s}  {'Error':>8s}")
        for i, block_size in enumerate(range(1,max_size+1)):
            n_blocks = n_total // block_size

            wt_truncated = wt_sp[:n_blocks * block_size]
            num_truncated = num_sp[:n_blocks * block_size]
            den_truncated = den_sp[:n_blocks * block_size]

            wt_num = wt_truncated * num_truncated
            wt_den = wt_truncated * den_truncated

            wt_num = wt_num.reshape(n_blocks, block_size)
            wt_den = wt_den.reshape(n_blocks, block_size)

            block_num = np.sum(wt_num, axis=1)
            block_den = np.sum(wt_den, axis=1)

            block_energy = (block_num / block_den).real
            block_mean = np.mean(block_energy)
            block_error = np.std(block_energy, ddof=1) / np.sqrt(n_blocks)
            if printE:
                print(f"{block_size:6d}  {n_blocks:6d}  {block_size*n_blocks:6d}  {block_mean:10.6f}  {block_error:10.6f}")
            err[i] = block_error
        return err
    
    # def filter_outliers(self, weights, num, den, zeta=5):

    #     weights_mean = weights.mean()
    #     sigma = np.std(weights)
    #     lower_bound = weights_mean - zeta*sigma
    #     upper_bound = weights_mean + zeta*sigma
    #     mask = (weights >= lower_bound) & (weights <= upper_bound)
        
    #     w_filtered = weights[mask]
    #     n_filtered = num[mask]
    #     d_filtered = den[mask]
        
    #     n_removed = len(weights) - len(w_filtered)
    #     print(f"Removed {n_removed} outliers")
    #     print(f"Weight bounds: [{lower_bound:.4e}, {upper_bound:.4e}]")
        
    #     return w_filtered, n_filtered, d_filtered

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class sampler_stoccsd2(sampler):
    n_prop_steps: int = 50
    n_sr_blocks: int = 1
    n_blocks: int = 50
    n_chol: int = 0

    @partial(jit, static_argnums=(0,3,4))
    def block_sample(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
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
    
    def jackknife(self, weights, n_ci, n_cr, d_ci, d_cr, block_size):
        """
        Performs Block Jackknife on the ratio estimator E = (<N_ci> + <N_cr>) / (<D_ci> + <D_cr>)
        """
        # 1. Ensure all inputs are numpy arrays
        w = np.array(weights)
        n1, n2 = np.array(n_ci), np.array(n_cr)
        d1, d2 = np.array(d_ci), np.array(d_cr)
        
        # 2. Calculate weighted components per sample
        # (Total Numerator and Total Denominator contributions per sample)
        num_samples = (len(w) // block_size) * block_size
        # print(num_samples)
        weighted_num = (w * (n1 + n2))[:num_samples]
        weighted_den = (w * (d1 + d2))[:num_samples]
        
        # 3. Reshape and sum into Blocks
        num_blocks = num_samples // block_size
        # We sum the weighted values within each block
        block_sums_num = weighted_num.reshape(num_blocks, block_size).sum(axis=1)
        block_sums_den = weighted_den.reshape(num_blocks, block_size).sum(axis=1)
        
        # 4. Calculate Global Totals
        total_num = np.sum(block_sums_num)
        total_den = np.sum(block_sums_den)
        
        # 5. Generate Jackknife Estimates (Leave-One-Block-Out)
        # e_jk[j] is the energy calculated excluding block j
        e_jk = (total_num - block_sums_num) / (total_den - block_sums_den)
        
        # 6. Calculate Jackknife Statistics
        m = num_blocks
        mean_jk = np.mean(e_jk)
        # The (m-1) factor is the standard Jackknife variance scaling
        err_jk = np.sqrt((m - 1) / m * np.sum((e_jk - mean_jk)**2))
        
        return err_jk.real
    
    def blocking_jackknife(self, whf_sp, numci_sp, numcr_sp, denci_sp, dencr_sp, nblk=None):
        if nblk is None:
            nblk = len(whf_sp) // 2
        err = np.zeros(nblk)
        thresh = 1.05
        # print(f'b_size Energy_JK Error_JK')
        for i in range(nblk):
            err[i] = self.block_jackknife(whf_sp, numci_sp, numcr_sp, denci_sp, dencr_sp, block_size=i+1)
            if err[i] < err[i-1]*thresh:
                break
        return err[i] # report the last error
    
    def sto_blocking_analysis(self, wt_sp, num_sp, den_sp, min_nblocks=20, final=False,):
        import numpy as np
        
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

        #     # wt_truncated = wt_sp[:n_blocks * block_size]
        #     # num_truncated = num_sp[:n_blocks * block_size]
        #     # den_truncated = den_sp[:n_blocks * block_size]

        #     # wt_num = wt_truncated * num_truncated
        #     # wt_den = wt_truncated * den_truncated

        #     # wt_num = wt_num.reshape(n_blocks, block_size)
        #     # wt_den = wt_den.reshape(n_blocks, block_size)

        #     # block_num = np.sum(wt_num, axis=1)
        #     # block_den = np.sum(wt_den, axis=1)

        #     # block_energy = (block_num / block_den).real
        #     # block_mean = np.mean(block_energy)
        #     # block_error = np.std(block_energy, ddof=1) / np.sqrt(n_blocks)
        #     # print(f"{block_size:6d}  {n_blocks:6d}  {block_size*n_blocks:6d}  {block_mean:10.6f}  {block_error:10.6f}")
        #     # print(f' {block_size}  {n_blocks}  {block_size*n_blocks}  {block_mean:.6f}  {block_error:.6f}')
        #     # err[i] = block_error
        # return err
    
    # def filter_outliers(self, weights, num, den, zeta=5):

    #     weights_mean = weights.mean()
    #     sigma = np.std(weights)
    #     lower_bound = weights_mean - zeta*sigma
    #     upper_bound = weights_mean + zeta*sigma
    #     mask = (weights >= lower_bound) & (weights <= upper_bound)
        
    #     w_filtered = weights[mask]
    #     n_filtered = num[mask]
    #     d_filtered = den[mask]
        
    #     n_removed = len(weights) - len(w_filtered)
    #     print(f"Removed {n_removed} outliers")
    #     print(f"Weight bounds: [{lower_bound:.4e}, {upper_bound:.4e}]")
        
    #     return w_filtered, n_filtered, d_filtered


    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))
    
# @dataclass
# class sampler_ustoccsd2(sampler_stoccsd2):
#     n_prop_steps: int = 50
#     n_ene_blocks: int = 50
#     n_sr_blocks: int = 1
#     n_blocks: int = 50
#     n_chol: int = 0

#     @partial(jit, static_argnums=(0,3,4))
#     def _block_scan(
#         self,
#         prop_data: dict,
#         ham_data: dict,
#         prop: propagator,
#         trial,
#         wave_data: dict,
#     ) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
#         """Block scan function. Propagation and energy calculation."""

#         prop_data["key"], subkey = random.split(prop_data["key"])
#         fields = random.normal(
#             subkey,
#             shape=(
#                 self.n_prop_steps,
#                 prop.n_walkers,
#                 self.n_chol,
#             ),
#         )
#         _step_scan_wrapper = lambda x, y: self._step_scan(
#             x, y, ham_data, prop, trial, wave_data
#         )
#         prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
#         prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
#             prop_data["weights"]
#         )

#         xtaus = trial.get_xtaus(prop_data, wave_data, prop)

#         prop_data = prop.orthonormalize_walkers(prop_data)
#         overlap_hf = trial.calc_overlap(prop_data["walkers"], wave_data)
#         prop_data["overlaps"] = overlap_hf
#         overlap_ci, energy_ci = trial.calc_energy_cid(prop_data["walkers"], ham_data, wave_data)
#         numerator_cr, denominator_cr = trial.calc_correction(prop_data["walkers"], xtaus, ham_data, wave_data)

#         num_ci = overlap_ci * energy_ci / overlap_hf
#         den_ci = overlap_ci / overlap_hf
#         num_cr = numerator_cr / overlap_hf
#         den_cr = denominator_cr / overlap_hf

#         whf = prop_data["weights"]

#         blk_whf = jnp.sum(whf)
#         blk_num_ci = jnp.sum(whf * num_ci) / blk_whf
#         blk_den_ci = jnp.sum(whf * den_ci) / blk_whf
#         blk_num_cr = jnp.sum(whf * num_cr) / blk_whf
#         blk_den_cr = jnp.sum(whf * den_cr) / blk_whf

#         return prop_data, (blk_whf, blk_num_ci, blk_den_ci, blk_num_cr, blk_den_cr)
    
#     def __hash__(self) -> int:
#         return hash(tuple(self.__dict__.values()))
