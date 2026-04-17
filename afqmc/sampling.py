from dataclasses import dataclass
from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
from jax import jit, lax, random
from afqmc.propagation import propagator
from afqmc import linalg_utils
import numpy as np

@dataclass
class sampler:
    n_prop_steps: int = 50
    n_ene_blocks: int = 50
    n_sr_blocks: int = 1
    n_blocks: int = 50
    n_chol: int = 0

    @partial(jit, static_argnums=(0, 4, 5))
    def _step_scan(
        self,
        prop_data: dict,
        fields: jax.Array,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, jax.Array]:
        """Phaseless propagation scan function over steps."""
        prop_data = prop.propagate(trial, ham_data, prop_data, fields, wave_data)
        return prop_data, fields

    @partial(jit, static_argnums=(0, 3, 4))
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

        prop_data = prop.orthonormalize_walkers(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        energies = jnp.real(trial.calc_energy(prop_data["walkers"], ham_data, wave_data))
        outlier = jnp.abs(energies - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt) # 20 Ha for dt = 0.005
        energies = jnp.where(outlier, prop_data["e_estimate"], energies)
        prop_data["weights"] = jnp.where(outlier, 0.0, prop_data["weights"])

        block_weight = jnp.sum(prop_data["weights"])
        block_energy = jnp.sum(prop_data["weights"] * energies) / block_weight

        prop_data["n_killed_walkers"] = prop_data["weights"].size - jnp.count_nonzero(prop_data["weights"])
        prop_data["pop_control_ene_shift"] = 0.9 * prop_data["pop_control_ene_shift"] + 0.1 * block_energy
        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

        return prop_data, (block_weight, block_energy)
    
    def blocking_analysis(self, wt_sp, en_sp, min_nblocks=20, final=False,):
        
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
            print(f"{'B':>4s}  {'NB':>4s}  {'NS':>4s}  {'Observable':>10s}  {'Error':>8s}  {'dError':>8s}")
        for i, block_size in enumerate(block_sizes):
            n_blocks = nsample // block_size
            sl = slice(0, n_blocks * block_size)
            wt = (wt_sp[sl]).reshape(n_blocks, block_size)
            wt_en = (wt_sp[sl] * en_sp[sl]).reshape(n_blocks, block_size)
            block_weight = np.sum(wt, axis=1)
            block_energy = np.sum(wt_en, axis=1) / block_weight
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
                      f'{block_mean:10.6f}  {block_error:8.6f}  {err_of_err:8.6f}')
        
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
    
    def filter_outliers(self, samples, zeta=20):

        median = np.median(samples)
        mad = 1.4826 * np.median(np.abs(samples - median))
        bound = zeta * mad
        mask = np.abs(samples - median) < bound
        print(f"Remove samples outside Zeta > {zeta}")
        print(f"Outlier bound [{median-bound:.6f}, {median+bound:.6f}]")
        
        return mask

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class sampler_fp:
    n_prop_steps: int = 50
    n_eql_blocks: int = 10
    n_trj: int = 100
    n_chol: int = 0
    
    @partial(jit, static_argnums=(0, 4, 5))
    def _step_scan(
        self,
        prop_data: dict,
        fields: jax.Array,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[dict, jax.Array]:
        """Phaseless propagation scan function over steps."""
        prop_data = prop.propagate_free(trial, ham_data, prop_data, fields)
        prop_data["walkers"], norms = linalg_utils.qr_vmap_uhf(prop_data["walkers"])
        norms = norms[0] * norms[1]
        norms_abs = jnp.abs(norms)
        phase = norms / norms_abs
        # weights = jnp.real(norms[0] * norms[1])
        nwalker = int(prop_data["weights"].shape[0])
        prop_data["weights"] *= norms_abs
        prop_data["weights"] = nwalker * prop_data["weights"] / jnp.sum(prop_data["weights"])
        nocca, noccb = prop_data["walkers"][0].shape[-1], prop_data["walkers"][1].shape[-1]
        phase_mo = jnp.stack((phase**(1/(2.0*nocca)), phase**(1/(2.0*noccb)))) # multiply the phase into the mo_coeff
        prop_data["walkers"] = prop._multiply_constant(prop_data["walkers"], phase_mo)

        # sr
        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        return prop_data, fields
    
    @partial(jit, static_argnums=(0, 3, 4))
    def fp_block(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict):
        """free propagation for a block of (n_prop_steps, n_walkers)."""
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

        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        energy_samples = trial.calc_energy(prop_data["walkers"], ham_data, wave_data)

        energy_samples = jnp.where(
            jnp.abs(energy_samples - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
            prop_data["e_estimate"],
            energy_samples,
        )
        
        blk_e = jnp.sum(
                energy_samples * prop_data["overlaps"] * prop_data["weights"]
                ) / jnp.sum(prop_data["overlaps"] * prop_data["weights"])

        blk_w = jnp.sum(prop_data["overlaps"] * prop_data["weights"])

        return prop_data, (blk_w, blk_e)
    
    @partial(jit, static_argnums=(0, 3, 4))
    def scan_eql_blocks(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
    ) -> Tuple[jax.Array, dict]:
        def _block_scan_wrapper(x,_):
            return self.fp_block(x, ham_data, prop, trial, wave_data)

        prop_data, (blk_w, blk_e) = lax.scan(
            _block_scan_wrapper, prop_data, None, length=self.n_eql_blocks
        )

        return prop_data, (blk_w, blk_e)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class sampler_mixed(sampler):

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
class sampler_pt(sampler):

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
        _step_scan_wrapper = lambda x, y: self._step_scan(x, y, ham_data, prop, trial, wave_data)
        prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)

        prop_data = prop.orthonormalize_walkers(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        t, e0, e1 = trial.calc_energy_pt(prop_data["walkers"], ham_data, wave_data)

        outlier = jnp.abs(e0 - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt) # 20 Ha for dt = 0.005
        e0 = jnp.where(outlier, prop_data["e_estimate"], e0)
        prop_data["weights"] = jnp.where(outlier, 0.0, prop_data["weights"])
        prop_data["n_killed_walkers"] = prop_data["weights"].size - jnp.count_nonzero(prop_data["weights"])

        blk_wt = jnp.sum(prop_data["weights"])
        blk_t = jnp.sum(prop_data["weights"] * t) / blk_wt
        blk_e0 = jnp.sum(prop_data["weights"] * e0) / blk_wt
        blk_e1 = jnp.sum(prop_data["weights"] * e1) / blk_wt

        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

        return prop_data, (blk_wt, blk_t, blk_e0, blk_e1)
    
    def ptblocking_analysis(
            self,
            wt_sp,
            t_sp, 
            e0_sp, 
            e1_sp, 
            h0,
            min_nblocks=20
            ):

        from scipy.optimize import curve_fit
        
        nsample = len(wt_sp)
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
        print(f"{'B':>4s}  {'NB':>4s}  {'NS':>4s}  {'Energy':>10s}  {'Error':>8s}  {'dError':>8s}")
        for i, block_size in enumerate(block_sizes):
            n_blocks = nsample // block_size
            sl = slice(0, n_blocks * block_size)
            wt = (wt_sp[sl]).reshape(n_blocks, block_size)
            wt_t = (wt_sp[sl] * t_sp[sl]).reshape(n_blocks, block_size)
            wt_e0 = (wt_sp[sl] * e0_sp[sl]).reshape(n_blocks, block_size)
            wt_e1 = (wt_sp[sl] * e1_sp[sl]).reshape(n_blocks, block_size)
            block_wt = np.sum(wt, axis=1)
            block_t = np.sum(wt_t, axis=1) / block_wt
            block_e0 = np.sum(wt_e0, axis=1) / block_wt
            block_e1 = np.sum(wt_e1, axis=1) / block_wt
            block_energy = (block_e0 + block_e1 - block_t * (block_e0 - h0)).real
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
                  f'{block_mean:10.6f}  {block_error:8.6f}  {err_of_err:8.6f}')

        def model(x, a, b, tau):
            return a - b * np.exp(-x / tau)
        p0 = [block_vars.max(), block_vars.max() - block_vars[0], 5.0]
        try:
            popt, pcov = curve_fit(model, 
                                   block_sizes,
                                   block_vars,
                                   sigma=block_var_errs,
                                   absolute_sigma=True,
                                   p0=p0,
                                   maxfev=10000)
            plateau_var = popt[0]
            plateau_var_unc = np.sqrt(pcov[0, 0])
            plateau_value = np.sqrt(plateau_var)
            # Error propagation: d(sqrt(v)) = dv / (2 sqrt(v))
            plateau_uncertainty = plateau_var_unc / (2.0 * plateau_value)
            tau = popt[2]
            ratio = 0.01 * popt[0] / popt[1]
            if ratio > 0:
                plateau_block_size = int(np.ceil(-popt[2] * np.log(ratio)))
            else:
                plateau_block_size = 1
            # plateau_block_size = min(plateau_block_size, max_size)
            # print(f"Fit (variance): plateau_var = {plateau_var:.6e} ± {plateau_var_unc:.6e}")
            print(f"Fit: plateau = {plateau_value:.6f} ± {plateau_uncertainty:.6f}")
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
        return plateau_value

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))

@dataclass
class sampler_pt2(sampler):
        
    @partial(jit, static_argnums=(0,3,4))
    def block_sample(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial,
        wave_data: dict,
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

        e_guide = jnp.real(trial.calc_energy(prop_data["walkers"], ham_data, wave_data))
        outlier = jnp.abs(e_guide - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt) # 20 Ha for dt = 0.005
        e_guide = jnp.where(outlier, prop_data["e_estimate"], e_guide)
        prop_data["weights"] = jnp.where(outlier, 0.0, prop_data["weights"])
        prop_data["n_killed_walkers"] = prop_data["weights"].size - jnp.count_nonzero(prop_data["weights"])

        t1, t2, e0, e1 = trial.calc_energy_pt(prop_data["walkers"], ham_data, wave_data)

        blk_wt = jnp.sum(prop_data["weights"])
        blk_eg = jnp.sum(prop_data["weights"] * e_guide) / blk_wt
        blk_t1 = jnp.sum(prop_data["weights"] * t1) / blk_wt
        blk_t2 = jnp.sum(prop_data["weights"] * t2) / blk_wt
        blk_e0 = jnp.sum(prop_data["weights"] * e0) / blk_wt
        blk_e1 = jnp.sum(prop_data["weights"] * e1) / blk_wt

        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

        return prop_data, (blk_wt, blk_eg, blk_t1, blk_t2, blk_e0, blk_e1)

    def pt2blocking_analysis(
            self,
            wt_clean, 
            t1_clean, 
            t2_clean, 
            e0_clean, 
            e1_clean, 
            h0, 
            min_nblocks=20
            ):

        from scipy.optimize import curve_fit
        
        nclean = len(wt_clean)
        max_size = nclean // min_nblocks
        if max_size < 10:
            min_nblocks = max(nclean // 10, 3)
            max_size = nclean // min_nblocks
            print(f"Warning: small dataset, relaxed min_nblocks to {min_nblocks}")
        block_sizes = np.arange(1, max_size + 1)
        block_vars = np.zeros(max_size)
        block_var_errs = np.zeros(max_size)
        block_means = np.zeros(max_size)
        print(f"nclean = {nclean}, max_block_size = {max_size}, min_nblocks = {min_nblocks}")
        print(f"{'Blk_SZ':>6s}  {'NBlk':>5s}  {'NSmp':>5s}  {'Energy':>10s}  {'Error':>8s}  {'dError':>8s}")
        for i, block_size in enumerate(block_sizes):
            n_blocks = nclean // block_size
            sl = slice(0, n_blocks * block_size)
            wt = (wt_clean[sl]).reshape(n_blocks, block_size)
            wt_t1 = (wt_clean[sl] * t1_clean[sl]).reshape(n_blocks, block_size)
            wt_t2 = (wt_clean[sl] * t2_clean[sl]).reshape(n_blocks, block_size)
            wt_e0 = (wt_clean[sl] * e0_clean[sl]).reshape(n_blocks, block_size)
            wt_e1 = (wt_clean[sl] * e1_clean[sl]).reshape(n_blocks, block_size)
            block_t1 = np.sum(wt_t1, axis=1)
            block_t2 = np.sum(wt_t2, axis=1)
            block_e0 = np.sum(wt_e0, axis=1)
            block_e1 = np.sum(wt_e1, axis=1)
            block_energy = (h0 + block_e0/block_t1 + block_e1/block_t1
                            - (block_t2 * block_e0) / block_t1**2).real
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
                f'{block_mean:10.6f}  {block_error:8.6f}  {err_of_err:8.6f}')

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
            # Error propagation: d(sqrt(v)) = dv / (2 sqrt(v))
            plateau_uncertainty = plateau_var_unc / (2.0 * plateau_value)
            tau = popt[2]
            ratio = 0.01 * popt[0] / popt[1]
            if ratio > 0:
                plateau_block_size = int(np.ceil(-popt[2] * np.log(ratio)))
            else:
                plateau_block_size = 1
            # plateau_block_size = min(plateau_block_size, max_size)
            print(f"Fit (variance): plateau_var = {plateau_var:.6e} ± {plateau_var_unc:.6e}")
            print(f"Fit (error):    plateau = {plateau_value:.6f} ± {plateau_uncertainty:.6f}")
            print(f"     autocorrelation length ~ {tau:.1f} blocks")
            print(f"     plateau reached at block size ~ {plateau_block_size}")
            if plateau_block_size > max_size:
                print(f"     !!!Failed to reach plateau in blocking")
                print(f"     Return max block error")
                plateau_value = np.sqrt(block_vars.max())
        except RuntimeError as e:
            print(f"\nFit failed: {e}")
            # idx_max = np.argmax(block_vars)
            # plateau_value = np.sqrt(block_vars[idx_max])
            plateau_value = np.sqrt(block_vars.max())
            # plateau_uncertainty = block_var_errs[idx_max] / (2.0 * plateau_value)
            # plateau_block_size = block_sizes[idx_max]
            # popt, pcov = None, None
            print(f"Fallback max error: {plateau_value:.6f}")
            # print(f"     plateau at block size ~ {plateau_block_size}")
        return plateau_value

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class sampler_stoccsd(sampler):
    n_prop_steps: int = 50
    n_sr_blocks: int = 1
    n_blocks: int = 50
    n_chol: int = 0


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

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))