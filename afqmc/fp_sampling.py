from dataclasses import dataclass
from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
from jax import jit, lax, random
from afqmc.propagation import propagator
from afqmc import linalg_utils
# import numpy as np


@dataclass
class fp_sampler:
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
class fp_sampler_pt2(fp_sampler):
  
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
                
        olps, wts = prop_data["overlaps"], prop_data["weights"]
        t1, t2, e0, e1 = trial.calc_energy_pt(prop_data["walkers"], ham_data, wave_data)

        e_estimate = ham_data["h0"] + e0/t1 + e1/t1 - t2 * e0 / t1**2
        outlier = jnp.abs(e_estimate - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt) # 20 Ha for dt = 0.005
        wts = jnp.where(outlier, 0.0, wts)
        
        blk_wt = jnp.sum(olps * wts)
        blk_t1 = jnp.sum(olps * wts * t1) / blk_wt
        blk_t2 = jnp.sum(olps * wts * t2) / blk_wt
        blk_e0 = jnp.sum(olps * wts * e0) / blk_wt
        blk_e1 = jnp.sum(olps * wts * e1) / blk_wt

        return prop_data, (blk_wt, blk_t1, blk_t2, blk_e0, blk_e1)
    
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

        prop_data, (blk_wt, blk_t1, blk_t2, blk_e0, blk_e1) = lax.scan(
            _block_scan_wrapper, prop_data, None, length=self.n_eql_blocks
        )

        return prop_data, (blk_wt, blk_t1, blk_t2, blk_e0, blk_e1)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))