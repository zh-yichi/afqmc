from dataclasses import dataclass
from functools import partial
# from typing import Tuple
# import jax
import jax.numpy as jnp
from jax import jit, lax, random
# from afqmc.propagation import propagator
# from afqmc.sampling import sampler

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
        prop_data["pop_control_ene_shift"] = 0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_e
        
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
        prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
            prop_data["weights"]
        )
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
        prop_data["pop_control_ene_shift"] = 0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_e0
        
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
        prop_data["n_killed_walkers"] = 0
        prop_data = prop.orthonormalize_walkers(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        e0, t1olp, eorb, t2eorb, t2orb, e0bar \
            = trial.calc_eorb_pt2(prop_data["walkers"],ham_data,wave_data)
        
        e0 = jnp.real(e0)
        outlier = jnp.abs(e0 - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt) # 20 Ha for dt = 0.005
        weights = jnp.where(outlier, 0.0, prop_data["weights"])

        eorb = t1olp * eorb
        t2eorb = t1olp * t2eorb
        t2orb = t1olp * t2orb
        e0bar = t1olp * e0bar

        blk_wt = jnp.sum(weights)
        blk_e0 = jnp.sum(e0 * weights) / blk_wt
        blk_eorb = jnp.sum(eorb * weights) / blk_wt
        blk_t2eorb = jnp.sum(t2eorb * weights) / blk_wt
        blk_t2orb = jnp.sum(t2orb * weights) / blk_wt
        blk_e0bar = jnp.sum(e0bar * weights) / blk_wt
        blk_t1olp = jnp.sum(t1olp * weights) / blk_wt
    
        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["pop_control_ene_shift"] = 0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_e0

        return prop_data, (blk_wt, blk_e0, blk_eorb, blk_t2eorb, blk_t2orb, blk_e0bar, blk_t1olp)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))
    

# @dataclass
# class sampler_eq(sampler):
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

#         prop_data = prop.orthonormalize_walkers(prop_data)
#         prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
#         e0 = jnp.real(trial.calc_energy(prop_data["walkers"],ham_data,wave_data))
#         outlier = jnp.abs(e0 - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt) # 20 Ha for dt = 0.005
#         e0 = jnp.where(outlier, prop_data["e_estimate"], e0)
#         prop_data["weights"] = jnp.where(outlier, 0.0, prop_data["weights"])
        
#         # e0 = jnp.where(
#         #     jnp.abs(e0 - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
#         #     prop_data["e_estimate"], e0
#         #     )

#         wt = prop_data["weights"]

#         blk_wt = jnp.sum(wt)
#         blk_e0 = jnp.sum(e0*wt)/blk_wt

#         prop_data["pop_control_ene_shift"] = 0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_e0

#         return prop_data, (blk_wt, blk_e0)


#     @partial(jit, static_argnums=(0,3,4))
#     def _sr_block_scan(
#         self,
#         prop_data: dict,
#         ham_data: dict,
#         prop: propagator,
#         trial,
#         wave_data: dict,
#     ) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
            
#         def _block_scan_wrapper(x,_):
#             return self._block_scan(x,ham_data,prop,trial,wave_data)
        
#         prop_data, (blk_wt, blk_e0) = lax.scan(
#             _block_scan_wrapper, prop_data, None, length = self.n_ene_blocks
#         )

#         prop_data = prop.stochastic_reconfiguration_local(prop_data)
#         prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

#         return prop_data, (blk_wt, blk_e0)


#     @partial(jit, static_argnums=(0,3,4))
#     def propagate_phaseless(
#         self,
#         prop_data: dict,
#         ham_data: dict,
#         prop: propagator,
#         trial,
#         wave_data: dict,
#     ) -> Tuple[jax.Array, dict]:
#         def _sr_block_scan_wrapper(x,_):
#             return self._sr_block_scan(x, ham_data, prop, trial, wave_data)

#         prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
#         prop_data["n_killed_walkers"] = 0
#         prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        
#         prop_data, (blk_wt, blk_e0) = lax.scan(
#             _sr_block_scan_wrapper, prop_data, None, length = self.n_sr_blocks
#         )
        
#         prop_data["n_killed_walkers"] /= (self.n_sr_blocks * self.n_ene_blocks * prop.n_walkers)

        
#         wt = jnp.sum(blk_wt)
#         e0 = jnp.sum(blk_e0 * blk_wt) / wt

#         return prop_data, (wt, e0)

#     def __hash__(self) -> int:
#         return hash(tuple(self.__dict__.values()))