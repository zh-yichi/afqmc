from dataclasses import dataclass
from typing import Callable, Tuple
from functools import partial
from jax import lax, vmap, jit, random
from jax import numpy as jnp
from afqmc.sampling import sampler

def map_over_walkers(single_fn, walkers, n_batch, *broadcast_args):
    """Lift a single-walker function to all walkers via batched scan + vmap.

    single_fn(walker, *broadcast_args) -> per-walker result (scalar or vector).
    Returns shape (n_walkers, *per_walker_trailing_dims).
    *broadcast_args = wave_data, ham_data ...
    """
    n_walkers = walkers.shape[0]
    batch_size = n_walkers // n_batch
    in_axes = (0,) + (None,) * len(broadcast_args)   # map walker, broadcast the rest

    def scan_walkers(carry, walker_batch):
        out_batch = vmap(single_fn, in_axes=in_axes)(walker_batch, *broadcast_args)
        return carry, out_batch

    _, out = lax.scan(
        scan_walkers, None,
        walkers.reshape(n_batch, batch_size, *walkers.shape[1:]),
    )
    return out.reshape(n_walkers, *out.shape[2:])

@dataclass(frozen=True)
class rwfn:
    guide_overlap_fn: Callable
    trial_overlap_fn: Callable
    force_bias_fn: Callable  # defined by guide
    energy_fn: Callable      # defined by trial
    nelec: Tuple[int, int]
    norb: int
    nchol: int
    nchol_chunk: int
    n_batch: int = 1

    def __post_init__(self):
        assert self.nelec[0] == self.nelec[1], \
            "Restricted Wavefunction requires equal number of up and down electrons."

    def _guide_overlap(self, walker, wave_data):
        return self.guide_overlap_fn(self, walker, wave_data)
    
    def _trial_overlap(self, walker, wave_data):
        return self.trial_overlap_fn(self, walker, wave_data)
    
    def _force_bias(self, walker, ham_data, wave_data):
        return self.force_bias_fn(self, walker, ham_data, wave_data)
    
    def _energy(self, walker, ham_data, wave_data):
        return self.energy_fn(self, walker, ham_data, wave_data)
    
    @partial(jit, static_argnums=0)
    def calc_overlap(self, walkers, wave_data):
        return map_over_walkers(self._guide_overlap, walkers, self.n_batch, wave_data)
    
    @partial(jit, static_argnums=0)
    def calc_trial_overlap(self, walkers, wave_data):
        return map_over_walkers(self._trial_overlap, walkers, self.n_batch, wave_data)

    @partial(jit, static_argnums=0)
    def calc_force_bias(self, walkers, ham_data, wave_data):
        return map_over_walkers(self._force_bias, walkers, self.n_batch, ham_data, wave_data)

    @partial(jit, static_argnums=0)
    def calc_energy(self, walkers, ham_data, wave_data):
        return map_over_walkers(self._energy, walkers, self.n_batch, ham_data, wave_data)


# @dataclass
# class sampler_exp(sampler):

#     @partial(jit, static_argnums=(0,1,2))
#     def block_sample(
#         self,
#         prop,
#         trial,
#         prop_data,
#         ham_data,
#         wave_data,
#         ):
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
#         prop_data = prop.orthonormalize_walkers(prop_data)
#         prop_data["n_killed_walkers"] = prop_data["weights"].size - jnp.count_nonzero(prop_data["weights"])

#         energies = jnp.real(trial.get_energy(prop_data["walkers"], ham_data, wave_data))
#         outlier = jnp.abs(energies - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt) # 20 Ha for dt = 0.005
#         weights = jnp.where(outlier, 0.0, prop_data["weights"])

#         guide_olps = trial.get_guide_overlap(prop_data["walkers"], wave_data)
#         trial_olps = trial.get_trial_overlap(prop_data["walkers"], wave_data)
#         prop_data["overlaps"] = guide_olps

#         olp_ratio = trial_olps / guide_olps
#         weights_p = weights * olp_ratio

#         blk_wt = jnp.sum(weights)
#         blk_wp = jnp.sum(weights_p)
#         blk_et = jnp.sum(weights_p * energies) / blk_wp

#         # prop_data["pop_control_ene_shift"] = 0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_eg
#         prop_data = prop.stochastic_reconfiguration_local(prop_data)
#         prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

#         return prop_data, (blk_wt, blk_wp, blk_et)
    
#     def __hash__(self) -> int:
#         return hash(tuple(self.__dict__.values()))