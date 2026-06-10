from dataclasses import dataclass
from typing import Callable, Tuple
from functools import partial
import jax
from jax import lax, vmap, jit

def map_over_walkers(single_fn, walkers, nbatch, *broadcast_args):
    """Lift a single-walker function to all walkers via batched scan + vmap.

    single_fn(walker, *broadcast_args) -> per-walker result (scalar or vector).
    Returns shape (n_walkers, *per_walker_trailing_dims).
    *broadcast_args = wave_data, ham_data ...
    """
    if isinstance(walkers, jax.Array) and len(walkers.shape) == 3:
        nwalker = walkers.shape[0]
        assert nwalker % nbatch == 0, \
            f"nwalker={nwalker} not divisible by nbatch={nbatch}"
        
        batch_size = nwalker // nbatch
        walkers = walkers.reshape(nbatch, batch_size, *walkers.shape[1:])

    elif isinstance(walkers, (tuple, list)) and len(walkers[0].shape) == 3:
        assert len(walkers[0].shape) == len(walkers[1].shape)
        nwalker = walkers[0].shape[0]
        assert nwalker % nbatch == 0, \
            f"nwalker={nwalker} not divisible by nbatch={nbatch}"
        
        batch_size = nwalker // nbatch
        walkers_a = walkers[0].reshape(nbatch, batch_size, *walkers[0].shape[1:])
        walkers_b = walkers[1].reshape(nbatch, batch_size, *walkers[1].shape[1:])
        walkers = (walkers_a, walkers_b)
    
    else:
        raise TypeError("walkers must be a 3D array for spin-restricted "
                        "and a tuple/list of 3D arrays for spin-unrestricted")

    in_axes = (0,) + (None,) * len(broadcast_args)   # map walker, broadcast the rest

    def scan_walkers(carry, walker_batch):
        out_batch = vmap(single_fn, in_axes=in_axes)(walker_batch, *broadcast_args)
        return carry, out_batch

    _, out = lax.scan(scan_walkers, None, walkers)
    return out.reshape(nwalker, *out.shape[2:])


@dataclass(frozen=True)
class wfn:
    guide_overlap_fn: Callable # defined by guide
    force_bias_fn: Callable    # defined by guide
    trial_overlap_fn: Callable # defined by trial
    energy_fn: Callable        # defined by trial
    nelec: tuple[int, int]
    norb: int | tuple[int, int]
    nchol: int
    nchol_chunk: int
    nwalker_batch: int = 1

    def __post_init__(self):
        
        object.__setattr__(self, "nelec", tuple(int(x) for x in self.nelec))
        object.__setattr__(self, "norb",
                        int(self.norb) if not isinstance(self.norb, tuple)
                        else tuple(int(x) for x in self.norb))
        
        assert self.guide_overlap_fn.__module__ == self.force_bias_fn.__module__, (
            f"guide_overlap_fn ({self.guide_overlap_fn.__module__}) and "
            f"force_bias_fn ({self.force_bias_fn.__module__}) must come from the same module"
        )
        assert self.trial_overlap_fn.__module__ == self.energy_fn.__module__, (
            f"trial_overlap_fn ({self.trial_overlap_fn.__module__}) and "
            f"energy_fn ({self.energy_fn.__module__}) must come from the same module"
        )

    def _guide_overlap(self, walker, wave_data):
        return self.guide_overlap_fn(self, walker, wave_data)
    
    def _force_bias(self, walker, ham_data, wave_data):
        return self.force_bias_fn(self, walker, ham_data, wave_data)

    def _trial_overlap(self, walker, wave_data):
        return self.trial_overlap_fn(self, walker, wave_data)
    
    def _energy(self, walker, ham_data, wave_data):
        return self.energy_fn(self, walker, ham_data, wave_data)
    
    @partial(jit, static_argnums=0)
    def calc_overlap(self, walkers, wave_data):
        return map_over_walkers(self._guide_overlap, walkers, self.nwalker_batch, wave_data)

    @partial(jit, static_argnums=0)
    def calc_force_bias(self, walkers, ham_data, wave_data):
        return map_over_walkers(self._force_bias, walkers, self.nwalker_batch, ham_data, wave_data)

    @partial(jit, static_argnums=0)
    def calc_trial_overlap(self, walkers, wave_data):
        return map_over_walkers(self._trial_overlap, walkers, self.nwalker_batch, wave_data)

    @partial(jit, static_argnums=0)
    def calc_energy(self, walkers, ham_data, wave_data):
        return map_over_walkers(self._energy, walkers, self.nwalker_batch, ham_data, wave_data)
