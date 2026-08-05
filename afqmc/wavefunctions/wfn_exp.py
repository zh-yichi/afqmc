from dataclasses import dataclass
from typing import Callable
from functools import partial
from jax import jit
from .. import walker_tools


@dataclass(frozen=True)
class wfn:
    guide_overlap_fn: Callable       # defined by guide
    guide_force_bias_fn: Callable    # defined by guide
    guide_energy_fn: Callable        # defined by guide
    trial_overlap_fn: Callable       # defined by trial
    trial_energy_fn: Callable        # defined by trial
    energy_formula_fn: Callable      # defined by trial
    intermediate_fn: Callable        # defined by both
    nelec: tuple[int, int]
    norb: int | tuple[int, int]
    nchol: int
    nchol_chunk: int
    nwalker_batch: int = 1
    mix_precision: bool = False

    def __post_init__(self):
        
        object.__setattr__(self, "nelec", tuple(int(x) for x in self.nelec))
        object.__setattr__(self, "norb", 
                           int(self.norb) if not isinstance(self.norb, tuple)
                           else tuple(int(x) for x in self.norb))
        
        assert self.guide_overlap_fn.__module__ == self.guide_force_bias_fn.__module__, (
            f"guide_overlap_fn ({self.guide_overlap_fn.__module__}) and "
            f"force_bias_fn ({self.guide_force_bias_fn.__module__}) must come from the same module"
        )
        assert self.trial_overlap_fn.__module__ == self.trial_energy_fn.__module__, (
            f"trial_overlap_fn ({self.trial_overlap_fn.__module__}) and "
            f"energy_fn ({self.trial_energy_fn.__module__}) must come from the same module"
        )

    @partial(jit, static_argnums=0)
    def _guide_overlap(self, walker, wave_data):
        return self.guide_overlap_fn(self, walker, wave_data)
    
    @partial(jit, static_argnums=0)
    def _guide_force_bias(self, walker, ham_data, wave_data):
        return self.guide_force_bias_fn(self, walker, ham_data, wave_data)

    @partial(jit, static_argnums=0)
    def _trial_overlap(self, walker, wave_data):
        return self.trial_overlap_fn(self, walker, wave_data)
    
    @partial(jit, static_argnums=0)
    def _trial_energy(self, walker, ham_data, wave_data):
        return self.trial_energy_fn(self, walker, ham_data, wave_data)
    
    @partial(jit, static_argnums=0)
    def build_trial_intermediate(self, ham_data, wave_data):
        return self.trial_intermediate_fn(self, ham_data, wave_data)
    
    def calc_sample_energy(self, weights, samples, ham_data):
        return self.energy_formula_fn(weights, samples, ham_data)
    
    @partial(jit, static_argnums=0)
    def calc_overlap(self, walkers, wave_data):
        return walker_tools.map_over_walkers(self._guide_overlap, walkers, self.nwalker_batch, wave_data)

    @partial(jit, static_argnums=0)
    def calc_force_bias(self, walkers, ham_data, wave_data):
        return walker_tools.map_over_walkers(self._guide_force_bias, walkers, self.nwalker_batch, ham_data, wave_data)

    @partial(jit, static_argnums=0)
    def calc_trial_overlap(self, walkers, wave_data):
        return walker_tools.map_over_walkers(self._trial_overlap, walkers, self.nwalker_batch, wave_data)

    @partial(jit, static_argnums=0)
    def calc_energy(self, walkers, ham_data, wave_data):
        return walker_tools.map_over_walkers(self._trial_energy, walkers, self.nwalker_batch, ham_data, wave_data)
