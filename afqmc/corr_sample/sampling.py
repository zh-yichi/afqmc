from dataclasses import dataclass
from functools import partial
import jax.numpy as jnp
from jax import jit, lax, random

from afqmc import sampling
from afqmc.corr_sample import sr

blocking_analysis = sampling.blocking_analysis

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
        wt_sp1 = jnp.where(outlier, 0.0, prop_data1["weights"])
        wt_sp2 = jnp.where(outlier, 0.0, prop_data2["weights"])

        wt1 = jnp.sum(wt_sp1)
        wt2 = jnp.sum(wt_sp2)
        en1 = jnp.sum(wt_sp1 * en_sp1) / wt1
        en2 = jnp.sum(wt_sp2 * en_sp2) / wt2

        prop_data1, prop_data2 = sr.stochastic_reconfiguration(prop_data1, prop_data2)
        prop_data1["overlaps"] = trial1.calc_overlap(prop_data1["walkers"], wave_data1)
        prop_data2["overlaps"] = trial2.calc_overlap(prop_data2["walkers"], wave_data2)
        prop_data1["e_estimate"] = 0.9 * prop_data1["e_estimate"] + 0.1 * en1
        prop_data2["e_estimate"] = 0.9 * prop_data2["e_estimate"] + 0.1 * en2
        prop_data1["pop_control_ene_shift"] = prop_data1["e_estimate"]
        prop_data2["pop_control_ene_shift"] = prop_data2["e_estimate"]

        return (prop_data1, prop_data2), (wt1, en1, wt2, en2)

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))