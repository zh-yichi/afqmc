import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, lax, random, vmap

from ad_afqmc import linalg_utils, sr, wavefunctions
from ad_afqmc.wavefunctions import wave_function


@dataclass
class propagator(ABC):
    """Abstract base class for propagator classes.
    Contains methods for propagation, orthogonalization, and reconfiguration.

    Attributes:
        dt: time step
        n_walkers: number of walkers
    """

    dt: float = 0.01
    n_walkers: int = 50
    n_exp_terms: int = 10

    @abstractmethod
    def init_prop_data(
        self,
        trial: wave_function,
        wave_data: Any,
        ham_data: dict,
        init_walkers: Optional[Union[jax.Array, List]] = None,
    ) -> dict:
        """Initialize propagation data. If walkers are not provided they are generated
        using the trial.

        Args:
            trial: trial wave function handler
            wave_data: dictionary containing the wave function data
            ham_data: dictionary containing the Hamiltonian data
            init_walkers: initial walkers

        Returns:
            prop_data: dictionary containing the propagation data
        """
        pass

    @abstractmethod
    def stochastic_reconfiguration_local(self, prop_data: dict) -> dict:
        """Perform stochastic reconfiguration locally on a process. Jax friendly."""
        pass

    @abstractmethod
    def stochastic_reconfiguration_global(self, prop_data: dict, comm: Any) -> dict:
        """Perform stochastic reconfiguration globally across processes using MPI. Not jax friendly."""
        pass

    @abstractmethod
    def orthonormalize_walkers(self, prop_data: dict) -> dict:
        """Orthonormalize walkers."""
        pass

    # defining this separately because calculating vhs for a batch seems to be faster
    @partial(jit, static_argnums=(0,))
    def _apply_trotprop_det(
        self, exp_h1: jax.Array, vhs_i: jax.Array, walker_i: jax.Array
    ) -> jax.Array:
        """Apply the Trotterized propagator to a det."""
        walker_i = exp_h1.dot(walker_i)

        def scanned_fun(carry, x):
            carry = vhs_i.dot(carry)
            return carry, carry

        _, vhs_n_walker = lax.scan(
            scanned_fun, walker_i, jnp.arange(1, self.n_exp_terms)
        )
        walker_i = walker_i + jnp.sum(
            jnp.stack(
                [
                    vhs_n_walker[n] / math.factorial(n + 1)
                    for n in range(self.n_exp_terms - 1)
                ]
            ),
            axis=0,
        )
        walker_i = exp_h1.dot(walker_i)
        return walker_i

    def _apply_trotprop(
        self, ham_data: dict, walkers: Sequence, fields: jax.Array
    ) -> jax.Array:
        """Apply the Trotterized propagator to a batch of walkers."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    @partial(jit, static_argnums=(0, 1))
    def propagate(
        self,
        trial: wave_function,
        ham_data: dict,
        prop_data: dict,
        fields: jax.Array,
        wave_data: dict,
    ) -> dict:
        """Phaseless AFQMC propagation.

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            fields: auxiliary fields
            wave_data: wave function data

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        force_bias = trial.calc_force_bias(prop_data["walkers"], ham_data, wave_data)
        field_shifts = -jnp.sqrt(self.dt) * (1.0j * force_bias - ham_data["mf_shifts"])
        shifted_fields = fields - field_shifts
        shift_term = jnp.sum(shifted_fields * ham_data["mf_shifts"], axis=1)
        fb_term = jnp.sum(
            fields * field_shifts - field_shifts * field_shifts / 2.0, axis=1
        )

        prop_data["walkers"] = self._apply_trotprop(
            ham_data, prop_data["walkers"], shifted_fields
        )

        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)
        # I(x,xbar,walkers,trial) = <trial|walkers_new>/<trial|walkers_old> 
        #                                 * exp(x_g xbar_g - 1/2 xbar_g xbar_g)
        imp_fun = (
            jnp.exp(
                -jnp.sqrt(self.dt) * shift_term
                + fb_term # pop_control_ene_shift = estimated ground state energy
                + self.dt * (prop_data["pop_control_ene_shift"] + ham_data["h0_prop"])
            )
            * overlaps_new
            / prop_data["overlaps"]
        )
        theta = jnp.angle(
            jnp.exp(-jnp.sqrt(self.dt) * shift_term)
            * overlaps_new
            / prop_data["overlaps"]
        )
        imp_fun_phaseless = jnp.abs(imp_fun) * jnp.cos(theta)
        imp_fun_phaseless = jnp.array(
            jnp.where(jnp.isnan(imp_fun_phaseless), 0.0, imp_fun_phaseless)
        )
        imp_fun_phaseless = jnp.where(
            imp_fun_phaseless < 1.0e-3, 0.0, imp_fun_phaseless
        )
        imp_fun_phaseless = jnp.where(imp_fun_phaseless > 100.0, 0.0, imp_fun_phaseless)
        # prop_data["imp_fun"] = imp_fun_phaseless
        prop_data["weights"] = imp_fun_phaseless * prop_data["weights"]
        prop_data["weights"] = jnp.array(
            jnp.where(prop_data["weights"] > 100, 0.0, prop_data["weights"])
        )
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"] - 0.1 * jnp.array(
            jnp.log(jnp.sum(prop_data["weights"]) / self.n_walkers) / self.dt
        )
        prop_data["overlaps"] = overlaps_new
        return prop_data

    def propagate_free(
        self,
        trial: wave_function,
        ham_data: dict,
        prop_data: dict,
        fields: jax.Array,
        wave_data: dict,
    ) -> dict:
        """Free projection AFQMC propagation.

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            fields: auxiliary fields
            wave_data: wave function data

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        raise NotImplementedError(
            "Free projection not implemented for this propagator."
        )

    def _build_propagation_intermediates(
        self, ham_data: dict, trial: wave_function, wave_data: dict
    ) -> dict:
        """Build intermediates for propagation."""
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class propagator_restricted(propagator):
    """Propagator for walkers with the same alpha and beta dets."""

    dt: float = 0.01
    n_walkers: int = 50
    n_exp_terms: int = 6
    n_batch: int = 1

    def init_prop_data(
        self,
        trial: wave_function,
        wave_data: dict,
        ham_data: dict,
        init_walkers: Optional[jax.Array] = None,
    ) -> dict:
        prop_data = {}
        prop_data["weights"] = jnp.ones(self.n_walkers)
        if init_walkers is not None:
            prop_data["walkers"] = init_walkers
        else:
            prop_data["walkers"] = trial.get_init_walkers(
                wave_data, self.n_walkers, restricted=True
            )
        energy_samples = jnp.real(
            trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
        )
        e_estimate = jnp.array(jnp.sum(energy_samples) / self.n_walkers)
        prop_data["e_estimate"] = e_estimate
        prop_data["pop_control_ene_shift"] = e_estimate
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        return prop_data

    @partial(jit, static_argnums=(0,))
    def stochastic_reconfiguration_local(self, prop_data: dict) -> dict:
        prop_data["key"], subkey = random.split(prop_data["key"])
        zeta = random.uniform(subkey)
        prop_data["walkers"], prop_data["weights"] = sr.stochastic_reconfiguration(
            prop_data["walkers"], prop_data["weights"], zeta
        )
        return prop_data

    def stochastic_reconfiguration_global(self, prop_data: dict, comm: Any) -> dict:
        prop_data["key"], subkey = random.split(prop_data["key"])
        zeta = random.uniform(subkey)
        prop_data["walkers"], prop_data["weights"] = sr.stochastic_reconfiguration_mpi(
            prop_data["walkers"], prop_data["weights"], zeta, comm
        )
        return prop_data

    def orthonormalize_walkers(self, prop_data: dict) -> dict:
        prop_data["walkers"], _ = linalg_utils.qr_vmap(prop_data["walkers"])
        return prop_data

    @partial(jit, static_argnums=(0,))
    def _apply_trotprop(
        self, ham_data: dict, walkers: jax.Array, fields: jax.Array
    ) -> jax.Array:
        """Apply the propagator to a batch of walkers."""
        n_walkers = walkers.shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, batch):
            field_batch, walker_batch = batch
            vhs = (
                1.0j
                * jnp.sqrt(self.dt)
                * field_batch.dot(ham_data["chol"]).reshape(
                    batch_size, walkers.shape[1], walkers.shape[1]
                )
            )
            walkers_new = vmap(self._apply_trotprop_det, in_axes=(None, 0, 0))(
                ham_data["exp_h1"],
                vhs,
                walker_batch,
            )
            return carry, walkers_new

        _, walkers_new = lax.scan(
            scanned_fun,
            None,
            (
                fields.reshape(self.n_batch, batch_size, -1),
                walkers.reshape(
                    self.n_batch, batch_size, walkers.shape[1], walkers.shape[2]
                ),
            ),
        )
        walkers = walkers_new.reshape(n_walkers, walkers.shape[1], walkers.shape[2])
        return walkers

    @partial(jit, static_argnums=(0, 2))
    def _build_propagation_intermediates(
        self, ham_data: dict, trial: wave_function, wave_data: dict
    ) -> dict:
        rdm1 = wave_data["rdm1"]
        rdm1 = rdm1[0] + rdm1[1]
        ham_data["mf_shifts"] = 1.0j * vmap(
            lambda x: jnp.sum(x.reshape(trial.norb, trial.norb) * rdm1)
        )(ham_data["chol"])
        ham_data["mf_shifts_fp"] = ham_data["mf_shifts"] / 2.0 / trial.nelec[0]
        ham_data["h0_prop"] = (
            -ham_data["h0"] - jnp.sum(ham_data["mf_shifts"] ** 2) / 2.0
        )
        v0 = 0.5 * jnp.einsum(
            "gik,gjk->ij",
            ham_data["chol"].reshape(-1, trial.norb, trial.norb),
            ham_data["chol"].reshape(-1, trial.norb, trial.norb),
            optimize="optimal",
        )
        h1_mod = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0 - v0
        h1_mod = h1_mod - jnp.real(
            1.0j
            * jnp.einsum(
                "g,gik->ik",
                ham_data["mf_shifts"],
                ham_data["chol"].reshape(-1, trial.norb, trial.norb),
            )
        )
        ham_data["exp_h1"] = jsp.linalg.expm(-self.dt * h1_mod / 2.0)
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class propagator_unrestricted(propagator_restricted):
    def init_prop_data(
        self,
        trial: wave_function,
        wave_data: dict,
        ham_data: dict,
        init_walkers: Optional[Sequence] = None,
    ) -> dict:
        prop_data = {}
        prop_data["weights"] = jnp.ones(self.n_walkers)
        if init_walkers is not None:
            prop_data["walkers"] = init_walkers
        else:
            prop_data["walkers"] = trial.get_init_walkers(
                wave_data, self.n_walkers, restricted=False
            )
        energy_samples = jnp.real(
            trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
        )
        e_estimate = jnp.array(jnp.sum(energy_samples) / self.n_walkers)
        prop_data["e_estimate"] = e_estimate
        prop_data["pop_control_ene_shift"] = e_estimate
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["normed_overlaps"] = prop_data["overlaps"]
        prop_data["norms"] = jnp.ones(self.n_walkers) + 0.0j
        return prop_data

    partial(jit, static_argnums=(0,))
    def _apply_trotprop(self, ham_data, walkers, fields):
        n_walkers = self.n_walkers
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, batch):
            field_batch, walker_batch_0, walker_batch_1 = batch
            # alpha
            vhs_a = (
                1.0j
                * jnp.sqrt(self.dt)
                * field_batch.dot(ham_data["chol"][0]).reshape(
                    batch_size, walkers[0].shape[1], walkers[0].shape[1]
                )
            )
            # beta
            vhs_b = (
                1.0j
                * jnp.sqrt(self.dt)
                * field_batch.dot(ham_data["chol"][1]).reshape(
                    batch_size, walkers[1].shape[1], walkers[1].shape[1]
                )
            )
            # alpha
            walkers_new_0 = vmap(self._apply_trotprop_det, in_axes=(None, 0, 0))(
                ham_data["exp_h1"][0], vhs_a, walker_batch_0
            )
            # beta
            walkers_new_1 = vmap(self._apply_trotprop_det, in_axes=(None, 0, 0))(
                ham_data["exp_h1"][1], vhs_b, walker_batch_1
            )
            return carry, [walkers_new_0, walkers_new_1]

        _, walkers_new = lax.scan(
            scanned_fun,
            None,
            (
                fields.reshape(self.n_batch, batch_size, -1),
                walkers[0].reshape(
                    self.n_batch, batch_size, walkers[0].shape[1], walkers[0].shape[2]
                ),
                walkers[1].reshape(
                    self.n_batch, batch_size, walkers[1].shape[1], walkers[1].shape[2]
                ),
            ),
        )
        walkers = [
            walkers_new[0].reshape(n_walkers, walkers[0].shape[1], walkers[0].shape[2]),
            walkers_new[1].reshape(n_walkers, walkers[1].shape[1], walkers[1].shape[2]),
        ]
        return walkers

    @partial(jit, static_argnums=(0,))
    def stochastic_reconfiguration_local(self, prop_data: dict) -> dict:
        prop_data["key"], subkey = random.split(prop_data["key"])
        zeta = random.uniform(subkey)
        prop_data["walkers"], prop_data["weights"] = sr.stochastic_reconfiguration_uhf(
            prop_data["walkers"], prop_data["weights"], zeta
        )
        return prop_data

    def stochastic_reconfiguration_global(self, prop_data: dict, comm: Any) -> dict:
        prop_data["key"], subkey = random.split(prop_data["key"])
        zeta = random.uniform(subkey)
        (
            prop_data["walkers"],
            prop_data["weights"],
        ) = sr.stochastic_reconfiguration_mpi_uhf(
            prop_data["walkers"], prop_data["weights"], zeta, comm
        )
        return prop_data

    def orthonormalize_walkers(self, prop_data: dict) -> dict:
        prop_data["walkers"], _ = linalg_utils.qr_vmap_uhf(prop_data["walkers"])
        return prop_data

    def _orthogonalize_walkers(self, prop_data: dict) -> Tuple:
        prop_data["walkers"], norms = linalg_utils.qr_vmap_uhf(prop_data["walkers"])
        return prop_data, norms

    @partial(jit, static_argnums=(0))
    def _multiply_constant(self, walkers: List, constants: jax.Array) -> Sequence:
        walkers[0] = constants[0].reshape(-1, 1, 1) * walkers[0]
        walkers[1] = constants[1].reshape(-1, 1, 1) * walkers[1]
        return walkers

    @partial(jit, static_argnums=(0, 1))
    def propagate_free(
        self,
        trial: wave_function,
        ham_data,
        prop_data: dict,
        fields: jax.Array,
        wave_data: Sequence,
    ) -> dict:
        shift_term = jnp.einsum("wg,sg->sw", fields, ham_data["mf_shifts_fp"])
        constants = jnp.einsum(
            "sw,s->sw",
            jnp.exp(-jnp.sqrt(self.dt) * shift_term),
            jnp.exp(self.dt * ham_data["h0_prop_fp"]),
        )
        prop_data["walkers"] = self._apply_trotprop(
            ham_data, prop_data["walkers"], fields
        )
        prop_data["walkers"] = self._multiply_constant(prop_data["walkers"], constants)
        prop_data, norms = self._orthogonalize_walkers(prop_data)
        prop_data["norms"] *= norms[0] * norms[1]
        prop_data["overlaps"] = (
            trial.calc_overlap(prop_data["walkers"], wave_data) * prop_data["norms"]
        )
        normed_walkers, _ = linalg_utils.qr_vmap_uhf(prop_data["walkers"])
        prop_data["normed_overlaps"] = trial.calc_overlap(normed_walkers, wave_data)
        return prop_data

    @partial(jit, static_argnums=(0, 2))
    def _build_propagation_intermediates(self, ham_data, trial, wave_data):
        rdm1 = wave_data["rdm1"]
        mf_shift_a = 1.0j * vmap(
            lambda x: jnp.sum(x.reshape(trial.norb[0], trial.norb[0])
                            * rdm1[0]))(ham_data["chol"][0])
        mf_shift_b = 1.0j * vmap(
            lambda x: jnp.sum(x.reshape(trial.norb[1], trial.norb[1]) 
                            * rdm1[1]))(ham_data["chol"][1])
        ham_data["mf_shifts"] = mf_shift_a + mf_shift_b
        ham_data["h0_prop"] = (
            - ham_data["h0"] - jnp.sum(ham_data["mf_shifts"]**2) / 2.0
                                    )
        # alpha
        v0_a = 0.5 * jnp.einsum(
            "gik,gjk->ij",
            ham_data["chol"][0].reshape(-1, trial.norb[0], trial.norb[0]),
            ham_data["chol"][0].reshape(-1, trial.norb[0], trial.norb[0]),
            optimize="optimal",
        )
        # beta
        v0_b = 0.5 * jnp.einsum(
            "gik,gjk->ij",
            ham_data["chol"][1].reshape(-1, trial.norb[1], trial.norb[1]),
            ham_data["chol"][1].reshape(-1, trial.norb[1], trial.norb[1]),
            optimize="optimal",
        )
        # alpha
        v1_a = jnp.real(
            1.0j
            * jnp.einsum(
                "g,gik->ik",ham_data["mf_shifts"],
                ham_data["chol"][0].reshape(-1, trial.norb[0], trial.norb[0]),
            )
        )
        # beta
        v1_b = jnp.real(
            1.0j
            * jnp.einsum(
                "g,gik->ik",ham_data["mf_shifts"],
                ham_data["chol"][1].reshape(-1, trial.norb[1], trial.norb[1]),
            )
        )
        h1_mod_a = jnp.array(ham_data["h1"][0] - v0_a - v1_a)
        h1_mod_b = jnp.array(ham_data["h1"][1] - v0_b - v1_b)
        ham_data["exp_h1"] = [
            jsp.linalg.expm(-self.dt * h1_mod_a / 2.0),
            jsp.linalg.expm(-self.dt * h1_mod_b / 2.0)
            ]
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))