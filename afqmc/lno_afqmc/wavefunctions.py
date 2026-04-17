from abc import ABC
from dataclasses import dataclass
from functools import partial, singledispatchmethod
from typing import Sequence, Tuple, Union

import jax
import jax.numpy as jnp
# import jax.scipy as jsp
import numpy as np
from jax import jit, jvp, lax, vmap
import opt_einsum as oe

# from ad_afqmc import linalg_utils


class wave_function(ABC):
    """Base class for wave functions. Contains methods for wave function measurements.

    The measurement methods support two types of walker batches:

    1) unrestricted: walkers is a list ([up, down]). up and down are jax.Arrays of shapes
    (nwalkers, norb, nelec[sigma]). In this case the _calc_<property> method is mapped over.

    2) restricted (up and down dets are assumed to be the same): walkers is a jax.Array of shape
    (nwalkers, max(nelec[0], nelec[1])). In this case the _calc_<property>_restricted method is mapped over. By default
    this method is defined to call _calc_<property>. For certain trial states, one can override
    it for computational efficiency.

    A minimal implementation of a wave function should define the _calc_<property> methods for
    property = overlap, force_bias, energy.

    The wave function data is stored in a separate wave_data dictionary. Its structure depends on the
    wave function type and is described in the corresponding class. It may contain "rdm1" which is a
    one-body spin RDM (2, norb, norb). If it is not provided, wave function specific methods are called.

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        n_batch: Number of batches used in scan.
    """

    norb: Union[int, Tuple[int, int]]
    nelec: Tuple[int, int]
    n_batch: int = 1

    @singledispatchmethod
    def calc_overlap(self, walkers, wave_data: dict) -> jax.Array:
        """Calculate the overlap < psi_t | walker > for a batch of walkers.

        Args:
            walkers : list or jax.Array
                The batched walkers.
            wave_data : dict
                The trial wave function data.

        Returns:
            jax.Array: The overlap of the walkers with the trial wave function.
        """
        raise NotImplementedError("Walker type not supported")

    @calc_overlap.register
    def _(self, walkers: list, wave_data: dict) -> jax.Array:
        n_walkers = walkers[0].shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, walker_batch):
            walker_batch_0, walker_batch_1 = walker_batch
            overlap_batch = vmap(self._calc_overlap, in_axes=(0, 0, None))(
                walker_batch_0, walker_batch_1, wave_data
            )
            return carry, overlap_batch

        _, overlaps = lax.scan(
            scanned_fun,
            None,
            (
                walkers[0].reshape(self.n_batch, batch_size, self.norb[0], self.nelec[0]),
                walkers[1].reshape(self.n_batch, batch_size, self.norb[1], self.nelec[1]),
            ),
        )
        return overlaps.reshape(n_walkers)

    @calc_overlap.register
    def _(self, walkers: jax.Array, wave_data: dict) -> jax.Array:
        n_walkers = walkers.shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, walker_batch):
            overlap_batch = vmap(self._calc_overlap_restricted, in_axes=(0, None))(
                walker_batch, wave_data
            )
            return carry, overlap_batch

        _, overlaps = lax.scan(
            scanned_fun, None, walkers.reshape(self.n_batch, batch_size, self.norb, -1)
        )
        return overlaps.reshape(n_walkers)

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        """Overlap for a single restricted walker."""
        return self._calc_overlap(
            walker[:, : self.nelec[0]], walker[:, : self.nelec[1]], wave_data
        )

    def _calc_overlap(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        """Overlap for a single walker."""
        raise NotImplementedError("Overlap not defined")

    @singledispatchmethod
    def calc_force_bias(self, walkers, ham_data: dict, wave_data: dict) -> jax.Array:
        """Calculate the force bias < psi_T | chol | walker > / < psi_T | walker > for a batch of walkers.

        Args:
            walkers : list or jax.Array
                The batched walkers.
            ham_data : dict
                The hamiltonian data.
            wave_data : dict
                The trial wave function data.

        Returns:
            jax.Array: The force bias.
        """
        raise NotImplementedError("Walker type not supported")

    @calc_force_bias.register
    def _(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
        n_walkers = walkers[0].shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, walker_batch):
            walker_batch_0, walker_batch_1 = walker_batch
            fb_batch = vmap(self._calc_force_bias, in_axes=(0, 0, None, None))(
                walker_batch_0, walker_batch_1, ham_data, wave_data
            )
            return carry, fb_batch

        _, fbs = lax.scan(
            scanned_fun,
            None,
            (
                walkers[0].reshape(self.n_batch, batch_size, self.norb[0], self.nelec[0]),
                walkers[1].reshape(self.n_batch, batch_size, self.norb[1], self.nelec[1]),
            ),
        )
        fbs = jnp.concatenate(fbs, axis=0)
        return fbs.reshape(n_walkers, -1)

    @calc_force_bias.register
    def _(self, walkers: jax.Array, ham_data: dict, wave_data: dict) -> jax.Array:
        n_walkers = walkers.shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, walker_batch):
            fb_batch = vmap(self._calc_force_bias_restricted, in_axes=(0, None, None))(
                walker_batch, ham_data, wave_data
            )
            return carry, fb_batch

        _, fbs = lax.scan(
            scanned_fun, None, walkers.reshape(self.n_batch, batch_size, self.norb, -1)
        )
        return fbs.reshape(n_walkers, -1)

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Force bias for a single restricted walker."""
        return self._calc_force_bias(
            walker[:, : self.nelec[0]], walker[:, : self.nelec[1]], ham_data, wave_data
        )

    def _calc_force_bias(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        """Force bias for a single walker."""
        raise NotImplementedError("Force bias not definedr")

    @singledispatchmethod
    def calc_energy(self, walkers, ham_data: dict, wave_data: dict) -> jax.Array:
        """Calculate the energy < psi_T | H | walker > / < psi_T | walker > for a batch of walkers.

        Args:
            walkers : list or jax.Array
                The batched walkers.
            ham_data : dict
                The hamiltonian data.
            wave_data : dict
                The trial wave function data.

        Returns:
            jax.Array: The energy.
        """
        raise NotImplementedError("Walker type not supported")

    @calc_energy.register
    def _(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
        n_walkers = walkers[0].shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, walker_batch):
            walker_batch_0, walker_batch_1 = walker_batch
            energy_batch = vmap(self._calc_energy, in_axes=(0, 0, None, None))(
                walker_batch_0, walker_batch_1, ham_data, wave_data
            )
            return carry, energy_batch

        _, energies = lax.scan(
            scanned_fun,
            None,
            (
                walkers[0].reshape(self.n_batch, batch_size, self.norb[0], self.nelec[0]),
                walkers[1].reshape(self.n_batch, batch_size, self.norb[1], self.nelec[1]),
            ),
        )
        return energies.reshape(n_walkers)

    @calc_energy.register
    def _(self, walkers: jax.Array, ham_data: dict, wave_data: dict) -> jax.Array:
        n_walkers = walkers.shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, walker_batch):
            energy_batch = vmap(self._calc_energy_restricted, in_axes=(0, None, None))(
                walker_batch, ham_data, wave_data
            )
            return carry, energy_batch

        _, energies = lax.scan(
            scanned_fun,
            None,
            walkers.reshape(self.n_batch, batch_size, self.norb, -1),
        )
        return energies.reshape(n_walkers)

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Energy for a single restricted walker."""
        return self._calc_energy(
            walker[:, : self.nelec[0]], walker[:, : self.nelec[1]], ham_data, wave_data
        )

    def _calc_energy(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        """Energy for a single walker."""
        raise NotImplementedError("Energy not defined")

    def get_rdm1(self, wave_data: dict) -> jax.Array:
        """Returns the one-body spin reduced density matrix of the trial.
        Used for calculating mean-field shift and as a default value in cases of large
        deviations in observable samples. If wave_data contains "rdm1" this value is used,
        calls otherwise _calc_rdm1.

        Args:
            wave_data : The trial wave function data.

        Returns:
            rdm1: The one-body spin reduced density matrix (2, norb, norb).
        """
        if "rdm1" in wave_data:
            return wave_data["rdm1"]
        else:
            return self._calc_rdm1(wave_data)

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        """Calculate the one-body spin reduced density matrix. Exact or approximate rdm1
        of the trial state.

        Args:
            wave_data : The trial wave function data.

        Returns:
            rdm1: The one-body spin reduced density matrix (2, norb, norb).
        """
        raise NotImplementedError(
            "One-body spin RDM not found in wave_data and not implemented for this trial."
        )

    def get_init_walkers(
        self, wave_data: dict, n_walkers: int, restricted: bool = False
    ) -> Union[Sequence, jax.Array]:
        """Get the initial walkers. Uses the rdm1 natural orbitals.

        Args:
            wave_data: The trial wave function data.
            n_walkers: The number of walkers.
            restricted: Whether the walkers should be restricted.

        Returns:
            walkers: The initial walkers.
                If restricted, a single jax.Array of shape (nwalkers, norb, nelec[0]).
                If unrestricted, a list of two jax.Arrays each of shape (nwalkers, norb, nelec[sigma]).
        """
        rdm1 = self.get_rdm1(wave_data)
        natorbs_up = jnp.linalg.eigh(rdm1[0])[1][:, ::-1][:, : self.nelec[0]]
        natorbs_dn = jnp.linalg.eigh(rdm1[1])[1][:, ::-1][:, : self.nelec[1]]
        if restricted:
            if self.nelec[0] == self.nelec[1]:
                det_overlap = np.linalg.det(
                    natorbs_up[:, : self.nelec[0]].T @ natorbs_dn[:, : self.nelec[1]]
                )
                if (
                    np.abs(det_overlap) > 1e-3
                ):  # probably should scale this threshold with number of electrons
                    return jnp.array([natorbs_up + 0.0j] * n_walkers)
                else:
                    overlaps = np.array(
                        [
                            natorbs_up[:, i].T @ natorbs_dn[:, i]
                            for i in range(self.nelec[0])
                        ]
                    )
                    new_vecs = natorbs_up[:, : self.nelec[0]] + np.einsum(
                        "ij,j->ij", natorbs_dn[:, : self.nelec[1]], np.sign(overlaps)
                    )
                    new_vecs = np.linalg.qr(new_vecs)[0]
                    det_overlap = np.linalg.det(
                        new_vecs.T @ natorbs_up[:, : self.nelec[0]]
                    ) * np.linalg.det(new_vecs.T @ natorbs_dn[:, : self.nelec[1]])
                    if np.abs(det_overlap) > 1e-3:
                        return jnp.array([new_vecs + 0.0j] * n_walkers)
                    else:
                        raise ValueError(
                            "Cannot find a set of RHF orbitals with good trial overlap."
                        )
            else:
                # bring the dn orbital projection onto up space to the front
                dn_proj = natorbs_up.T.conj() @ natorbs_dn
                proj_orbs = jnp.linalg.qr(dn_proj, mode="complete")[0]
                orbs = natorbs_up @ proj_orbs
                return jnp.array([orbs + 0.0j] * n_walkers)
        else:
            return [
                jnp.array([natorbs_up + 0.0j] * n_walkers),
                jnp.array([natorbs_dn + 0.0j] * n_walkers),
            ]

    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        """Build intermediates for measurements in ham_data. This method is called by the hamiltonian class.

        Args:
            ham_data: The hamiltonian data.
            wave_data: The trial wave function data.

        Returns:
            ham_data: The updated Hamiltonian data.
        """
        return ham_data



    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))



# we assume afqmc is performed in the rhf orbital basis
@dataclass
class rhf(wave_function):
    """Class for the restricted Hartree-Fock wave function.

    The corresponding wave_data should contain "mo_coeff", a jax.Array of shape (norb, nelec).
    The measurement methods make use of half-rotated integrals which are stored in ham_data.
    ham_data should contain "rot_h1" and "rot_chol" intermediates which are the half-rotated
    one-body and two-body integrals respectively.

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        n_opt_iter: Number of optimization scf iterations.
    """

    norb: int
    nelec: Tuple[int, int]
    n_opt_iter: int = 30
    n_batch: int = 1

    def __post_init__(self):
        assert (
            self.nelec[0] == self.nelec[1]
        ), "RHF requires equal number of up and down electrons."

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        return jnp.linalg.det(wave_data["mo_coeff"].T.conj() @ walker) ** 2

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        return jnp.linalg.det(
            wave_data["mo_coeff"].T.conj() @ walker_up
        ) * jnp.linalg.det(wave_data["mo_coeff"].T.conj() @ walker_dn)

    @partial(jit, static_argnums=0)
    def _calc_green(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        """Calculates the half green's function.

        Args:
            walker: The walker.
            wave_data: The trial wave function data.

        Returns:
            green: The half green's function.
        """
        return (walker.dot(jnp.linalg.inv(wave_data["mo_coeff"].T.conj() @ walker))).T

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: Sequence, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        green_walker = self._calc_green(walker, wave_data)
        fb = 2.0 * oe.contract("gij,ij->g", ham_data["rot_chol"], green_walker, 
                               backend="jax")
        return fb

    @partial(jit, static_argnums=0)
    def _calc_force_bias(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        green_walker_up = self._calc_green(walker_up, wave_data)
        green_walker_dn = self._calc_green(walker_dn, wave_data)
        green_walker = green_walker_up + green_walker_dn
        fb = oe.contract("gij,ij->g", ham_data["rot_chol"], green_walker, 
                         backend="jax")
        return fb

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        green_walker = self._calc_green(walker, wave_data)
        ene1 = 2.0 * jnp.sum(green_walker * rot_h1)
        f = oe.contract("gij,jk->gik", rot_chol, green_walker.T, backend="jax")
        c = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = 2.0 * jnp.sum(c * c) - exc
        return h0 + ene1 + ene2

    @partial(jit, static_argnums=0)
    def _calc_energy(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ene0 = h0
        green_walker_up = self._calc_green(walker_up, wave_data)
        green_walker_dn = self._calc_green(walker_dn, wave_data)
        green_walker = green_walker_up + green_walker_dn
        ene1 = jnp.sum(green_walker * rot_h1)
        f = oe.contract("gij,jk->gik", rot_chol, green_walker.T, backend="jax")
        c = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = jnp.sum(c * c) - exc
        return ene2 + ene1 + ene0
    
    @partial(jit, static_argnums=0)
    def _calc_ecorr(self, walker: jax.Array, ham_data: dict, wave_data: dict):
        '''hf correlation energy'''
        # <HF|H-E0|walker>/<HF|walker>
        rot_h1, rot_chol = ham_data['rot_h1'], ham_data['rot_chol']
        nocc = rot_h1.shape[0]
        green_walker = self._calc_green(walker, wave_data)
        f = oe.contract('gij,jk->gik', rot_chol[:,:nocc,nocc:],
                        green_walker.T[nocc:,:nocc], backend="jax")
        c = vmap(jnp.trace)(f)
        eneo2Jt = oe.contract('g,g->',c,c, backend="jax")*2 
        eneo2ext = oe.contract('gij,gji->',f,f, backend="jax")
        e_corr = eneo2Jt - eneo2ext
        return jnp.real(e_corr)
    
    @partial(jit, static_argnums=0)
    def _calc_eorb(self, walker: jax.Array, ham_data: dict, wave_data: dict):
        '''hf orbital correlation energy'''
        # <HF|H_i|walker>/<HF|walker>
        rot_h1, rot_chol = ham_data['rot_h1'], ham_data['rot_chol']
        m = wave_data["prjlo"]
        nocc = rot_h1.shape[0]
        green_walker = self._calc_green(walker, wave_data)
        f = oe.contract('gij,jk->gik', rot_chol[:,:nocc,nocc:],
                        green_walker.T[nocc:,:nocc], backend="jax")
        c = vmap(jnp.trace)(f)
        eneo2Jt = oe.contract('Gxk,xk,G->',f,m,c, backend="jax")*2 
        eneo2ext = oe.contract('Gxy,Gyk,xk->',f,f,m, backend="jax")
        e_orb = eneo2Jt - eneo2ext
        return jnp.real(e_orb)

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        rdm1 = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T] * 2)
        return rdm1

    @partial(jit, static_argnums=0)
    def _calc_energy_ref(self, walker, ham_data, trial_coeff):
        ''' straight-ahead <HF|H|walker>/<HF|walker>
            without half rotating integrals and gf '''
        h0, h1 = ham_data["h0"], ham_data["h1"][0]
        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        gf = (walker.dot(jnp.linalg.inv(trial_coeff.T @ walker)) @ trial_coeff.T).T
        ene1 = 2.0 * jnp.sum(gf * h1)
        f = oe.contract("gij,jk->gik", chol, gf.T, backend="jax")
        c = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = 2.0 * jnp.sum(c * c) - exc
        return h0 + ene1 + ene2

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        """Builds half rotated integrals for efficient force bias and energy calculations."""
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ (
            (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        )
        ham_data["rot_chol"] = oe.contract(
            "pi,gij->gpj",
            wave_data["mo_coeff"].T.conj(),
            ham_data["chol"].reshape(-1, self.norb, self.norb), 
            backend="jax")
        return ham_data
        
    # @partial(jit, static_argnums=0)
    # def _calc_orbenergy(self, rot_h1, rot_chol, walker, wave_data=None):
    #     ene0 =0
    #     m = wave_data["prjlo"]
    #     nocc = rot_h1.shape[0]
    #     green_walker = self._calc_green(walker, wave_data) # in ao
    #     f = oe.contract('gij,jk->gik', rot_chol[:,:nocc,nocc:], green_walker.T[nocc:,:nocc], 
    #                     backend="jax")
    #     c = vmap(jnp.trace)(f)

    #     eneo2Jt = oe.contract('Gxk,xk,G->',f,m,c, backend="jax")*2 
    #     eneo2ext = oe.contract('Gxy,Gyk,xk->',f,f,m, backend="jax") 
    #     return eneo2Jt - eneo2ext
    
    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class uhf(wave_function):
    """Class for the unrestricted Hartree-Fock wave function.

    The corresponding wave_data contains "mo_coeff", a list of two jax.Arrays of shape (norb, nelec[sigma]).
    The measurement methods make use of half-rotated integrals which are stored in ham_data.
    ham_data should contain "rot_h1" and "rot_chol" intermediates which are the half-rotated
    one-body and two-body integrals respectively.

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        n_opt_iter: Number of optimization scf iterations.
    """

    norb: Tuple[int, int]
    nelec: Tuple[int, int]
    n_opt_iter: int = 30
    n_batch: int = 1

    def _calc_rdm1(self, wave_data: dict):
        dm_up = jnp.array(wave_data["mo_coeff"][0] @ wave_data["mo_coeff"][0].T.conj())
        dm_dn = jnp.array(wave_data["mo_coeff"][1] @ wave_data["mo_coeff"][1].T.conj())
        return [dm_up, dm_dn]

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: dict,
    ) -> complex:
        nocca, noccb = self.nelec
        o0 = jnp.linalg.det(walker_up[: nocca, :]) \
                * jnp.linalg.det(walker_dn[: noccb, :])
        return o0

    @partial(jit, static_argnums=0)
    def _calc_green(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: dict,
    ) -> list:
        """Calculates the half green's function.

        Args:
            walker_up: The walker for spin up.
            walker_dn: The walker for spin down.
            wave_data: The trial wave function data.

        Returns:
            green: The half green's function for spin up and spin down.
        """
        nocca, noccb = self.nelec
        green_up = (walker_up.dot(jnp.linalg.inv(walker_up[:nocca,:nocca]))).T
        green_dn = (walker_dn.dot(jnp.linalg.inv(walker_dn[:noccb,:noccb]))).T
        return [green_up, green_dn]

    @partial(jit, static_argnums=0)
    def _calc_force_bias(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        nocca, noccb = self.nelec
        norba, norbb = self.norb
        rot_chola = ham_data["chol"][0].reshape(-1,norba,norba)[:,:nocca,:]
        rot_cholb = ham_data["chol"][1].reshape(-1,norbb,norbb)[:,:noccb,:]
        greena, greenb = self._calc_green(walker_up, walker_dn, wave_data)
        fb_up = oe.contract("gij,ij->g", rot_chola, greena, backend="jax")
        fb_dn = oe.contract("gij,ij->g", rot_cholb, greenb, backend="jax")
        return fb_up + fb_dn

    # @partial(jit, static_argnums=0)
    # def _calc_energy(
    #     self,
    #     walker_up: jax.Array,
    #     walker_dn: jax.Array,
    #     ham_data: dict,
    #     wave_data: dict,
    # ) -> complex:
    #     h0 = ham_data["h0"]
    #     nocca, noccb = self.nelec
    #     norba, norbb = self.norb
    #     rot_h1a = ham_data['h1'][0][:nocca,:]
    #     rot_h1b = ham_data['h1'][1][:noccb,:]
    #     rot_chola = ham_data["chol"][0].reshape(-1,norba,norba)[:,:nocca,:]
    #     rot_cholb = ham_data["chol"][1].reshape(-1,norbb,norbb)[:,:noccb,:]
    #     greena, greenb = self._calc_green(walker_up, walker_dn, wave_data)
    #     ene1 = jnp.sum(greena * rot_h1a) + jnp.sum(greenb * rot_h1b)
    #     # f_up = oe.contract("gij,jk->gik", rot_chola, greena.T, backend="jax")
    #     # f_dn = oe.contract("gij,jk->gik", rot_cholb, greenb.T, backend="jax")
    #     # c_up = vmap(jnp.trace)(f_up)
    #     # c_dn = vmap(jnp.trace)(f_dn)
    #     # exc_up = jnp.sum(vmap(lambda x: x * x.T)(f_up))
    #     # exc_dn = jnp.sum(vmap(lambda x: x * x.T)(f_dn))
    #     # ene2 = (jnp.sum(c_up * c_up)
    #     #       + jnp.sum(c_dn * c_dn)
    #     #       + 2.0 * jnp.sum(c_up * c_dn)
    #     #       - exc_up - exc_dn) / 2.0
    #     gl_a = oe.contract("ir,gjr->gij", greena, rot_chola, backend="jax")
    #     gl_b = oe.contract("ir,gjr->gij", greenb, rot_cholb, backend="jax")
    #     tr_gl_a = oe.contract("gii->g", gl_a, backend="jax")
    #     tr_gl_b = oe.contract("gii->g", gl_b, backend="jax")
    #     tr_gl_ab = tr_gl_a + tr_gl_b
    #     e_col = oe.contract('g,g->', tr_gl_ab, tr_gl_ab, backend="jax") / 2
    #     e_exc = (oe.contract('gij,gji->',gl_a,gl_a, backend="jax")
    #              + oe.contract('gij,gji->',gl_b,gl_b, backend="jax")) / 2
    #     ene2 = e_col - e_exc

    #     return h0 + ene1 + ene2
    
    @partial(jit, static_argnums=0)
    def _calc_energy(self, walker_up, walker_dn, ham_data, wave_data)-> complex:
        '''
        uhf trial correlation energy 
        <HF|H-E0|walker>/<HF|walker> 
        '''
        nocca, noccb = self.nelec
        norba, norbb = self.norb
        e0 = ham_data['E0']
        rot_chola = ham_data["chol"][0].reshape(-1,norba,norba)[:,:nocca,nocca:]
        rot_cholb = ham_data["chol"][1].reshape(-1,norbb,norbb)[:,:noccb,noccb:]
        greena, greenb = self._calc_green(walker_up, walker_dn, wave_data)
        lga = oe.contract('gia,ka->gik', rot_chola, greena[:nocca,nocca:], backend="jax")
        lgb = oe.contract('gia,ka->gik', rot_cholb, greenb[:noccb,noccb:], backend="jax")
        tr_lga = oe.contract('gii->g',lga, backend="jax")
        tr_lgb = oe.contract('gii->g',lgb, backend="jax")
        tr_lg = tr_lga + tr_lgb
        e_col = oe.contract('g,g->', tr_lg, tr_lg, backend="jax") / 2
        e_exc = (oe.contract('gij,gji->',lga,lga, backend="jax")
                 + oe.contract('gij,gji->',lgb,lgb, backend="jax")) / 2
        ecorr = e_col - e_exc
        # lglg_aa = oe.contract('g,g->',tr_lga,tr_lga, backend="jax") \
        #       - oe.contract('gij,gji->',lga,lga, backend="jax")
        # lglg_ab = oe.contract('g,g->',tr_lga,tr_lgb, backend="jax")*2
        # lglg_bb = oe.contract('g,g->',tr_lgb,tr_lgb, backend="jax") \
        #       - oe.contract('gij,gji->',lgb,lgb, backend="jax")
        # ecorr = 0.5*(lglg_aa + lglg_ab + lglg_bb)
        return e0 + ecorr
    
    @partial(jit, static_argnums=0)
    def _calc_eorb(self, walker_up, walker_dn, ham_data, wave_data)-> complex:
        '''
        uhf trial orbital correlation energy
        <HF|(H-E0)_I|walker>/<HF|walker>
        '''
        nocca, noccb = self.nelec
        norba, norbb = self.norb
        prjloa, prjlob = wave_data["prjlo"]
        rot_chola = ham_data["chol"][0].reshape(-1,norba,norba)[:,:nocca,nocca:]
        rot_cholb = ham_data["chol"][1].reshape(-1,norbb,norbb)[:,:noccb,noccb:]
        greena, greenb = self._calc_green(walker_up, walker_dn, wave_data)
        lga = oe.contract('gia,ak->gik', rot_chola, greena.T[nocca:,:nocca], backend="jax")
        lgb = oe.contract('gia,ak->gik', rot_cholb, greenb.T[noccb:,:noccb], backend="jax")
        tr_lga = oe.contract('gii->g',lga, backend="jax")
        tr_lgb = oe.contract('gii->g',lgb ,backend="jax")
        lga_orb = oe.contract('gik,ik->g',lga, prjloa, backend="jax")
        lgb_orb = oe.contract('gik,ik->g',lgb, prjlob, backend="jax")
        eorb_aa = oe.contract('g,g->',lga_orb, tr_lga, backend="jax") \
            - oe.contract('gij,gjk,ik->',lga, lga, prjloa, backend="jax")
        eorb_ab = oe.contract('g,g->', lga_orb, tr_lgb, backend="jax") 
        eorb_ba = oe.contract('g,g->', lgb_orb, tr_lga, backend="jax")
        eorb_bb = oe.contract('g,g->',lgb_orb, tr_lgb, backend="jax") \
            - oe.contract('gij,gjk,ik->',lgb, lgb, prjlob, backend="jax")
        eorb = 0.5 * (eorb_aa + eorb_ab + eorb_ba + eorb_bb)
        return eorb

    @partial(jit, static_argnums=(0,))
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["h1"] = [(ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0,
                          (ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0]
        # ham_data["h1"] = [wave_data["mo_coeff"][0].T.conj() @ ham_data["h1"][0],
        #                       wave_data["mo_coeff"][1].T.conj() @ ham_data["h1"][1]]
        # ham_data["rot_chol"] = [oe.contract("ip,gqp->giq",
        #                                      wave_data["mo_coeff"][0].T.conj(),
        #                                      ham_data["chol"][0].reshape(-1, self.norb[0], self.norb[0]),
        #                                      backend="jax"),
        #                         oe.contract("ip,gpq->giq",
        #                                      wave_data["mo_coeff"][1].T.conj(),
        #                                      ham_data["chol"][1].reshape(-1, self.norb[1], self.norb[1]),
        #                                      backend="jax")]
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))

@dataclass
class cisd(wave_function):
    """A manual implementation of the CISD wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        rdm1 = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T] * 2)
        return rdm1

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
        GF = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        o1 = oe.contract("ia,ia", ci1, GF[:, nocc:], backend="jax")
        o2 = 2 * oe.contract(
            "iajb, ia, jb", ci2, GF[:, nocc:], GF[:, nocc:], backend="jax"
        ) - oe.contract("iajb, ib, ja", ci2, GF[:, nocc:], GF[:, nocc:], backend="jax")
        return (1.0 + 2 * o1 + o2) * o0

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
        ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]
        lg = oe.contract("gpj,pj->g", rot_chol, green, backend="jax")

        # ref
        fb_0 = 2 * lg

        # single excitations
        ci1g = oe.contract("pt,pt->", ci1, green_occ, backend="jax")
        ci1gp = oe.contract("pt,it->pi", ci1, greenp, backend="jax")
        gci1gp = oe.contract("pj,pi->ij", green, ci1gp, backend="jax")
        fb_1_1 = 4 * ci1g * lg
        fb_1_2 = -2 * oe.contract("gij,ij->g", chol, gci1gp, backend="jax")
        fb_1 = fb_1_1 + fb_1_2

        # double excitations
        ci2g_c = oe.contract("ptqu,pt->qu", ci2, green_occ, backend="jax")
        ci2g_e = oe.contract("ptqu,pu->qt", ci2, green_occ, backend="jax")
        cisd_green_c = (greenp @ ci2g_c.T) @ green
        cisd_green_e = (greenp @ ci2g_e.T) @ green
        cisd_green = -4 * cisd_green_c + 2 * cisd_green_e
        ci2g = 4 * ci2g_c - 2 * ci2g_e
        gci2g = oe.contract("qu,qu->", ci2g, green_occ, backend="jax")
        fb_2_1 = lg * gci2g
        fb_2_2 = oe.contract("gij,ij->g", chol, cisd_green, backend="jax")
        fb_2 = fb_2_1 + fb_2_2

        # overlap
        overlap_1 = 2 * ci1g
        overlap_2 = gci2g / 2.0
        overlap = 1.0 + overlap_1 + overlap_2

        return (fb_0 + fb_1 + fb_2) / overlap
    
    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        # rot_chol = ham_data["rot_chol"]
        rot_chol = chol[:, : self.nelec[0], :]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        hg = oe.contract("pj,pj->", h1[:nocc, :], green, backend="jax")

        # 0 body energy
        h0 = ham_data["h0"]

        # 1 body energy
        # ref
        e1_0 = 2 * hg

        # single excitations
        ci1g = oe.contract("pt,pt->", ci1, green_occ, backend="jax")
        e1_1_1 = 4 * ci1g * hg
        gpci1 = greenp @ ci1.T
        ci1_green = gpci1 @ green
        e1_1_2 = -2 * oe.contract("ij,ij->", h1, ci1_green, backend="jax")
        e1_1 = e1_1_1 + e1_1_2

        # double excitations
        ci2g_c = oe.contract("ptqu,pt->qu", ci2, green_occ, backend="jax")
        ci2g_e = oe.contract("ptqu,pu->qt", ci2, green_occ, backend="jax")
        ci2_green_c = (greenp @ ci2g_c.T) @ green
        ci2_green_e = (greenp @ ci2g_e.T) @ green
        ci2_green = 2 * ci2_green_c - ci2_green_e
        ci2g = 2 * ci2g_c - ci2g_e
        gci2g = oe.contract("qu,qu->", ci2g, green_occ, backend="jax")
        e1_2_1 = 2 * hg * gci2g
        e1_2_2 = -2 * oe.contract("ij,ij->", h1, ci2_green, backend="jax")
        e1_2 = e1_2_1 + e1_2_2

        # two body energy
        # ref
        lg = oe.contract("gpj,pj->g", rot_chol, green, backend="jax")
        # lg1 = jnp.einsum("gpj,pk->gjk", rot_chol, green, optimize="optimal")
        lg1 = oe.contract("gpj,qj->gpq", rot_chol, green, backend="jax")
        e2_0_1 = 2 * lg @ lg
        e2_0_2 = -jnp.sum(vmap(lambda x: x * x.T)(lg1))
        e2_0 = e2_0_1 + e2_0_2

        # single excitations
        e2_1_1 = 2 * e2_0 * ci1g
        lci1g = oe.contract("gij,ij->g", chol, ci1_green, backend="jax")
        e2_1_2 = -2 * (lci1g @ lg)

        ci1g1 = ci1 @ green_occ.T
        # e2_1_3 = jnp.einsum("gpq,gpq->", glgpci1, lg1, optimize="optimal")
        e2_1_3_1 = oe.contract("gpq,gqr,rp->", lg1, lg1, ci1g1, backend="jax")
        lci1g = oe.contract("gip,qi->gpq", ham_data["lci1"], green, backend="jax")
        e2_1_3_2 = -oe.contract("gpq,gqp->", lci1g, lg1, backend="jax")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3)

        # double excitations
        e2_2_1 = e2_0 * gci2g
        lci2g = oe.contract("gij,ij->g", chol, ci2_green, backend="jax")
        e2_2_2_1 = -lci2g @ lg

        def scanned_fun(carry, x):
            chol_i, rot_chol_i = x
            gl_i = oe.contract("pj,ji->pi", green, chol_i, backend="jax")
            lci2_green_i = oe.contract(
                "pi,ji->pj", rot_chol_i, ci2_green, backend="jax"
            )
            carry[0] += 0.5 * oe.contract(
                "pi,pi->", gl_i, lci2_green_i, backend="jax"
            )
            glgp_i = oe.contract("pi,it->pt", gl_i, greenp, backend="jax")
            l2ci2_1 = oe.contract(
                "pt,qu,ptqu->",
                glgp_i,
                glgp_i,
                ci2,
                backend="jax"
            )
            l2ci2_2 = oe.contract(
                "pu,qt,ptqu->",
                glgp_i,
                glgp_i,
                ci2,
                backend="jax"
            )
            carry[1] += 2 * l2ci2_1 - l2ci2_2
            return carry, 0.0

        [e2_2_2_2, e2_2_3], _ = lax.scan(scanned_fun, [0.0, 0.0], (chol, rot_chol))
        e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)

        e2_2 = e2_2_1 + e2_2_2 + e2_2_3

        e1 = e1_1 + e1_2
        e2 = e2_1 + e2_2

        e0 = e1_0 + e2_0
        e12 = e1 + e2

        # overlap
        o12 = 2 * ci1g + gci2g

        # E0 = h0 + e0
        # E1 = e12 - o12*e0
        # E2 = -o12 * (e12 - o12*e0)
        # E3 = o12**2 * (e12 - o12*e0)
        # E4 = -o12**3 * (e12 - o12*e0)
        # E5 = o12**4 * (e12 - o12*e0)
        # E6 = -o12**5 * (e12 - o12*e0)
        # E7 = o12**6 * (e12 - o12*e0)
        # E8 = -o12**7 * (e12 - o12*e0)

        e_cisd = h0 + e0 + (e12-o12*e0)/(1+o12)

        return e_cisd #E0+E1+E2+E3+E4+E5+E6+E7+E8

    @partial(jit, static_argnums=0)
    def _ehf12(self, walker: jax.Array, ham_data: dict, wave_data: dict):
        '''<HF|h1+h2|walker>/<HF|walker>'''
        # <HF|H-E0|walker>/<HF|walker>
        rot_h1, rot_chol = ham_data['rot_h1'], ham_data['rot_chol']
        # nocc = rot_h1.shape[0]
        # green_walker = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        # f = oe.contract('gij,jk->gik', rot_chol[:,:nocc,nocc:],
        #                 green_walker.T[nocc:,:nocc], backend="jax")
        # c = vmap(jnp.trace)(f)
        # eneo2Jt = oe.contract('g,g->',c,c, backend="jax")*2 
        # eneo2ext = oe.contract('gij,gji->',f,f, backend="jax")
        # e_corr = eneo2Jt - eneo2ext
        green_walker = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        ene1 = 2.0 * jnp.sum(green_walker * rot_h1)
        f = oe.contract("gij,jk->gik", rot_chol, green_walker.T, backend="jax")
        c = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = 2.0 * jnp.sum(c * c) - exc
        return ene1 + ene2
    
    @partial(jit, static_argnums=0)
    def _ci_olp(self, walker: jax.Array, wave_data: dict) -> complex:
        '''
        <HF|C1+C2|walker> 
        = (C_ia <HF|i+ a|walker>/<HF|walker> + C_iajb <HF|i+ j+ a b|walker>/<HF|walker>) * <HF|walker>
        = (C_ia G_ia + C_iajb (G_ia G_jb-G_ib G_ja)) * <HF|walker>
        '''
        nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
        gf = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        o0 = jnp.linalg.det(walker[: nocc, :]) ** 2
        o1 = oe.contract("ia,ia->", ci1, gf[:, nocc:], backend="jax")
        o2 = 2 * oe.contract("iajb,ia,jb->", ci2, gf[:, nocc:], gf[:, nocc:], backend="jax") \
            - oe.contract("iajb,ib,ja->", ci2, gf[:, nocc:], gf[:, nocc:], backend="jax")
        olp = (2*o1+o2) * o0
        return olp

    @partial(jit, static_argnums=0)
    def _ci_orb_olp(self, walker: jax.Array, wave_data: dict) -> complex:
        '''
        <HF|C1+C2|walker>_i 
        = (C_ia <HF|i+ a|walker>/<HF|walker> + C_iajb <HF|i+ j+ a b|walker>/<HF|walker>) * <HF|walker>
        = (C_ia G_ia + C_iajb (G_ia G_jb-G_ib G_ja)) * <HF|walker>
        prj onto orbital i
        '''
        m = wave_data["prjlo"]
        nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
        gf = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        o0 = jnp.linalg.det(walker[: nocc, :]) ** 2
        o1 = oe.contract("ia,ka,ik->", ci1, gf[:, nocc:],m, backend="jax")
        o2 = 2 * oe.contract("iajb,ka,jb,ik->", ci2, gf[:, nocc:], gf[:, nocc:],m, backend="jax") \
            - oe.contract("iajb,kb,ja,ik->", ci2, gf[:, nocc:], gf[:, nocc:],m, backend="jax")
        olp = (2*o1+o2) * o0
        return olp

    @partial(jit, static_argnums=0)
    def _ci_orb_olp1(self, x: float, h1_mod: jax.Array, walker: jax.Array,
                    wave_data: dict) -> complex:
        '''
        <HF|(C1+C2)_i exp(x*h1_mod)|walker>
        '''
        walker_1x = walker + x*h1_mod.dot(walker)
        olp = self._ci_orb_olp(walker_1x, wave_data)
        return olp

    @partial(jit, static_argnums=0)
    def _ci_orb_olp2(self, x: float, chol_i: jax.Array, 
                     walker: jax.Array, wave_data: dict) -> complex:
        '''
        <HF|(C1+C2)_i exp(x*h2_mod)|walker>
        '''
        walker_2x = (
                walker
                + x * chol_i.dot(walker)
                + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
            )
        olp = self._ci_orb_olp(walker_2x, wave_data)
        return olp

    @partial(jit, static_argnums=0)
    def _hf_eorb(self, walker: jax.Array, ham_data: dict, wave_data: dict):
        '''hf orbital correlation energy'''
        # <HF|H_i|walker>/<HF|walker>
        rot_h1, rot_chol = ham_data['rot_h1'], ham_data['rot_chol']
        m = wave_data["prjlo"]
        nocc = rot_h1.shape[0]
        green_walker = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        f = oe.contract('gij,jk->gik', rot_chol[:,:nocc,nocc:],
                        green_walker.T[nocc:,:nocc], backend="jax")
        c = vmap(jnp.trace)(f)
        eneo2Jt = oe.contract('Gxk,xk,G->',f,m,c, backend="jax")*2 
        eneo2ext = oe.contract('Gxy,Gyk,xk->',f,f,m, backend="jax")
        hf_orb_en = eneo2Jt - eneo2ext
        return hf_orb_en

    @partial(jit, static_argnums=0)
    def _d2_olp2_i(self, chol_i: jax.Array,walker: jax.Array, wave_data: dict):
        x = 0.0
        f = lambda a: self._ci_orb_olp2(a,chol_i,walker,wave_data)
        _, d2f = jax.jvp(lambda x: jax.jvp(f, [x], [1.0])[1], [x], [1.0])
        return d2f

    @partial(jit, static_argnums=0)
    def _calc_orb_energy(self, walker: jax.Array, ham_data: dict, wave_data: dict):
        '''
        eorb0 = <HF|(H-E0)_i|walker>/<HF|walker>
        ehf12 = <HF|h1+h2|walker>/<HF|walker>
        eorb12 = <HF|(c1+c2)_i H|walker>/<HF|walker>
        corb12 = <HF|(c1+c2)_i|walker>/<HF|walker>
        c12 = <HF|(c1+c2)|walker>/<HF|walker>
        '''

        norb = self.norb
        chol = ham_data["chol"].reshape(-1, norb, norb)
        h1_mod = ham_data['h1_mod']
        # h0 = ham_data["h0"]

        nocc = walker.shape[1]
        o0 = jnp.linalg.det(walker[: nocc, :]) ** 2

        eorb0 = self._hf_eorb(walker, ham_data, wave_data)
        ehf12 = self._ehf12(walker, ham_data, wave_data)

        x = 0.0
        # one body
        f1 = lambda a: self._ci_orb_olp1(a,h1_mod,walker,wave_data)
        olp_orb12, d_overlap = jvp(f1, [x], [1.0])

        # two body
        def scanned_fun(carry, c):
            walker, wave_data = carry
            return carry, self._d2_olp2_i(c,walker,wave_data)

        _, d2_olp2_i = lax.scan(scanned_fun, (walker, wave_data), chol)
        d_2_overlap = jnp.sum(d2_olp2_i)/2

        eorb12 = (d_overlap + d_2_overlap) / o0

        corb12 = olp_orb12 / o0
        c12 = self._ci_olp(walker,wave_data) / o0

        E0 = eorb0
        E1 = (eorb12 - corb12*ehf12)/(1+c12)

        return jnp.real(E0+E1)
    
    @partial(jit, static_argnums=(0)) 
    def calc_orb_energy(self,walkers,ham_data,wave_data):
        eorb = vmap(
            self._calc_orb_energy,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        return eorb
    
    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ (
            (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        )
        ham_data["rot_chol"] = oe.contract(
            "pi,gij->gpj",
            wave_data["mo_coeff"].T.conj(),
            ham_data["chol"].reshape(-1, self.norb, self.norb), 
            backend="jax")
        ham_data["lci1"] = oe.contract(
            "git,pt->gip",
            ham_data["chol"].reshape(-1, self.norb, self.norb)[:, :, self.nelec[0] :],
            wave_data["ci1"],
            backend="jax"
        )
        return ham_data
    
    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class ccsd_pt(rhf):

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _te_orb(self, walker, ham_data, wave_data):
        t1, t2 = wave_data["t1"], wave_data["t2"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:]
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, :nocc, :]
        h1 = ham_data["h1"][0]
        hg = oe.contract("pj,pj->", h1[:nocc, :], green, backend="jax")

        # 0 body energy
        h0 = ham_data["h0"]

        # 1 body energy
        # ref
        e1_0 = 2 * hg

        # single excitations
        t1g = oe.contract("pt,pt->", t1, green_occ, backend="jax")
        e1_1_1 = 4 * t1g * hg
        gpt1 = greenp @ t1.T
        t1_green = gpt1 @ green
        e1_1_2 = -2 * oe.contract("ij,ij->", h1, t1_green, backend="jax")
        e1_1 = e1_1_1 + e1_1_2

        # double excitations
        t2g_c = oe.contract("ptqu,pt->qu", t2, green_occ, backend="jax")
        t2g_e = oe.contract("ptqu,pu->qt", t2, green_occ, backend="jax")
        t2_green_c = (greenp @ t2g_c.T) @ green
        t2_green_e = (greenp @ t2g_e.T) @ green
        t2_green = 2 * t2_green_c - t2_green_e
        t2g = 2 * t2g_c - t2g_e
        gt2g = oe.contract("qu,qu->", t2g, green_occ, backend="jax")
        e1_2_1 = 2 * hg * gt2g
        e1_2_2 = -2 * oe.contract("ij,ij->", h1, t2_green, backend="jax")
        e1_2 = e1_2_1 + e1_2_2

        # two body energy
        # ref
        lg = oe.contract("gpj,pj->g", rot_chol, green, backend="jax")
        # lg1 = jnp.einsum("gpj,pk->gjk", rot_chol, green, optimize="optimal")
        lg1 = oe.contract("gpj,qj->gpq", rot_chol, green, backend="jax")
        e2_0_1 = 2 * lg @ lg
        e2_0_2 = -jnp.sum(vmap(lambda x: x * x.T)(lg1))
        e2_0 = e2_0_1 + e2_0_2

        # single excitations
        e2_1_1 = 2 * e2_0 * t1g
        lt1g = oe.contract("gij,ij->g", chol, t1_green, backend="jax")
        e2_1_2 = -2 * (lt1g @ lg)
        t1g1 = t1 @ green_occ.T
        # e2_1_3 = jnp.einsum("gpq,gpq->", glgpci1, lg1, optimize="optimal")
        e2_1_3_1 = oe.contract("gpq,gqr,rp->", lg1, lg1, t1g1, backend="jax")
        lt1g = oe.contract("gip,qi->gpq", ham_data["lt1"], green, backend="jax")
        e2_1_3_2 = -oe.contract("gpq,gqp->", lt1g, lg1, backend="jax")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3)

        # double excitations
        e2_2_1 = e2_0 * gt2g
        lt2g = oe.contract("gij,ij->g", chol, t2_green, backend="jax")
        e2_2_2_1 = -lt2g @ lg

        def scanned_fun(carry, x):
            chol_i, rot_chol_i = x
            gl_i = oe.contract("pj,ji->pi", green, chol_i, backend="jax")
            lt2_green_i = oe.contract(
                "pi,ji->pj", rot_chol_i, t2_green, backend="jax"
            )
            carry[0] += 0.5 * oe.contract(
                "pi,pi->", gl_i, lt2_green_i, backend="jax"
            )
            glgp_i = oe.contract("pi,it->pt", gl_i, greenp, backend="jax")
            l2t2_1 = oe.contract(
                "pt,qu,ptqu->",
                glgp_i,
                glgp_i,
                t2, backend="jax"
            )
            l2t2_2 = oe.contract(
                "pu,qt,ptqu->",
                glgp_i,
                glgp_i,
                t2, backend="jax"
            )
            carry[1] += 2 * l2t2_1 - l2t2_2
            return carry, 0.0

        [e2_2_2_2, e2_2_3], _ = lax.scan(scanned_fun, [0.0, 0.0], (chol, rot_chol))
        e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)

        e2_2 = e2_2_1 + e2_2_2 + e2_2_3

        e0 = h0 + e1_0 + e2_0 # h0 + <psi|(h1+h2)|phi>/<psi|phi>
        te = e1_1 + e1_2 + e2_1 + e2_2 # <psi|(t1+t2)(h1+h2)|phi>/<psi|phi>

        t = 2 * t1g + gt2g # <psi|(t1+t2)|phi>/<psi|phi>

        return jnp.real(t), jnp.real(te), jnp.real(e0)

    @partial(jit, static_argnums=0)
    def _calc_eorb_pt(self, walker: jax.Array, ham_data: dict, wave_data: dict):
        
        eorb = self._calc_eorb(walker, ham_data, wave_data)
        torb, teorb, e0 = self._te_orb(walker, ham_data, wave_data)

        return eorb, teorb, torb, e0
    
    @partial(jit, static_argnums=(0)) 
    def calc_eorb_pt(self,walkers,ham_data,wave_data):
        eorb, teorb, torb, e0 = vmap(
            self._calc_eorb_pt,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        return eorb, teorb, torb, e0

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        norb = self.norb

        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ (
            (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        )
        ham_data["rot_chol"] = oe.contract(
            "pi,gij->gpj",
            wave_data["mo_coeff"].T.conj(),
            ham_data["chol"].reshape(-1, norb, norb), backend="jax"
        )

        ham_data["lt1"] = oe.contract(
            "git,pt->gip",
            ham_data["chol"].reshape(-1, self.norb, self.norb)[:, :, self.nelec[0] :],
            wave_data["t1"],
            optimize="optimal", backend="jax"
        )
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class ccsd_pt_ad(rhf):

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _t_orb(self, walker: jax.Array, wave_data: dict) -> complex:
        '''
        <HF|t1+t2|walker>_i 
        = (C_ia <HF|i+ a|walker>/<HF|walker> + C_iajb <HF|i+ j+ a b|walker>/<HF|walker>) * <HF|walker>
        = (C_ia G_ia + C_iajb (G_ia G_jb-G_ib G_ja)) * <HF|walker>
        prj onto orbital i
        '''
        nocc = walker.shape[1]
        t1, t2 = wave_data["t1"], wave_data["t2"]
        gf = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        o0 = jnp.linalg.det(walker[: nocc, :]) ** 2
        o1 = oe.contract("ia,ia->", t1, gf[:, nocc:], backend="jax")
        o2 = 2 * oe.contract("iajb,ia,jb->", t2, gf[:, nocc:], gf[:, nocc:], backend="jax") \
            - oe.contract("iajb,ib,ja->", t2, gf[:, nocc:], gf[:, nocc:], backend="jax")
        olp = (2*o1+o2) * o0
        return olp

    @partial(jit, static_argnums=0)
    def _t_orb_exp1(self, x: float, h1_mod: jax.Array, walker: jax.Array,
                    wave_data: dict) -> complex:
        '''
        <HF|(t1+t2)_i exp(x*h1_mod)|walker>
        '''
        walker_1x = walker + x*h1_mod.dot(walker)
        olp = self._t_orb(walker_1x, wave_data)
        return olp

    @partial(jit, static_argnums=0)
    def _t_orb_exp2(self, x: float, chol_i: jax.Array, 
                     walker: jax.Array, wave_data: dict) -> complex:
        '''
        <HF|(t1+t2)_i exp(x*h2_mod)|walker>
        '''
        walker_2x = (
                walker
                + x * chol_i.dot(walker)
                + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
            )
        olp = self._t_orb(walker_2x, wave_data)
        return olp
    

    @partial(jit, static_argnums=0)
    def _d2_exp2_i(self, chol_i: jax.Array,walker: jax.Array, wave_data: dict):
        x = 0.0
        f = lambda a: self._t_orb_exp2(a,chol_i,walker,wave_data)
        _, d2f = jax.jvp(lambda x: jax.jvp(f, [x], [1.0])[1], [x], [1.0])
        return d2f

    @partial(jit, static_argnums=0)
    def _te_orb(self, walker: jax.Array, ham_data: dict, wave_data: dict):
        '''
        <HF|(t1+t2)_i (H-E0)|walker>/<HF|walker>
        '''

        norb = self.norb
        chol = ham_data["chol"].reshape(-1, norb, norb)
        h1_mod = ham_data['h1_mod']
        # h0_E0 = ham_data["h0"]-ham_data["E0"]

        nocc = walker.shape[1]
        o0 = jnp.linalg.det(walker[: nocc, :]) ** 2

        x = 0.0
        # one body
        f1 = lambda a: self._t_orb_exp1(a,h1_mod,walker,wave_data)
        tolp, d_overlap = jvp(f1, [x], [1.0])

        # two body
        def scanned_fun(carry, c):
            walker, wave_data = carry
            return carry, self._d2_exp2_i(c,walker,wave_data)

        _, d2_olp2_i = lax.scan(scanned_fun, (walker, wave_data), chol)
        d_2_overlap = jnp.sum(d2_olp2_i)/2

        # <hf|(t1+t2)_i (h1+h2)|walker>/<hf|walker>
        teorb = (d_overlap + d_2_overlap) / o0
        torb = tolp/o0 # <(t1+t2)_i>

        return jnp.real(teorb), jnp.real(torb)

    @partial(jit, static_argnums=0)
    def _calc_eorb_pt(self, walker: jax.Array, ham_data: dict, wave_data: dict):
        
        eorb = self._calc_eorb(walker, ham_data, wave_data)
        teorb, torb = self._te_orb(walker, ham_data, wave_data)
        ecorr = self._calc_ecorr(walker, ham_data, wave_data)

        return eorb, teorb, torb, ecorr

    @partial(jit, static_argnums=(0)) 
    def calc_eorb_pt(self,walkers,ham_data,wave_data):
        eorb, teorb, torb, ecorr = vmap(
            self._calc_eorb_pt,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        return eorb, teorb, torb, ecorr
    
    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class ccsd_pt2_ad(rhf):

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: Sequence, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        nocc, norb = self.nelec[0], self.norb
        rot_chol = ham_data["chol"].reshape(-1,norb,norb)[:,:nocc,:]
        green_walker = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        fb = 2.0 * oe.contract("gij,ij->g", rot_chol, green_walker, 
                               backend="jax")
        return fb

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ):
        nocc, norb = self.nelec[0], self.norb
        h0 = ham_data["h0"]
        rot_h1 = ham_data["h1"][0][:nocc,:] 
        rot_chol = ham_data["chol"].reshape(-1,norb,norb)[:,:nocc,:]
        green_walker = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        ene1 = 2.0 * jnp.sum(green_walker * rot_h1)
        f = oe.contract("gip,jp->gij", rot_chol, green_walker, backend="jax")
        c = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = 2.0 * jnp.sum(c * c) - exc
        return h0 + ene1 + ene2

    @partial(jit, static_argnums=0)
    def _calc_eorb_bar(self, walker, ham_data, wave_data):
        '''
        calculate the correlation energy of the Hamiltonian
        transformed by exp(T1^dagger):
        ecorr_bar = <psi_0|H_bar|walker_bar>/<psi_0|walker_bar>
        |walker_bar> = exp(T1^dagger) |walker>
        H_bar = exp(T1^dagger) H exp(-T1^dagger)
        |psi_0> is the mean-field solution of H
        '''
        nocc, norb = self.nelec[0], self.norb
        prjlo = wave_data['prjlo']
        e0 = ham_data['e0t1orb'] # <psi_0|H_bar|psi_0>
        rot_fock = ham_data['fock_bar'][:nocc,:]
        rot_chol = ham_data['chol_bar'][:,:nocc,:]

        gf = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        e1 = oe.contract('ia,ia->',gf[:nocc,nocc:],
                        rot_fock[:nocc,nocc:], backend="jax") * 2
        lg = oe.contract('gia,ka->gik', rot_chol[:,:nocc,nocc:],
                        gf[:nocc,nocc:], backend="jax")
        e2 = oe.contract('gik,ik,gjj->', lg, prjlo, lg, backend="jax")*2 \
            - oe.contract('gij,gjk,ik->',lg, lg, prjlo, backend="jax")
        e_corr = e0 + e1 + e2
        return e_corr
    
    @partial(jit, static_argnums=0)
    def _calc_energy_bar(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        '''
        <HF|h1_bar+h2_bar|walker_bar>/<HF|walker_bar>
        '''
        nocc = self.nelec[0]
        rot_h1 = ham_data["h1_bar"][:nocc,:]
        rot_chol = ham_data["chol_bar"][:,:nocc,:]
        green_walker = (walker.dot(jnp.linalg.inv(walker[:walker.shape[1], :]))).T
        ene1 = 2.0 * jnp.sum(green_walker * rot_h1)
        f = oe.contract("gij,jk->gik", rot_chol, green_walker.T, backend="jax")
        c = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = 2.0 * jnp.sum(c * c) - exc
        return ene1 + ene2

    @partial(jit, static_argnums=0)
    def _t2_orb(self, walker: jax.Array, wave_data: dict) -> complex:
        '''
        <HF|t2|walker>_i 
        = t_iajb <HF|i+ j+ a b|walker>/<HF|walker> * <HF|walker>
        = t_iajb (G_ia G_jb-G_ib G_ja) * <HF|walker>
        prj onto orbital i
        '''
        nocc = walker.shape[1]
        t2 = wave_data["t2"]
        gf = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        o0 = jnp.linalg.det(walker[: nocc, :]) ** 2
        # o1 = oe.contract("ia,ia->", t1, gf[:, nocc:], backend="jax")
        o2 = 2 * oe.contract("iajb,ia,jb->", t2, gf[:, nocc:], gf[:, nocc:], backend="jax") \
            - oe.contract("iajb,ib,ja->", t2, gf[:, nocc:], gf[:, nocc:], backend="jax")
        olp = o2 * o0
        return olp

    @partial(jit, static_argnums=0)
    def _t2_orb_exp1(self, x: float, h1_mod: jax.Array, walker: jax.Array,
                    wave_data: dict) -> complex:
        '''
        <HF|t2_i exp(x*h1_mod)|walker>
        '''
        walker_1x = walker + x*h1_mod.dot(walker)
        olp = self._t2_orb(walker_1x, wave_data)
        return olp

    @partial(jit, static_argnums=0)
    def _t2_orb_exp2(self, x: float, chol_i: jax.Array, 
                     walker: jax.Array, wave_data: dict) -> complex:
        '''
        <HF|t2_i exp(x*h2_mod)|walker>
        '''
        walker_2x = (
                walker
                + x * chol_i.dot(walker)
                + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
            )
        olp = self._t2_orb(walker_2x, wave_data)
        return olp
    

    @partial(jit, static_argnums=0)
    def _d2_exp2_i(self, chol_i: jax.Array,walker: jax.Array, wave_data: dict):
        x = 0.0
        f = lambda a: self._t2_orb_exp2(a,chol_i,walker,wave_data)
        _, d2f = jax.jvp(lambda x: jax.jvp(f, [x], [1.0])[1], [x], [1.0])
        return d2f

    @partial(jit, static_argnums=0)
    def _t2e_orb_ad(self, walker: jax.Array, ham_data: dict, wave_data: dict):
        '''
        <HF|t2_i (h1_bar+h2_bar)|walker_bar>/<HF|walker_bar>
        '''
        
        chol = ham_data["chol_bar"]
        h1_mod = ham_data['h1_mod_bar']

        nocc = walker.shape[1]
        o0 = jnp.linalg.det(walker[: nocc, :]) ** 2

        # one body
        f1 = lambda a: self._t2_orb_exp1(a,h1_mod,walker,wave_data)
        t2_olp, d_overlap = jvp(f1, [0.0], [1.0])

        # two body
        def scanned_fun(carry, c):
            walker, wave_data = carry
            return carry, self._d2_exp2_i(c,walker,wave_data)

        _, d2_olp2_i = lax.scan(scanned_fun, (walker, wave_data), chol)
        d_2_overlap = jnp.sum(d2_olp2_i)/2

        # <hf|t2_i (h1+h2)|walker>/<hf|walker>
        t2e_orb = (d_overlap + d_2_overlap) / o0
        t2_orb = t2_olp /o0 # <t2_i>

        return t2e_orb, t2_orb

    @partial(jit, static_argnums=0)
    def _calc_eorb_pt2(self, walker: jax.Array, ham_data: dict, wave_data: dict):
        
        walker_bar = wave_data['exp_t1'] @ walker
        o0 = jnp.linalg.det(walker[:walker.shape[1], :]) ** 2
        o_bar = jnp.linalg.det(walker_bar[:walker_bar.shape[1], :]) ** 2
        t1olp = o_bar/o0 # <exp(T1)HF|walker>/<HF|walker>
        e0 = self._calc_energy_restricted(walker, ham_data, wave_data)
        eorb = self._calc_eorb_bar(walker_bar, ham_data, wave_data)
        t2eorb, t2orb = self._t2e_orb_ad(walker_bar, ham_data, wave_data)
        e12bar = self._calc_energy_bar(walker_bar, ham_data, wave_data)

        return e0, t1olp, eorb, t2eorb, t2orb, e12bar

    @partial(jit, static_argnums=(0)) 
    def calc_eorb_pt2(self,walkers,ham_data,wave_data):
        e0, t1olp, eorb, t2eorb, t2orb, e12bar = vmap(
            self._calc_eorb_pt2,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        return e0, t1olp, eorb, t2eorb, t2orb, e12bar
    
    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        """Builds half rotated integrals for efficient force bias and energy calculations."""
        norb, nocc = self.norb, self.nelec[0]
        chol = ham_data["chol"].reshape(-1, norb, norb)
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        # ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ (
        #     (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        # )
        # ham_data["rot_chol"] = oe.contract(
        #     "pi,gij->gpj",
        #     wave_data["mo_coeff"].T.conj(),
        #     ham_data["chol"].reshape(-1, norb, norb), 
        #     backend="jax")
        
        # exp(T1^dagger) H exp(-T1^dagger)
        h1_bar = wave_data['exp_t1'] @ ham_data['h1'][0] @ wave_data['exp_mt1']
        ham_data['h1_bar'] = h1_bar
        chol_bar = oe.contract(
            'pr,grs,sq->gpq', wave_data['exp_t1'], chol, wave_data['exp_mt1'], backend='jax')
        ham_data["chol_bar"] = chol_bar
        chol_bar = ham_data["chol_bar"].reshape(-1,norb,norb)
        v0_bar = 0.5 * oe.contract("gpr,grq->pq", chol_bar, chol_bar, backend="jax")
        h1e_mod_bar = h1_bar - v0_bar
        ham_data['h1_mod_bar'] = h1e_mod_bar
        # exp(T1^dagger) Fock exp(-T1^dagger)
        jeff = oe.contract('gpq,gjj->pq', chol_bar, chol_bar[:,:nocc,:nocc], backend="jax")
        keff = oe.contract('gpj,gjq->pq', chol_bar[:,:,:nocc],
                        chol_bar[:,:nocc,:], backend="jax")
        fock_bar = h1_bar + 2 * jeff - keff
        ham_data['fock_bar'] = oe.contract(
            'ip,ik->kp', fock_bar[:nocc, :], wave_data['prjlo'], backend="jax")
        
        h1_bar = chol_bar = chol = jeff = keff = fock_bar = h1e_mod_bar = v0_bar = None 
        ham_data['h1_mod'] = None
        
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class ccsd_pt2(rhf):

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: Sequence, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        nocc, norb = self.nelec[0], self.norb
        rot_chol = ham_data["chol"].reshape(-1,norb,norb)[:,:nocc,:]
        green_walker = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        fb = 2.0 * oe.contract("gij,ij->g", rot_chol, green_walker, 
                               backend="jax")
        return fb

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ):
        nocc, norb = self.nelec[0], self.norb
        h0 = ham_data["h0"]
        rot_h1 = ham_data["h1"][0][:nocc,:]
        rot_chol = ham_data["chol"].reshape(-1,norb,norb)[:,:nocc,:]
        green_walker = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        ene1 = 2.0 * jnp.sum(green_walker * rot_h1)
        f = oe.contract("gip,jp->gij", rot_chol, green_walker, backend="jax")
        c = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = 2.0 * jnp.sum(c * c) - exc
        return h0 + ene1 + ene2

    @partial(jit, static_argnums=0)
    def _calc_eorb_bar(self, walker, ham_data, wave_data):
        '''
        calculate the correlation energy of the Hamiltonian
        transformed by exp(T1^dagger):
        ecorr_bar = <psi_0|H_bar|walker_bar>/<psi_0|walker_bar>
        |walker_bar> = exp(T1^dagger) |walker>
        H_bar = exp(T1^dagger) H exp(-T1^dagger)
        |psi_0> is the mean-field solution of H
        '''
        nocc, norb = self.nelec[0], self.norb
        prjlo = wave_data['prjlo']
        e0 = ham_data['e0t1orb'] # <psi_0|H_bar|psi_0>
        rot_fock = ham_data['fock_bar'][:nocc,:]
        rot_chol = ham_data['chol_bar'].reshape(-1,norb,norb)[:,:nocc,:]

        gf = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        e1 = oe.contract('ia,ia->',gf[:nocc,nocc:],
                        rot_fock[:nocc,nocc:], backend="jax") * 2
        lg = oe.contract('gia,ka->gik', rot_chol[:,:nocc,nocc:],
                        gf[:nocc,nocc:], backend="jax")
        e2 = oe.contract('gik,ik,gjj->', lg, prjlo, lg, backend="jax")*2 \
            - oe.contract('gij,gjk,ik->',lg, lg, prjlo, backend="jax")
        e_corr = e0 + e1 + e2

        return e_corr


    @partial(jit, static_argnums=0)
    def _t2eorb_tc(self, walker, ham_data, wave_data):
        nocc, norb = self.nelec[0], self.norb
        t2 = wave_data["t2"]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:]
        greenp = jnp.vstack((green_occ, -jnp.eye(norb - nocc)))

        chol = ham_data["chol_bar"]
        rot_chol = chol[:, :nocc, :]
        h1 = ham_data["h1_bar"]

        # 1 body energy
        hg = oe.contract("pi,pi->", h1[:nocc, :], green, backend="jax")
        e1_0 = 2 * hg

        # double excitations
        # t_iajb =! t_jbia since the i axis is projected onto LNO !!!
        t2g_c_1 = oe.contract("iajb,ia->jb", t2, green_occ, backend="jax")
        t2g_c_2 = oe.contract("iajb,jb->ia", t2, green_occ, backend="jax")
        t2g_e_1 = oe.contract("iajb,ib->ja", t2, green_occ, backend="jax")
        t2g_e_2 = oe.contract("iajb,ja->ib", t2, green_occ, backend="jax")
        t2_green_c_1 = oe.contract("pb,jb,jq->pq", greenp, t2g_c_1, green, backend="jax") # t_iajb G_ia G_jq Gp_pb (-)
        t2_green_c_2 = oe.contract("pa,ia,iq->pq", greenp, t2g_c_2, green, backend="jax") # t_iajb G_jb G_iq Gp_pa (-)
        t2_green_e_1 = oe.contract("pa,ja,jq->pq", greenp, t2g_e_1, green, backend="jax") # t_iajb G_ib G_jq Gp_pa (+)
        t2_green_e_2 = oe.contract("pb,ib,iq->pq", greenp, t2g_e_2, green, backend="jax") # t_iajb G_ja G_iq Gp_pb (+)
        t2g_c = t2g_c_1 + t2g_c_2
        t2g_e = t2g_e_1 + t2g_e_2
        t2_green_c = t2_green_c_1 + t2_green_c_2
        t2_green_e = t2_green_e_1 + t2_green_e_2
        t2_green = t2_green_c - t2_green_e * 0.5
        t2g = t2g_c - t2g_e * 0.5
        gt2g = oe.contract("ia,ia->", t2g, green_occ, backend="jax")
        e1_2_1 = 2 * hg * gt2g
        e1_2_2 = -2 * oe.contract("pq,pq->", h1, t2_green, backend="jax")
        e1_2 = e1_2_1 + e1_2_2

        # two body
        # e2_2_1 = e2_0 * gt2g
        # e2_2_2_1 = -lt2g @ lg_c # t_iajb |G_ia G_jq Gp_pb L_pr| G_qs L_qs
        def scan_chol(carry, x):
            chol_i, rot_chol_i = x
            gl_i = oe.contract("ir,qr->iq", green, chol_i, backend="jax")
            gl_c_i = oe.contract("ii->",gl_i[:,:nocc], backend="jax")
            e2_0_c_i = gl_c_i**2
            e2_0_e_i = oe.contract("ij,ji->",gl_i[:,:nocc], gl_i[:,:nocc], backend="jax")
            carry[0] += 2 * e2_0_c_i - e2_0_e_i
            lt2g_i = oe.contract("pr,pr->", chol_i, t2_green, backend="jax")
            carry[1] += -lt2g_i * gl_c_i
            lt2_green_i = oe.contract("ir,qr->iq", rot_chol_i, t2_green, backend="jax")
            # t_iajb |G_ia G_js Gp_pb| G_qr L_pr L_qs
            carry[2] += 0.5 * oe.contract("iq,iq->", gl_i, lt2_green_i, backend="jax")
            # t_iajb G_ir G_js Gp_pa Gp_qb L_pr L_qs type
            glgp_i = oe.contract("ir,rb->ib", gl_i, greenp, backend="jax")
            l2t2_c = oe.contract("iajb,ia,jb->", t2, glgp_i, glgp_i, backend="jax")
            l2t2_e = oe.contract("iajb,ib,ja->", t2, glgp_i, glgp_i, backend="jax")
            carry[3] += 2*l2t2_c - l2t2_e
            return carry, 0.0

        [e2_0, e2_2_2_1, e2_2_2_2, e2_2_3], _ \
            = lax.scan(scan_chol, [0.0, 0.0, 0.0, 0.0], (chol, rot_chol))
        e2_2_1 = e2_0 * gt2g
        e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3

        e0 = e1_0 + e2_0 # <psi|(h1+h2)|phi>/<psi|phi>
        te = e1_2 + e2_2 # <psi|t2(h1+h2)|phi>/<psi|phi>

        t = gt2g # <psi|t2|phi>/<psi|phi>

        return te, t, e0


    @partial(jit, static_argnums=0)
    def _calc_eorb_pt2(self, walker: jax.Array, ham_data: dict, wave_data: dict):
        
        e0 = self._calc_energy_restricted(walker, ham_data, wave_data)
        walker_bar = wave_data['exp_t1'] @ walker
        o0 = jnp.linalg.det(walker[:walker.shape[1], :]) ** 2
        o_bar = jnp.linalg.det(walker_bar[:walker_bar.shape[1], :]) ** 2
        t1olp = o_bar/o0 # <exp(T1)HF|walker>/<HF|walker>
        eorb = self._calc_eorb_bar(walker_bar, ham_data, wave_data)
        t2eorb, t2orb, e12bar = self._t2eorb_tc(walker_bar, ham_data, wave_data)

        return e0, t1olp, eorb, t2eorb, t2orb, e12bar

    @partial(jit, static_argnums=(0))
    def calc_eorb_pt2(self,walkers,ham_data,wave_data):
        e0, t1olp, eorb, t2eorb, t2orb, e12bar = vmap(
            self._calc_eorb_pt2,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        return e0, t1olp, eorb, t2eorb, t2orb, e12bar
    
    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        """Builds half rotated integrals for efficient force bias and energy calculations."""
        norb, nocc = self.norb, self.nelec[0]
        chol = ham_data["chol"].reshape(-1, norb, norb)
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )

        # exp(T1^dagger) H exp(-T1^dagger)
        h1_bar = wave_data['exp_t1'] @ ham_data['h1'][0] @ wave_data['exp_mt1']
        ham_data["h1_bar"] = h1_bar
        chol_bar = jnp.einsum(
            'pr,grs,sq->gpq', wave_data['exp_t1'], chol, wave_data['exp_mt1'])
        ham_data["chol_bar"] = chol_bar        
        # exp(T1^dagger) Fock exp(-T1^dagger)
        jeff = oe.contract('gpq,gjj->pq', chol_bar, chol_bar[:,:nocc,:nocc], backend="jax")
        keff = oe.contract('gpj,gjq->pq', chol_bar[:,:,:nocc],
                        chol_bar[:,:nocc,:], backend="jax")
        fock_bar = h1_bar + 2 * jeff - keff
        ham_data['fock_bar'] = oe.contract(
            'ip,ik->kp', fock_bar[:nocc, :], wave_data['prjlo'], backend="jax")
        
        h1_bar = chol_bar = chol = jeff = keff = fock_bar = None 
        ham_data['h1_mod'] = None
        
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class ccsd_pt2_fast(ccsd_pt2):

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _t2eorb_tc(self, walker, ham_data, wave_data):
        nocc, norb = self.nelec[0], self.norb
        t2 = wave_data["t2"]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:]
        greenp = jnp.vstack((green_occ, -jnp.eye(norb - nocc)))

        chol = ham_data["chol_bar"]
        rot_chol = chol[:, :nocc, :]
        h1 = ham_data["h1_bar"]

        # 1 body energy
        hg = oe.contract("pi,pi->", h1[:nocc, :], green, backend="jax")
        e1_0 = 2 * hg

        # double excitations
        # t_iajb =! t_jbia since the i axis is projected onto LNO !!!
        t2g_c_1 = oe.contract("iajb,ia->jb", t2, green_occ, backend="jax")
        t2g_c_2 = oe.contract("iajb,jb->ia", t2, green_occ, backend="jax")
        t2g_e_1 = oe.contract("iajb,ib->ja", t2, green_occ, backend="jax")
        t2g_e_2 = oe.contract("iajb,ja->ib", t2, green_occ, backend="jax")
        t2_green_c_1 = oe.contract("pb,jb,jq->pq", greenp, t2g_c_1, green, backend="jax") # t_iajb G_ia G_jq Gp_pb (-)
        t2_green_c_2 = oe.contract("pa,ia,iq->pq", greenp, t2g_c_2, green, backend="jax") # t_iajb G_jb G_iq Gp_pa (-)
        t2_green_e_1 = oe.contract("pa,ja,jq->pq", greenp, t2g_e_1, green, backend="jax") # t_iajb G_ib G_jq Gp_pa (+)
        t2_green_e_2 = oe.contract("pb,ib,iq->pq", greenp, t2g_e_2, green, backend="jax") # t_iajb G_ja G_iq Gp_pb (+)
        t2g_c = t2g_c_1 + t2g_c_2
        t2g_e = t2g_e_1 + t2g_e_2
        t2_green_c = t2_green_c_1 + t2_green_c_2
        t2_green_e = t2_green_e_1 + t2_green_e_2
        t2_green = t2_green_c - t2_green_e * 0.5
        t2g = t2g_c - t2g_e * 0.5
        gt2g = oe.contract("ia,ia->", t2g, green_occ, backend="jax")
        e1_2_1 = 2 * hg * gt2g
        e1_2_2 = -2 * oe.contract("pq,pq->", h1, t2_green, backend="jax")
        e1_2 = e1_2_1 + e1_2_2
        
        gl = oe.contract("ir,gqr->giq", green, chol, backend="jax")
        gl_c = oe.contract("gii->g", gl[:,:,:nocc], backend="jax")
        e2_0_c = oe.contract("g,g->", gl_c, gl_c, backend="jax") * 2
        e2_0_e = -oe.contract("gij,gji->",gl[:,:,:nocc], gl[:,:,:nocc], backend="jax")
        e2_0 = e2_0_c + e2_0_e

        lt2g = oe.contract("gpr,pr->g", chol, t2_green, backend="jax")
        e2_2_2_1 = -oe.contract("g,g->", lt2g, gl_c, backend="jax")

        lt2_green = oe.contract("gir,qr->giq", rot_chol, t2_green, backend="jax")
        # t_iajb |G_ia G_js Gp_pb| G_qr L_pr L_qs
        e2_2_2_2 = 0.5 * oe.contract("giq,giq->", gl, lt2_green, backend="jax")
        # t_iajb G_ir G_js Gp_pa Gp_qb L_pr L_qs type
        glgp = oe.contract("gir,rb->gib", gl, greenp, backend="jax")
        # glgp_t2_c = oe.contract("iajb,gia->gjb", glgp, t2, backend="jax")
        l2t2_c = oe.contract("gia,iajb,gjb->", glgp, t2, glgp, backend="jax")
        l2t2_e = oe.contract("gib,iajb,gja->", glgp, t2, glgp, backend="jax")
        e2_2_3 = 2*l2t2_c - l2t2_e

        e2_2_1 = e2_0 * gt2g
        e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3

        t2orb = gt2g # <psi|t2|phi>/<psi|phi>
        e12orb = e1_0 + e2_0 # <psi|(h1+h2)|phi>/<psi|phi>
        t2eorb = e1_2 + e2_2 # <psi|t2(h1+h2)|phi>/<psi|phi>

        return t2eorb, t2orb, e12orb


    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
    

@dataclass
class uccsd_pt_ad(uhf):

    norb: Tuple[int, int]
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _t_orb(self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict) -> complex:
        '''
        <HF|t1+t2|walker>_i 
        = (C_ia <HF|i+ a|walker>/<HF|walker> + C_iajb <HF|i+ j+ a b|walker>/<HF|walker>) * <HF|walker>
        = (C_ia G_ia + C_iajb (G_ia G_jb-G_ib G_ja)) * <HF|walker>
        prj onto spin-orbit i
        '''

        nocca, noccb = self.nelec
        t1a, t1b = wave_data["t1a"], wave_data["t1b"]
        t2aa, t2ab = wave_data["t2aa"], wave_data["t2ab"]
        t2ba, t2bb = wave_data["t2ba"], wave_data["t2bb"]
        greena, greenb = self._calc_green(walker_up, walker_dn, wave_data)
        greena, greenb = greena[:nocca, nocca:], greenb[:noccb, noccb:]
        o0 = jnp.linalg.det(walker_up[:nocca,:]) * jnp.linalg.det(walker_dn[:noccb,:])
        o1 = oe.contract("ia,ia->", t1a, greena, backend="jax") \
              + oe.contract("ia,ia->", t1b, greenb, backend="jax")
        o2 = (oe.contract("iajb,ia,jb->", t2aa, greena, greena, backend="jax")
              + oe.contract("iajb,ia,jb->", t2ab, greena, greenb, backend="jax")
              + oe.contract("iajb,ia,jb->", t2ba, greenb, greena, backend="jax")
              + oe.contract("iajb,ia,jb->", t2bb, greenb, greenb, backend="jax")) * 0.5
        return (o1 + o2) * o0

    @partial(jit, static_argnums=0)
    def _t_exp1_orb(self, x, h1_mod, walker_up, walker_dn, wave_data):
        '''
        unrestricted t_ia <psi_i^a|exp(x*h1_mod)|walker>/<HF|walker>
        '''
        walker_up_1x = walker_up + x * h1_mod[0].dot(walker_up)
        walker_dn_1x = walker_dn + x * h1_mod[1].dot(walker_dn)
        olp = self._t_orb(walker_up_1x, walker_dn_1x, wave_data)
        return olp

    @partial(jit, static_argnums=0)
    def _t_exp2_orb(self, x, chol_i, walker_up, walker_dn, wave_data) -> complex:
        '''
        t_ia <psi_i^a|exp(x*h2_mod)|walker>/<HF|walker>
        '''
        walker_up_2x = (
            walker_up
            + x * chol_i[0].dot(walker_up)
            + x**2 / 2.0 * chol_i[0].dot(chol_i[0].dot(walker_up))
        )
        walker_dn_2x = (
            walker_dn
            + x * chol_i[1].dot(walker_dn)
            + x**2 / 2.0 * chol_i[1].dot(chol_i[1].dot(walker_dn))
        )
        olp = self._t_orb(walker_up_2x,walker_dn_2x,wave_data)
        return olp
    
    @partial(jit, static_argnums=0)
    def _d2_exp2_orb_i(self,chol_i,walker_up,walker_dn,wave_data):
        x = 0.0
        f = lambda a: self._t_exp2_orb(a,chol_i,walker_up,walker_dn,wave_data)
        _, d2f = jax.jvp(lambda x: jax.jvp(f, [x], [1.0])[1], [x], [1.0])
        return d2f


    @partial(jit, static_argnums=0)
    def _te_orb(self, walker_up, walker_dn, ham_data, wave_data):
        '''
        <HF|(t1+t2) (H-E0)|walker>/<HF|walker>
        '''
        norba, norbb = self.norb
        chola, cholb = ham_data["chol"]
        chola = chola.reshape(-1, norba, norba)
        cholb = cholb.reshape(-1, norbb, norbb)
        chol = [chola, cholb]
        h1_mod = ham_data['h1_mod']
        # h0_E0 = ham_data["h0"]-ham_data["E0"]

        o0 = self._calc_overlap(walker_up,walker_dn,wave_data)

        x = 0.0
        # one body
        f1 = lambda a: self._t_exp1_orb(a,h1_mod,walker_up,walker_dn,wave_data)
        olp_orb12, d_overlap = jvp(f1, [x], [1.0])

        # two body
        def scanned_fun(carry, c):
            walker_up, walker_dn, wave_data = carry
            return carry, self._d2_exp2_orb_i(c,walker_up,walker_dn,wave_data)

        _, d2_olp2_i = lax.scan(scanned_fun, (walker_up,walker_dn,wave_data), chol)
        d_2_overlap = jnp.sum(d2_olp2_i)/2

        # <hf|(t1+t2)_i (h0-E0+h1+h2)|walker>/<hf|walker>
        # et_orb = (h0_E0*olp_orb12 + d_overlap + d_2_overlap) / o0
        et_orb = (d_overlap + d_2_overlap) / o0 # <hf|(t1+t2)_i(h1+h2)|walker>/<hf|walker>
        t_orb = olp_orb12 /o0 # <(t1+t2)_i>

        return jnp.real(et_orb), jnp.real(t_orb)

    @partial(jit, static_argnums=0)
    def _calc_eorb_pt(self,
                      walker_up: jax.Array,
                      walker_dn: jax.Array,
                      ham_data: dict,
                      wave_data: dict):
        
        eorb = self._calc_eorb(walker_up, walker_dn, ham_data, wave_data)
        teorb, torb = self._te_orb(walker_up, walker_dn, ham_data, wave_data)
        # ecorr = self._calc_ecorr(walker_up, walker_dn, ham_data, wave_data)
        e0 = self._calc_energy(walker_up, walker_dn, ham_data, wave_data)

        return eorb, teorb, torb, jnp.real(e0)

    @partial(jit, static_argnums=(0)) 
    def calc_eorb_pt(self,
                     walkers: list,
                     ham_data: dict, 
                     wave_data: dict) -> jax.Array:
        eorb, teorb, torb, e0 = vmap(
            self._calc_eorb_pt,in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data, wave_data)
        return eorb, teorb, torb, e0
    
    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
    

@dataclass
class uccsd_pt(uhf):

    norb: Tuple[int, int]
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=(0)) 
    def _calc_eorb_pt(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict):
        
        norb_a, norb_b = self.norb
        nocc_a, nocc_b = self.nelec
        h0, E0 = ham_data["h0"], ham_data["E0"]
        h1a, h1b = ham_data["h1"]
        t1a, t1b = wave_data["t1a"], wave_data["t1b"]
        t2aa, t2ab = wave_data["t2aa"], wave_data["t2ab"]
        t2ba, t2bb = wave_data["t2ba"], wave_data["t2bb"]
        chol_a, chol_b = ham_data["chol"]
        chol_a = chol_a.reshape(-1, norb_a, norb_a)
        chol_b = chol_b.reshape(-1, norb_b, norb_b)
        rot_chol_a = chol_a[:, :nocc_a, :]
        rot_chol_b = chol_b[:, :nocc_b, :]

        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a, :]))).T # G_ip
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[:nocc_b, :]))).T
        green_occ_a = green_a[:, nocc_a:].copy() # G_ia
        green_occ_b = green_b[:, nocc_b:].copy()
        greenp_a = jnp.vstack((green_occ_a, -jnp.eye(norb_a - nocc_a)))
        greenp_b = jnp.vstack((green_occ_b, -jnp.eye(norb_b - nocc_b)))

        # 1 body energy    
        hg_a = oe.contract("pj,pj->", h1a[:nocc_a, :], green_a, backend="jax")
        hg_b = oe.contract("pj,pj->", h1b[:nocc_b, :], green_b, backend="jax")
        e1_0 = hg_a + hg_b #  <HF|h1|walker>/<HF|walker>

        # single excitations = t_ia (G_ia G_pq - G_iq Gp_pa) h_pq
        t1g_a = oe.contract("ia,ia->", t1a, green_occ_a, backend="jax")
        t1g_b = oe.contract("ia,ia->", t1b, green_occ_b, backend="jax")
        t1g = t1g_a + t1g_b
        e1_1_1 = t1g * e1_0
        t1_green_a = oe.contract("pa,ia,iq->pq", greenp_a, t1a, green_a, backend="jax")
        t1_green_b = oe.contract("pa,ia,iq->pq", greenp_b, t1b, green_b, backend="jax")
        e1_1_2 = -(oe.contract("pq,pq->", t1_green_a, h1a, backend="jax")
                + oe.contract("pq,pq->", t1_green_b, h1b, backend="jax"))
        e1_1 = e1_1_1 + e1_1_2 # <HF|T1 h1|walker>/<HF|walker>

        # double excitations
        t2g_a = oe.contract("iajb,ia->jb", t2aa, green_occ_a, backend="jax") / 4
        t2g_b = oe.contract("iajb,ia->jb", t2bb, green_occ_b, backend="jax") / 4
        t2g_ab_a = oe.contract("iajb,ia->jb", t2ab, green_occ_a, backend="jax") / 2
        t2g_ab_b = oe.contract("iajb,jb->ia", t2ab, green_occ_b, backend="jax") / 2
        t2g_ba_a = oe.contract("iajb,jb->ia", t2ba, green_occ_a, backend="jax") / 2
        t2g_ba_b = oe.contract("iajb,ia->jb", t2ba, green_occ_b, backend="jax") / 2
        gt2g_aa = oe.contract("jb,jb->", t2g_a, green_occ_a, backend="jax")
        gt2g_bb = oe.contract("jb,jb->", t2g_b, green_occ_b, backend="jax")
        gt2g_ab = oe.contract("jb,jb->", t2g_ab_a, green_occ_b, backend="jax")
        gt2g_ba = oe.contract("jb,jb->", t2g_ba_b, green_occ_a, backend="jax")
        gt2g = 2 * (gt2g_aa + gt2g_bb) + (gt2g_ab + gt2g_ba)
        e1_2_1 = gt2g * e1_0
        # t_iajb G_ia G_jq Gp_pb
        t2_green_aaa = oe.contract('pb,jb,jq->pq', greenp_a, t2g_a, green_a, backend="jax")
        t2_green_bbb = oe.contract('pb,jb,jq->pq', greenp_b, t2g_b, green_b, backend="jax")
        t2_green_aba = oe.contract('pa,ia,iq->pq', greenp_a, t2g_ab_b, green_a, backend="jax")
        t2_green_baa = oe.contract('pb,jb,jq->pq', greenp_a, t2g_ba_b, green_a, backend="jax")
        t2_green_bab = oe.contract('pa,ia,iq->pq', greenp_b, t2g_ba_a, green_b, backend="jax")
        t2_green_abb = oe.contract('pb,jb,jq->pq', greenp_b, t2g_ab_a, green_b, backend="jax")
        e1_2_2_a = -oe.contract(
            "pq,pq->", 4*t2_green_aaa + t2_green_aba + t2_green_baa, h1a, backend="jax")
        e1_2_2_b = -oe.contract(
            "pq,pq->", 4*t2_green_bbb + t2_green_bab + t2_green_abb, h1b, backend="jax")
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2 # <HF|T2 h1|walker>/<HF|walker>

        # two body energy
        lg_a = oe.contract("gpj,qj->gpq", rot_chol_a, green_a, backend="jax")
        lg_b = oe.contract("gpj,qj->gpq", rot_chol_b, green_b, backend="jax")
        tr_lg_a = oe.contract("gpp->g", lg_a, backend="jax")
        tr_lg_b = oe.contract("gpp->g", lg_b, backend="jax")
        lg_0 = tr_lg_a + tr_lg_b
        e2_0_1 = oe.contract('g,g->', lg_0, lg_0) / 2.0
        e2_0_2 = - (oe.contract("gpq,gqp->", lg_a, lg_a, backend="jax")
                    + oe.contract("gpq,gqp->", lg_b, lg_b, backend="jax")) / 2.0
        e2_0 = e2_0_1 + e2_0_2 # <HF|h2|walker>/<HF|walker>

        # single excitations
        e2_1_1 = e2_0 * t1g
        lt1g_a = oe.contract("gpq,pq->g", chol_a, t1_green_a, backend="jax")
        lt1g_b = oe.contract("gpq,pq->g", chol_b, t1_green_b, backend="jax")
        e2_1_2 = -((lt1g_a + lt1g_b) @ lg_0)
        t1g1_a = t1a @ green_occ_a.T
        t1g1_b = t1b @ green_occ_b.T
        e2_1_3_1 = oe.contract("gpq,gqr,rp->", lg_a, lg_a, t1g1_a, backend="jax") \
            + oe.contract("gpq,gqr,rp->", lg_b, lg_b, t1g1_b, backend="jax")
        lt1g_a = oe.contract("gip,qi->gpq", ham_data["lt1_a"], green_a, backend="jax")
        lt1g_b = oe.contract("gip,qi->gpq", ham_data["lt1_b"], green_b, backend="jax")
        e2_1_3_2 = -oe.contract("gpq,gqp->", lt1g_a, lg_a, backend="jax") \
            - oe.contract("gpq,gqp->", lt1g_b, lg_b, backend="jax")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + e2_1_2 + e2_1_3 # <HF|T1 h2|walker>/<HF|walker>

        # double excitations
        e2_2_1 = e2_0 * gt2g
        lt2g_a = oe.contract(
            "gpq,pq->g", chol_a, 8*t2_green_aaa + 2*(t2_green_aba + t2_green_baa),
            backend="jax")
        lt2g_b = oe.contract(
            "gpq,pq->g", chol_b, 8*t2_green_bbb + 2*(t2_green_bab + t2_green_abb),
            backend="jax")
        e2_2_2_1 = -((lt2g_a + lt2g_b) @ lg_0) / 2.0

        def scanned_fun(carry, x):
            chol_a_i, rot_chol_a_i, chol_b_i, rot_chol_b_i = x
            gl_a_i = oe.contract("ir,pr->ip", green_a, chol_a_i, backend="jax")
            gl_b_i = oe.contract("ir,pr->ip", green_b, chol_b_i, backend="jax")
            lt2_green_a_i = oe.contract(
                "pi,ji->pj", rot_chol_a_i, 8*t2_green_aaa + 2*(t2_green_aba + t2_green_baa), 
                backend="jax")
            lt2_green_b_i = oe.contract(
                "pi,ji->pj", rot_chol_b_i, 8*t2_green_bbb + 2*(t2_green_bab + t2_green_abb),
                backend="jax")
            carry[0] += (oe.contract("ip,ip->", gl_a_i, lt2_green_a_i, backend="jax")
                        + oe.contract("ip,ip->", gl_b_i, lt2_green_b_i, backend="jax")) / 2
            glgp_a_i = oe.contract("ip,pa->ia", gl_a_i, greenp_a, backend="jax")
            glgp_b_i = oe.contract("ip,pa->ia", gl_b_i, greenp_b, backend="jax")
            l2t2_aa = 0.5 * oe.contract(
                "ia,jb,iajb->", glgp_a_i, glgp_a_i, t2aa, backend="jax")
            l2t2_ab = 0.5 * oe.contract(
                "ia,jb,iajb->", glgp_a_i, glgp_b_i, t2ab, backend="jax")
            l2t2_ba = 0.5 * oe.contract(
                "ia,jb,iajb->", glgp_b_i, glgp_a_i, t2ba, backend="jax")
            l2t2_bb = 0.5 * oe.contract(
                "ia,jb,iajb->", glgp_b_i, glgp_b_i, t2bb, backend="jax")
            carry[1] += l2t2_aa + l2t2_ab + l2t2_ba + l2t2_bb
            return carry, 0.0

        [e2_2_2_2, e2_2_3], _ = lax.scan(
            scanned_fun, [0.0, 0.0], (chol_a, rot_chol_a, chol_b, rot_chol_b)
        )
        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <HF|T2 h2|walker>/<HF|walker>

        torb = t1g + gt2g # <HF|T1+T2|walker>/<HF|walker>
        e0 = h0 + e1_0 + e2_0 # <HF|h0+h1+h2|walker>/<HF|walker> - E0
        teorb = e1_1 + e1_2 + e2_1 + e2_2 # <HF|(T1+T2)(h1+h2)|walker>/<HF|walker>
        eorb = self._calc_eorb(walker_up, walker_dn, ham_data, wave_data)

        return eorb, jnp.real(teorb), jnp.real(torb), jnp.real(e0)

    @partial(jit, static_argnums=(0)) 
    def calc_eorb_pt(self,
                     walkers: list,
                     ham_data: dict, 
                     wave_data: dict) -> jax.Array:
        eorb, teorb, torb, e0 = vmap(
            self._calc_eorb_pt,in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data, wave_data)
        return eorb, teorb, torb, e0
    
    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        norba, norbb = self.norb
        nocca, noccb = self.nelec
        ham_data["h1"] = [(ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0,
                          (ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0]
        ham_data["rot_h1"] = [wave_data["mo_coeff"][0].T.conj() @ ham_data["h1"][0],
                              wave_data["mo_coeff"][1].T.conj() @ ham_data["h1"][1]]
        ham_data["rot_chol"] = [oe.contract("ip,gqp->giq",
                                             wave_data["mo_coeff"][0].T.conj(),
                                             ham_data["chol"][0].reshape(-1, norba, norba),
                                             backend="jax"),
                                oe.contract("ip,gpq->giq",
                                             wave_data["mo_coeff"][1].T.conj(),
                                             ham_data["chol"][1].reshape(-1, norbb, norbb),
                                             backend="jax")]
        ham_data["lt1_a"] = oe.contract(
            "gpa,ia->gpi",
            ham_data["chol"][0].reshape(-1, norba, norba)[:, :, nocca:],
            wave_data["t1a"],backend="jax")
        ham_data["lt1_b"] = oe.contract(
            "gpa,ia->gpi",
            ham_data["chol"][1].reshape(-1, norbb, norbb)[:, :, noccb:],
            wave_data["t1b"],backend="jax")
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
    

@dataclass
class uccsd_pt2_ad(uhf):

    norb: Tuple[int, int]
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_energy_bar(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
        nocca, noccb = self.nelec
        norba, norbb = self.norb
        rot_h1a = ham_data['h1bar'][0][:nocca,:]
        rot_h1b = ham_data['h1bar'][1][:noccb,:]
        rot_chola = ham_data["chol_bar"][0].reshape(-1,norba,norba)[:,:nocca,:]
        rot_cholb = ham_data["chol_bar"][1].reshape(-1,norbb,norbb)[:,:noccb,:]
        greena, greenb = self._calc_green(walker_up, walker_dn, wave_data)
        ene1 = jnp.sum(greena * rot_h1a) + jnp.sum(greenb * rot_h1b)
        f_up = oe.contract("gij,jk->gik", rot_chola, greena.T, backend="jax")
        f_dn = oe.contract("gij,jk->gik", rot_cholb, greenb.T, backend="jax")
        c_up = vmap(jnp.trace)(f_up)
        c_dn = vmap(jnp.trace)(f_dn)
        exc_up = jnp.sum(vmap(lambda x: x * x.T)(f_up))
        exc_dn = jnp.sum(vmap(lambda x: x * x.T)(f_dn))
        ene2 = (jnp.sum(c_up * c_up)
              + jnp.sum(c_dn * c_dn)
              + 2.0 * jnp.sum(c_up * c_dn)
              - exc_up - exc_dn) / 2.0

        return ene1 + ene2

    @partial(jit, static_argnums=0)
    def _calc_eorb_bar(self, walker_up, walker_dn, ham_data, wave_data):
        '''
        calculate the correlation energy of the Hamiltonian
        transformed by exp(T1^dagger):
        ecorr_bar = <psi_0|H_bar|walker_bar>/<psi_0|walker_bar>
        |walker_bar> = exp(T1^dagger) |walker>
        H_bar = exp(T1^dagger) H exp(-T1^dagger)
        |psi_0> is the mean-field solution of H
        '''
        nocca, noccb = self.nelec 
        norba, norbb = self.norb
        prjloa, prjlob = wave_data['prjlo']
        e0 = ham_data['e0t1orb'] # <psi_0|H_bar|psi_0>
        rot_focka = ham_data['fock_bar'][0][:nocca,:]
        rot_fockb = ham_data['fock_bar'][1][:noccb,:]
        rot_chola = ham_data['chol_bar'][0].reshape(-1,norba,norba)[:,:nocca,:]
        rot_cholb = ham_data['chol_bar'][1].reshape(-1,norbb,norbb)[:,:noccb,:]

        gfa = (walker_up.dot(jnp.linalg.inv(walker_up[:nocca, :]))).T
        gfb = (walker_dn.dot(jnp.linalg.inv(walker_dn[:noccb, :]))).T
        e1a = oe.contract('ia,ia->',gfa[:nocca,nocca:],rot_focka[:nocca,nocca:], backend="jax")
        e1b = oe.contract('ia,ia->',gfb[:noccb,noccb:],rot_fockb[:noccb,noccb:], backend="jax")
        e1 = e1a + e1b
        
        lga = oe.contract('gia,ka->gik', rot_chola[:,:nocca,nocca:], gfa[:nocca,nocca:], backend="jax")
        lgb = oe.contract('gia,ka->gik', rot_cholb[:,:noccb,noccb:], gfb[:noccb,noccb:], backend="jax")
        e2aa = oe.contract('gik,ik,gjj->', lga, prjloa, lga, backend="jax") \
            - oe.contract('gij,gjk,ik->',lga, lga, prjloa, backend="jax")
        e2ab = oe.contract('gik,ik,gjj->', lga, prjloa, lgb, backend="jax")
        e2ba = oe.contract('gik,ik,gjj->', lgb, prjlob, lga, backend="jax")
        e2bb = oe.contract('gik,ik,gjj->', lgb, prjlob, lgb, backend="jax") \
            - oe.contract('gij,gjk,ik->',lgb, lgb, prjlob, backend="jax")
        e2 = 0.5 * (e2aa + e2ab + e2ba + e2bb)
        
        e_corr = e0 + e1 + e2
        return e_corr

    @partial(jit, static_argnums=0)
    def _t2_orb(self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict) -> complex:
        '''
        <HF|t2|walker>_i 
        = t_iajb <HF|i+ j+ a b|walker>/<HF|walker> * <HF|walker>
        = t_iajb (G_ia G_jb-G_ib G_ja) * <HF|walker>
        prj onto spin-orbit i
        '''

        nocca, noccb = self.nelec
        t2aa, t2ab = wave_data["t2aa"], wave_data["t2ab"]
        t2ba, t2bb = wave_data["t2ba"], wave_data["t2bb"]
        gf_ta = walker_up.dot(jnp.linalg.inv(walker_up[:nocca, :]))[nocca:,:nocca]
        gf_tb = walker_dn.dot(jnp.linalg.inv(walker_dn[:noccb, :]))[noccb:,:noccb]
        o0 = jnp.linalg.det(walker_up[:nocca,:]) * jnp.linalg.det(walker_dn[:noccb,:])
        o2 = (oe.contract("ai,iajb,bj->", gf_ta, t2aa, gf_ta, backend="jax")
              + oe.contract("ai,iajb,bj->", gf_ta, t2ab, gf_tb, backend="jax")
              + oe.contract("ai,iajb,bj->", gf_tb, t2ba, gf_ta, backend="jax")
              + oe.contract("ai,iajb,bj->", gf_tb, t2bb, gf_tb, backend="jax")) * 0.5
        return o2 * o0

    @partial(jit, static_argnums=0)
    def _t2_exp1_orb(self, x, h1_mod, walker_up, walker_dn, wave_data):
        '''
        one-body term
        unrestricted t_ia <psi_i^a|exp(x*h1_mod)|walker>/<HF|walker>
        '''
        walker_up_1x = walker_up + x * h1_mod[0].dot(walker_up)
        walker_dn_1x = walker_dn + x * h1_mod[1].dot(walker_dn)
        olp = self._t2_orb(walker_up_1x, walker_dn_1x, wave_data)
        return olp

    @partial(jit, static_argnums=0)
    def _t2_exp2_orb(self, x, chol_i, walker_up, walker_dn, wave_data) -> complex:
        '''
        two-body term
        t_ia <psi_i^a|exp(x*h2_mod)|walker>/<HF|walker>
        '''
        walker_up_2x = (
            walker_up
            + x * chol_i[0].dot(walker_up)
            + x**2 / 2.0 * chol_i[0].dot(chol_i[0].dot(walker_up))
        )
        walker_dn_2x = (
            walker_dn
            + x * chol_i[1].dot(walker_dn)
            + x**2 / 2.0 * chol_i[1].dot(chol_i[1].dot(walker_dn))
        )
        olp = self._t2_orb(walker_up_2x,walker_dn_2x,wave_data)
        return olp
    
    @partial(jit, static_argnums=0)
    def _d2_exp2_orb_i(self,chol_i,walker_up,walker_dn,wave_data):
        x = 0.0
        f = lambda a: self._t2_exp2_orb(a,chol_i,walker_up,walker_dn,wave_data)
        _, d2f = jax.jvp(lambda x: jax.jvp(f, [x], [1.0])[1], [x], [1.0])
        return d2f


    @partial(jit, static_argnums=0)
    def _t2e_orb_ad(self, walker_up, walker_dn, ham_data, wave_data):
        '''
        <HF|t2_i (h1mod+h2mod)|walker>/<HF|walker>
        note h1mod_pq = h1_pq - 1/2 v_prrq
        '''
        nocca, noccb = self.nelec
        norba, norbb = self.norb
        h1_mod = ham_data['h1_mod_bar']
        chola, cholb = ham_data["chol_bar"]
        chola = chola.reshape(-1, norba, norba)
        cholb = cholb.reshape(-1, norbb, norbb)
        chol = [chola, cholb]

        o0 = jnp.linalg.det(walker_up[:nocca,:]) * jnp.linalg.det(walker_dn[:noccb,:])

        # one body
        f1 = lambda a: self._t2_exp1_orb(a,h1_mod,walker_up,walker_dn,wave_data)
        t2olp, d_overlap = jvp(f1, [0.0], [1.0])

        # two body
        def scanned_fun(carry, c):
            walker_up, walker_dn, wave_data = carry
            return carry, self._d2_exp2_orb_i(c,walker_up,walker_dn,wave_data)

        _, d2_olp2_i = lax.scan(scanned_fun, (walker_up,walker_dn,wave_data), chol)
        d2_overlap = jnp.sum(d2_olp2_i)/2

        e1mod = d_overlap / o0
        e2mod = d2_overlap / o0
        t2eorb = e1mod + e2mod # <hf|t2_i(h1+h2)|walker>/<hf|walker>
        t2orb = t2olp / o0 # <t2_i>

        return t2eorb, t2orb

    @partial(jit, static_argnums=0)
    def _calc_eorb_pt2(self,
                      walker_up: jax.Array,
                      walker_dn: jax.Array,
                      ham_data: dict,
                      wave_data: dict):

        o0 = jnp.linalg.det(walker_up[:walker_up.shape[1],:]) \
            * jnp.linalg.det(walker_dn[:walker_dn.shape[1],:])
        e0 = self._calc_energy(walker_up, walker_dn, ham_data, wave_data)

        walker_up_bar = wave_data['exp_t1a'] @ walker_up
        walker_dn_bar = wave_data['exp_t1b'] @ walker_dn
        
        obar = jnp.linalg.det(walker_up_bar[:walker_up_bar.shape[1], :]) \
            * jnp.linalg.det(walker_dn_bar[:walker_dn_bar.shape[1], :])
        t1olp = obar/o0 # <exp(T1)HF|walker>/<HF|walker>
        
        eorb_bar = self._calc_eorb_bar(walker_up_bar, walker_dn_bar, ham_data, wave_data)
        t2eorb, t2orb = self._t2e_orb_ad(walker_up_bar, walker_dn_bar, ham_data, wave_data)
        e12bar = self._calc_energy_bar(walker_up_bar, walker_dn_bar, ham_data, wave_data)

        return e0, t1olp, eorb_bar, t2eorb, t2orb, e12bar

    @partial(jit, static_argnums=(0)) 
    def calc_eorb_pt2(self,
                     walkers: list,
                     ham_data: dict, 
                     wave_data: dict) -> jax.Array:
        e0, t1olp, eorb_bar, t2eorb, t2orb, e12bar = vmap(
            self._calc_eorb_pt2,in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data, wave_data)
        return e0, t1olp, eorb_bar, t2eorb, t2orb, e12bar

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        """Builds half rotated integrals for efficient force bias and energy calculations."""
        norba, norbb = self.norb
        nocca, noccb = self.nelec
        prjloa, prjlob = wave_data['prjlo']
        chola = ham_data["chol"][0].reshape(-1, norba, norba)
        cholb = ham_data["chol"][1].reshape(-1, norbb, norbb)
        # exp(T1^dagger) H exp(-T1^dagger)
        h1bar_a = wave_data['exp_t1a'] @ ham_data['h1'][0] @ wave_data['exp_mt1a']
        h1bar_b = wave_data['exp_t1b'] @ ham_data['h1'][1] @ wave_data['exp_mt1b']
        ham_data["h1bar"] = [h1bar_a, h1bar_b]
        chol_bar_a = oe.contract('pr,grs,sq->gpq', wave_data['exp_t1a'], chola, wave_data['exp_mt1a'], backend='jax')
        chol_bar_b = oe.contract('pr,grs,sq->gpq', wave_data['exp_t1b'], cholb, wave_data['exp_mt1b'], backend='jax')
        ham_data["chol_bar"] = [chol_bar_a, chol_bar_b]        
        # exp(T1^dagger) Fock exp(-T1^dagger)
        v0bar_a = 0.5 * jnp.einsum("gpr,grq->pq", chol_bar_a, chol_bar_a, optimize="optimal")
        v0bar_b = 0.5 * jnp.einsum("gpr,grq->pq", chol_bar_b, chol_bar_b, optimize="optimal")
        h1mod_bar_a = h1bar_a - v0bar_a
        h1mod_bar_b = h1bar_b - v0bar_b
        ham_data['h1_mod_bar'] = [h1mod_bar_a,h1mod_bar_b]
        la = oe.contract('gjj->g', chol_bar_a[:,:nocca,:nocca], backend="jax")
        lb = oe.contract('gjj->g', chol_bar_b[:,:noccb,:noccb], backend="jax")
        jeff_a = oe.contract('gpq,g->pq', chol_bar_a, la+lb, backend="jax")
        jeff_b = oe.contract('gpq,g->pq', chol_bar_b, la+lb, backend="jax")
        keff_a = oe.contract('gpj,gjq->pq', chol_bar_a[:,:,:nocca], chol_bar_a[:,:nocca,:], backend="jax")
        keff_b = oe.contract('gpj,gjq->pq', chol_bar_b[:,:,:noccb], chol_bar_b[:,:noccb,:], backend="jax")
        fock_bar_a = h1bar_a + jeff_a - keff_a
        fock_bar_b = h1bar_b + jeff_b - keff_b
        fock_bar_a = oe.contract('ip,ik->kp', fock_bar_a[:nocca, :], prjloa, backend="jax")
        fock_bar_b = oe.contract('ip,ik->kp', fock_bar_b[:noccb, :], prjlob, backend="jax")
        ham_data['fock_bar'] = [fock_bar_a, fock_bar_b]
        
        h1bar_a = chol_bar_a = la = jeff_a = keff_a = fock_bar_a = h1mod_bar_a = v0bar_a = None
        h1bar_b = chol_bar_b = lb = jeff_b = keff_b = fock_bar_b = h1mod_bar_a = v0bar_b = None  
        ham_data['h1_mod'] = None
        
        return ham_data
    
    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
    

@dataclass
class uccsd_pt2(uhf):

    norb: Tuple[int, int]
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_eorb_bar(self, walker_up, walker_dn, ham_data, wave_data):
        '''
        calculate the correlation energy of the Hamiltonian
        transformed by exp(T1^dagger):
        ecorr_bar = <psi_0|H_bar|walker_bar>/<psi_0|walker_bar>
        |walker_bar> = exp(T1^dagger) |walker>
        H_bar = exp(T1^dagger) H exp(-T1^dagger)
        |psi_0> is the mean-field solution of H
        '''
        nocca, noccb = self.nelec 
        norba, norbb = self.norb
        prjloa, prjlob = wave_data['prjlo']
        e0 = ham_data['e0t1orb'] # <psi_0|H_bar|psi_0>
        rot_focka = ham_data['fock_bar'][0][:nocca,:]
        rot_fockb = ham_data['fock_bar'][1][:noccb,:]
        rot_chola = ham_data['chol_bar'][0].reshape(-1,norba,norba)[:,:nocca,:]
        rot_cholb = ham_data['chol_bar'][1].reshape(-1,norbb,norbb)[:,:noccb,:]

        gfa = (walker_up.dot(jnp.linalg.inv(walker_up[:nocca, :]))).T
        gfb = (walker_dn.dot(jnp.linalg.inv(walker_dn[:noccb, :]))).T
        e1a = oe.contract('ia,ia->',gfa[:nocca,nocca:],rot_focka[:nocca,nocca:], backend="jax")
        e1b = oe.contract('ia,ia->',gfb[:noccb,noccb:],rot_fockb[:noccb,noccb:], backend="jax")
        e1 = e1a + e1b
        
        lga = oe.contract('gia,ka->gik', rot_chola[:,:nocca,nocca:], gfa[:nocca,nocca:], backend="jax")
        lgb = oe.contract('gia,ka->gik', rot_cholb[:,:noccb,noccb:], gfb[:noccb,noccb:], backend="jax")
        e2aa = oe.contract('gik,ik,gjj->', lga, prjloa, lga, backend="jax") \
            - oe.contract('gij,gjk,ik->',lga, lga, prjloa, backend="jax")
        e2ab = oe.contract('gik,ik,gjj->', lga, prjloa, lgb, backend="jax")
        e2ba = oe.contract('gik,ik,gjj->', lgb, prjlob, lga, backend="jax")
        e2bb = oe.contract('gik,ik,gjj->', lgb, prjlob, lgb, backend="jax") \
            - oe.contract('gij,gjk,ik->',lgb, lgb, prjlob, backend="jax")
        e2 = 0.5 * (e2aa + e2ab + e2ba + e2bb)
        
        e_corr = e0 + e1 + e2
        return e_corr

    @partial(jit, static_argnums=0)
    def _t2eorb_tc(self, walker_up, walker_dn, ham_data, wave_data):
        
        norb_a, norb_b = self.norb
        nocc_a, nocc_b = self.nelec
        # h0 = ham_data["h0"], ham_data["E0"]
        h1a, h1b = ham_data["h1bar"]
        t2aa, t2ab = wave_data["t2aa"], wave_data["t2ab"]
        t2ba, t2bb = wave_data["t2ba"], wave_data["t2bb"]
        chol_a, chol_b = ham_data["chol_bar"]
        chol_a = chol_a.reshape(-1, norb_a, norb_a)
        chol_b = chol_b.reshape(-1, norb_b, norb_b)
        rot_chol_a = chol_a[:, :nocc_a, :]
        rot_chol_b = chol_b[:, :nocc_b, :]

        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a, :]))).T # G_ip
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[:nocc_b, :]))).T
        green_occ_a = green_a[:, nocc_a:]
        green_occ_b = green_b[:, nocc_b:]
        greenp_a = jnp.vstack((green_occ_a, -jnp.eye(norb_a - nocc_a)))
        greenp_b = jnp.vstack((green_occ_b, -jnp.eye(norb_b - nocc_b)))

        # 1 body energy    
        hg_a = oe.contract("pj,pj->", h1a[:nocc_a, :], green_a, backend="jax")
        hg_b = oe.contract("pj,pj->", h1b[:nocc_b, :], green_b, backend="jax")
        e1_0 = hg_a + hg_b #  <HF|h1|walker>/<HF|walker>

        # double excitations
        # i <-> j does not have anti-sym in LNO!!!
        t2g_aa_a_c = oe.contract("iajb,ia->jb", t2aa, green_occ_a, backend="jax") / 4
        t2g_aa_a_e = oe.contract("iajb,ja->ib", t2aa, green_occ_a, backend="jax") / 4
        t2g_bb_b_c = oe.contract("iajb,ia->jb", t2bb, green_occ_b, backend="jax") / 4
        t2g_bb_b_e = oe.contract("iajb,ja->ib", t2bb, green_occ_b, backend="jax") / 4
        t2g_ab_a = oe.contract("iajb,ia->jb", t2ab, green_occ_a, backend="jax") / 2
        t2g_ab_b = oe.contract("iajb,jb->ia", t2ab, green_occ_b, backend="jax") / 2
        t2g_ba_a = oe.contract("iajb,jb->ia", t2ba, green_occ_a, backend="jax") / 2
        t2g_ba_b = oe.contract("iajb,ia->jb", t2ba, green_occ_b, backend="jax") / 2
        gt2g_aa = oe.contract("jb,jb->", t2g_aa_a_c, green_occ_a, backend="jax")
        gt2g_bb = oe.contract("jb,jb->", t2g_bb_b_c, green_occ_b, backend="jax")
        gt2g_ab = oe.contract("jb,jb->", t2g_ab_a, green_occ_b, backend="jax")
        gt2g_ba = oe.contract("jb,jb->", t2g_ba_b, green_occ_a, backend="jax")
        gt2g =  (gt2g_aa + gt2g_bb) * 2 + (gt2g_ab + gt2g_ba)
        e1_2_1 = gt2g * e1_0

        # t_iajb G_ia G_jq Gp_pb
        t2_green_aaa_c = oe.contract('pb,jb,jq->pq', greenp_a, t2g_aa_a_c, green_a, backend="jax") # t_iajb G_ia G_jq Gp_pb (-)
        t2_green_aaa_e = oe.contract('pb,ib,iq->pq', greenp_a, t2g_aa_a_e, green_a, backend="jax") # t_iajb G_ja G_iq Gp_pb (+)
        t2_green_bbb_c = oe.contract('pb,jb,jq->pq', greenp_b, t2g_bb_b_c, green_b, backend="jax")
        t2_green_bbb_e = oe.contract('pb,ib,iq->pq', greenp_b, t2g_bb_b_e, green_b, backend="jax")
        t2_green_aba = oe.contract('pa,ia,iq->pq', greenp_a, t2g_ab_b, green_a, backend="jax")
        t2_green_baa = oe.contract('pb,jb,jq->pq', greenp_a, t2g_ba_b, green_a, backend="jax")
        t2_green_bab = oe.contract('pa,ia,iq->pq', greenp_b, t2g_ba_a, green_b, backend="jax")
        t2_green_abb = oe.contract('pb,jb,jq->pq', greenp_b, t2g_ab_a, green_b, backend="jax")
        t2_green_aaa = 2 * (t2_green_aaa_c - t2_green_aaa_e) # use the anti-sym of a <-> b
        t2_green_bbb = 2 * (t2_green_bbb_c - t2_green_bbb_e)
        e1_2_2_a = -oe.contract("pq,pq->", t2_green_aaa + t2_green_aba + t2_green_baa, h1a, backend="jax")
        e1_2_2_b = -oe.contract("pq,pq->", t2_green_bbb + t2_green_bab + t2_green_abb, h1b, backend="jax")
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2 # <HF|T2 h1|walker>/<HF|walker>

        # two body energy
        lg_a = oe.contract("gpj,qj->gpq", rot_chol_a, green_a, backend="jax")
        lg_b = oe.contract("gpj,qj->gpq", rot_chol_b, green_b, backend="jax")
        tr_lg_a = oe.contract("gpp->g", lg_a, backend="jax")
        tr_lg_b = oe.contract("gpp->g", lg_b, backend="jax")
        lg_0 = tr_lg_a + tr_lg_b
        e2_0_1 = oe.contract('g,g->', lg_0, lg_0) / 2.0
        e2_0_2 = - (oe.contract("gpq,gqp->", lg_a, lg_a, backend="jax")
                    + oe.contract("gpq,gqp->", lg_b, lg_b, backend="jax")) / 2.0
        e2_0 = e2_0_1 + e2_0_2 # <HF|h2|walker>/<HF|walker>

        # double excitations
        e2_2_1 = e2_0 * gt2g
        lt2g_a = oe.contract("gpq,pq->g", chol_a, 2*t2_green_aaa + 2*(t2_green_aba + t2_green_baa),backend="jax")
        lt2g_b = oe.contract("gpq,pq->g", chol_b, 2*t2_green_bbb + 2*(t2_green_bab + t2_green_abb),backend="jax")
        e2_2_2_1 = -((lt2g_a + lt2g_b) @ lg_0) / 2.0

        def scanned_fun(carry, x):
            chol_a_i, rot_chol_a_i, chol_b_i, rot_chol_b_i = x
            gl_a_i = oe.contract("ir,pr->ip", green_a, chol_a_i, backend="jax")
            gl_b_i = oe.contract("ir,pr->ip", green_b, chol_b_i, backend="jax")
            lt2_green_a_i = oe.contract(
                "pi,ji->pj", rot_chol_a_i, 2*t2_green_aaa + 2*(t2_green_aba + t2_green_baa), backend="jax")
            lt2_green_b_i = oe.contract(
                "pi,ji->pj", rot_chol_b_i, 2*t2_green_bbb + 2*(t2_green_bab + t2_green_abb),backend="jax")
            carry[0] += (oe.contract("ip,ip->", gl_a_i, lt2_green_a_i, backend="jax")
                        + oe.contract("ip,ip->", gl_b_i, lt2_green_b_i, backend="jax")) / 2
            glgp_a_i = oe.contract("ip,pa->ia", gl_a_i, greenp_a, backend="jax")
            glgp_b_i = oe.contract("ip,pa->ia", gl_b_i, greenp_b, backend="jax")
            l2t2_aa = 0.5 * oe.contract("ia,jb,iajb->", glgp_a_i, glgp_a_i, t2aa, backend="jax")
            l2t2_ab = 0.5 * oe.contract("ia,jb,iajb->", glgp_a_i, glgp_b_i, t2ab, backend="jax")
            l2t2_ba = 0.5 * oe.contract("ia,jb,iajb->", glgp_b_i, glgp_a_i, t2ba, backend="jax")
            l2t2_bb = 0.5 * oe.contract("ia,jb,iajb->", glgp_b_i, glgp_b_i, t2bb, backend="jax")
            carry[1] += l2t2_aa + l2t2_ab + l2t2_ba + l2t2_bb
            return carry, 0.0

        [e2_2_2_2, e2_2_3], _ = lax.scan(
            scanned_fun, [0.0, 0.0], (chol_a, rot_chol_a, chol_b, rot_chol_b)
        )
        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <HF|T2 h2|walker>/<HF|walker>

        t2orb = gt2g # <HF|T1+T2|walker>/<HF|walker>
        e12bar = e1_0 + e2_0 # <HF|h1+h2|walker>/<HF|walker>
        t2eorb = e1_2 + e2_2 # <HF|T2(h1+h2)|walker>/<HF|walker>

        return t2eorb, t2orb, e12bar
    
    @partial(jit, static_argnums=0)
    def _calc_eorb_pt2(self, walker_up: jax.Array, walker_dn: jax.Array, ham_data: dict, wave_data: dict):
        
        o0 = jnp.linalg.det(walker_up[:walker_up.shape[1],:]) \
            * jnp.linalg.det(walker_dn[:walker_dn.shape[1],:])
        e0 = self._calc_energy(walker_up, walker_dn, ham_data, wave_data)
        
        walker_up_bar = wave_data['exp_t1a'] @ walker_up
        walker_dn_bar = wave_data['exp_t1b'] @ walker_dn
        
        obar = jnp.linalg.det(walker_up_bar[:walker_up_bar.shape[1], :]) \
            * jnp.linalg.det(walker_dn_bar[:walker_dn_bar.shape[1], :])
        t1olp = obar/o0 # <exp(T1)HF|walker>/<HF|walker>
        
        eorb_bar = self._calc_eorb_bar(walker_up_bar, walker_dn_bar, ham_data, wave_data)
        t2eorb, t2orb, e12bar = self._t2eorb_tc(walker_up_bar, walker_dn_bar, ham_data, wave_data)

        return e0, t1olp, eorb_bar, t2eorb, t2orb, e12bar

    @partial(jit, static_argnums=(0)) 
    def calc_eorb_pt2(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:

        n_walkers = walkers[0].shape[0]
        batch_size = n_walkers // self.n_batch
        
        def scan_batch(carry, walker_batch):
            batch_walker_up, batch_walker_dn = walker_batch
            e0, t1olp, eorb_bar, t2eorb, t2orb, e12bar \
                = vmap(self._calc_eorb_pt2, in_axes=(0, 0, None, None))(
                batch_walker_up, batch_walker_dn, ham_data, wave_data
            )
            return carry, (e0, t1olp, eorb_bar, t2eorb, t2orb, e12bar)
        
        _, (e0, t1olp, eorb_bar, t2eorb, t2orb, e12bar) \
            = lax.scan(scan_batch, None,
            (
                walkers[0].reshape(self.n_batch, batch_size, self.norb[0], self.nelec[0]),
                walkers[1].reshape(self.n_batch, batch_size, self.norb[1], self.nelec[1]),
            ),
        )

        e0 = e0.reshape(n_walkers)
        t1olp = t1olp.reshape(n_walkers)
        eorb_bar = eorb_bar.reshape(n_walkers)
        t2eorb = t2eorb.reshape(n_walkers)
        t2orb = t2orb.reshape(n_walkers)
        e12bar = e12bar.reshape(n_walkers)

        return e0, t1olp, eorb_bar, t2eorb, t2orb, e12bar

    
    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        """Builds half rotated integrals for efficient force bias and energy calculations."""
        norba, norbb = self.norb
        nocca, noccb = self.nelec
        prjloa, prjlob = wave_data['prjlo']
        chola = ham_data["chol"][0].reshape(-1, norba, norba)
        cholb = ham_data["chol"][1].reshape(-1, norbb, norbb)
        # exp(T1^dagger) H exp(-T1^dagger)
        h1bar_a = wave_data['exp_t1a'] @ ham_data['h1'][0] @ wave_data['exp_mt1a']
        h1bar_b = wave_data['exp_t1b'] @ ham_data['h1'][1] @ wave_data['exp_mt1b']
        ham_data["h1bar"] = [h1bar_a, h1bar_b]
        chol_bar_a = oe.contract('pr,grs,sq->gpq', wave_data['exp_t1a'], chola, wave_data['exp_mt1a'], backend='jax')
        chol_bar_b = oe.contract('pr,grs,sq->gpq', wave_data['exp_t1b'], cholb, wave_data['exp_mt1b'], backend='jax')
        ham_data["chol_bar"] = [chol_bar_a, chol_bar_b]        
        # exp(T1^dagger) Fock exp(-T1^dagger)
        la = oe.contract('gjj->g', chol_bar_a[:,:nocca,:nocca], backend="jax")
        lb = oe.contract('gjj->g', chol_bar_b[:,:noccb,:noccb], backend="jax")
        jeff_a = oe.contract('gpq,g->pq', chol_bar_a, la+lb, backend="jax")
        jeff_b = oe.contract('gpq,g->pq', chol_bar_b, la+lb, backend="jax")
        keff_a = oe.contract('gpj,gjq->pq', chol_bar_a[:,:,:nocca], chol_bar_a[:,:nocca,:], backend="jax")
        keff_b = oe.contract('gpj,gjq->pq', chol_bar_b[:,:,:noccb], chol_bar_b[:,:noccb,:], backend="jax")
        fock_bar_a = h1bar_a + jeff_a - keff_a
        fock_bar_b = h1bar_b + jeff_b - keff_b
        fock_bar_a = oe.contract('ip,ik->kp', fock_bar_a[:nocca, :], prjloa, backend="jax")
        fock_bar_b = oe.contract('ip,ik->kp', fock_bar_b[:noccb, :], prjlob, backend="jax")
        ham_data['fock_bar'] = [fock_bar_a, fock_bar_b]
        
        h1bar_a = chol_bar_a = la = jeff_a = keff_a = fock_bar_a = None
        h1bar_b = chol_bar_b = lb = jeff_b = keff_b = fock_bar_b = None  
        ham_data['h1_mod'] = None
        
        return ham_data
    
    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class uccsd_pt2_alpha(uccsd_pt2):
    '''
    Alpha LNO UCCSD_PT2 Trial:
    separate Alpha and Beta LNO projection 
    since they don't project onto each other
    should be about 2 time faster
    maybe able to build larger intermediate
    '''

    norb: Tuple[int, int]
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_eorb_bar(self, walker_up, walker_dn, ham_data, wave_data):
        '''
        calculate the correlation energy of the Hamiltonian
        transformed by exp(T1^dagger):
        ecorr_bar = <psi_0|H_bar|walker_bar>/<psi_0|walker_bar>
        |walker_bar> = exp(T1^dagger) |walker>
        H_bar = exp(T1^dagger) H exp(-T1^dagger)
        |psi_0> is the mean-field solution of H
        '''
        nocca, noccb = self.nelec 
        norba, norbb = self.norb
        prjloa, prjlob = wave_data['prjlo']
        e0 = ham_data['e0t1orb'] # <psi_0|H_bar|psi_0>
        rot_focka = ham_data['fock_bar'][0][:nocca,:]
        rot_chola = ham_data['chol_bar'][0].reshape(-1,norba,norba)[:,:nocca,:]
        rot_cholb = ham_data['chol_bar'][1].reshape(-1,norbb,norbb)[:,:noccb,:]

        gfa = (walker_up.dot(jnp.linalg.inv(walker_up[:nocca, :]))).T
        gfb = (walker_dn.dot(jnp.linalg.inv(walker_dn[:noccb, :]))).T
        e1a = oe.contract('ia,ia->',gfa[:nocca,nocca:],rot_focka[:nocca,nocca:], backend="jax")
        e1 = e1a
        
        lga = oe.contract('gia,ka->gik', rot_chola[:,:nocca,nocca:], gfa[:nocca,nocca:], backend="jax")
        lgb = oe.contract('gia,ka->gik', rot_cholb[:,:noccb,noccb:], gfb[:noccb,noccb:], backend="jax")
        e2aa = oe.contract('gik,ik,gjj->', lga, prjloa, lga, backend="jax") \
            - oe.contract('gij,gjk,ik->',lga, lga, prjloa, backend="jax")
        e2ab = oe.contract('gik,ik,gjj->', lga, prjloa, lgb, backend="jax")
        e2 = 0.5 * (e2aa + e2ab)
        
        e_corr = e0 + e1 + e2
        return e_corr

    @partial(jit, static_argnums=0)
    def _t2eorb_tc(self,walker_up,walker_dn,ham_data,wave_data):
        norb_a, norb_b = self.norb
        nocc_a, nocc_b = self.nelec
        h1a, h1b = ham_data["h1bar"]
        t2aa, t2ab = wave_data["t2aa"], wave_data["t2ab"]
        chol_a, chol_b = ham_data["chol_bar"]
        chol_a = chol_a.reshape(-1, norb_a, norb_a)
        chol_b = chol_b.reshape(-1, norb_b, norb_b)
        rot_chol_a = chol_a[:, :nocc_a, :]
        rot_chol_b = chol_b[:, :nocc_b, :]

        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a, :]))).T # G_ip
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[:nocc_b, :]))).T
        green_occ_a = green_a[:, nocc_a:]
        green_occ_b = green_b[:, nocc_b:]
        greenp_a = jnp.vstack((green_occ_a, -jnp.eye(norb_a - nocc_a)))
        greenp_b = jnp.vstack((green_occ_b, -jnp.eye(norb_b - nocc_b)))

        # 1 body energy    
        hg_a = oe.contract("pj,pj->", h1a[:nocc_a, :], green_a, backend="jax")
        hg_b = oe.contract("pj,pj->", h1b[:nocc_b, :], green_b, backend="jax")
        e1_0 = hg_a + hg_b #  <HF|h1|walker>/<HF|walker>

        # double excitations
        # i <-> j does not have anti-sym in LNO!!!
        t2g_aa_a_c = oe.contract("iajb,ia->jb", t2aa, green_occ_a, backend="jax") / 4
        t2g_aa_a_e = oe.contract("iajb,ja->ib", t2aa, green_occ_a, backend="jax") / 4
        t2g_ab_a = oe.contract("iajb,ia->jb", t2ab, green_occ_a, backend="jax") / 2
        t2g_ab_b = oe.contract("iajb,jb->ia", t2ab, green_occ_b, backend="jax") / 2
        gt2g_aa = oe.contract("jb,jb->", t2g_aa_a_c, green_occ_a, backend="jax")
        gt2g_ab = oe.contract("jb,jb->", t2g_ab_a, green_occ_b, backend="jax")
        gt2g =  2*gt2g_aa + gt2g_ab
        e1_2_1 = gt2g * e1_0

        # t_iajb G_ia G_jq Gp_pb
        t2_green_aaa_c = oe.contract('pb,jb,jq->pq', greenp_a, t2g_aa_a_c, green_a, backend="jax") # t_iajb G_ia G_jq Gp_pb (-)
        t2_green_aaa_e = oe.contract('pb,ib,iq->pq', greenp_a, t2g_aa_a_e, green_a, backend="jax") # t_iajb G_ja G_iq Gp_pb (+)
        t2_green_aba = oe.contract('pa,ia,iq->pq', greenp_a, t2g_ab_b, green_a, backend="jax")
        t2_green_abb = oe.contract('pb,jb,jq->pq', greenp_b, t2g_ab_a, green_b, backend="jax")
        t2_green_aaa = 2 * (t2_green_aaa_c - t2_green_aaa_e) # use the anti-sym of a <-> b
        e1_2_2_a = -oe.contract("pq,pq->", t2_green_aaa + t2_green_aba, h1a, backend="jax")
        e1_2_2_b = -oe.contract("pq,pq->", t2_green_abb, h1b, backend="jax")
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2 # <HF|T2 h1|walker>/<HF|walker>

        # two body energy
        # scan over chol vecs
        def scan_chol(carry, x):
            chol_a_i, rot_chol_a_i, chol_b_i, rot_chol_b_i = x
            gl_a_i = oe.contract("ir,pr->ip", green_a, chol_a_i, backend="jax")
            gl_b_i = oe.contract("ir,pr->ip", green_b, chol_b_i, backend="jax")
            tr_gl_a_i = oe.contract("ii->", gl_a_i[:nocc_a,:nocc_a], backend="jax")
            tr_gl_b_i = oe.contract("ii->", gl_b_i[:nocc_b,:nocc_b], backend="jax")
            gl_c_i = tr_gl_a_i + tr_gl_b_i
            e2_0_c_i = gl_c_i * gl_c_i / 2.0
            e2_0_e_i = - (oe.contract("ij,ji->", gl_a_i[:nocc_a,:nocc_a], gl_a_i[:nocc_a,:nocc_a], backend="jax")
                        + oe.contract("ij,ji->", gl_b_i[:nocc_b,:nocc_b], gl_b_i[:nocc_b,:nocc_b], backend="jax")) / 2.0
            carry[0] += e2_0_e_i + e2_0_c_i

            # double excitations
            lt2g_a_i = oe.contract("pq,pq->", chol_a_i, 2*t2_green_aaa + 2*t2_green_aba,backend="jax")
            lt2g_b_i = oe.contract("pq,pq->", chol_b_i, 2*t2_green_abb, backend="jax")
            carry[1] += - (lt2g_a_i + lt2g_b_i) * gl_c_i / 2.0

            lt2_green_a_i = oe.contract("pi,ji->pj", rot_chol_a_i, 2*t2_green_aaa + 2*t2_green_aba, backend="jax")
            lt2_green_b_i = oe.contract("pi,ji->pj", rot_chol_b_i, 2*t2_green_abb, backend="jax")
            carry[2] += (oe.contract("ip,ip->", gl_a_i, lt2_green_a_i, backend="jax")
                        + oe.contract("ip,ip->", gl_b_i, lt2_green_b_i, backend="jax")) / 2
            
            glgp_a_i = oe.contract("ip,pa->ia", gl_a_i, greenp_a, backend="jax")
            glgp_b_i = oe.contract("ip,pa->ia", gl_b_i, greenp_b, backend="jax")
            # l2t2_aa_a_i = oe.contract("ia,iajb->jb", glgp_a_i, t2aa, backend="jax")
            # l2t2_ab_a = oe.contract("gia,iajb->gjb", glgp_a, t2ab, backend="jax")
            l2t2_aa_i = 0.5 * oe.contract("ia,iajb,jb->", glgp_a_i, t2aa, glgp_a_i, backend="jax")
            l2t2_ab_i = 0.5 * oe.contract("ia,iajb,jb->", glgp_a_i, t2ab, glgp_b_i, backend="jax")
            carry[3] += l2t2_aa_i + l2t2_ab_i
            
            return carry, 0.0

        [e2_0, e2_2_2_1, e2_2_2_2, e2_2_3], _ \
            = lax.scan(scan_chol, [0.0,0.0,0.0,0.0], (chol_a, rot_chol_a, chol_b, rot_chol_b))
        
        e2_2_1 = e2_0 * gt2g
        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <HF|T2 h2|walker>/<HF|walker>

        t2orb = gt2g # <HF|T1+T2|walker>/<HF|walker>
        e12bar = e1_0 + e2_0 # <HF|h1+h2|walker>/<HF|walker>
        t2eorb = e1_2 + e2_2 # <HF|T2(h1+h2)|walker>/<HF|walker>

        return t2eorb, t2orb, e12bar
    
    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        """Builds half rotated integrals for efficient force bias and energy calculations."""
        norba, norbb = self.norb
        nocca, noccb = self.nelec
        prjloa, prjlob = wave_data['prjlo']
        chola = ham_data["chol"][0].reshape(-1, norba, norba)
        cholb = ham_data["chol"][1].reshape(-1, norbb, norbb)
        # exp(T1^dagger) H exp(-T1^dagger)
        h1bar_a = wave_data['exp_t1a'] @ ham_data['h1'][0] @ wave_data['exp_mt1a']
        h1bar_b = wave_data['exp_t1b'] @ ham_data['h1'][1] @ wave_data['exp_mt1b']
        ham_data["h1bar"] = [h1bar_a, h1bar_b]
        chol_bar_a = oe.contract('pr,grs,sq->gpq', wave_data['exp_t1a'], chola, wave_data['exp_mt1a'], backend='jax')
        chol_bar_b = oe.contract('pr,grs,sq->gpq', wave_data['exp_t1b'], cholb, wave_data['exp_mt1b'], backend='jax')
        ham_data["chol_bar"] = [chol_bar_a, chol_bar_b]        
        # exp(T1^dagger) Fock exp(-T1^dagger)
        la = oe.contract('gjj->g', chol_bar_a[:,:nocca,:nocca], backend="jax")
        lb = oe.contract('gjj->g', chol_bar_b[:,:noccb,:noccb], backend="jax")
        jeff_a = oe.contract('gpq,g->pq', chol_bar_a, la+lb, backend="jax")
        jeff_b = oe.contract('gpq,g->pq', chol_bar_b, la+lb, backend="jax")
        keff_a = oe.contract('gpj,gjq->pq', chol_bar_a[:,:,:nocca], chol_bar_a[:,:nocca,:], backend="jax")
        keff_b = oe.contract('gpj,gjq->pq', chol_bar_b[:,:,:noccb], chol_bar_b[:,:noccb,:], backend="jax")
        fock_bar_a = h1bar_a + jeff_a - keff_a
        fock_bar_b = h1bar_b + jeff_b - keff_b
        fock_bar_a = oe.contract('ip,ik->kp', fock_bar_a[:nocca, :], prjloa, backend="jax")
        fock_bar_b = oe.contract('ip,ik->kp', fock_bar_b[:noccb, :], prjlob, backend="jax")
        ham_data['fock_bar'] = [fock_bar_a, fock_bar_b]
        
        h1bar_a = chol_bar_a = la = jeff_a = keff_a = fock_bar_a = None
        h1bar_b = chol_bar_b = lb = jeff_b = keff_b = fock_bar_b = None  
        ham_data['h1_mod'] = None
        
        return ham_data
    
    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
    
@dataclass
class uccsd_pt2_beta(uccsd_pt2):
    '''
    Beta LNO UCCSD_PT2 Trial
    Checkout the definition and comment 
    in uccsd_pt2 and uccsd_pt2_alpha
    '''

    norb: Tuple[int, int]
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_eorb_bar(self, walker_up, walker_dn, ham_data, wave_data):
        '''
        calculate the correlation energy of the Hamiltonian
        transformed by exp(T1^dagger):
        ecorr_bar = <psi_0|H_bar|walker_bar>/<psi_0|walker_bar>
        |walker_bar> = exp(T1^dagger) |walker>
        H_bar = exp(T1^dagger) H exp(-T1^dagger)
        |psi_0> is the mean-field solution of H
        '''
        nocca, noccb = self.nelec 
        norba, norbb = self.norb
        prjloa, prjlob = wave_data['prjlo']
        e0 = ham_data['e0t1orb'] # <psi_0|H_bar|psi_0>
        rot_fockb = ham_data['fock_bar'][1][:noccb,:]
        rot_chola = ham_data['chol_bar'][0].reshape(-1,norba,norba)[:,:nocca,:]
        rot_cholb = ham_data['chol_bar'][1].reshape(-1,norbb,norbb)[:,:noccb,:]

        gfa = (walker_up.dot(jnp.linalg.inv(walker_up[:nocca, :]))).T
        gfb = (walker_dn.dot(jnp.linalg.inv(walker_dn[:noccb, :]))).T
        e1b = oe.contract('ia,ia->',gfb[:noccb,noccb:],rot_fockb[:noccb,noccb:], backend="jax")
        e1 = e1b
        
        lga = oe.contract('gia,ka->gik', rot_chola[:,:nocca,nocca:], gfa[:nocca,nocca:], backend="jax")
        lgb = oe.contract('gia,ka->gik', rot_cholb[:,:noccb,noccb:], gfb[:noccb,noccb:], backend="jax")
        e2ba = oe.contract('gik,ik,gjj->', lgb, prjlob, lga, backend="jax")
        e2bb = oe.contract('gik,ik,gjj->', lgb, prjlob, lgb, backend="jax") \
            - oe.contract('gij,gjk,ik->',lgb, lgb, prjlob, backend="jax")
        e2 = 0.5 * (e2ba + e2bb)
        
        e_corr = e0 + e1 + e2
        return e_corr

    @partial(jit, static_argnums=0)
    def _t2eorb_tc(self,walker_up,walker_dn,ham_data,wave_data):
        
        norb_a, norb_b = self.norb
        nocc_a, nocc_b = self.nelec
        h1a, h1b = ham_data["h1bar"]
        t2ba, t2bb = wave_data["t2ba"], wave_data["t2bb"]
        chol_a, chol_b = ham_data["chol_bar"]
        chol_a = chol_a.reshape(-1, norb_a, norb_a)
        chol_b = chol_b.reshape(-1, norb_b, norb_b)
        rot_chol_a = chol_a[:, :nocc_a, :]
        rot_chol_b = chol_b[:, :nocc_b, :]

        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a, :]))).T # G_ip
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[:nocc_b, :]))).T
        green_occ_a = green_a[:, nocc_a:]
        green_occ_b = green_b[:, nocc_b:]
        greenp_a = jnp.vstack((green_occ_a, -jnp.eye(norb_a - nocc_a)))
        greenp_b = jnp.vstack((green_occ_b, -jnp.eye(norb_b - nocc_b)))

        # 1 body energy    
        hg_a = oe.contract("pj,pj->", h1a[:nocc_a, :], green_a, backend="jax")
        hg_b = oe.contract("pj,pj->", h1b[:nocc_b, :], green_b, backend="jax")
        e1_0 = hg_a + hg_b #  <HF|h1|walker>/<HF|walker>

        # double excitations
        # i <-> j does not have anti-sym in LNO!!!
        t2g_bb_b_c = oe.contract("iajb,ia->jb", t2bb, green_occ_b, backend="jax") / 4
        t2g_bb_b_e = oe.contract("iajb,ja->ib", t2bb, green_occ_b, backend="jax") / 4
        t2g_ba_a = oe.contract("iajb,jb->ia", t2ba, green_occ_a, backend="jax") / 2
        t2g_ba_b = oe.contract("iajb,ia->jb", t2ba, green_occ_b, backend="jax") / 2
        gt2g_bb = oe.contract("jb,jb->", t2g_bb_b_c, green_occ_b, backend="jax")
        gt2g_ba = oe.contract("jb,jb->", t2g_ba_b, green_occ_a, backend="jax")
        gt2g =  2 * gt2g_bb + gt2g_ba
        e1_2_1 = gt2g * e1_0

        # t_iajb G_ia G_jq Gp_pb
        t2_green_bbb_c = oe.contract('pb,jb,jq->pq', greenp_b, t2g_bb_b_c, green_b, backend="jax")
        t2_green_bbb_e = oe.contract('pb,ib,iq->pq', greenp_b, t2g_bb_b_e, green_b, backend="jax")
        t2_green_baa = oe.contract('pb,jb,jq->pq', greenp_a, t2g_ba_b, green_a, backend="jax")
        t2_green_bab = oe.contract('pa,ia,iq->pq', greenp_b, t2g_ba_a, green_b, backend="jax")
        t2_green_bbb = 2 * (t2_green_bbb_c - t2_green_bbb_e) # use the anti-sym of a <-> b
        e1_2_2_a = -oe.contract("pq,pq->", t2_green_baa, h1a, backend="jax")
        e1_2_2_b = -oe.contract("pq,pq->", t2_green_bbb + t2_green_bab, h1b, backend="jax")
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2 # <HF|T2 h1|walker>/<HF|walker>

        # two body energy
        # scan over chol vecs
        def scan_chol(carry, x):
            chol_a_i, rot_chol_a_i, chol_b_i, rot_chol_b_i = x
            gl_a_i = oe.contract("ir,pr->ip", green_a, chol_a_i, backend="jax")
            gl_b_i = oe.contract("ir,pr->ip", green_b, chol_b_i, backend="jax")
            tr_gl_a_i = oe.contract("ii->", gl_a_i[:nocc_a,:nocc_a], backend="jax")
            tr_gl_b_i = oe.contract("ii->", gl_b_i[:nocc_b,:nocc_b], backend="jax")
            gl_c_i = tr_gl_a_i + tr_gl_b_i
            e2_0_c_i = gl_c_i * gl_c_i / 2.0
            e2_0_e_i = - (oe.contract("ij,ji->", gl_a_i[:nocc_a,:nocc_a], gl_a_i[:nocc_a,:nocc_a], backend="jax")
                        + oe.contract("ij,ji->", gl_b_i[:nocc_b,:nocc_b], gl_b_i[:nocc_b,:nocc_b], backend="jax")) / 2.0
            carry[0] += e2_0_c_i + e2_0_e_i

            # double excitations
            lt2g_a_i = oe.contract("pq,pq->", chol_a_i, 2*t2_green_baa,backend="jax")
            lt2g_b_i = oe.contract("pq,pq->", chol_b_i, 2*t2_green_bbb + 2*t2_green_bab, backend="jax")
            carry[1] += - (lt2g_a_i + lt2g_b_i) * gl_c_i / 2.0

            lt2_green_a_i = oe.contract("pi,ji->pj", rot_chol_a_i, 2*t2_green_baa, backend="jax")
            lt2_green_b_i = oe.contract("pi,ji->pj", rot_chol_b_i, 2*t2_green_bbb + 2*t2_green_bab, backend="jax")
            carry[2] += (oe.contract("ip,ip->", gl_a_i, lt2_green_a_i, backend="jax")
                        + oe.contract("ip,ip->", gl_b_i, lt2_green_b_i, backend="jax")) / 2
            
            glgp_a_i = oe.contract("ip,pa->ia", gl_a_i, greenp_a, backend="jax")
            glgp_b_i = oe.contract("ip,pa->ia", gl_b_i, greenp_b, backend="jax")
            # l2t2_ba_b = oe.contract("gia,iajb->gjb", glgp_b, t2ba, backend="jax")
            # l2t2_bb_b = oe.contract("gia,iajb->gjb", glgp_b, t2bb, backend="jax")
            l2t2_ba = 0.5 * oe.contract("ia,iajb,jb->", glgp_b_i, t2ba, glgp_a_i, backend="jax")
            l2t2_bb = 0.5 * oe.contract("ia,iajb,jb->", glgp_b_i, t2bb, glgp_b_i, backend="jax")
            carry[3] += l2t2_ba + l2t2_bb
            
            return carry, 0.0

        [e2_0, e2_2_2_1, e2_2_2_2, e2_2_3], _ \
            = lax.scan(scan_chol, [0.0,0.0,0.0,0.0], (chol_a, rot_chol_a, chol_b, rot_chol_b))
        
        e2_2_1 = e2_0 * gt2g
        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <HF|T2 h2|walker>/<HF|walker>

        t2orb = gt2g # <HF|T1+T2|walker>/<HF|walker>
        e12bar = e1_0 + e2_0 # <HF|h1+h2|walker>/<HF|walker>
        t2eorb = e1_2 + e2_2 # <HF|T2(h1+h2)|walker>/<HF|walker>

        return t2eorb, t2orb, e12bar

    
    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        """Builds half rotated integrals for efficient force bias and energy calculations."""
        norba, norbb = self.norb
        nocca, noccb = self.nelec
        prjloa, prjlob = wave_data['prjlo']
        chola = ham_data["chol"][0].reshape(-1, norba, norba)
        cholb = ham_data["chol"][1].reshape(-1, norbb, norbb)
        # exp(T1^dagger) H exp(-T1^dagger)
        h1bar_a = wave_data['exp_t1a'] @ ham_data['h1'][0] @ wave_data['exp_mt1a']
        h1bar_b = wave_data['exp_t1b'] @ ham_data['h1'][1] @ wave_data['exp_mt1b']
        ham_data["h1bar"] = [h1bar_a, h1bar_b]
        chol_bar_a = oe.contract('pr,grs,sq->gpq', wave_data['exp_t1a'], chola, wave_data['exp_mt1a'], backend='jax')
        chol_bar_b = oe.contract('pr,grs,sq->gpq', wave_data['exp_t1b'], cholb, wave_data['exp_mt1b'], backend='jax')
        ham_data["chol_bar"] = [chol_bar_a, chol_bar_b]        
        # exp(T1^dagger) Fock exp(-T1^dagger)
        la = oe.contract('gjj->g', chol_bar_a[:,:nocca,:nocca], backend="jax")
        lb = oe.contract('gjj->g', chol_bar_b[:,:noccb,:noccb], backend="jax")
        jeff_a = oe.contract('gpq,g->pq', chol_bar_a, la+lb, backend="jax")
        jeff_b = oe.contract('gpq,g->pq', chol_bar_b, la+lb, backend="jax")
        keff_a = oe.contract('gpj,gjq->pq', chol_bar_a[:,:,:nocca], chol_bar_a[:,:nocca,:], backend="jax")
        keff_b = oe.contract('gpj,gjq->pq', chol_bar_b[:,:,:noccb], chol_bar_b[:,:noccb,:], backend="jax")
        fock_bar_a = h1bar_a + jeff_a - keff_a
        fock_bar_b = h1bar_b + jeff_b - keff_b
        fock_bar_a = oe.contract('ip,ik->kp', fock_bar_a[:nocca, :], prjloa, backend="jax")
        fock_bar_b = oe.contract('ip,ik->kp', fock_bar_b[:noccb, :], prjlob, backend="jax")
        ham_data['fock_bar'] = [fock_bar_a, fock_bar_b]
        
        h1bar_a = chol_bar_a = la = jeff_a = keff_a = fock_bar_a = None
        h1bar_b = chol_bar_b = lb = jeff_b = keff_b = fock_bar_b = None  
        ham_data['h1_mod'] = None
        
        return ham_data
    
    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
    

@dataclass
class uccsd_pt2_alpha_fast(uccsd_pt2_alpha):
    '''
    direct tensor contraction, no scan over the chol
    '''

    norb: Tuple[int, int]
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _t2eorb_tc(self,walker_up,walker_dn,ham_data,wave_data):
        norb_a, norb_b = self.norb
        nocc_a, nocc_b = self.nelec
        h1a, h1b = ham_data["h1bar"]
        t2aa, t2ab = wave_data["t2aa"], wave_data["t2ab"]
        chol_a, chol_b = ham_data["chol_bar"]
        chol_a = chol_a.reshape(-1, norb_a, norb_a)
        chol_b = chol_b.reshape(-1, norb_b, norb_b)
        rot_chol_a = chol_a[:, :nocc_a, :]
        rot_chol_b = chol_b[:, :nocc_b, :]

        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a, :]))).T # G_ip
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[:nocc_b, :]))).T
        green_occ_a = green_a[:, nocc_a:]
        green_occ_b = green_b[:, nocc_b:]
        greenp_a = jnp.vstack((green_occ_a, -jnp.eye(norb_a - nocc_a)))
        greenp_b = jnp.vstack((green_occ_b, -jnp.eye(norb_b - nocc_b)))

        # 1 body energy    
        hg_a = oe.contract("pj,pj->", h1a[:nocc_a, :], green_a, backend="jax")
        hg_b = oe.contract("pj,pj->", h1b[:nocc_b, :], green_b, backend="jax")
        e1_0 = hg_a + hg_b #  <HF|h1|walker>/<HF|walker>

        # double excitations
        # i <-> j does not have anti-sym in LNO!!!
        t2g_aa_a_c = oe.contract("iajb,ia->jb", t2aa, green_occ_a, backend="jax") / 4
        t2g_aa_a_e = oe.contract("iajb,ja->ib", t2aa, green_occ_a, backend="jax") / 4
        t2g_ab_a = oe.contract("iajb,ia->jb", t2ab, green_occ_a, backend="jax") / 2
        t2g_ab_b = oe.contract("iajb,jb->ia", t2ab, green_occ_b, backend="jax") / 2
        gt2g_aa = oe.contract("jb,jb->", t2g_aa_a_c, green_occ_a, backend="jax")
        gt2g_ab = oe.contract("jb,jb->", t2g_ab_a, green_occ_b, backend="jax")
        gt2g =  2*gt2g_aa + gt2g_ab
        e1_2_1 = gt2g * e1_0

        # t_iajb G_ia G_jq Gp_pb
        t2_green_aaa_c = oe.contract('pb,jb,jq->pq', greenp_a, t2g_aa_a_c, green_a, backend="jax") # t_iajb G_ia G_jq Gp_pb (-)
        t2_green_aaa_e = oe.contract('pb,ib,iq->pq', greenp_a, t2g_aa_a_e, green_a, backend="jax") # t_iajb G_ja G_iq Gp_pb (+)
        t2_green_aba = oe.contract('pa,ia,iq->pq', greenp_a, t2g_ab_b, green_a, backend="jax")
        t2_green_abb = oe.contract('pb,jb,jq->pq', greenp_b, t2g_ab_a, green_b, backend="jax")
        t2_green_aaa = 2 * (t2_green_aaa_c - t2_green_aaa_e) # use the anti-sym of a <-> b
        e1_2_2_a = -oe.contract("pq,pq->", t2_green_aaa + t2_green_aba, h1a, backend="jax")
        e1_2_2_b = -oe.contract("pq,pq->", t2_green_abb, h1b, backend="jax")
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2 # <HF|T2 h1|walker>/<HF|walker>

        # two body energy
        # explicit tensor contraction
        gl_a = oe.contract("ir,gpr->gip", green_a, chol_a, backend="jax")
        gl_b = oe.contract("ir,gpr->gip", green_b, chol_b, backend="jax")
        tr_gl_a = oe.contract("gii->g", gl_a[:,:nocc_a,:nocc_a], backend="jax")
        tr_gl_b = oe.contract("gii->g", gl_b[:,:nocc_b,:nocc_b], backend="jax")
        gl_c = tr_gl_a + tr_gl_b
        e2_0_1 = oe.contract('g,g->', gl_c, gl_c) / 2.0
        e2_0_2 = - (oe.contract("gij,gji->", gl_a[:,:nocc_a,:nocc_a], gl_a[:,:nocc_a,:nocc_a], backend="jax")
                    + oe.contract("gij,gji->", gl_b[:,:nocc_b,:nocc_b], gl_b[:,:nocc_b,:nocc_b], backend="jax")) / 2.0
        e2_0 = e2_0_1 + e2_0_2

        # double excitations
        e2_2_1 = e2_0 * gt2g
        lt2g_a = oe.contract("gpq,pq->g", chol_a, 2*t2_green_aaa + 2*t2_green_aba,backend="jax")
        lt2g_b = oe.contract("gpq,pq->g", chol_b, 2*t2_green_abb, backend="jax")
        e2_2_2_1 = -oe.contract('g,g->', lt2g_a + lt2g_b, gl_c, backend="jax") / 2.0

        lt2_green_a = oe.contract("gpi,ji->gpj", rot_chol_a, 2*t2_green_aaa + 2*t2_green_aba, backend="jax")
        lt2_green_b = oe.contract("gpi,ji->gpj", rot_chol_b, 2*t2_green_abb, backend="jax")
        e2_2_2_2 = (oe.contract("gip,gip->", gl_a, lt2_green_a, backend="jax")
                    + oe.contract("gip,gip->", gl_b, lt2_green_b, backend="jax")) / 2
        
        glgp_a = oe.contract("gip,pa->gia", gl_a, greenp_a, backend="jax")
        glgp_b = oe.contract("gip,pa->gia", gl_b, greenp_b, backend="jax")
        l2t2_aa_a = oe.contract("gia,iajb->gjb", glgp_a, t2aa, backend="jax")
        l2t2_ab_a = oe.contract("gia,iajb->gjb", glgp_a, t2ab, backend="jax")
        l2t2_aa = 0.5 * oe.contract("gjb,gjb->", l2t2_aa_a, glgp_a, backend="jax")
        l2t2_ab = 0.5 * oe.contract("gjb,gjb->", l2t2_ab_a, glgp_b, backend="jax")
        e2_2_3 = l2t2_aa + l2t2_ab
        
        e2_2_1 = e2_0 * gt2g
        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <HF|T2 h2|walker>/<HF|walker>

        t2orb = gt2g # <HF|T1+T2|walker>/<HF|walker>
        e12bar = e1_0 + e2_0 # <HF|h1+h2|walker>/<HF|walker>
        t2eorb = e1_2 + e2_2 # <HF|T2(h1+h2)|walker>/<HF|walker>

        return t2eorb, t2orb, e12bar
    
    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
    

@dataclass
class uccsd_pt2_beta_fast(uccsd_pt2_beta):
    '''
    Beta LNO UCCSD_PT2 Trial
    Checkout the definition and comment 
    in uccsd_pt2 and uccsd_pt2_alpha
    '''

    norb: Tuple[int, int]
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _t2eorb_tc(self,walker_up,walker_dn,ham_data,wave_data):
        
        norb_a, norb_b = self.norb
        nocc_a, nocc_b = self.nelec
        h1a, h1b = ham_data["h1bar"]
        t2ba, t2bb = wave_data["t2ba"], wave_data["t2bb"]
        chol_a, chol_b = ham_data["chol_bar"]
        chol_a = chol_a.reshape(-1, norb_a, norb_a)
        chol_b = chol_b.reshape(-1, norb_b, norb_b)
        rot_chol_a = chol_a[:, :nocc_a, :]
        rot_chol_b = chol_b[:, :nocc_b, :]

        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a, :]))).T # G_ip
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[:nocc_b, :]))).T
        green_occ_a = green_a[:, nocc_a:]
        green_occ_b = green_b[:, nocc_b:]
        greenp_a = jnp.vstack((green_occ_a, -jnp.eye(norb_a - nocc_a)))
        greenp_b = jnp.vstack((green_occ_b, -jnp.eye(norb_b - nocc_b)))

        # 1 body energy    
        hg_a = oe.contract("pj,pj->", h1a[:nocc_a, :], green_a, backend="jax")
        hg_b = oe.contract("pj,pj->", h1b[:nocc_b, :], green_b, backend="jax")
        e1_0 = hg_a + hg_b #  <HF|h1|walker>/<HF|walker>

        # double excitations
        # i <-> j does not have anti-sym in LNO!!!
        t2g_bb_b_c = oe.contract("iajb,ia->jb", t2bb, green_occ_b, backend="jax") / 4
        t2g_bb_b_e = oe.contract("iajb,ja->ib", t2bb, green_occ_b, backend="jax") / 4
        t2g_ba_a = oe.contract("iajb,jb->ia", t2ba, green_occ_a, backend="jax") / 2
        t2g_ba_b = oe.contract("iajb,ia->jb", t2ba, green_occ_b, backend="jax") / 2
        gt2g_bb = oe.contract("jb,jb->", t2g_bb_b_c, green_occ_b, backend="jax")
        gt2g_ba = oe.contract("jb,jb->", t2g_ba_b, green_occ_a, backend="jax")
        gt2g =  2 * gt2g_bb + gt2g_ba
        e1_2_1 = gt2g * e1_0

        # t_iajb G_ia G_jq Gp_pb
        t2_green_bbb_c = oe.contract('pb,jb,jq->pq', greenp_b, t2g_bb_b_c, green_b, backend="jax")
        t2_green_bbb_e = oe.contract('pb,ib,iq->pq', greenp_b, t2g_bb_b_e, green_b, backend="jax")
        t2_green_baa = oe.contract('pb,jb,jq->pq', greenp_a, t2g_ba_b, green_a, backend="jax")
        t2_green_bab = oe.contract('pa,ia,iq->pq', greenp_b, t2g_ba_a, green_b, backend="jax")
        t2_green_bbb = 2 * (t2_green_bbb_c - t2_green_bbb_e) # use the anti-sym of a <-> b
        e1_2_2_a = -oe.contract("pq,pq->", t2_green_baa, h1a, backend="jax")
        e1_2_2_b = -oe.contract("pq,pq->", t2_green_bbb + t2_green_bab, h1b, backend="jax")
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2 # <HF|T2 h1|walker>/<HF|walker>

        # two body energy
        # explicity tensor contraction faster high memory
        gl_a = oe.contract("ir,gpr->gip", green_a, chol_a, backend="jax")
        gl_b = oe.contract("ir,gpr->gip", green_b, chol_b, backend="jax")
        tr_gl_a = oe.contract("gii->g", gl_a[:,:nocc_a,:nocc_a], backend="jax")
        tr_gl_b = oe.contract("gii->g", gl_b[:,:nocc_b,:nocc_b], backend="jax")
        gl_c = tr_gl_a + tr_gl_b
        e2_0_1 = oe.contract('g,g->', gl_c, gl_c) / 2.0
        e2_0_2 = - (oe.contract("gij,gji->", gl_a[:,:nocc_a,:nocc_a], gl_a[:,:nocc_a,:nocc_a], backend="jax")
                    + oe.contract("gij,gji->", gl_b[:,:nocc_b,:nocc_b], gl_b[:,:nocc_b,:nocc_b], backend="jax")) / 2.0
        e2_0 = e2_0_1 + e2_0_2

        # double excitations
        e2_2_1 = e2_0 * gt2g
        lt2g_a = oe.contract("gpq,pq->g", chol_a, 2*t2_green_baa,backend="jax")
        lt2g_b = oe.contract("gpq,pq->g", chol_b, 2*t2_green_bbb + 2*t2_green_bab, backend="jax")
        e2_2_2_1 = -oe.contract('g,g->', lt2g_a + lt2g_b, gl_c, backend="jax") / 2.0

        lt2_green_a = oe.contract("gpi,ji->gpj", rot_chol_a, 2*t2_green_baa, backend="jax")
        lt2_green_b = oe.contract("gpi,ji->gpj", rot_chol_b, 2*t2_green_bbb + 2*t2_green_bab, backend="jax")
        e2_2_2_2 = (oe.contract("gip,gip->", gl_a, lt2_green_a, backend="jax")
                    + oe.contract("gip,gip->", gl_b, lt2_green_b, backend="jax")) / 2
        
        glgp_a = oe.contract("gip,pa->gia", gl_a, greenp_a, backend="jax")
        glgp_b = oe.contract("gip,pa->gia", gl_b, greenp_b, backend="jax")
        l2t2_ba_b = oe.contract("gia,iajb->gjb", glgp_b, t2ba, backend="jax")
        l2t2_bb_b = oe.contract("gia,iajb->gjb", glgp_b, t2bb, backend="jax")
        l2t2_ba = 0.5 * oe.contract("gjb,gjb->", l2t2_ba_b, glgp_a, backend="jax")
        l2t2_bb = 0.5 * oe.contract("gjb,gjb->", l2t2_bb_b, glgp_b, backend="jax")
        e2_2_3 = l2t2_ba + l2t2_bb
        
        e2_2_1 = e2_0 * gt2g
        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <HF|T2 h2|walker>/<HF|walker>

        t2orb = gt2g # <HF|T1+T2|walker>/<HF|walker>
        e12bar = e1_0 + e2_0 # <HF|h1+h2|walker>/<HF|walker>
        t2eorb = e1_2 + e2_2 # <HF|T2(h1+h2)|walker>/<HF|walker>

        return t2eorb, t2orb, e12bar
    
    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
