from abc import ABC
from dataclasses import dataclass
from functools import partial
from typing import Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, jvp, lax, vmap, random
import opt_einsum as oe

class wave_function_unrestricted(ABC):
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

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    # @singledispatchmethod
    # def calc_overlap(self, walkers, wave_data: dict) -> jax.Array:
    #     """Calculate the overlap < psi_t | walker > for a batch of walkers.

    #     Args:
    #         walkers : list or jax.Array
    #             The batched walkers.
    #         wave_data : dict
    #             The trial wave function data.

    #     Returns:
    #         jax.Array: The overlap of the walkers with the trial wave function.
    #     """
    #     raise NotImplementedError("Walker type not supported")

    def calc_overlap(self, walkers: list, wave_data: dict) -> jax.Array:
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
                walkers[0].reshape(self.n_batch, batch_size, self.norb, self.nelec[0]),
                walkers[1].reshape(self.n_batch, batch_size, self.norb, self.nelec[1]),
            ),
        )
        return overlaps.reshape(n_walkers)

    def calc_force_bias(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
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
                walkers[0].reshape(self.n_batch, batch_size, self.norb, self.nelec[0]),
                walkers[1].reshape(self.n_batch, batch_size, self.norb, self.nelec[1]),
            ),
        )
        fbs = jnp.concatenate(fbs, axis=0)
        return fbs.reshape(n_walkers, -1)

    def calc_energy(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
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
                walkers[0].reshape(self.n_batch, batch_size, self.norb, self.nelec[0]),
                walkers[1].reshape(self.n_batch, batch_size, self.norb, self.nelec[1]),
            ),
        )
        return energies.reshape(n_walkers)

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
            return jnp.array(wave_data["rdm1"])
        else:
            return self._calc_rdm1(wave_data)

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

    def decompose_t2(trial, t2, thresh=1e-8):
        # adapted from Yann
        norb = trial.norb
        nocca, noccb = trial.nelec
        nvira, nvirb = (norb - nocca, norb - noccb)

        # Number of excitation pairs
        nex_a = nocca * nvira
        nex_b = noccb * nvirb

        t2aa, t2ab, t2bb = t2

        assert t2aa.shape == (nocca, nvira, nocca, nvira)
        assert t2ab.shape == (nocca, nvira, noccb, nvirb)
        assert t2bb.shape == (noccb, nvirb, noccb, nvirb)

        print('Decomposing Unrestricted T2 amplitudes')

        t2aa = t2aa.reshape(nex_a, nex_a)
        t2ab = t2ab.reshape(nex_a, nex_b)
        t2bb = t2bb.reshape(nex_b, nex_b)

        # Symmetric full t2 
        # [[ t2aa/2  t2ab   ]]
        # [[ t2ab^T  t2bb/2 ]]
        t2full = np.zeros((nex_a + nex_b, nex_a + nex_b))
        t2full[:nex_a, :nex_a] = 0.5 * t2aa
        t2full[nex_a:, :nex_a] = t2ab.T
        t2full[:nex_a, nex_a:] = t2ab
        t2full[nex_a:, nex_a:] = 0.5 * t2bb
        t2full = jnp.array(t2full)

        # t2 = LL^T
        e_val, e_vec = jnp.linalg.eigh(t2full)

        # Keep only important modes
        mask = jnp.abs(e_val) > thresh
        e_val_trunc = e_val[mask]
        e_vec_trunc = e_vec[:, mask]
        
        tau = e_vec_trunc @ jnp.diag(np.sqrt(e_val_trunc + 0.0j))
        err = jnp.linalg.norm(t2full - tau @ tau.T)
        assert err < 10 * thresh
        print(f'Throw {len(e_val)-len(e_val_trunc)} vectors in T2 deomposition')
        print(f'SVD cutoff = {thresh:.2e} | error = {err:.2e}')
        print(f'number of T2 decomposition vectors {len(e_val_trunc)}')

        # alpha/beta operators for HS
        # Summation on the left to have a list of operators
        taua = tau.T[:,:nex_a]
        taub = tau.T[:, nex_a:]
        taua = taua.reshape(-1, nocca, nvira)
        taub = taub.reshape(-1, noccb, nvirb)

        return [taua, taub]
    
    @partial(jit, static_argnums=0)
    def _thouless(self, slater, t):
        # calculate |psi'> = exp(t_ia a+ i)|psi>
        
        slater_up, slater_dn = slater
        ta, tb = t
        
        norb = self.norb
        nocc_a, nocc_b = self.nelec
        nvir_a = norb - nocc_a
        nvir_b = norb - nocc_b
        
        assert ta.shape == (nocc_a, nvir_a)
        assert tb.shape == (nocc_b, nvir_b)

        ta_full = jnp.eye(norb, dtype=jnp.complex128)
        tb_full = jnp.eye(norb, dtype=jnp.complex128)
        exp_ta = ta_full.at[:nocc_a, nocc_a:].set(ta)
        exp_tb = tb_full.at[:nocc_b, nocc_b:].set(tb)
        # exp_tau = jsp.linalg.expm(t_full) 
        slater_ta = exp_ta.T @ slater_up
        slater_tb = exp_tb.T @ slater_dn
        return [slater_ta, slater_tb]

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class uhf(wave_function_unrestricted):
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

    norb: int
    nelec: Tuple[int, int]
    n_opt_iter: int = 30
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: dict,
    ) -> complex:
        return jnp.linalg.det(
            wave_data["mo_coeff"][0].T.conj() @ walker_up
        ) * jnp.linalg.det(wave_data["mo_coeff"][1].T.conj() @ walker_dn)

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
        green_up = (
            walker_up.dot(jnp.linalg.inv(wave_data["mo_coeff"][0].T.conj() @ walker_up))
        ).T
        green_dn = (
            walker_dn.dot(jnp.linalg.inv(wave_data["mo_coeff"][1].T.conj() @ walker_dn))
        ).T
        return [green_up, green_dn]

    @partial(jit, static_argnums=0)
    def _calc_force_bias(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        green_walker = self._calc_green(walker_up, walker_dn, wave_data)
        fb_up = oe.contract("gij,ij->g", ham_data["rot_chol"][0], green_walker[0], backend="jax")
        fb_dn = oe.contract("gij,ij->g", ham_data["rot_chol"][1], green_walker[1], backend="jax")
        return fb_up + fb_dn

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
        green_walker = self._calc_green(walker_up, walker_dn, wave_data)
        ene1 = jnp.sum(green_walker[0] * rot_h1[0]) \
             + jnp.sum(green_walker[1] * rot_h1[1])
        f_up = oe.contract("gij,jk->gik", rot_chol[0], green_walker[0].T,
                           backend="jax")
        f_dn = oe.contract("gij,jk->gik", rot_chol[1], green_walker[1].T,
                           backend="jax")
        c_up = vmap(jnp.trace)(f_up)
        c_dn = vmap(jnp.trace)(f_dn)
        exc_up = jnp.sum(vmap(lambda x: x * x.T)(f_up))
        exc_dn = jnp.sum(vmap(lambda x: x * x.T)(f_dn))
        ene2 = (jnp.sum(c_up * c_up)
              + jnp.sum(c_dn * c_dn)
              + 2.0 * jnp.sum(c_up * c_dn)
              - exc_up - exc_dn) / 2.0

        return ene2 + ene1 + ene0

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        dm_up = wave_data["mo_coeff"][0] @ wave_data["mo_coeff"][0].T.conj()
        dm_dn = wave_data["mo_coeff"][1] @ wave_data["mo_coeff"][1].T.conj()
        return jnp.array([dm_up, dm_dn])

    # @partial(jit, static_argnums=0)
    # def optimize(self, ham_data: dict, wave_data: dict) -> dict:
    #     h1 = ham_data["h1"]
    #     h1 = h1.at[0].set((h1[0] + h1[0].T) / 2.0)
    #     h1 = h1.at[1].set((h1[1] + h1[1].T) / 2.0)
    #     h2 = ham_data["chol"]
    #     h2 = h2.reshape((h2.shape[0], h1.shape[1], h1.shape[1]))
    #     nelec = self.nelec

    #     def scanned_fun(carry, x):
    #         dm = carry
    #         f_up = oe.contract("gij,ik->gjk", h2, dm[0], backend="jax")
    #         c_up = vmap(jnp.trace)(f_up)
    #         vj_up = oe.contract("g,gij->ij", c_up, h2, backend="jax")
    #         vk_up = oe.contract("glj,gjk->lk", f_up, h2, backend="jax")
    #         f_dn = oe.contract("gij,ik->gjk", h2, dm[1], backend="jax")
    #         c_dn = vmap(jnp.trace)(f_dn)
    #         vj_dn = oe.contract("g,gij->ij", c_dn, h2, backend="jax")
    #         vk_dn = oe.contract("glj,gjk->lk", f_dn, h2, backend="jax")
    #         fock_up = h1[0] + vj_up + vj_dn - vk_up
    #         fock_dn = h1[1] + vj_up + vj_dn - vk_dn
    #         mo_energy_up, mo_coeff_up = linalg_utils._eigh(fock_up)
    #         mo_energy_dn, mo_coeff_dn = linalg_utils._eigh(fock_dn)

    #         nmo = mo_energy_up.size

    #         idx_up = jnp.argmax(abs(mo_coeff_up.real), axis=0)
    #         mo_coeff_up = jnp.where(
    #             mo_coeff_up[idx_up, jnp.arange(len(mo_energy_up))].real < 0,
    #             -mo_coeff_up,
    #             mo_coeff_up,
    #         )
    #         e_idx_up = jnp.argsort(mo_energy_up)
    #         mo_occ_up = jnp.zeros(nmo)
    #         nocc_up = nelec[0]
    #         mo_occ_up = mo_occ_up.at[e_idx_up[:nocc_up]].set(1)
    #         mocc_up = mo_coeff_up[:, jnp.nonzero(mo_occ_up, size=nocc_up)[0]]
    #         dm_up = (mocc_up * mo_occ_up[jnp.nonzero(mo_occ_up, size=nocc_up)[0]]).dot(
    #             mocc_up.T
    #         )

    #         idx_dn = jnp.argmax(abs(mo_coeff_dn.real), axis=0)
    #         mo_coeff_dn = jnp.where(
    #             mo_coeff_dn[idx_dn, jnp.arange(len(mo_energy_dn))].real < 0,
    #             -mo_coeff_dn,
    #             mo_coeff_dn,
    #         )
    #         e_idx_dn = jnp.argsort(mo_energy_dn)
    #         mo_occ_dn = jnp.zeros(nmo)
    #         nocc_dn = nelec[1]
    #         mo_occ_dn = mo_occ_dn.at[e_idx_dn[:nocc_dn]].set(1)
    #         mocc_dn = mo_coeff_dn[:, jnp.nonzero(mo_occ_dn, size=nocc_dn)[0]]
    #         dm_dn = (mocc_dn * mo_occ_dn[jnp.nonzero(mo_occ_dn, size=nocc_dn)[0]]).dot(
    #             mocc_dn.T
    #         )

    #         return jnp.array([dm_up, dm_dn]), jnp.array([mo_coeff_up, mo_coeff_dn])

    #     dm0 = self._calc_rdm1(wave_data)
    #     _, mo_coeff = lax.scan(scanned_fun, dm0, None, length=self.n_opt_iter)

    #     wave_data["mo_coeff"] = [
    #         mo_coeff[-1][0][:, : nelec[0]],
    #         mo_coeff[-1][1][:, : nelec[1]],
    #     ]
    #     return wave_data

    @partial(jit, static_argnums=(0,))
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        ham_data["rot_h1"] = [
            wave_data["mo_coeff"][0].T.conj() @ ham_data["h1"][0],
            wave_data["mo_coeff"][1].T.conj() @ ham_data["h1"][1],
        ]
        ham_data["rot_chol"] = [
            oe.contract(
                "pi,gij->gpj",
                wave_data["mo_coeff"][0].T.conj(),
                ham_data["chol"][0].reshape(-1, self.norb, self.norb), 
                backend="jax"),
            oe.contract(
                "pi,gij->gpj",
                wave_data["mo_coeff"][1].T.conj(),
                ham_data["chol"][1].reshape(-1, self.norb, self.norb), 
                backend="jax")]
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class ucisd(wave_function_unrestricted):
    """A manual implementation of the UCISD wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        noccA, noccB = self.nelec[0], self.nelec[1]
        dm_up = (wave_data["mo_coeff"][0][:,:noccA] 
                 @ wave_data["mo_coeff"][0][:,:noccA].T.conj())
        dm_dn = (wave_data["mo_coeff"][1][:,:noccB] 
                 @ wave_data["mo_coeff"][1][:,:noccB].T.conj())
        return jnp.array([dm_up, dm_dn])

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> complex:
        noccA, ci1A, ci2AA = self.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
        noccB, ci1B, ci2BB = self.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
        ci2AB = wave_data["ci2AB"]
        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[: noccA, :]))).T
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[: noccB, :]))).T
        green_a, green_b = green_a[:, noccA:], green_b[:, noccB:]
        o0 = jnp.linalg.det(walker_up[:noccA, :]) * jnp.linalg.det(walker_dn[:noccB, :])
        o1 = oe.contract("ia,ia", ci1A, green_a, backend="jax") \
            + oe.contract("ia,ia", ci1B, green_b, backend="jax")
        o2 = 0.5 * oe.contract("iajb, ia, jb", ci2AA, green_a, green_a, backend="jax")\
            + 0.5 * oe.contract("iajb, ia, jb", ci2BB, green_b, green_b, backend="jax")\
            + oe.contract("iajb, ia, jb", ci2AB, green_a, green_b, backend="jax")
        return (1.0 + o1 + o2) * o0

    @partial(jit, static_argnums=0)
    def _calc_force_bias(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
        nocc_a, ci1_a, ci2_aa = self.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
        nocc_b, ci1_b, ci2_bb = self.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
        ci2_ab = wave_data["ci2AB"]
        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a, :]))).T
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[:nocc_b, :]))).T
        green_occ_a = green_a[:, nocc_a:].copy()
        green_occ_b = green_b[:, nocc_b:].copy()
        greenp_a = jnp.vstack((green_occ_a, -jnp.eye(self.norb - nocc_a)))
        greenp_b = jnp.vstack((green_occ_b, -jnp.eye(self.norb - nocc_b)))

        chol_a = ham_data["chol"][0].reshape(-1, self.norb, self.norb)
        chol_b = ham_data["chol"][1].reshape(-1, self.norb, self.norb)
        rot_chol_a = chol_a[:, : self.nelec[0], :]
        rot_chol_b = chol_b[:, : self.nelec[1], :]
        lg_a = oe.contract("gpj,pj->g", rot_chol_a, green_a, backend="jax")
        lg_b = oe.contract("gpj,pj->g", rot_chol_b, green_b, backend="jax")
        lg = lg_a + lg_b

        # ref
        fb_0 = lg_a + lg_b

        # single excitations
        ci1g_a = oe.contract("pt,pt->", ci1_a, green_occ_a, backend="jax")
        ci1g_b = oe.contract("pt,pt->", ci1_b, green_occ_b, backend="jax")
        ci1g = ci1g_a + ci1g_b
        fb_1_1 = ci1g * lg
        ci1gp_a = oe.contract("pt,it->pi", ci1_a, greenp_a, backend="jax")
        ci1gp_b = oe.contract("pt,it->pi", ci1_b, greenp_b, backend="jax")
        gci1gp_a = oe.contract("pj,pi->ij", green_a, ci1gp_a, backend="jax")
        gci1gp_b = oe.contract("pj,pi->ij", green_b, ci1gp_b, backend="jax")
        fb_1_2 = -oe.contract(
            "gij,ij->g", chol_a, gci1gp_a, backend="jax")\
                - oe.contract("gij,ij->g", chol_b, gci1gp_b, backend="jax")
        fb_1 = fb_1_1 + fb_1_2

        # double excitations
        ci2g_a = oe.contract("ptqu,pt->qu", ci2_aa, green_occ_a, backend="jax")
        ci2g_b = oe.contract("ptqu,pt->qu", ci2_bb, green_occ_b, backend="jax")
        ci2g_ab_a = oe.contract("ptqu,qu->pt", ci2_ab, green_occ_b, backend="jax")
        ci2g_ab_b = oe.contract("ptqu,pt->qu", ci2_ab, green_occ_a, backend="jax")
        gci2g_a = 0.5 * oe.contract("qu,qu->", ci2g_a, green_occ_a, backend="jax")
        gci2g_b = 0.5 * oe.contract("qu,qu->", ci2g_b, green_occ_b, backend="jax")
        gci2g_ab = oe.contract("pt,pt->", ci2g_ab_a, green_occ_a, backend="jax")
        gci2g = gci2g_a + gci2g_b + gci2g_ab
        fb_2_1 = lg * gci2g
        ci2_green_a = (greenp_a @ (ci2g_a + ci2g_ab_a).T) @ green_a
        ci2_green_b = (greenp_b @ (ci2g_b + ci2g_ab_b).T) @ green_b
        fb_2_2_a = -oe.contract("gij,ij->g", chol_a, ci2_green_a, backend="jax")
        fb_2_2_b = -oe.contract("gij,ij->g", chol_b, ci2_green_b, backend="jax")
        fb_2_2 = fb_2_2_a + fb_2_2_b
        fb_2 = fb_2_1 + fb_2_2

        # overlap
        overlap_1 = ci1g
        overlap_2 = gci2g
        overlap = 1.0 + overlap_1 + overlap_2

        return (fb_0 + fb_1 + fb_2) / overlap

    @partial(jit, static_argnums=0)
    def _calc_energy(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
        nocc_a, ci1_a, ci2_aa = self.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
        nocc_b, ci1_b, ci2_bb = self.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
        ci2_ab = wave_data["ci2AB"]
        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a, :]))).T
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[:nocc_b, :]))).T
        green_occ_a = green_a[:, nocc_a:].copy()
        green_occ_b = green_b[:, nocc_b:].copy()
        greenp_a = jnp.vstack((green_occ_a, -jnp.eye(self.norb - nocc_a)))
        greenp_b = jnp.vstack((green_occ_b, -jnp.eye(self.norb - nocc_b)))

        chol_a = ham_data["chol"][0].reshape(-1, self.norb, self.norb)
        chol_b = ham_data["chol"][1].reshape(-1, self.norb, self.norb)
        rot_chol_a = chol_a[:, :nocc_a, :]
        rot_chol_b = chol_b[:, :nocc_b, :]
        h1_a = ham_data["h1"][0]
        h1_b = ham_data["h1"][1]
        hg_a = oe.contract("pj,pj->", h1_a[:nocc_a, :], green_a, backend="jax")
        hg_b = oe.contract("pj,pj->", h1_b[:nocc_b, :], green_b, backend="jax")
        hg = hg_a + hg_b

        # 0 body energy
        e0 = ham_data["h0"]

        # 1 body energy
        # ref
        e1_0 = hg

        # single excitations
        ci1g_a = oe.contract("pt,pt->", ci1_a, green_occ_a, backend="jax")
        ci1g_b = oe.contract("pt,pt->", ci1_b, green_occ_b, backend="jax")
        ci1g = ci1g_a + ci1g_b
        e1_1_1 = ci1g * hg
        gpci1_a = greenp_a @ ci1_a.T
        gpci1_b = greenp_b @ ci1_b.T
        ci1_green_a = gpci1_a @ green_a
        ci1_green_b = gpci1_b @ green_b
        e1_1_2 = -(
            oe.contract("ij,ij->", h1_a, ci1_green_a, backend="jax")
            + oe.contract("ij,ij->", h1_b, ci1_green_b, backend="jax")
        )
        e1_1 = e1_1_1 + e1_1_2

        # double excitations
        ci2g_a = oe.contract("ptqu,pt->qu", ci2_aa, green_occ_a, backend="jax") / 4
        ci2g_b = oe.contract("ptqu,pt->qu", ci2_bb, green_occ_b, backend="jax") / 4
        ci2g_ab_a = oe.contract("ptqu,qu->pt", ci2_ab, green_occ_b, backend="jax")
        ci2g_ab_b = oe.contract("ptqu,pt->qu", ci2_ab, green_occ_a, backend="jax")
        gci2g_a = oe.contract("qu,qu->", ci2g_a, green_occ_a, backend="jax")
        gci2g_b = oe.contract("qu,qu->", ci2g_b, green_occ_b, backend="jax")
        gci2g_ab = oe.contract("pt,pt->", ci2g_ab_a, green_occ_a, backend="jax")
        gci2g = 2 * (gci2g_a + gci2g_b) + gci2g_ab
        e1_2_1 = hg * gci2g
        ci2_green_a = (greenp_a @ ci2g_a.T) @ green_a
        ci2_green_ab_a = (greenp_a @ ci2g_ab_a.T) @ green_a
        ci2_green_b = (greenp_b @ ci2g_b.T) @ green_b
        ci2_green_ab_b = (greenp_b @ ci2g_ab_b.T) @ green_b
        e1_2_2_a = -oe.contract(
            "ij,ij->", h1_a, 4 * ci2_green_a + ci2_green_ab_a, backend="jax")
        e1_2_2_b = -oe.contract(
            "ij,ij->", h1_b, 4 * ci2_green_b + ci2_green_ab_b, backend="jax")
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2

        e1 = e1_0 + e1_1 + e1_2

        # two body energy
        # ref
        lg_a = oe.contract("gpj,pj->g", rot_chol_a, green_a, backend="jax")
        lg_b = oe.contract("gpj,pj->g", rot_chol_b, green_b, backend="jax")
        e2_0_1 = ((lg_a + lg_b) @ (lg_a + lg_b)) / 2.0
        lg1_a = oe.contract("gpj,qj->gpq", rot_chol_a, green_a, backend="jax")
        lg1_b = oe.contract("gpj,qj->gpq", rot_chol_b, green_b, backend="jax")
        e2_0_2 = (
            -(
                jnp.sum(vmap(lambda x: x * x.T)(lg1_a))
                + jnp.sum(vmap(lambda x: x * x.T)(lg1_b))
            )
            / 2.0
        )
        e2_0 = e2_0_1 + e2_0_2

        # single excitations
        e2_1_1 = e2_0 * ci1g
        lci1g_a = oe.contract("gij,ij->g", chol_a, ci1_green_a, backend="jax")
        lci1g_b = oe.contract("gij,ij->g", chol_b, ci1_green_b, backend="jax")
        e2_1_2 = -((lci1g_a + lci1g_b) @ (lg_a + lg_b))
        ci1g1_a = ci1_a @ green_occ_a.T
        ci1g1_b = ci1_b @ green_occ_b.T
        e2_1_3_1 = oe.contract(
            "gpq,gqr,rp->", lg1_a, lg1_a, ci1g1_a, backend="jax"
        ) + oe.contract("gpq,gqr,rp->", lg1_b, lg1_b, ci1g1_b, backend="jax")
        lci1g_a = oe.contract(
            "gip,qi->gpq", ham_data["lci1_a"], green_a, backend="jax"
        )
        lci1g_b = oe.contract(
            "gip,qi->gpq", ham_data["lci1_b"], green_b, backend="jax"
        )
        e2_1_3_2 = -oe.contract(
            "gpq,gqp->", lci1g_a, lg1_a, backend="jax"
        ) - oe.contract("gpq,gqp->", lci1g_b, lg1_b, backend="jax")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + e2_1_2 + e2_1_3

        # double excitations
        e2_2_1 = e2_0 * gci2g
        lci2g_a = oe.contract("gij,ij->g",
            chol_a, 8 * ci2_green_a + 2 * ci2_green_ab_a, backend="jax")
        lci2g_b = oe.contract("gij,ij->g",
            chol_b, 8 * ci2_green_b + 2 * ci2_green_ab_b, backend="jax")
        e2_2_2_1 = -((lci2g_a + lci2g_b) @ (lg_a + lg_b)) / 2.0

        def scanned_fun(carry, x):
            chol_a_i, rot_chol_a_i, chol_b_i, rot_chol_b_i = x
            gl_a_i = oe.contract("pj,ji->pi", green_a, chol_a_i, backend="jax")
            gl_b_i = oe.contract("pj,ji->pi", green_b, chol_b_i, backend="jax")
            lci2_green_a_i = oe.contract(
                "pi,ji->pj",
                rot_chol_a_i,
                8 * ci2_green_a + 2 * ci2_green_ab_a, backend="jax"
            )
            lci2_green_b_i = oe.contract(
                "pi,ji->pj",
                rot_chol_b_i,
                8 * ci2_green_b + 2 * ci2_green_ab_b, backend="jax"
            )
            carry[0] += 0.5 * (
                oe.contract("pi,pi->", gl_a_i, lci2_green_a_i, backend="jax")
                + oe.contract("pi,pi->", gl_b_i, lci2_green_b_i, backend="jax")
            )
            glgp_a_i = oe.contract(
                "pi,it->pt", gl_a_i, greenp_a, backend="jax"
            )
            glgp_b_i = oe.contract(
                "pi,it->pt", gl_b_i, greenp_b, backend="jax"
            )
            l2ci2_a = 0.5 * oe.contract(
                "pt,qu,ptqu->",
                glgp_a_i, glgp_a_i, ci2_aa, backend="jax")
            l2ci2_b = 0.5 * oe.contract(
                "pt,qu,ptqu->",
                glgp_b_i, glgp_b_i, ci2_bb, backend="jax")
            l2ci2_ab = oe.contract(
                "pt,qu,ptqu->",
                glgp_a_i, glgp_b_i, ci2_ab, backend="jax")
            carry[1] += l2ci2_a + l2ci2_b + l2ci2_ab
            return carry, 0.0

        [e2_2_2_2, e2_2_3], _ = lax.scan(
            scanned_fun, [0.0, 0.0], (chol_a, rot_chol_a, chol_b, rot_chol_b)
        )
        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3

        e2 = e2_0 + e2_1 + e2_2

        # overlap
        overlap_1 = ci1g
        overlap_2 = gci2g
        overlap = 1.0 + overlap_1 + overlap_2
        return (e1 + e2) / overlap + e0

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["lci1_a"] = oe.contract(
            "git,pt->gip",
            ham_data["chol"][0].reshape(-1, self.norb, self.norb)[:, :, self.nelec[0] :],
            wave_data["ci1A"],
            backend="jax")
        ham_data["lci1_b"] = oe.contract(
            "git,pt->gip",
            ham_data["chol"][1].reshape(-1, self.norb, self.norb)[:, :, self.nelec[1] :],
            wave_data["ci1B"],
            backend="jax")
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))



@dataclass
class uccsd_pt(uhf):
    """A manual implementation of the UCCSD_PT wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_energy_pt(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
        nocc_a, t1_a, t2_aa = self.nelec[0], wave_data["t1a"], wave_data["t2aa"]
        nocc_b, t1_b, t2_bb = self.nelec[1], wave_data["t1b"], wave_data["t2bb"]
        t2_ab = wave_data["t2ab"]
        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a, :]))).T
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[:nocc_b, :]))).T
        green_occ_a = green_a[:, nocc_a:].copy()
        green_occ_b = green_b[:, nocc_b:].copy()
        greenp_a = jnp.vstack((green_occ_a, -jnp.eye(self.norb - nocc_a)))
        greenp_b = jnp.vstack((green_occ_b, -jnp.eye(self.norb - nocc_b)))

        chol_a = ham_data["chol"][0].reshape(-1, self.norb, self.norb)
        chol_b = ham_data["chol"][1].reshape(-1, self.norb, self.norb)
        rot_chol_a = chol_a[:, :nocc_a, :]
        rot_chol_b = chol_b[:, :nocc_b, :]
        h1_a = ham_data["h1"][0]
        h1_b = ham_data["h1"][1]
        hg_a = oe.contract("pj,pj->", h1_a[:nocc_a, :], green_a, backend="jax")
        hg_b = oe.contract("pj,pj->", h1_b[:nocc_b, :], green_b, backend="jax")
        hg = hg_a + hg_b

        # 0 body energy
        h0 = ham_data["h0"]

        # 1 body energy
        # ref
        e1_0 = hg # <HF|h1|walker>/<HF|walker>

        # single excitations
        t1g_a = oe.contract("ia,ia->", t1_a, green_occ_a, backend="jax")
        t1g_b = oe.contract("ia,ia->", t1_b, green_occ_b, backend="jax")
        t1g = t1g_a + t1g_b
        e1_1_1 = t1g * hg
        gpt1_a = greenp_a @ t1_a.T
        gpt1_b = greenp_b @ t1_b.T
        t1_green_a = gpt1_a @ green_a
        t1_green_b = gpt1_b @ green_b
        e1_1_2 = -(
            oe.contract("pq,pq->", h1_a, t1_green_a, backend="jax")
            + oe.contract("pq,pq->", h1_b, t1_green_b, backend="jax")
        )
        e1_1 = e1_1_1 + e1_1_2 # <HF|T1 h1|walker>/<HF|walker>

        # double excitations
        t2g_a = oe.contract("ptqu,pt->qu", t2_aa, green_occ_a, backend="jax") / 4
        t2g_b = oe.contract("ptqu,pt->qu", t2_bb, green_occ_b, backend="jax") / 4
        t2g_ab_a = oe.contract("ptqu,qu->pt", t2_ab, green_occ_b, backend="jax")
        t2g_ab_b = oe.contract("ptqu,pt->qu", t2_ab, green_occ_a, backend="jax")
        gt2g_a = oe.contract("qu,qu->", t2g_a, green_occ_a, backend="jax")
        gt2g_b = oe.contract("qu,qu->", t2g_b, green_occ_b, backend="jax")
        gt2g_ab = oe.contract("pt,pt->", t2g_ab_a, green_occ_a, backend="jax")
        gt2g = 2 * (gt2g_a + gt2g_b) + gt2g_ab
        e1_2_1 = hg * gt2g
        t2_green_a = (greenp_a @ t2g_a.T) @ green_a
        t2_green_ab_a = (greenp_a @ t2g_ab_a.T) @ green_a
        t2_green_b = (greenp_b @ t2g_b.T) @ green_b
        t2_green_ab_b = (greenp_b @ t2g_ab_b.T) @ green_b
        e1_2_2_a = -oe.contract(
            "ij,ij->", h1_a, 4 * t2_green_a + t2_green_ab_a, backend="jax"
        )
        e1_2_2_b = -oe.contract(
            "ij,ij->", h1_b, 4 * t2_green_b + t2_green_ab_b, backend="jax"
        )
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2 # <HF|T2 h1|walker>/<HF|walker>

        # two body energy
        # ref
        lg_a = oe.contract("gpj,pj->g", rot_chol_a, green_a, backend="jax")
        lg_b = oe.contract("gpj,pj->g", rot_chol_b, green_b, backend="jax")
        e2_0_1 = ((lg_a + lg_b) @ (lg_a + lg_b)) / 2.0
        lg1_a = oe.contract("gpj,qj->gpq", rot_chol_a, green_a, backend="jax")
        lg1_b = oe.contract("gpj,qj->gpq", rot_chol_b, green_b, backend="jax")
        e2_0_2 = (
            -(
                jnp.sum(vmap(lambda x: x * x.T)(lg1_a))
                + jnp.sum(vmap(lambda x: x * x.T)(lg1_b))
            )
            / 2.0
        )
        e2_0 = e2_0_1 + e2_0_2 # <HF|h2|walker>/<HF|walker>

        # single excitations
        e2_1_1 = e2_0 * t1g
        lt1g_a = oe.contract("gij,ij->g", chol_a, t1_green_a, backend="jax")
        lt1g_b = oe.contract("gij,ij->g", chol_b, t1_green_b, backend="jax")
        e2_1_2 = -((lt1g_a + lt1g_b) @ (lg_a + lg_b))
        t1g1_a = t1_a @ green_occ_a.T
        t1g1_b = t1_b @ green_occ_b.T
        e2_1_3_1 = oe.contract(
            "gpq,gqr,rp->", lg1_a, lg1_a, t1g1_a, backend="jax"
        ) + oe.contract("gpq,gqr,rp->", lg1_b, lg1_b, t1g1_b, backend="jax")
        lt1g_a = oe.contract(
            "gip,qi->gpq", ham_data["lt1_a"], green_a, backend="jax"
        )
        lt1g_b = oe.contract(
            "gip,qi->gpq", ham_data["lt1_b"], green_b, backend="jax"
        )
        e2_1_3_2 = -oe.contract(
            "gpq,gqp->", lt1g_a, lg1_a, backend="jax"
        ) - oe.contract("gpq,gqp->", lt1g_b, lg1_b, backend="jax")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + e2_1_2 + e2_1_3 # <HF|T1 h2|walker>/<HF|walker>

        # double excitations
        e2_2_1 = e2_0 * gt2g
        lt2g_a = oe.contract(
            "gij,ij->g",
            chol_a,
            8 * t2_green_a + 2 * t2_green_ab_a,
            backend="jax",
        )
        lt2g_b = oe.contract(
            "gij,ij->g",
            chol_b,
            8 * t2_green_b + 2 * t2_green_ab_b,
            backend="jax",
        )
        e2_2_2_1 = -((lt2g_a + lt2g_b) @ (lg_a + lg_b)) / 2.0

        def scanned_fun(carry, x):
            chol_a_i, rot_chol_a_i, chol_b_i, rot_chol_b_i = x
            gl_a_i = oe.contract("pj,ji->pi", green_a, chol_a_i, backend="jax")
            gl_b_i = oe.contract("pj,ji->pi", green_b, chol_b_i, backend="jax")
            lt2_green_a_i = oe.contract(
                "pi,ji->pj",
                rot_chol_a_i,
                8 * t2_green_a + 2 * t2_green_ab_a,
                backend="jax",
            )
            lt2_green_b_i = oe.contract(
                "pi,ji->pj",
                rot_chol_b_i,
                8 * t2_green_b + 2 * t2_green_ab_b,
                backend="jax",
            )
            carry[0] += 0.5 * (
                oe.contract("pi,pi->", gl_a_i, lt2_green_a_i, backend="jax")
                + oe.contract("pi,pi->", gl_b_i, lt2_green_b_i, backend="jax")
            )
            glgp_a_i = oe.contract(
                "pi,it->pt", gl_a_i, greenp_a, backend="jax"
            )
            glgp_b_i = oe.contract(
                "pi,it->pt", gl_b_i, greenp_b, backend="jax"
            )
            l2t2_a = 0.5 * oe.contract(
                "pt,qu,ptqu->",
                glgp_a_i,
                glgp_a_i,
                t2_aa,
                backend="jax",
            )
            l2t2_b = 0.5 * oe.contract(
                "pt,qu,ptqu->",
                glgp_b_i,
                glgp_b_i,
                t2_bb,
                backend="jax",
            )
            l2t2_ab = oe.contract(
                "pt,qu,ptqu->",
                glgp_a_i,
                glgp_b_i,
                t2_ab,
                backend="jax",
            )
            carry[1] += l2t2_a + l2t2_b + l2t2_ab
            return carry, 0.0

        [e2_2_2_2, e2_2_3], _ = lax.scan(
            scanned_fun, [0.0, 0.0], (chol_a, rot_chol_a, chol_b, rot_chol_b)
        )
        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <HF|T2 h2|walker>/<HF|walker>

        t = t1g + gt2g # <HF|T1+T2|walker>/<HF|walker>
        e0 = h0 + e1_0 + e2_0 # h0 + <HF|h1+h2|walker>/<HF|walker>
        e1 = e1_1 + e1_2 + e2_1 + e2_2 # <HF|(T1+T2)(h1+h2)|walker>/<HF|walker>

        return t, e0, e1

    def calc_energy_pt(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
        t, e0, e1 = vmap(
            self._calc_energy_pt, in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data, wave_data)
        return t, e0, e1

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        ham_data["rot_h1"] = [
            wave_data["mo_coeff"][0].T.conj() @ ham_data["h1"][0],
            wave_data["mo_coeff"][1].T.conj() @ ham_data["h1"][1],
        ]
        ham_data["rot_chol"] = [
            oe.contract(
                "pi,gij->gpj",
                wave_data["mo_coeff"][0].T.conj(),
                ham_data["chol"][0].reshape(-1, self.norb, self.norb), backend="jax"
            ),
            oe.contract(
                "pi,gij->gpj",
                wave_data["mo_coeff"][1].T.conj(),
                ham_data["chol"][1].reshape(-1, self.norb, self.norb), backend="jax"
            ),
        ]
        ham_data["lt1_a"] = oe.contract(
            "gpa,ia->gpi",
            ham_data["chol"][0].reshape(-1, self.norb, self.norb)[:, :, self.nelec[0] :],
            wave_data["t1a"],
            backend="jax"
        )
        ham_data["lt1_b"] = oe.contract(
            "gpa,ia->gpi",
            ham_data["chol"][1].reshape(-1, self.norb, self.norb)[:, :, self.nelec[1] :],
            wave_data["t1b"],
            backend="jax"
        )
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class uccsd_pt2(uhf):
    """Tensor contraction form of the UCCSD_PT2 (exact T1) trial wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1


    def hs_op(self, wave_data: dict): # t2aa, t2ab, t2bb) -> dict:
        # adapted from Yann

        nOa, nOb = self.nelec
        n = self.norb
        nVa, nVb = (n - nOa, n - nOb)

        # Number of excitations
        nex_a = nOa * nVa
        nex_b = nOb * nVb

        t2aa = wave_data['t2aa']
        t2ab = wave_data["t2ab"]
        t2bb = wave_data["t2bb"]

        assert n == nOb + nVb
        assert t2aa.shape == (nOa, nOa, nVa, nVa)
        assert t2ab.shape == (nOa, nOb, nVa, nVb)
        assert t2bb.shape == (nOb, nOb, nVb, nVb)

        # t2(i,j,a,b) -> t2(ai,bj)
        t2aa = jnp.einsum("ijab->aibj", t2aa)
        t2ab = jnp.einsum("ijab->aibj", t2ab)
        t2bb = jnp.einsum("ijab->aibj", t2bb)

        t2aa = t2aa.reshape(nex_a, nex_a)
        t2ab = t2ab.reshape(nex_a, nex_b)
        t2bb = t2bb.reshape(nex_b, nex_b)

        # Symmetric t2 =
        # t2aa/2 t2ab
        # t2ab^T t2bb
        t2 = np.zeros((nex_a + nex_b, nex_a + nex_b))
        t2[:nex_a, :nex_a] = 0.5 * t2aa
        t2[nex_a:, :nex_a] = t2ab.T
        t2[:nex_a, nex_a:] = t2ab
        t2[nex_a:, nex_a:] = 0.5 * t2bb

        # t2 = LL^T
        e_val, e_vec = jnp.linalg.eigh(t2)
        L = e_vec @ jnp.diag(np.sqrt(e_val + 0.0j))
        assert abs(jnp.linalg.norm(t2 - L @ L.T)) < 1e-12

        # alpha/beta operators for HS
        # Summation on the left to have a list of operators
        La = L[:nex_a, :]
        Lb = L[nex_a:, :]
        La = La.T.reshape(nex_a + nex_b, nVa, nOa)
        Lb = Lb.T.reshape(nex_a + nex_b, nVb, nOb)

        # wave_data["T2_La"] = La
        # wave_data["T2_Lb"] = Lb

        return [La, Lb]
    
    @partial(jax.jit, static_argnums=(0, 2))
    def get_stocc(self, wave_data: dict, nslater: int):
        # adapted from Yann

        nOa, nOb = self.nelec
        # nVa, nVb = self.nvirt
        n = self.norb
        # nVa, nVb = (n - nOa, n - nOb)

        # nex_a = nOa * nVa
        # nex_b = nOb * nVb

        t1a = wave_data["t1a"]
        t1b = wave_data["t1b"]

        Ca_occ, Ca_vir = jnp.split(wave_data["mo_coeff"][0], [nOa], axis=1)
        Cb_occ, Cb_vir = jnp.split(wave_data["mo_coeff"][1], [nOb], axis=1)

        # e^T1
        e_t1a = t1a.T + 0.0j
        e_t1b = t1b.T + 0.0j

        ops_a = jnp.array([e_t1a] * nslater)
        ops_b = jnp.array([e_t1b] * nslater)

        # La = wave_data["T2_La"]
        # Lb = wave_data["T2_Lb"]
        La, Lb = self.hs_op(wave_data)

        wave_data["key"], subkey = random.split(wave_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                nslater,
                La.shape[0],
            ),
        )

        # e^{T1+T2}
        ops_a = ops_a + jnp.einsum("wg,gai->wai", fields, La)
        ops_b = ops_b + jnp.einsum("wg,gai->wai", fields, Lb)

        # Initial determinants
        rdm1 = self.get_rdm1(wave_data)
        natorbs_up = jnp.linalg.eigh(rdm1[0])[1][:, ::-1][:, :nOa]
        natorbs_dn = jnp.linalg.eigh(rdm1[1])[1][:, ::-1][:, :nOb]

        stocc_a = jnp.array([natorbs_up + 0.0j] * nslater)
        stocc_b = jnp.array([natorbs_dn + 0.0j] * nslater)

        id_a = jnp.array([np.identity(n) + 0.0j] * nslater)
        id_b = jnp.array([np.identity(n) + 0.0j] * nslater)

        # e^{T1+T2} \ket{\phi}
        stocc_a = (
            id_a + jnp.einsum("pa,wai,iq -> wpq", Ca_vir, ops_a, Ca_occ.T)
        ) @ stocc_a
        stocc_b = (
            id_b + jnp.einsum("pa,wai,iq -> wpq", Cb_vir, ops_b, Cb_occ.T)
        ) @ stocc_b

        stocc_a = jnp.array(stocc_a)
        stocc_b = jnp.array(stocc_b)

        # stocc = UHFWalkers([walkers_a, walkers_b])
        return [stocc_a, stocc_b]
    

    @partial(jit, static_argnums=0)
    def thouless_trans(self, t1):
        ''' thouless transformation |psi'> = exp(t1)|psi>
            gives the transformed mo_occrep in the 
            original mo basis <psi_p|psi'_i>
            t = t_ia
            t_ia = c_ik c.T_ka
            c_ik = <psi_i|psi'_k>
        '''
        q, r = jnp.linalg.qr(t1,mode='complete')
        u_ji = q
        u_ai = r.T
        mo_t = jnp.vstack((u_ji,u_ai))
        mo_t, _ = jnp.linalg.qr(mo_t)# ,mode='complete')
        # this sgn is a problem when
        # turn on mol point group symmetry
        # sgn = jnp.sign((mo_t).diagonal())
        # choose the mo_t s.t it has positive olp with the original mo
        # <psi'_i|psi_i>
        # mo_t = jnp.einsum("ij,j->ij", mo_t, sgn)
        return mo_t
    
    @partial(jit, static_argnums=0)
    def _calc_energy_pt(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
        nocc_a, t2_aa = self.nelec[0], wave_data["t2aa"]
        nocc_b, t2_bb = self.nelec[1], wave_data["t2bb"]
        t2_ab = wave_data["t2ab"]
        mo_a, mo_b = wave_data['mo_ta'], wave_data['mo_tb']
        chol_a = ham_data["chol"][0].reshape(-1, self.norb, self.norb)
        chol_b = ham_data["chol"][1].reshape(-1, self.norb, self.norb)
        h1_a = ham_data["h1"][0]
        h1_b = ham_data["h1"][1]

        # full green's function G_pq
        green_a = (walker_up @ (jnp.linalg.inv(mo_a.T @ walker_up)) @ mo_a.T).T
        green_b = (walker_dn @ (jnp.linalg.inv(mo_b.T @ walker_dn)) @ mo_b.T).T
        greenp_a = (green_a - jnp.eye(self.norb))[:,nocc_a:]
        greenp_b = (green_b - jnp.eye(self.norb))[:,nocc_b:]

        hg_a = oe.contract("pq,pq->", h1_a, green_a, backend="jax")
        hg_b = oe.contract("pq,pq->", h1_b, green_b, backend="jax")
        hg = hg_a + hg_b # <exp(T1)HF|h1|walker>/<exp(T1)HF|walker>

        # <exp(T1)HF|h1|walker>/<exp(T1)HF|walker>
        # one body energy
        e1_0 = hg

        # <exp(T1)HF|T2 h1|walker>/<exp(T1)HF|walker>
        # double excitations
        t2g_a = oe.contract("iajb,ia->jb", t2_aa, green_a[:nocc_a,nocc_a:],
                            backend="jax") / 4
        t2g_b = oe.contract("iajb,ia->jb", t2_bb, green_b[:nocc_b,nocc_b:], 
                            backend="jax") / 4
        t2g_ab_a = oe.contract("iajb,jb->ia", t2_ab, green_b[:nocc_b,nocc_b:],
                               backend="jax")
        t2g_ab_b = oe.contract("iajb,ia->jb", t2_ab, green_a[:nocc_a,nocc_a:],
                               backend="jax")
        # t_iajb (G_ia G_jb - G_ib G_ja)
        gt2g_a = oe.contract("jb,jb->", t2g_a, green_a[:nocc_a,nocc_a:], 
                            backend="jax")
        gt2g_b = oe.contract("jb,jb->", t2g_b, green_b[:nocc_b,nocc_b:], 
                            backend="jax")
        gt2g_ab = oe.contract("ia,ia->", t2g_ab_a, green_a[:nocc_a,nocc_a:], 
                              backend="jax")
        gt2g = 2 * (gt2g_a + gt2g_b) + gt2g_ab # <exp(T1)HF|T2|walker>/<exp(T1)HF|walker>

        e1_2_1 = hg * gt2g
        
        t2_green_a = (greenp_a @ t2g_a.T) @ green_a[:nocc_a,:] # Gp_pb t_iajb G_ia G_jq
        t2_green_ab_a = (greenp_a @ t2g_ab_a.T) @ green_a[:nocc_a,:]
        t2_green_b = (greenp_b @ t2g_b.T) @ green_b[:nocc_b,:]
        t2_green_ab_b = (greenp_b @ t2g_ab_b.T) @ green_b[:nocc_b,:]
        e1_2_2_a = -oe.contract(
            "pq,pq->", h1_a, 4 * t2_green_a + t2_green_ab_a, backend="jax")
        e1_2_2_b = -oe.contract(
            "pq,pq->", h1_b, 4 * t2_green_b + t2_green_ab_b, backend="jax")
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2  # <exp(T1)HF|T2 h1|walker>/<exp(T1)HF|walker>

        # <exp(T1)HF|T2 h2|walker>/<exp(T1)HF|walker>
        # double excitations
        # e2_2_1 = e2_0 * gt2g
        lg_a = oe.contract("gpq,pq->g", chol_a, green_a, backend="jax")
        lg_b = oe.contract("gpq,pq->g", chol_b, green_b, backend="jax")
        lt2g_a = oe.contract("gpq,pq->g",
                            chol_a, 8 * t2_green_a + 2 * t2_green_ab_a,
                            backend="jax")
        lt2g_b = oe.contract("gpq,pq->g",
            chol_b, 8 * t2_green_b + 2 * t2_green_ab_b,
            backend="jax")
        e2_2_2_1 = -((lt2g_a + lt2g_b) @ (lg_a + lg_b)) / 2.0

        def scanned_fun(carry, x):
            chol_a_i, chol_b_i = x
            # e2_0
            lg_a_i = oe.contract("pr,qr->pq", chol_a_i, green_a, backend="jax")
            lg_b_i = oe.contract("pr,qr->pq", chol_b_i, green_b, backend="jax")
            e2_0_1_i = (jnp.trace(lg_a_i) + jnp.trace(lg_b_i))**2 / 2.0
            e2_0_2_i = -(oe.contract('pq,qp->',lg_a_i,lg_a_i, backend="jax") 
                        + oe.contract('pq,qp->',lg_b_i,lg_b_i, backend="jax")
                        ) / 2.0
            carry[0] += e2_0_1_i + e2_0_2_i
            # e2_2
            gl_a_i = oe.contract("pr,rq->pq", green_a, chol_a_i,
                                backend="jax")
            gl_b_i = oe.contract("pr,rq->pq", green_b, chol_b_i,
                                backend="jax")
            lt2_green_a_i = oe.contract(
                "pr,qr->pq", chol_a_i, 8 * t2_green_a + 2 * t2_green_ab_a,
                backend="jax")
            lt2_green_b_i = oe.contract(
                "pr,qr->pq", chol_b_i, 8 * t2_green_b + 2 * t2_green_ab_b,
                backend="jax")
            carry[1] += 0.5 * (
                oe.contract("pq,pq->", gl_a_i, lt2_green_a_i, backend="jax")
                + oe.contract("pq,pq->", gl_b_i, lt2_green_b_i, backend="jax")
            )
            glgp_a_i = oe.contract(
                "iq,qa->ia", gl_a_i[:nocc_a,:], greenp_a, backend="jax"
            )
            glgp_b_i = oe.contract(
                "iq,qa->ia", gl_b_i[:nocc_b,:], greenp_b, backend="jax"
            )
            l2t2_a = 0.5 * oe.contract(
                "ia,jb,iajb->",glgp_a_i,glgp_a_i,t2_aa,
                backend="jax")
            l2t2_b = 0.5 * oe.contract(
                "ia,jb,iajb->",glgp_b_i,glgp_b_i,t2_bb,
                backend="jax")
            l2t2_ab = oe.contract(
                "ia,jb,iajb->",glgp_a_i,glgp_b_i,t2_ab,
                backend="jax")
            carry[2] += l2t2_a + l2t2_b + l2t2_ab
            return carry, 0.0

        [e2_0, e2_2_2_2, e2_2_3], _ = lax.scan(scanned_fun, [0.0, 0.0, 0.0], (chol_a, chol_b))
        e2_2_1 = e2_0 * gt2g
        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <exp(T1)HF|T2 h2|walker>/<exp(T1)HF|walker>

        o0 = jnp.linalg.det(walker_up[:nocc_a,:nocc_a]
            ) * jnp.linalg.det(walker_dn[:nocc_b,:nocc_b])
        # <exp(T1)HF|walker>/<HF|walker>
        t1 = jnp.linalg.det(wave_data["mo_ta"].T.conj() @ walker_up
            ) * jnp.linalg.det(wave_data["mo_tb"].T.conj() @ walker_dn) / o0
        t2 = gt2g * t1 # <exp(T1)HF|T2|walker>/<HF|walker>
        e0 = (e1_0 + e2_0) * t1 # <exp(T1)HF|h1+h2|walker>/<HF|walker>
        e1 = (e1_2 + e2_2) * t1 # <exp(T1)HF|T2 (h1+h2)|walker>/<HF|walker>

        return t1, t2, e0, e1

    # @singledispatchmethod
    # def calc_energy_pt(self, walkers, ham_data: dict, wave_data: dict) -> jax.Array:
    #     raise NotImplementedError("Walker type not supported")

    def calc_energy_pt(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
        t1, t2, e0, e1 = vmap(
            self._calc_energy_pt, in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data, wave_data)
        return t1, t2, e0, e1
    

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
    

@dataclass
class uccsd_pt2_eff(uccsd_pt2):
    """Tensor contraction form of the UCCSD_PT2 (exact T1) trial wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_energy_pt(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
        # CISD trial with half Green
        nocc_a, nocc_b = self.nelec
        norb = self.norb
        nvir_a = norb - nocc_a
        nvir_b = norb - nocc_b
        t2_aa = wave_data["rot_t2aa"]
        t2_bb = wave_data["rot_t2bb"]
        t2_ab = wave_data["rot_t2ab"]
        mo_a, mo_b = wave_data['mo_ta'], wave_data['mo_tb']
        h1_a, h1_b = ham_data["rot_h1_ci"]
        dh1_a, dh1_b = ham_data["d_h1_ci"] # delta_pb h_pq
        chol_a = ham_data["rot_chol_ci"][0].reshape(-1, nocc_a, norb)
        chol_b = ham_data["rot_chol_ci"][1].reshape(-1, nocc_b, norb)
        dchol_a = ham_data["d_chol_ci"][0].reshape(-1, nvir_a, norb) # delta_pb L_gpq
        dchol_b = ham_data["d_chol_ci"][1].reshape(-1, nvir_b, norb)

        # half green's function G_pq
        green_a = (walker_up @ (jnp.linalg.inv(mo_a.T @ walker_up))).T
        green_b = (walker_dn @ (jnp.linalg.inv(mo_b.T @ walker_dn))).T

        # ref one-body energy
        hg_a = oe.contract("pq,rq->pr", h1_a, green_a, backend="jax")
        hg_b = oe.contract("pq,rq->pr", h1_b, green_b, backend="jax")
        trhg_a = oe.contract("pp->", hg_a, backend="jax")
        trhg_b = oe.contract("pp->", hg_b, backend="jax")
        e1_0 = trhg_a + trhg_b # <psi|h1|walker>/<psi|walker>

        # <psi|T2 h1|walker>/<exp(T1)HF|walker>
        # double excitations
        t2g_aa_a = oe.contract("iajb,ia->jb", t2_aa, green_a[:nocc_a,nocc_a:], backend="jax") / 4
        t2g_bb_b = oe.contract("iajb,ia->jb", t2_bb, green_b[:nocc_b,nocc_b:], backend="jax") / 4
        t2g_ab_a = oe.contract("iajb,jb->ia", t2_ab, green_b[:nocc_b,nocc_b:], backend="jax")
        t2g_ab_b = oe.contract("iajb,ia->jb", t2_ab, green_a[:nocc_a,nocc_a:], backend="jax")

        t2gg_aa = oe.contract("jb,jq->bq", t2g_aa_a, green_a[:nocc_a,:]) # t_iajb G_ia G_jq
        t2gg_bb = oe.contract("jb,jq->bq", t2g_bb_b, green_b[:nocc_b,:]) 
        t2gg_ab = oe.contract("jb,jq->bq", t2g_ab_a, green_a[:nocc_a,:])
        t2gg_ba = oe.contract("jb,jq->bq", t2g_ab_b, green_b[:nocc_b,:])

        # t_iajb (G_ia G_jb - G_ib G_ja)
        gt2g_a = oe.contract("jb,jb->", t2g_aa_a, green_a[:nocc_a,nocc_a:], backend="jax")
        gt2g_b = oe.contract("jb,jb->", t2g_bb_b, green_b[:nocc_b,nocc_b:], backend="jax")
        gt2g_ab = oe.contract("ia,ia->", t2g_ab_a, green_a[:nocc_a,nocc_a:], backend="jax")
        gt2g = 2 * (gt2g_a + gt2g_b) + gt2g_ab # <exp(T1)HF|T2|walker>/<exp(T1)HF|walker>

        e1_2_1 = e1_0 * gt2g

        # t2_green_a = (greenp_a @ t2g_a.T) @ green_a[:nocc_a,:] # t_iajb G_ia G_jq Gp_pb
        t2ggg_aaa = oe.contract('pb,jb,jq->pq', green_a[:,nocc_a:], t2g_aa_a, green_a[:nocc_a,:], backend="jax") # t_iajb G_ia G_jq G_pb
        t2ggg_aba = oe.contract('pb,jb,jq->pq', green_a[:,nocc_a:], t2g_ab_a, green_a[:nocc_a,:], backend="jax") # (greenp_a @ t2g_ab_a.T) @ green_a[:nocc_a,:]
        t2ggg_bbb = oe.contract('pb,jb,jq->pq', green_b[:,nocc_b:], t2g_bb_b, green_b[:nocc_b,:], backend="jax")
        t2ggg_bab = oe.contract('pb,jb,jq->pq', green_b[:,nocc_b:], t2g_ab_b, green_b[:nocc_b,:], backend="jax") # (greenp_b @ t2g_ab_b.T) @ green_b[:nocc_b,:]
        e1_2_2_a = -(oe.contract("pq,pq->", 4 * t2ggg_aaa + t2ggg_aba, h1_a, backend="jax")
                    -oe.contract('bq,bq->', 4 * t2gg_aa + t2gg_ab, dh1_a))
        e1_2_2_b = -(oe.contract("pq,pq->", 4 * t2ggg_bbb + t2ggg_bab, h1_b, backend="jax")
                    -oe.contract('bq,bq->', 4 * t2gg_bb + t2gg_ba, dh1_b))
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2  # <exp(T1)HF|T2 h1|walker>/<exp(T1)HF|walker>

        # <exp(T1)HF|T2 h2|walker>/<exp(T1)HF|walker>
        # double excitations
        # e2_2_1 = e2_0 * gt2g
        lg_a = oe.contract("gpq,pq->g", chol_a, green_a, backend="jax")
        lg_b = oe.contract("gpq,pq->g", chol_b, green_b, backend="jax")
        lt2ggg_a = (oe.contract("gpq,pq->g", chol_a, 8 * t2ggg_aaa + 2 * t2ggg_aba, backend="jax")
                    -oe.contract("gbq,bq->g", dchol_a, 8 * t2gg_aa + 2 * t2gg_ab, backend="jax"))
        lt2ggg_b = (oe.contract("gpq,pq->g", chol_b, 8 * t2ggg_bbb + 2 * t2ggg_bab, backend="jax")
                    -oe.contract("gbq,bq->g", dchol_b, 8 * t2gg_bb + 2 * t2gg_ba, backend="jax"))
        e2_2_2_1 = -((lt2ggg_a + lt2ggg_b) @ (lg_a + lg_b)) / 2.0

        def scanned_fun(carry, x):
            chol_a_i, chol_b_i, dchol_a_i, dchol_b_i = x
            # e2_0
            lg_a_i = oe.contract("pr,qr->pq", chol_a_i, green_a, backend="jax")
            lg_b_i = oe.contract("pr,qr->pq", chol_b_i, green_b, backend="jax")
            e2_0_1_i = (jnp.trace(lg_a_i) + jnp.trace(lg_b_i))**2 / 2.0
            e2_0_2_i = -(oe.contract('pq,qp->', lg_a_i, lg_a_i, backend="jax") 
                        + oe.contract('pq,qp->', lg_b_i, lg_b_i, backend="jax")
                        ) / 2.0
            carry[0] += e2_0_1_i + e2_0_2_i
            # e2_2
            gl_a_i = oe.contract("ps,pr->sr", green_a, chol_a_i, backend="jax")
            gl_b_i = oe.contract("ps,pr->sr", green_b, chol_b_i, backend="jax")
            lt2_green_a_i = (oe.contract("qs,qr->sr", chol_a_i, 8 * t2ggg_aaa + 2 * t2ggg_aba, backend="jax")
                            -oe.contract("bs,br->sr", dchol_a_i, 8 * t2gg_aa + 2 * t2gg_ab, backend="jax"))
            lt2_green_b_i = (oe.contract("qs,qr->sr", chol_b_i, 8 * t2ggg_bbb + 2 * t2ggg_bab, backend="jax")
                            -oe.contract("bs,br->sr", dchol_b_i, 8 * t2gg_bb + 2 * t2gg_ba, backend="jax"))
            carry[1] += 0.5 * (
                oe.contract("sr,sr->", gl_a_i, lt2_green_a_i, backend="jax")
                + oe.contract("sr,sr->", gl_b_i, lt2_green_b_i, backend="jax")
            )
            
            gl_a_i = oe.contract("ir,pr->ip", green_a[:nocc_a,:], chol_a_i, backend="jax")
            gl_b_i = oe.contract("ir,pr->ip", green_b[:nocc_a,:], chol_b_i, backend="jax")
            glgp_a_i = (oe.contract("ip,pa->ia", gl_a_i[:nocc_a,:], green_a[:,nocc_a:], backend="jax")
                        -oe.contract("ir,ar->ia", green_a[:nocc_a,:], dchol_a_i, backend="jax"))
            glgp_b_i = (oe.contract("ip,pa->ia", gl_b_i[:nocc_b,:], green_b[:,nocc_b:], backend="jax")
                        -oe.contract("ir,ar->ia", green_b[:nocc_b,:], dchol_b_i, backend="jax"))
            l2t2_a = 0.5 * oe.contract("ia,jb,iajb->", glgp_a_i, glgp_a_i, t2_aa, backend="jax")
            l2t2_b = 0.5 * oe.contract("ia,jb,iajb->", glgp_b_i, glgp_b_i, t2_bb, backend="jax")
            l2t2_ab = oe.contract("ia,jb,iajb->", glgp_a_i, glgp_b_i, t2_ab, backend="jax")
            carry[2] += l2t2_a + l2t2_b + l2t2_ab
            return carry, 0.0

        [e2_0, e2_2_2_2, e2_2_3], _ = lax.scan(scanned_fun, [0.0, 0.0, 0.0], (chol_a, chol_b, dchol_a, dchol_b))
        e2_2_1 = e2_0 * gt2g
        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <exp(T1)HF|T2 h2|walker>/<exp(T1)HF|walker>

        o0 = jnp.linalg.det(walker_up[:nocc_a,:nocc_a]
            ) * jnp.linalg.det(walker_dn[:nocc_b,:nocc_b])
        # <exp(T1)HF|walker>/<HF|walker>
        t1 = jnp.linalg.det(wave_data["mo_ta"].T.conj() @ walker_up
            ) * jnp.linalg.det(wave_data["mo_tb"].T.conj() @ walker_dn) / o0
        t2 = gt2g * t1 # <exp(T1)HF|T2|walker>/<HF|walker>
        e0 = (e1_0 + e2_0) * t1 # <exp(T1)HF|h1+h2|walker>/<HF|walker>
        e1 = (e1_2 + e2_2) * t1 # <exp(T1)HF|T2 (h1+h2)|walker>/<HF|walker>
        # eci = ham_data['h0'] + (e1_0 + e2_0 + e1_2 + e2_2) / (1 + gt2g)
        return t1, t2, e0, e1 
    
    @partial(jit, static_argnums=(0,))
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        nocc_a, nocc_b = self.nelec
        norb = self.norb
        nvir_a = norb - nocc_a
        nvir_b = norb - nocc_b
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )

        ham_data["rot_h1"] = [
            wave_data["mo_coeff"][0].T.conj() @ ham_data["h1"][0],
            wave_data["mo_coeff"][1].T.conj() @ ham_data["h1"][1],
        ]
        ham_data["rot_chol"] = [
            oe.contract(
                "pi,gij->gpj",
                wave_data["mo_coeff"][0].T.conj(),
                ham_data["chol"][0].reshape(-1, norb, norb), 
                backend="jax"),
            oe.contract(
                "pi,gij->gpj",
                wave_data["mo_coeff"][1].T.conj(),
                ham_data["chol"][1].reshape(-1, norb, norb), 
                backend="jax")]
        
        ham_data["rot_h1_ci"] = [
            wave_data["mo_ta"].T.conj() @ ham_data["h1"][0],
            wave_data["mo_tb"].T.conj() @ ham_data["h1"][1],
        ]
        ham_data["rot_chol_ci"] = [
            oe.contract(
                "ip,gpq->giq",
                wave_data["mo_ta"].T.conj(),
                ham_data["chol"][0].reshape(-1, norb, norb), 
                backend="jax"),
            oe.contract(
                "ip,gpq->giq",
                wave_data["mo_tb"].T.conj(),
                ham_data["chol"][1].reshape(-1, norb, norb), 
                backend="jax")]
        
        ham_data['d_h1_ci'] = [ham_data['h1'][0][nocc_a:,:],
                               ham_data['h1'][1][nocc_b:,:]]
        ham_data['d_chol_ci'] = [ham_data['chol'][0].reshape(-1, norb, norb)[:,nocc_a:,:],
                                 ham_data['chol'][1].reshape(-1, norb, norb)[:,nocc_b:,:]]
        
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class uccsd_pt_ad(uhf):
    """differential form of the CCSD_PT wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _t1t2_walker_olp(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> complex:
        '''<HF|(t1+t2)|walker> = (t_ia G_ia + t_iajb G_iajb) * <HF|walker>'''
        noccA, t1A, t2AA = self.nelec[0], wave_data["rot_t1A"], wave_data["rot_t2AA"]
        noccB, t1B, t2BB = self.nelec[1], wave_data["rot_t1B"], wave_data["rot_t2BB"]
        t2AB = wave_data["rot_t2AB"]
        # green_a = (walker_up.dot(jnp.linalg.inv(wave_data["mo_coeff"][0].T.conj() @ walker_up))).T
        # green_b = (walker_dn.dot(jnp.linalg.inv(wave_data["mo_coeff"][1].T.conj() @ walker_dn))).T
        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:noccA,:noccA]))).T
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[:noccB,:noccB]))).T
        green_a, green_b = green_a[:noccA, noccA:], green_b[:noccB, noccB:]
        o0 = self._calc_overlap(walker_up,walker_dn,wave_data)
        o1 = oe.contract("ia,ia", t1A, green_a, backend="jax") \
              + oe.contract("ia,ia", t1B, green_b, backend="jax")
        o2 = (
            0.5 * oe.contract("iajb, ia, jb", t2AA, green_a, green_a, backend="jax")
            + 0.5 * oe.contract("iajb, ia, jb", t2BB, green_b, green_b, backend="jax")
            + oe.contract("iajb, ia, jb", t2AB, green_a, green_b, backend="jax")
        )
        return (o1 + o2) * o0
    
    @partial(jit, static_argnums=0)
    def _t1t2_exp1(self, x: float, h1_mod: jax.Array, walker_up: jax.Array,
                        walker_dn: jax.Array, wave_data: dict):
        '''
        unrestricted t_ia <psi_i^a|exp(x*h1_mod)|walker>/<HF|walker>
        '''
        walker_up_1x = walker_up + x * h1_mod[0].dot(walker_up)
        walker_dn_1x = walker_dn + x * h1_mod[1].dot(walker_dn)
        
        olp = self._t1t2_walker_olp(walker_up_1x, walker_dn_1x, wave_data)
        o0 = self._calc_overlap(walker_up,walker_dn,wave_data)

        return olp/o0

    @partial(jit, static_argnums=0)
    def _t1t2_exp2(self, x: float, 
                   chol_i: jax.Array,
                   walker_up: jax.Array, 
                   walker_dn: jax.Array,
                   wave_data: dict) -> complex:
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
        
        olp = self._t1t2_walker_olp(walker_up_2x,walker_dn_2x,wave_data)
        o0 = self._calc_overlap(walker_up,walker_dn,wave_data)
        
        return olp/o0

    @partial(jit, static_argnums=0)
    def _d2_t1t2_exp2_i(self,chol_i,walker_up,walker_dn,wave_data):
        x = 0.0
        f = lambda a: self._t1t2_exp2(a,chol_i,walker_up,walker_dn,wave_data)
        _, d2f = jax.jvp(lambda x: jax.jvp(f, [x], [1.0])[1], [x], [1.0])
        return d2f

    @partial(jit, static_argnums=0)
    def _d2_t1t2_exp2(self,walker_up,walker_dn,ham_data,wave_data):
        norb = self.norb
        chol = ham_data["chol"].reshape(2, -1, norb, norb)
        chol = chol.transpose(1,0,2,3)
        d2_exp2_batch = jax.vmap(self._d2_t1t2_exp2_i, in_axes=(0,None,None,None))
        d2_exp2s = d2_exp2_batch(chol,walker_up,walker_dn,wave_data)
        h2 = jnp.sum(d2_exp2s)/2
        return h2

    @partial(jit, static_argnums=0)
    def _calc_energy_pt(self, walker_up, walker_dn, ham_data, wave_data):
        '''
        t = <psi|T1+T2|phi>/<psi|phi>
        e0 = <psi|H|phi>/<psi|phi>
        e1 = <psi|(T1+T2)(h1+h2)|phi>/<psi|phi>
        '''

        norb = self.norb
        h1_mod = ham_data['h1_mod']
        chol = ham_data["chol"].reshape(2, -1, norb, norb)
        chol = chol.transpose(1,0,2,3)

        # one body
        x = 0.0
        f1 = lambda a: self._t1t2_exp1(a,h1_mod,walker_up,walker_dn,wave_data)
        t, d_exp1 = jvp(f1, [x], [1.0])

        # two body
        d2_exp2 = self._d2_t1t2_exp2(walker_up,walker_dn,ham_data,wave_data)

        e0 = self._calc_energy(walker_up,walker_dn,ham_data,wave_data)
        e1 = (d_exp1 + d2_exp2)

        return t, e0, e1

    def calc_energy_pt(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
        t, e0, e1 = vmap(
            self._calc_energy_pt, in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data, wave_data)
        return t, e0, e1

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class uccsd_pt2_ad(uhf):
    """differential form of the CCSD_PT2 (exact T1) wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def thouless_trans(self, t1):
        ''' thouless transformation |psi'> = exp(t1)|psi>
            gives the transformed mo_occrep in the 
            original mo basis <psi_p|psi'_i>
            t = t_ia
            t_ia = c_ik c.T_ka
            c_ik = <psi_i|psi'_k>
        '''
        q, r = jnp.linalg.qr(t1,mode='complete')
        u_ji = q
        u_ai = r.T
        mo_t = jnp.vstack((u_ji,u_ai))
        mo_t, _ = jnp.linalg.qr(mo_t)# ,mode='complete')
        return mo_t
    
    @partial(jit, static_argnums=0)
    def _tls_olp(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: dict,
    ) -> complex:
        '''<exp(T1)HF|walker>'''

        olp = jnp.linalg.det(wave_data["mo_ta"].T.conj() @ walker_up
            ) * jnp.linalg.det(wave_data["mo_tb"].T.conj() @ walker_dn)

        return olp

    @partial(jit, static_argnums=0)
    def _tls_exp1(self, x: float, h1_mod: jax.Array, walker_up: jax.Array,
                        walker_dn: jax.Array, wave_data: dict):
        '''
        unrestricted <ep(T1)HF|exp(x*h1_mod)|walker>
        '''

        walker_up_1x = walker_up + x * h1_mod[0].dot(walker_up)
        walker_dn_1x = walker_dn + x * h1_mod[1].dot(walker_dn)

        olp = self._tls_olp(walker_up_1x, walker_dn_1x, wave_data)
        o0 = self._calc_overlap(walker_up,walker_dn,wave_data)

        return olp/o0

    @partial(jit, static_argnums=0)
    def _tls_exp2(self, x: float, chol_i: jax.Array, walker_up: jax.Array,
                    walker_dn: jax.Array, wave_data: dict) -> complex:
        '''
        <exp(T1)HF|exp(x*h2_mod)|walker>
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

        olp = self._tls_olp(walker_up_2x,walker_dn_2x,wave_data)
        o0 = self._calc_overlap(walker_up,walker_dn,wave_data)
        
        return olp/o0
    
    @partial(jit, static_argnums=0)
    def _ut2_walker_olp(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> complex:
        '''<exp(T1)HF|(t1+t2)|walker> = (t_ia G_ia + t_iajb G_iajb) * <exp(T1)HF|walker>'''
        noccA, t2AA = self.nelec[0], wave_data["rot_t2aa"]
        noccB, t2BB = self.nelec[1], wave_data["rot_t2bb"]
        t2AB = wave_data["rot_t2ab"]
        mo_A = wave_data['mo_ta'] # in alpha basis
        mo_B = wave_data['mo_tb'] # in beta basis
        green_a = (walker_up.dot(jnp.linalg.inv(mo_A.T.conj() @ walker_up))).T
        green_b = (walker_dn.dot(jnp.linalg.inv(mo_B.T.conj() @ walker_dn))).T
        green_a, green_b = green_a[:noccA, noccA:], green_b[:noccB, noccB:]
        o0 = self._tls_olp(walker_up,walker_dn,wave_data)
        o2 = (0.5 * oe.contract("iajb, ia, jb", t2AA, green_a, green_a, backend="jax")
            + 0.5 * oe.contract("iajb, ia, jb", t2BB, green_b, green_b, backend="jax")
            + oe.contract("iajb, ia, jb", t2AB, green_a, green_b, backend="jax"))
        return o2 * o0

    @partial(jit, static_argnums=0)
    def _ut2_exp1(self, x: float, h1_mod: jax.Array, walker_up: jax.Array,
                  walker_dn: jax.Array, wave_data: dict):
        '''
        unrestricted <ep(T1)HF|T2 exp(x*h1_mod)|walker>
        '''
        walker_up_1x = walker_up + x * h1_mod[0].dot(walker_up)
        walker_dn_1x = walker_dn + x * h1_mod[1].dot(walker_dn)
        
        olp = self._ut2_walker_olp(walker_up_1x, walker_dn_1x, wave_data)
        o0 = self._calc_overlap(walker_up,walker_dn,wave_data)

        return olp/o0

    @partial(jit, static_argnums=0)
    def _ut2_exp2(self, x: float, chol_i: jax.Array, walker_up: jax.Array,
                  walker_dn: jax.Array, wave_data: dict) -> complex:
        '''
        t_ia <psi_i^a|exp(x*h2_mod)|walker>
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
        
        olp = self._ut2_walker_olp(walker_up_2x,walker_dn_2x,wave_data)
        o0 = self._calc_overlap(walker_up,walker_dn,wave_data)

        return olp/o0
    
    @partial(jit, static_argnums=0)
    def _d2_tls_exp2_i(self,chol_i,walker_up,walker_dn,wave_data):
        x = 0.0
        f = lambda a: self._tls_exp2(a,chol_i,walker_up,walker_dn,wave_data)
        _, d2f = jax.jvp(lambda x: jax.jvp(f, [x], [1.0])[1], [x], [1.0])
        return d2f

    @partial(jit, static_argnums=0)
    def _d2_ut2_exp2_i(self,chol_i,walker_up,walker_dn,wave_data):
        x = 0.0
        f = lambda a: self._ut2_exp2(a,chol_i,walker_up,walker_dn,wave_data)
        _, d2f = jax.jvp(lambda x: jax.jvp(f, [x], [1.0])[1], [x], [1.0])
        return d2f

    @partial(jit, static_argnums=0)
    def _d2_tls_exp2(self,walker_up,walker_dn,ham_data,wave_data):
        norb = self.norb
        chol = ham_data["chol"].reshape(2, -1, norb, norb)
        chol = chol.transpose(1,0,2,3)
        d2_exp2_batch = jax.vmap(self._d2_tls_exp2_i, in_axes=(0,None,None,None))
        d2_exp2s = d2_exp2_batch(chol,walker_up,walker_dn,wave_data)
        h2 = jnp.sum(d2_exp2s)/2
        return h2

    @partial(jit, static_argnums=0)
    def _d2_ut2_exp2(self,walker_up,walker_dn,ham_data,wave_data):
        norb = self.norb
        chol = ham_data["chol"].reshape(2, -1, norb, norb)
        chol = chol.transpose(1,0,2,3)
        d2_exp2_batch = jax.vmap(self._d2_ut2_exp2_i, in_axes=(0,None,None,None))
        d2_exp2s = d2_exp2_batch(chol,walker_up,walker_dn,wave_data)
        h2 = jnp.sum(d2_exp2s)/2
        return h2

    @partial(jit, static_argnums=0)
    def _calc_energy_pt(self, walker_up, walker_dn, ham_data, wave_data):
        '''
        t1 = <exp(T1)HF|walker>/<HF|walker>
        t2 = <exp(T1)HF|T1+T2|walker>/<HF|walker>
        e0 = <exp(T1)HF|h1+h2|walker>/<HF|walker>
        e1 = <exp(T1)HF|(T1+T2)(h1+h2)|walker>/<HF|walker>
        '''

        norb = self.norb
        h1_mod = ham_data['h1_mod']
        chol = ham_data["chol"].reshape(2, -1, norb, norb)
        chol = chol.transpose(1,0,2,3)

        # e0 = <exp(T1)HF|h1+h2|walker>/<HF|walker> #
        # one body
        x = 0.0
        f1 = lambda a: self._tls_exp1(a,h1_mod,walker_up,walker_dn,wave_data)
        t1, d_exp1_0 = jvp(f1, [x], [1.0])

        # two body
        d2_exp2_0 = self._d2_tls_exp2(walker_up,walker_dn,ham_data,wave_data)

        e0 = d_exp1_0 + d2_exp2_0
        
        # e1 = <exp(T1)HF|(T1+T2)(h1+h2)|walker>/<HF|walker>
        # one body
        x = 0.0
        f1 = lambda a: self._ut2_exp1(a,h1_mod,walker_up,walker_dn,wave_data)
        t2, d_exp1_1 = jvp(f1, [x], [1.0])

        # two body
        d2_exp2_1 = self._d2_ut2_exp2(walker_up,walker_dn,ham_data,wave_data)

        e1 = d_exp1_1 + d2_exp2_1
        
        return t1, t2, e0, e1

    def calc_energy_pt(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
        t1, t2, e0, e1 = vmap(
            self._calc_energy_pt, in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data, wave_data)
        return t1, t2, e0, e1

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
