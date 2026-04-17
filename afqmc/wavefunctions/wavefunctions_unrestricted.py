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


class ustoccsd(uhf):
    '''
    Trial = Stochastically sampled CCSD wavefunction
    Guide = UHF
    '''

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1
    nslater: int = 1000

    @partial(jit, static_argnums=(0))
    def get_stocc(self, wave_data: dict, prop_data: dict):
        nO = self.nelec[0]
        nslater = self.nslater
        t1 = wave_data["t1"]

        # L, e_val = hs_op_yann(self, wave_data)
        # L = L.transpose(0,2,1)
        L, _ = self.decompose_t2(wave_data)

        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                nslater,
                L.shape[0],
            ),
        )

        # e^{t1+x*tau2}
        t1s = jnp.array([t1 + 0.0j] * nslater)
        taus = t1s + jnp.einsum("wg,gia->wia", fields, L)

        # from jax import scipy as jsp
        def _exp_tau(tau, sd):
            # tau_full = jnp.zeros((self.norb, self.norb),dtype=jnp.complex128)
            # for matrix that only have one block nonzero exp(tau_ia) = 1 + tau_ia true
            tau_full = jnp.eye(self.norb,dtype=jnp.complex128)
            exp_tau = tau_full.at[:nO, nO:].set(tau)
            # exp_tau = jsp.linalg.expm(tau_full)
            return exp_tau.T @ sd

        # Initial slater determinants
        init_sd = jnp.array([jnp.eye(self.norb)[:,:nO] + 0.0j] * nslater)
        stocc = vmap(_exp_tau)(taus, init_sd)

        return stocc

    @partial(jit, static_argnums=0)
    def get_green_slater(self, trial_slater: jax.Array, walker: jax.Array) -> jax.Array:
        
        green = (
            walker @ (
                jnp.linalg.inv(trial_slater.T.conj() @ walker)
                    ) @ trial_slater.T.conj()
            ).T
        
        return green

    @partial(jit, static_argnums=0)
    def get_energy_slater(self, slater: jax.Array, walker: jax.Array, ham_data: dict) -> jax.Array:
        norb = self.norb

        h0, chol = ham_data["h0"], ham_data["chol"]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        chol = chol.reshape(-1,norb,norb)

        green = self.get_green_slater(slater, walker)
        hg = oe.contract("pq,pq->", h1, green, backend="jax")
        e1 = 2 * hg
        lg = oe.contract("gpr,qr->gpq", chol, green, backend="jax")
        e2_1 = 2 * jnp.sum(oe.contract('gpp->g', lg, backend="jax")**2)
        e2_2 = oe.contract('gpq,gqp->',lg,lg, backend="jax")
        e2 = e2_1 - e2_2

        return h0 + e1 + e2

    @partial(jit, static_argnums=0)
    def get_overlap_slater(self, slater: jax.Array, walker: jax.Array) -> jax.Array:
        return jnp.linalg.det(slater.T.conj() @ walker) ** 2

    @partial(jit, static_argnums=0)
    def get_energy_slaters_one_walker(
        self, 
        slaters: jax.Array,
        walker: jax.Array,
        ham_data: dict
        ):
        """
        slaters: (N, norb, nocc)
        walker:  (norb, nocc)

        returns: (N,) energies
        """

        def scan_slaters(carry, slater):
            # carry is unused; we keep it for scan API
            energy = self.get_energy_slater(slater, walker, ham_data)
            return carry, energy

        # Initial dummy carry (None not allowed)
        init_carry = 0.0

        _, energies = lax.scan(scan_slaters, init_carry, slaters)

        return energies

    @partial(jit, static_argnums=0)
    def get_overlap_slaters_one_walker(
        self,
        slaters: jax.Array,
        walker: jax.Array,
        ):
        """
        slaters: (N, norb, nocc)
        walker:  (norb, nocc)

        returns: (N,) energies
        """

        def scan_slaters(carry, slater):
            # carry is unused; we keep it for scan API
            overlap = self.get_overlap_slater(slater, walker)
            return carry, overlap

        # Initial dummy carry (None not allowed)
        init_carry = 0.0

        _, overlaps = lax.scan(scan_slaters, init_carry, slaters)

        return overlaps
    
    @partial(jit, static_argnums=0)
    def get_eloc_oloc_stocc(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        slaters = wave_data['stocc']
        energies = self.get_energy_slaters_one_walker(slaters, walker, ham_data)
        overlaps = self.get_overlap_slaters_one_walker(slaters, walker) / slaters.shape[0]
        oloc = jnp.sum(overlaps)
        eloc = jnp.sum(overlaps * energies) / oloc
        return (oloc, eloc) 
    
    @partial(jit, static_argnums=0)
    def calc_energy_mixed(
            self, walkers: jax.Array, ham_data: jax.Array, wave_data: dict
            ):

        (overlaps, energies) =  vmap(
            lambda walker: self.get_eloc_oloc_stocc(walker, ham_data, wave_data
            ))(walkers)
        
        return (overlaps, energies)


    def __hash__(self):
        return hash(tuple(self.__dict__.values()))



@dataclass
class ustoccsd2(uhf):
    """
    use CISD Trial and HF Guide 
    abosrb the overlap ratio <Trial|walker>/<Guide/walker> into the weight
    w'(walker)  = weight (for measurements) 
                = weight accumulated by HF importance sampling * <CISD|walker>/<HF|walker>
    E_local(walker) = <CISD|H|walker>/<CISD|walker>
    <E> = {sum_walker w'(walker) * E_local(walker)} / {sum_walker w'(walker)}
    """

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1
    nslater: int = 100

    @partial(jit, static_argnums=(0,3))
    def get_xtaus(self, prop_data, wave_data, prop):
        prop_data["key"], subkey = random.split(prop_data["key"])
        
        fieldx = random.normal(
            subkey,
            shape=(
                prop.n_walkers,
                self.nslater,
                wave_data['tau'][0].shape[0],
            ),
        )
        # xtaus shape (nwalker, nslater, nocc, nvir)
        xtaus_up = oe.contract("wsg,gia->wsia", fieldx, wave_data['tau'][0], backend='jax')
        xtaus_dn = oe.contract("wsg,gia->wsia", fieldx, wave_data['tau'][1], backend='jax')

        return [xtaus_up, xtaus_dn], prop_data

    @partial(jit, static_argnums=(0))
    def _green(
        self,
        walker_up: jax.Array, 
        walker_dn: jax.Array, 
        slater_up: jax.Array,
        slater_dn: jax.Array
        ):
        '''
        full green's function 
        <psi|a_p^dagger a_q|walker>/<psi|walker>
        '''
        green_a = (walker_up @ (jnp.linalg.inv(slater_up.T.conj() @ walker_up)) @ slater_up.T.conj()).T
        green_b = (walker_dn @ (jnp.linalg.inv(slater_dn.T.conj() @ walker_dn)) @ slater_dn.T.conj()).T
        return [green_a, green_b]
    
    @partial(jit, static_argnums=(0))
    def _slater_olp(
        self,
        walker_up: jax.Array, 
        walker_dn: jax.Array, 
        slater_up: jax.Array,
        slater_dn: jax.Array
        ) -> complex:
        ''' 
        <psi|walker>
        '''
        olp = jnp.linalg.det(slater_up.T.conj() @ walker_up) * \
                jnp.linalg.det(slater_dn.T.conj() @ walker_dn)
        return olp

    @partial(jit, static_argnums=0)
    def _calc_energy_slater(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        slater_up: jax.Array,
        slater_dn: jax.Array,
        ham_data: dict,
        ) -> jax.Array:
        
        norb = self.norb
        nocc_a, nocc_b = self.nelec
        h0  = ham_data['h0']
        h1_a, h1_b = ham_data["h1"]
        chol_a = ham_data["chol"][0].reshape(-1, norb, norb)
        chol_b = ham_data["chol"][1].reshape(-1, norb, norb)
        green_a, green_b = self._green(walker_up, walker_dn, slater_up, slater_dn)
        
        hg_a = oe.contract("pq,pq->", h1_a, green_a, backend="jax")
        hg_b = oe.contract("pq,pq->", h1_b, green_b, backend="jax")
        e1 = hg_a + hg_b
    
        # gl_a = oe.contract("pr,gqr->gpq", green_a, chol_a, backend="jax")
        # gl_b = oe.contract("pr,gqr->gpq", green_b, chol_b, backend="jax")
        # trgl_a = oe.contract('gpp->g', gl_a, backend="jax")
        # trgl_b = oe.contract('gpp->g', gl_b, backend="jax")
        # e2_1 = jnp.sum((trgl_a + trgl_b)**2) / 2
        # e2_2 = -(oe.contract('gpq,gqp->', gl_a, gl_a, backend="jax")
        #         + oe.contract('gpq,gqp->', gl_b, gl_b, backend="jax")) / 2
        # e2 = e2_1 + e2_2

        def scan_chol(carry, x):
            chol_a_i, chol_b_i = x
            gl_a_i = oe.contract("pr,qr->pq", green_a, chol_a_i, backend="jax")
            gl_b_i = oe.contract("pr,qr->pq", green_b, chol_b_i, backend="jax")
            trgl_a_i = oe.contract('pp->', gl_a_i, backend="jax")
            trgl_b_i = oe.contract('pp->', gl_b_i, backend="jax")
            e2_c_i = (trgl_a_i + trgl_b_i)**2 / 2
            e2_e_i = -(oe.contract('pq,qp->', gl_a_i, gl_a_i, backend="jax")
                     + oe.contract('pq,qp->', gl_b_i, gl_b_i, backend="jax")) / 2
            carry += e2_c_i + e2_e_i
            return carry, 0.0
        
        e2, _ = lax.scan(scan_chol, 0.0, (chol_a, chol_b))
        
        overlap = self._slater_olp(walker_up, walker_dn, slater_up, slater_dn)
        energy = h0 + e1 + e2

        return overlap, energy

    @partial(jit, static_argnums=0)
    def _ci_walker_olp(
        self,
        walker_up: jax.Array, 
        walker_dn: jax.Array, 
        slater_up: jax.Array,
        slater_dn: jax.Array,
        ci1, ci2
        ) -> complex:
        ''' 
        unrestricted cisd walker overlap
        <(1+ci1+ci2)psi|walker>
        = c_ia* <psi|ia|walker> + 1/4 c_iajb* <psi|ijab|walker>
        '''
        c1a, c1b = ci1
        c2aa, c2ab, c2bb = ci2
        c1a = c1a.conj()
        c1b = c1b.conj()
        c2aa = c2aa.conj()
        c2ab = c2ab.conj()
        c2bb = c2bb.conj()
        nocca, noccb = self.nelec
        norb = self.norb
        greena, greenb = self._green(walker_up, walker_dn, slater_up, slater_dn)
        greena_ov = greena[:nocca, nocca:]
        greenb_ov = greenb[:noccb, noccb:]
        o0 = self._slater_olp(walker_up, walker_dn, slater_up, slater_dn)
        o1 = oe.contract("ia,ia", c1a, greena_ov, backend="jax") \
            + oe.contract("ia,ia", c1b, greenb_ov, backend="jax")
        o2 = 0.5 * oe.contract("iajb, ia, jb", c2aa, greena_ov, greena_ov, backend="jax") \
            + 0.5 * oe.contract("iajb, ia, jb", c2bb, greenb_ov, greenb_ov, backend="jax") \
            + oe.contract("iajb, ia, jb", c2ab, greena_ov, greenb_ov, backend="jax")
        return (1.0 + o1 + o2) * o0

    @partial(jit, static_argnums=0)
    def _ci_walker_olp_disconnected(self,
                                    walker_up: jax.Array,
                                    walker_dn: jax.Array, 
                                    slater_up: jax.Array, 
                                    slater_dn: jax.Array,
                                    ci1) -> complex:
        ''' 
        <(1+ci1+ci2)psi|walker> for disconnected doubles
        = (cA + cB) <psi|ia|walker> + 1/2 (cAcA + cAcB + cBcA + cBcB) <psi|i+j+ab|walker>
        '''
        c1a, c1b = ci1
        c1a = c1a.conj()
        c1b = c1b.conj()
        nocca = walker_up.shape[1]
        noccb = walker_dn.shape[1]
        greena, greenb = self._green(walker_up, walker_dn, slater_up, slater_dn)
        greena_ov = greena[:nocca, nocca:]
        greenb_ov = greenb[:noccb, noccb:]
        ciga = oe.contract('ia,ja->ij', c1a, greena_ov, backend='jax')
        cigb = oe.contract('ia,ja->ij', c1b, greenb_ov, backend='jax')
        o0 = self._slater_olp(walker_up, walker_dn, slater_up, slater_dn)
        o1a = oe.contract("ii->", ciga, backend="jax")
        o1b = oe.contract("ii->", cigb, backend="jax")
        o1 = o1a + o1b
        o2_c = o1**2 / 2
        o2_e = -(oe.contract("ij,ji->", ciga, ciga, backend="jax")
                +oe.contract("ij,ji->", cigb, cigb, backend="jax")) / 2
        o2 = o2_c + o2_e
        return (1.0 + o1 + o2) * o0

    @partial(jit, static_argnums=0)
    def _exp_h1(self,
                x,
                h1_mod, 
                walker_up: jax.Array, 
                walker_dn: jax.Array,
                slater_up: jax.Array, 
                slater_dn: jax.Array, 
                ci1
                ) -> complex:
        '''
        <exp(T1)HF|(1+ci1+ci2) exp(x*h1_mod)|walker>
        '''
        # t = x * h1_mod
        # walker_1x = walker + t.dot(walker)
        walker_up_1x = walker_up + (x * h1_mod[0]) @ walker_up
        walker_dn_1x = walker_dn + (x * h1_mod[1]) @ walker_dn
        o_exp = self._ci_walker_olp_disconnected(walker_up_1x, walker_dn_1x, slater_up, slater_dn, ci1)
        # o_exp = _ci_walker_olp(trial, walker_up_1x, walker_dn_1x, slater_up, slater_dn, ci1, ci2)
        # o_exp = _walker_olp(trial, walker_up_1x, walker_dn_1x, slater_up, slater_dn)
        return o_exp 

    @partial(jit, static_argnums=0)
    def _exp_h2(self, 
                x, 
                chol_i, 
                walker_up: jax.Array,
                walker_dn: jax.Array,
                slater_up: jax.Array,
                slater_dn: jax.Array,
                ci1
                ) -> complex:
        '''
        <exp(T1)HF|(1+ci1+ci2) exp(x*h2)|walker>
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
        o_exp = self._ci_walker_olp_disconnected(walker_up_2x, walker_dn_2x, slater_up, slater_dn, ci1)
        # o_exp = _ci_walker_olp(trial, walker_up_2x, walker_dn_2x, slater_up, slater_dn, ci1, ci2)
        # o_exp = _walker_olp(trial, walker_up_2x, walker_dn_2x, slater_up, slater_dn)
        return o_exp

    @partial(jit, static_argnums=0)
    def _d2_exp_h2i(self,
                    chol_i, 
                    walker_up: jax.Array,
                    walker_dn: jax.Array, 
                    slater_up: jax.Array,
                    slater_dn: jax.Array, 
                    ci1):
        x = 0.0
        f = lambda a: self._exp_h2(a, chol_i, walker_up, walker_dn, slater_up, slater_dn, ci1)
        _, d2f = jax.jvp(lambda x: jax.jvp(f, [x], [1.0])[1], [x], [1.0])
        return d2f


    @partial(jit, static_argnums=0)
    def _calc_energy_cisd_disconnected_ad(self, walker_up, walker_dn, ham_data, wave_data, ci1):

        norb = self.norb
        h0 = ham_data['h0']
        h1_mod = ham_data['h1_mod']
        chol = ham_data["chol"].reshape(2, -1, norb, norb)
        chol = chol.transpose(1,0,2,3)
        slater_up, slater_dn = wave_data['mo_ta'], wave_data['mo_tb']

        # one body
        f1 = lambda a: self._exp_h1(a, h1_mod, walker_up, walker_dn, slater_up, slater_dn, ci1)
        olp, d1_overlap = jvp(f1, [0.0], [1.0])

        # two body
        def scan_chol(carry, c):
            walker_up, walker_dn, slater_up, slater_dn = carry
            return carry, self._d2_exp_h2i(c, walker_up, walker_dn, slater_up, slater_dn, ci1)

        _, d2_overlap_i = lax.scan(scan_chol, (walker_up, walker_dn, slater_up, slater_dn), chol)
        d2_overlap = jnp.sum(d2_overlap_i)/2

        # <psi|(1+ci1+ci2) (h1+h2)|walker> / <psi|1+ci1+ci2|walker>
        e12 = (d1_overlap + d2_overlap) / olp

        return olp, h0 + e12

    @partial(jit, static_argnums=0)
    def _calc_energy_cid(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
        nocc_a = self.nelec[0]
        nocc_b = self.nelec[1]
        c2_aa, c2_ab, c2_bb = wave_data['t2aa'], wave_data['t2ab'], wave_data['t2bb']
        c2_aa = c2_aa.conj()
        c2_ab = c2_ab.conj()
        c2_bb = c2_bb.conj()

        h0 = ham_data['h0']
        h1_a, h1_b = ham_data["h1"]
        chol_a = ham_data["chol"][0].reshape(-1, self.norb, self.norb)
        chol_b = ham_data["chol"][1].reshape(-1, self.norb, self.norb)
        slater_up, slater_dn = wave_data['mo_ta'], wave_data['mo_tb']

        # full green's function G_pq
        green_a, green_b = self._green(walker_up, walker_dn, slater_up, slater_dn)
        greenov_a = green_a[:nocc_a, nocc_a:]
        greenov_b = green_b[:nocc_b, nocc_b:]
        greenp_a = (green_a - jnp.eye(self.norb))[:,nocc_a:]
        greenp_b = (green_b - jnp.eye(self.norb))[:,nocc_b:]

        ################## overlaps #########################
        o0 = self._slater_olp(walker_up, walker_dn, slater_up, slater_dn)
        o2 = 0.5 * oe.contract("iajb,ia,jb->", c2_aa, greenov_a, greenov_a, backend="jax") \
            + 0.5 * oe.contract("iajb,ia,jb->", c2_bb, greenov_b, greenov_b, backend="jax") \
            + oe.contract("iajb,ia,jb->", c2_ab, greenov_a, greenov_b, backend="jax")
        overlap =  (1.0 + o2) * o0

        ################## ref ###############################
        hg_a = oe.contract("pq,pq->", h1_a, green_a, backend="jax")
        hg_b = oe.contract("pq,pq->", h1_b, green_b, backend="jax")
        e1_0 = hg_a + hg_b

        # gl_a = oe.contract("pr,gqr->gpq", green_a, chol_a, backend="jax")
        # gl_b = oe.contract("pr,gqr->gpq", green_b, chol_b, backend="jax")
        
        # reduce memory cost in scan_chol
        # trgl_a = oe.contract('gpp->g', gl_a, backend="jax")
        # trgl_b = oe.contract('gpp->g', gl_b, backend="jax")
        # e2_0_1 = jnp.sum((trgl_a + trgl_b)**2) / 2
        # e2_0_2 = - (oe.contract('gpq,gqp->', gl_a, gl_a, backend="jax")
        #             + oe.contract('gpq,gqp->', gl_b, gl_b, backend="jax")) / 2
        # e2_0 = e2_0_1 + e2_0_2
        ########################################################

        # <exp(T1)HF|T2 h1|walker>/<exp(T1)HF|walker>
        # double excitations
        c2g_a = oe.contract("iajb,ia->jb", c2_aa, greenov_a, backend="jax") / 4
        c2g_b = oe.contract("iajb,ia->jb", c2_bb, greenov_b, backend="jax") / 4
        c2g_ab_a = oe.contract("iajb,jb->ia", c2_ab, greenov_b, backend="jax")
        c2g_ab_b = oe.contract("iajb,ia->jb", c2_ab, greenov_a, backend="jax")

        e1_2_1 = o2 * e1_0
        
        c2_ggg_aaa = (greenp_a @ c2g_a.T) @ green_a[:nocc_a,:] # Gp_pb t_iajb G_ia G_jq
        c2_ggg_aba = (greenp_a @ c2g_ab_a.T) @ green_a[:nocc_a,:]
        c2_ggg_bbb = (greenp_b @ c2g_b.T) @ green_b[:nocc_b,:]
        c2_ggg_bab = (greenp_b @ c2g_ab_b.T) @ green_b[:nocc_b,:]
        e1_2_2_a = -oe.contract("pq,pq->", h1_a, 4*c2_ggg_aaa + c2_ggg_aba, backend="jax")
        e1_2_2_b = -oe.contract("pq,pq->", h1_b, 4*c2_ggg_bbb + c2_ggg_bab, backend="jax")
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2  # <exp(T1)HF|T2 h1|walker>/<exp(T1)HF|walker>

        # two body double excitations
        # e2_2_1 = o2 * e2_0

        # in scan_chol
        # lc2ggg_a = oe.contract("gpr,qr->gpq", chol_a, 8 * c2_ggg_aaa + 2 * c2_ggg_aba, backend="jax")
        # lc2ggg_b = oe.contract("gpr,qr->gpq", chol_b, 8 * c2_ggg_bbb + 2 * c2_ggg_bab, backend="jax")
        # trlc2ggg_a = oe.contract("gpp->g", lc2ggg_a, backend="jax")
        # trlc2ggg_b = oe.contract("gpp->g", lc2ggg_b, backend="jax")
        # e2_2_2_c = -jnp.sum((trlc2ggg_a + trlc2ggg_b) * (trgl_a + trgl_b)) / 2.0
        # e2_2_2_e = (oe.contract("gpq,gpq->", gl_a, lc2ggg_a, backend="jax")
        #             + oe.contract("gpq,gpq->", gl_b, lc2ggg_b, backend="jax")) / 2
        # e2_2_2 = e2_2_2_c + e2_2_2_e

        def scan_chol(carry, x):
            chol_a_i, chol_b_i = x
            gl_a_i = oe.contract("pr,qr->pq", green_a, chol_a_i, backend="jax")
            gl_b_i = oe.contract("pr,qr->pq", green_b, chol_b_i, backend="jax")
            trgl_a_i = oe.contract('pp->', gl_a_i, backend="jax")
            trgl_b_i = oe.contract('pp->', gl_b_i, backend="jax")

            e2_0_c_i = (trgl_a_i + trgl_b_i)**2 / 2
            e2_0_e_i = -(oe.contract('pq,qp->', gl_a_i, gl_a_i, backend="jax")
                        + oe.contract('pq,qp->', gl_b_i, gl_b_i, backend="jax")) / 2
            e2_0_i = e2_0_c_i + e2_0_e_i
            carry[0] += e2_0_i

            lc2ggg_a_i = oe.contract("pr,qr->pq", chol_a_i, 8 * c2_ggg_aaa + 2 * c2_ggg_aba, backend="jax")
            lc2ggg_b_i = oe.contract("pr,qr->pq", chol_b_i, 8 * c2_ggg_bbb + 2 * c2_ggg_bab, backend="jax")
            trlc2ggg_a_i = oe.contract("pp->", lc2ggg_a_i, backend="jax")
            trlc2ggg_b_i = oe.contract("pp->", lc2ggg_b_i, backend="jax")
            e2_2_2_c_i = -((trlc2ggg_a_i + trlc2ggg_b_i) * (trgl_a_i + trgl_b_i)) / 2.0
            e2_2_2_e_i = (oe.contract("pq,pq->", gl_a_i, lc2ggg_a_i, backend="jax")
                        + oe.contract("pq,pq->", gl_b_i, lc2ggg_b_i, backend="jax")) / 2
            e2_2_2_i = e2_2_2_c_i + e2_2_2_e_i
            carry[1] += e2_2_2_i

            glgp_a_i = oe.contract("iq,qa->ia", gl_a_i[:nocc_a,:], greenp_a, backend="jax")
            glgp_b_i = oe.contract("iq,qa->ia", gl_b_i[:nocc_b,:], greenp_b, backend="jax")
            l2c2_aa = 0.5 * oe.contract("ia,jb,iajb->", glgp_a_i, glgp_a_i, c2_aa, backend="jax")
            l2c2_bb = 0.5 * oe.contract("ia,jb,iajb->", glgp_b_i, glgp_b_i, c2_bb, backend="jax")
            l2c2_ab = oe.contract("ia,jb,iajb->", glgp_a_i, glgp_b_i, c2_ab, backend="jax")
            e2_2_3_i = l2c2_aa + l2c2_bb + l2c2_ab
            carry[2] += e2_2_3_i
            return carry, 0.0

        [e2_0, e2_2_2, e2_2_3], _ = lax.scan(scan_chol, [0.0, 0.0, 0.0], (chol_a, chol_b))

        e2_2_1 = o2 * e2_0
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <C2 psi|h2|walker>/<psi|walker>

        energy = h0 + (e1_0 + e2_0 + e1_2 + e2_2) / (1 + o2)
        return overlap, energy
    
    @partial(jit, static_argnums=0)
    def _calc_energy_cisd(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
        ci1, ci2,
    ) -> complex:
        
        '''
        A local energy evaluator for <(1+C1+C2)psi|H|walker> / <(1+C1+C2)psi|walker>
        all operators and the walkers and psi are in the same basis (normally MO)
        |psi> is not necesarily diagonal
        
        all green's function and the chol and ci coeff are as their original definition
        no half rotation performed
        '''
        nocc_a = self.nelec[0]
        nocc_b = self.nelec[1]
        c1_a, c1_b = ci1
        c2_aa, c2_ab, c2_bb = ci2
        c1_a = c1_a.conj()
        c1_b = c1_b.conj()
        c2_aa = c2_aa.conj()
        c2_ab = c2_ab.conj()
        c2_bb = c2_bb.conj()
        
        slater_up, slater_dn = wave_data['mo_ta'], wave_data['mo_tb']
        h0 = ham_data["h0"]
        h1_a = ham_data["h1"][0]
        h1_b = ham_data["h1"][1]
        chol_a = ham_data["chol"][0].reshape(-1, self.norb, self.norb)
        chol_b = ham_data["chol"][1].reshape(-1, self.norb, self.norb)

        # full green's function G_pq
        green_a, green_b = self._green(walker_up, walker_dn, slater_up, slater_dn)
        greenov_a = green_a[:nocc_a, nocc_a:]
        greenov_b = green_b[:nocc_b, nocc_b:]
        greenp_a = (green_a - jnp.eye(self.norb))[:,nocc_a:]
        greenp_b = (green_b - jnp.eye(self.norb))[:,nocc_b:]

        ################## overlaps #########################
        o0 = self._slater_olp(walker_up, walker_dn, slater_up, slater_dn)
        o1 = oe.contract("ia,ia->", c1_a, greenov_a, backend="jax") \
            + oe.contract("ia,ia->", c1_b, greenov_b, backend="jax")
        o2 = 0.5 * oe.contract("iajb,ia,jb->", c2_aa, greenov_a, greenov_a, backend="jax") \
            + 0.5 * oe.contract("iajb,ia,jb->", c2_bb, greenov_b, greenov_b, backend="jax") \
            + oe.contract("iajb,ia,jb->", c2_ab, greenov_a, greenov_b, backend="jax")
        overlap =  (1.0 + o1 + o2) * o0

        ################## ref ###############################
        hg_a = oe.contract("pq,pq->", h1_a, green_a, backend="jax")
        hg_b = oe.contract("pq,pq->", h1_b, green_b, backend="jax")
        e1_0 = hg_a + hg_b # <exp(T1)HF|h1|walker>/<exp(T1)HF|walker>

        # two-body 
        gla = oe.contract("pr,gqr->gpq", green_a, chol_a, backend="jax")
        glb = oe.contract("pr,gqr->gpq", green_b, chol_b, backend="jax")
        trgla = oe.contract('gpp->g', gla, backend="jax")
        trglb = oe.contract('gpp->g', glb, backend="jax")
        e2_0_1 = 0.5 * jnp.sum((trgla + trglb)**2)
        e2_0_2 = - 0.5 * (oe.contract('gpq,gqp->', gla, gla, backend="jax")
                        + oe.contract('gpq,gqp->', glb, glb, backend="jax"))
        e2_0 = e2_0_1 + e2_0_2
        ########################################################

        # one body single excitations  <psi|T1 h1|walker>/<psi|HF|walker>
        e1_1_1 = o1 * e1_0

        gpc1_a = oe.contract("pa,ia->pi", greenp_a, c1_a, backend="jax") # greenp_a @ t1_a.T
        gpc1_b = oe.contract("pa,ia->pi", greenp_b, c1_b, backend="jax")
        c1_green_a = oe.contract("pi,iq->pq", gpc1_a, green_a[:nocc_a,:], backend="jax")
        c1_green_b = oe.contract("pi,iq->pq", gpc1_b, green_b[:nocc_b,:], backend="jax") # gpt1_b @ green_b
        e1_1_2 = -(oe.contract("pq,pq->", h1_a, c1_green_a, backend="jax")
                + oe.contract("pq,pq->", h1_b, c1_green_b, backend="jax"))
        
        e1_1 = e1_1_1 + e1_1_2 # <HF|T1 h1|walker>/<HF|walker>

        # one body double excitations  <psi|T2 h1|walker>/<psi|HF|walker>
        c2g_aa_a = oe.contract("iajb,ia->jb", c2_aa, greenov_a, backend="jax") / 4
        c2g_bb_b = oe.contract("iajb,ia->jb", c2_bb, greenov_b, backend="jax") / 4
        c2g_ab_a = oe.contract("iajb,jb->ia", c2_ab, greenov_b, backend="jax")
        c2g_ab_b = oe.contract("iajb,ia->jb", c2_ab, greenov_a, backend="jax")

        e1_2_1 = o2 * e1_0
        
        c2_ggg_aaa = (greenp_a @ c2g_aa_a.T) @ green_a[:nocc_a,:] # Gp_pb t_iajb G_ia G_jq
        c2_ggg_aba = (greenp_a @ c2g_ab_a.T) @ green_a[:nocc_a,:]
        c2_ggg_bbb = (greenp_b @ c2g_bb_b.T) @ green_b[:nocc_b,:] 
        c2_ggg_bab = (greenp_b @ c2g_ab_b.T) @ green_b[:nocc_b,:]
        e1_2_2_a = -oe.contract("pq,pq->", h1_a, 4 * c2_ggg_aaa + c2_ggg_aba, backend="jax")
        e1_2_2_b = -oe.contract("pq,pq->", h1_b, 4 * c2_ggg_bbb + c2_ggg_bab, backend="jax")
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2  # <psi|T2 h1|walker>/<psi|walker>

        # two body single excitations <psi|T1 h2|walker>/<psi|walker>
        e2_1_1 = o1 * e2_0

        # c_ia Gp_pa G_ir L_pr G_qs L_qs
        lc1g_a = oe.contract("gpq,pq->g", chol_a, c1_green_a, backend="jax")
        lc1g_b = oe.contract("gpq,pq->g", chol_b, c1_green_b, backend="jax")
        e2_1_2 = -((lc1g_a + lc1g_b) @ (trgla + trglb))

        # t_ia Gp_pa G_qr G_is L_pr L_qs
        c1gp_a = oe.contract("ia,pa->ip", c1_a, greenp_a, backend="jax") # t_ia Gp_pa 
        c1gp_b = oe.contract("ia,pa->ip", c1_b, greenp_b, backend="jax")
        glgpc1_a = jnp.einsum("gpq,iq->gpi", gla, c1gp_a, optimize="optimal") # t_ia Gp_pa G_qr L_pr
        glgpc1_b = jnp.einsum("gpq,iq->gpi", glb, c1gp_b, optimize="optimal")
        e2_1_3 = jnp.einsum("gpi,gip->", glgpc1_a, gla[:,:nocc_a,:], optimize="optimal") \
                + jnp.einsum("gpi,gip->", glgpc1_b, glb[:,:nocc_b,:], optimize="optimal") # t_ia Gp_pa L_pr G_qr L_qs G_is
        
        e2_1 = e2_1_1 + e2_1_2 + e2_1_3 # <psi|ci1 h2|walker> / <psi|walker>

        # two body double excitations <psi|T2 h2|walker>/<psi|walker>
        e2_2_1 = o2 * e2_0

        lc2g_a = oe.contract("gpq,pq->g", chol_a, 8*c2_ggg_aaa + 2*c2_ggg_aba, backend="jax")
        lc2g_b = oe.contract("gpq,pq->g", chol_b, 8*c2_ggg_bbb + 2*c2_ggg_bab, backend="jax")
        e2_2_2_1 = -((lc2g_a + lc2g_b) @ (trgla + trglb)) / 2.0

        def scanned_fun(carry, x):
            chol_a_i, chol_b_i, gl_a_i, gl_b_i = x
            lc2_ggg_a_i = oe.contract("pr,qr->pq", chol_a_i, 8*c2_ggg_aaa + 2*c2_ggg_aba, backend="jax")
            lc2_ggg_b_i = oe.contract("pr,qr->pq", chol_b_i, 8*c2_ggg_bbb + 2*c2_ggg_bab, backend="jax")
            carry[0] += (oe.contract("pq,pq->", gl_a_i, lc2_ggg_a_i, backend="jax")
                        + oe.contract("pq,pq->", gl_b_i, lc2_ggg_b_i, backend="jax")) / 2 
            glgp_a_i = oe.contract("iq,qa->ia", gl_a_i[:nocc_a,:], greenp_a, backend="jax")
            glgp_b_i = oe.contract("iq,qa->ia", gl_b_i[:nocc_b,:], greenp_b, backend="jax")
            l2c2_aa = oe.contract("ia,jb,iajb->", 
                                  glgp_a_i.astype(jnp.complex64), # be carefull with single precision
                                  glgp_a_i.astype(jnp.complex64),
                                  c2_aa.astype(jnp.complex64), 
                                  backend="jax") / 2
            l2c2_bb = oe.contract("ia,jb,iajb->", 
                                  glgp_b_i.astype(jnp.complex64), 
                                  glgp_b_i.astype(jnp.complex64), 
                                  c2_bb.astype(jnp.complex64), 
                                  backend="jax") / 2
            l2c2_ab = oe.contract("ia,jb,iajb->", 
                                  glgp_a_i.astype(jnp.complex64), 
                                  glgp_b_i.astype(jnp.complex64), 
                                  c2_ab.astype(jnp.complex64), 
                                  backend="jax")
            carry[1] += l2c2_aa + l2c2_ab + l2c2_bb
            return carry, 0.0

        [e2_2_2_2, e2_2_3], _ = lax.scan(scanned_fun, [0.0, 0.0], (chol_a, chol_b, gla, glb))

        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <psi|T2 h2|walker>/<psi|walker>

        energy = h0 + (e1_0 + e2_0 + e1_1 + e2_1 + e1_2 + e2_2) / (1 + o1 + o2)
        return overlap, energy

    @partial(jit, static_argnums=0)
    def _calc_energy_cisd_disconnected(
        self, 
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict, 
        wave_data: dict,
        ci1,
        ):

        '''
        Disconnected Doubles!!!
        <(1+ci1+ci2)psi|H|walker>
        = (cA + cB) <psi|ia H|walker> + 1/2 (cAcA + cAcB + cBcA + cBcB) <psi|i+j+ab H|walker>
        A local energy evaluator for <(1+C1+C2)psi|H|walker> / <(1+C1+C2)psi|walker>
        all operators and the walkers and psi are in the same basis (normally MO)
        |psi> is not necesarily diagonal
        
        all green's function and the chol and ci coeff are as their original definition
        no half rotation performed
        '''
        norb = self.norb
        nocc_a, nocc_b = self.nelec
        h0  = ham_data['h0']
        h1_a, h1_b = ham_data["h1"]
        slater_up, slater_dn = wave_data["mo_ta"], wave_data["mo_tb"]
        chol_a = ham_data["chol"][0].reshape(-1, self.norb, self.norb)
        chol_b = ham_data["chol"][1].reshape(-1, self.norb, self.norb)
        green_a, green_b = self._green(walker_up, walker_dn, slater_up, slater_dn) # full green
        greenov_a = green_a[:nocc_a, nocc_a:]
        greenov_b = green_b[:nocc_b, nocc_b:]
        greenp_a = (green_a - jnp.eye(norb))[:, nocc_a:]
        greenp_b = (green_b - jnp.eye(norb))[:, nocc_b:]
        
        # applied to the bra
        c1_a, c1_b = ci1
        c1_a = c1_a.conj()
        c1_b = c1_b.conj()

        ######################## universal terms #########################
        c1g_a = oe.contract("ia,ja->ij", c1_a, greenov_a, backend="jax")
        c1g_b = oe.contract("ia,ja->ij", c1_b, greenov_b, backend="jax")
        c1gp_a = oe.contract("ia,pa->ip", c1_a, greenp_a, backend="jax")
        c1gp_b = oe.contract("ia,pa->ip", c1_b, greenp_b, backend="jax")
        c1gg_a = oe.contract("ij,iq->jq", c1g_a, green_a[:nocc_a,:], backend="jax") # c_ia G_ja G_iq
        c1gg_b = oe.contract("ij,iq->jq", c1g_b, green_b[:nocc_b,:], backend="jax")
        c1gpg_a = oe.contract("ip,iq->pq", c1gp_a, green_a[:nocc_a,:], backend="jax") # c_ia Gp_pa G_iq
        c1gpg_b = oe.contract("ip,iq->pq", c1gp_b, green_b[:nocc_b,:], backend="jax")
        
        ########################## overlap terms #########################
        o0 = self._slater_olp(walker_up, walker_dn, slater_up, slater_dn)
        o1_a = oe.contract("ii->", c1g_a, backend="jax")
        o1_b = oe.contract("ii->", c1g_b, backend="jax")
        o1 = o1_a + o1_b
        o2_c = o1**2 / 2
        o2_e = -(oe.contract("ij,ji->", c1g_a, c1g_a, backend="jax")
                +oe.contract("ij,ji->", c1g_b, c1g_b, backend="jax")) / 2
        o2 = o2_c + o2_e
        overlap =  (1.0 + o1 + o2) * o0

        ########################### ref energy ############################
        gh_a = oe.contract("pr,qr->pq", green_a, h1_a, backend="jax")
        gh_b = oe.contract("pr,qr->pq", green_b, h1_b, backend="jax")
        trgh_a = oe.contract("pp->", gh_a, backend="jax")
        trgh_b = oe.contract("pp->", gh_b, backend="jax")
        e1_0 = trgh_a + trgh_b

        ############################ ci terms #############################

        ###### one-body single excitations ######
        e1_1_1 = o1 * e1_0

        e1_1_2 = -(oe.contract("pq,pq->", c1gpg_a, h1_a, backend="jax")
                + oe.contract("pq,pq->", c1gpg_b, h1_b, backend="jax"))
        
        e1_1 = e1_1_1 + e1_1_2 # <C1 psi|h1|walker>/<psi|walker>

        ###### one-body double excitations ######
        e1_2_1 = o2 * e1_0

        c2ggg_aaa_c = o1_a * c1gpg_a # cA_ia cA_jb GA_ia GA_jq GpA_pb (-)
        c2ggg_aaa_e = oe.contract('jp,jq->pq', c1gp_a, c1gg_a, backend='jax') # cA_ia cA_jb GA_ja GA_iq GpA_pb (+)
        c2ggg_aaa = 2 * (c2ggg_aaa_c - c2ggg_aaa_e) # swap ia, jb pairwise
        c2ggg_aba = 2* o1_b * c1gpg_a # cB_jb GB_jb  cA_ia GpA_pa  GA_iq
        # c2ggg_baa = c2ggg_aba # cB_ia GB_ia  cA_jb GpA_pb  GA_jq
        c2ggg_bbb_c = o1_b * c1gpg_b
        c2ggg_bbb_e = oe.contract('jp,jq->pq', c1gp_b, c1gg_b, backend='jax')
        c2ggg_bbb = 2 * (c2ggg_bbb_c - c2ggg_bbb_e)
        c2ggg_bab = 2 * o1_a * c1gpg_b
        # c2ggg_abb = c2ggg_bab
        e1_2_2_a = -oe.contract("pq,pq->", c2ggg_aaa + c2ggg_aba, h1_a, backend="jax") / 2
        e1_2_2_b = -oe.contract("pq,pq->", c2ggg_bbb + c2ggg_bab, h1_b, backend="jax") / 2
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2  # <C2 psi|h1|walker>/<psi|walker>

        def scan_chol(carry, x):
            chol_a_i, chol_b_i = x

            gl_a_i = oe.contract("pr,qr->pq", green_a, chol_a_i, backend="jax")
            gl_b_i = oe.contract("pr,qr->pq", green_b, chol_b_i, backend="jax")
            trgl_a_i = oe.contract('pp->', gl_a_i, backend="jax")
            trgl_b_i = oe.contract('pp->', gl_b_i, backend="jax")
            e2_0_c_i = (trgl_a_i + trgl_b_i)**2 / 2
            e2_0_e_i = -(oe.contract('pq,qp->', gl_a_i, gl_a_i, backend="jax")
                        + oe.contract('pq,qp->', gl_b_i, gl_b_i, backend="jax")) / 2
            e2_0_i = e2_0_c_i + e2_0_e_i
            carry[0] += e2_0_i

            c1gpgl_a = oe.contract("pr,qr->pq", c1gpg_a, chol_a_i, backend="jax")
            c1gpgl_b = oe.contract("pr,qr->pq", c1gpg_b, chol_b_i, backend="jax")
            trc1gpgl_a = oe.contract("pp->", c1gpgl_a, backend="jax")
            trc1gpgl_b = oe.contract("pp->", c1gpgl_b, backend="jax")
            e2_1_2_c_i = -(trc1gpgl_a + trc1gpgl_b) * (trgl_a_i + trgl_b_i)
            e2_1_2_e_i = oe.contract("pq,qp->", c1gpgl_a, gl_a_i, backend="jax") \
                    + oe.contract("pq,qp->", c1gpgl_b, gl_b_i, backend="jax") # t_ia Gp_pa G_is L_qs G_qr L_pr
            e2_1_2_i =  e2_1_2_c_i + e2_1_2_e_i
            carry[1] += e2_1_2_i

            lc2ggg_a_i = oe.contract("pr,qr->pq", chol_a_i, 2*(c2ggg_aaa + c2ggg_aba), backend="jax")
            lc2ggg_b_i = oe.contract("pr,qr->pq", chol_b_i, 2*(c2ggg_bbb + c2ggg_bab), backend="jax")
            trlc2ggg_a_i = oe.contract("pp->", lc2ggg_a_i, backend="jax")
            trlc2ggg_b_i = oe.contract("pp->", lc2ggg_b_i, backend="jax")
            e2_2_2_c_i = -(trlc2ggg_a_i + trlc2ggg_b_i)*(trgl_a_i + trgl_b_i) / 4
            e2_2_2_e_i = (oe.contract("pq,pq->", gl_a_i, lc2ggg_a_i, backend="jax")
                        + oe.contract("pq,pq->", gl_b_i, lc2ggg_b_i, backend="jax")) / 4
            e2_2_2_i = e2_2_2_c_i + e2_2_2_e_i
            carry[2] += e2_2_2_i
            
            c1glgp_a_i = oe.contract("ip,jp->ij", c1gp_a, gl_a_i[:nocc_a,:], backend="jax")
            c1glgp_b_i = oe.contract("ip,jp->ij", c1gp_b, gl_b_i[:nocc_b,:], backend="jax")
            trc1glgp_a_i = oe.contract("ii->", c1glgp_a_i, backend="jax")
            trc1glgp_b_i = oe.contract("ii->", c1glgp_b_i, backend="jax")
            e2_2_3_c_i = (trc1glgp_a_i + trc1glgp_b_i)**2 / 2
            e2_2_3_e_i = (oe.contract("ij,ji->", c1glgp_a_i, c1glgp_a_i, backend="jax")
                        + oe.contract("ij,ji->", c1glgp_b_i, c1glgp_b_i, backend="jax")) / 2
            e2_2_3_i = e2_2_3_c_i - e2_2_3_e_i
            carry[3] += e2_2_3_i
            return carry, 0.0

        [e2_0, e2_1_2, e2_2_2, e2_2_3], _ = lax.scan(scan_chol, [0.0, 0.0, 0.0, 0.0], (chol_a, chol_b))

        e2_1_1 = o1 * e2_0
        e2_1 = e2_1_1 + e2_1_2

        e2_2_1 = o2 * e2_0
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <C2 psi|h2|walker>/<psi|walker>

        energy = h0 + (e1_0 + e2_0 + e1_1 + e2_1 + e1_2 + e2_2) / (1 + o1 + o2)

        return overlap, energy
    
    @partial(jit, static_argnums=0)
    def _calc_energy_exp_xtau(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict, 
        wave_data: dict, 
        xtau,
        ) -> jax.Array:
        
        # xtau_a, xtau_b = xtau
        slater_up, slater_dn = self._thouless([wave_data['mo_ta'], wave_data['mo_tb']], xtau)
        overlap, energy = self._calc_energy_slater(walker_up, walker_dn, slater_up, slater_dn, ham_data)

        return overlap, energy

    @partial(jit, static_argnums=0)
    def _calc_energy_cisd_xtau(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict, 
        wave_data: dict, 
        xtau,
        ) -> jax.Array:
        
        # overlap, energy = self._calc_energy_cisd_disconnected_ad(walker_up, walker_dn, ham_data, wave_data, xtau)
        overlap, energy = self._calc_energy_cisd_disconnected(walker_up, walker_dn, ham_data, wave_data, xtau)

        return overlap, energy
    
    @partial(jit, static_argnums=0)
    def _calc_correction_xtau(self, walker_up, walker_dn, xtau_up, xtau_dn, ham_data, wave_data):
        # numerator correction = <[exp(xtau)-cisd] psi|H|walker>
        # denominator correction = <[exp(xtau)-cisd] psi|walker>
        xtau = [xtau_up, xtau_dn]
        o_exp, e_exp = self._calc_energy_exp_xtau(walker_up, walker_dn, ham_data, wave_data, xtau)
        o_ci, e_ci =  self._calc_energy_cisd_xtau(walker_up, walker_dn, ham_data, wave_data, xtau)
        numerator = o_exp*e_exp - o_ci*e_ci
        denominator = o_exp - o_ci

        return numerator, denominator
    
    @partial(jit, static_argnums=0)
    def _calc_correction_xtaus(self, walker_up, walker_dn, xtaus_up, xtaus_dn, ham_data, wave_data):
        # calculating corrections for more than one xtau

        nslater = self.nslater
        norb = self.norb
        nocc_a, nocc_b = self.nelec
        nvir_a = norb - nocc_a
        nvir_b = norb - nocc_b

        assert xtaus_up.shape == (nslater, nocc_a, nvir_a)
        assert xtaus_dn.shape == (nslater, nocc_b, nvir_b)

        def _scan_xtaus(carry, xs):
            xtau_up, xtau_dn = xs 
            num, den = self._calc_correction_xtau(walker_up, walker_dn, xtau_up, xtau_dn, ham_data, wave_data)
            return carry, (num, den)

        init_carry = 0.0
        _, (nums, dens) = lax.scan(_scan_xtaus, init_carry, (xtaus_up, xtaus_dn))

        # intermediately normalize stocc
        numerator = jnp.sum(nums) / nslater
        denominator = jnp.sum(dens) / nslater

        return numerator, denominator

    @partial(jit, static_argnums=(0))
    def calc_correction(self, walkers, xtaus, ham_data, wave_data):
        # xtaus shape (nwalker, nslater, nocc, nvir)
        walkers_up, walkers_dn = walkers
        xtaus_up, xtaus_dn = xtaus

        nslater = self.nslater # samples of T2 per walker
        norb = self.norb
        nocc_a, nocc_b = self.nelec
        nvir_a = norb - nocc_a
        nvir_b = norb - nocc_b
        nwalker = walkers_up.shape[0]
        batch_size = nwalker // self.n_batch

        assert xtaus_up.shape == (nwalker, nslater, nocc_a, nvir_a)
        assert xtaus_dn.shape == (nwalker, nslater, nocc_b, nvir_b)

        def scan_batch(carry, xs):
            walker_up_batch, walker_dn_batch, xtaus_up_batch, xtaus_dn_batch = xs
            num, den = vmap(self._calc_correction_xtaus, in_axes=(0, 0, 0, 0, None, None))(
                walker_up_batch, walker_dn_batch, xtaus_up_batch, xtaus_dn_batch, 
                ham_data, wave_data
            )
            return carry, (num, den)

        _, (num, den) = lax.scan(
            scan_batch, None,
            (walkers_up.reshape(self.n_batch, batch_size, norb, nocc_a),
             walkers_dn.reshape(self.n_batch, batch_size, norb, nocc_b),
             xtaus_up.reshape(self.n_batch, batch_size, nslater, nocc_a, nvir_a),
             xtaus_dn.reshape(self.n_batch, batch_size, nslater, nocc_b, nvir_b))
            )
        
        num = num.reshape(nwalker)
        den = den.reshape(nwalker)
        
        return num, den
    
    @partial(jit, static_argnums=(0))
    def calc_energy_cid(self, walkers, ham_data, wave_data):
        nwalker = walkers[0].shape[0]
        nocc_a, nocc_b = self.nelec
        batch_size = nwalker // self.n_batch

        def scan_batch(carry, walker_batch):
            walker_up_batch, walker_dn_batch = walker_batch
            overlap, energy = vmap(self._calc_energy_cid, in_axes=(0, 0, None, None))(
                walker_up_batch, walker_dn_batch, ham_data, wave_data
            )
            return carry, (overlap, energy)

        _, (overlaps, energies) = lax.scan(
            scan_batch,
            None, 
            (walkers[0].reshape(self.n_batch, batch_size, self.norb, nocc_a),
             walkers[1].reshape(self.n_batch, batch_size, self.norb, nocc_b)))

        overlaps = overlaps.reshape(nwalker)
        energies = energies.reshape(nwalker)
        
        return overlaps, energies

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))