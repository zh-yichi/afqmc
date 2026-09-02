from abc import ABC
from dataclasses import dataclass
from functools import partial
from typing import Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, jvp, lax, vmap, random
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import opt_einsum as oe

from .wavefunctions_restricted import _chol_chunking

class uwfn(ABC):
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
    nchol_chunk: int
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
    def _thouless(self, slater, tau):
        # calculate |psi'> = exp(t_ia a+ i)|psi>
        
        slater_up, slater_dn = slater
        ta, tb = tau
        
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
    
    @partial(jit, static_argnums=(0,3))
    def get_ccsd_walkers(self, prop_data, wave_data, prop):
        prop_data["key"], subkey = random.split(prop_data["key"])
        
        fieldy = random.normal(
            subkey,
            shape=(
                prop.n_walkers,
                wave_data['tau'][0].shape[0],
            ),
        )
        # ytaus shape (nwalker, nocc, nvir)
        ytaus_up = oe.contract("wg,gia->wia", fieldy, wave_data['tau'][0], backend='jax')
        ytaus_dn = oe.contract("wg,gia->wia", fieldy, wave_data['tau'][1], backend='jax')

        mo_t = [wave_data['mo_ta'], wave_data['mo_tb']]

        def scan_body(carry, ytau):
            ytau_up, ytau_dn = ytau
            slater_up, slater_dn = self._thouless(mo_t, [ytau_up, ytau_dn])
            return carry, (slater_up, slater_dn)

        # scan iterates over leading axis (n_walkers) of (ytaus_up, ytaus_dn)
        _, (slaters_up, slaters_dn) = lax.scan(scan_body, None, (ytaus_up, ytaus_dn),)

        return [slaters_up, slaters_dn], prop_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class uhf(uwfn):
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
    nchol_chunk: int = 100
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
    def _calc_overlap_delta(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: dict,
    ) -> complex:
        o0 = jnp.linalg.det(walker_up[:walker_up.shape[1], :]) \
            * jnp.linalg.det(walker_dn[:walker_dn.shape[1], :])
        return o0

    def calc_overlap_delta(self, walkers: list, wave_data: dict) -> jax.Array:
        n_walkers = walkers[0].shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, walker_batch):
            walker_batch_0, walker_batch_1 = walker_batch
            overlap_batch = vmap(self._calc_overlap_delta, in_axes=(0, 0, None))(
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
        h0 = ham_data["h0"]
        rot_h1_a, rot_h1_b = ham_data["rot_h1"]
        rot_chol_a, rot_chol_b = ham_data["rot_chol"]
        green_a, green_b = self._calc_green(walker_up, walker_dn, wave_data)

        e1 = oe.contract("pq,pq->", rot_h1_a, green_a) \
            + oe.contract("pq,pq->", rot_h1_b, green_b)
        
        nchol = rot_chol_a.shape[0]
        nchol_chunk = self.nchol_chunk
        nchunks = -(-nchol // nchol_chunk)
        pad = nchunks * nchol_chunk - nchol
        rot_chol_a = jnp.pad(rot_chol_a, ((0, pad), (0, 0), (0, 0)))
        rot_chol_b = jnp.pad(rot_chol_b, ((0, pad), (0, 0), (0, 0)))
        rot_chol_a_chunks = rot_chol_a.reshape(nchunks, nchol_chunk, *rot_chol_a.shape[1:])
        rot_chol_b_chunks = rot_chol_b.reshape(nchunks, nchol_chunk, *rot_chol_b.shape[1:])

        def scanned_fun(carry, x):
            chol_a_c, chol_b_c = x  # (nchol_chunk, nocc, norb) each
            lg_a_c = oe.contract("gpr,qr->gpq", chol_a_c, green_a, backend="jax")
            lg_b_c = oe.contract("gpr,qr->gpq", chol_b_c, green_b, backend="jax")
            trlg_a_c = oe.contract("gpp->g", lg_a_c, backend="jax")
            trlg_b_c = oe.contract("gpp->g", lg_b_c, backend="jax")

            e2aa_c_c = jnp.sum(trlg_a_c ** 2)
            e2aa_e_c = oe.contract("gpq,gqp->", lg_a_c, lg_a_c, backend="jax")
            e2aa_c = e2aa_c_c - e2aa_e_c

            e2ab_c = jnp.sum(trlg_a_c * trlg_b_c) * 2

            e2bb_c_c = jnp.sum(trlg_b_c ** 2)
            e2bb_e_c = oe.contract("gpq,gqp->", lg_b_c, lg_b_c, backend="jax")
            e2bb_c = e2bb_c_c - e2bb_e_c

            carry += (e2aa_c + e2ab_c + e2bb_c) / 2
            return carry, 0.0

        e2, _ = lax.scan(scanned_fun, 0.0, (rot_chol_a_chunks, rot_chol_b_chunks))

        return h0 + e1 + e2

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        dm_up = wave_data["mo_coeff"][0] @ wave_data["mo_coeff"][0].T.conj()
        dm_dn = wave_data["mo_coeff"][1] @ wave_data["mo_coeff"][1].T.conj()
        return jnp.array([dm_up, dm_dn])

    @partial(jit, static_argnums=(0,))
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        chola, cholb = ham_data["chol"]
        chola = chola.reshape(-1, self.norb, self.norb)
        cholb = cholb.reshape(-1, self.norb, self.norb)
        ham_data["rot_h1"] = [
            wave_data["mo_coeff"][0].T.conj() @ ham_data["h1"][0],
            wave_data["mo_coeff"][1].T.conj() @ ham_data["h1"][1],
        ]
        ham_data["rot_chol"] = [
            oe.contract("pi,gij->gpj", wave_data["mo_coeff"][0].T.conj(), chola, backend="jax"),
            oe.contract("pi,gij->gpj", wave_data["mo_coeff"][1].T.conj(), cholb, backend="jax")
            ]
        
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class ucisd(uwfn):
    """A manual implementation of the UCISD wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1
    nchol_chunk: int = 1
    mix_precision: bool = False

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
        
        # e2_0_1 = ((lg_a + lg_b) @ (lg_a + lg_b)) / 2.0
        
        lg1_a = oe.contract("gpj,qj->gpq", rot_chol_a, green_a, backend="jax")
        lg1_b = oe.contract("gpj,qj->gpq", rot_chol_b, green_b, backend="jax")

        # e2_0_2 = (
        #     -(
        #         jnp.sum(vmap(lambda x: x * x.T)(lg1_a))
        #         + jnp.sum(vmap(lambda x: x * x.T)(lg1_b))
        #     )
        #     / 2.0
        # )
        # e2_0 = e2_0_1 + e2_0_2

        nchol = rot_chol_a.shape[0]
        # nchol_chunk = self.nchol_chunk
        # nchunks = -(-nchol // nchol_chunk)
        # pad = nchunks * nchol_chunk - nchol
        # rot_chol_a_pad = jnp.pad(rot_chol_a, ((0, pad), (0, 0), (0, 0)))
        # rot_chol_b_pad = jnp.pad(rot_chol_b, ((0, pad), (0, 0), (0, 0)))
        # rot_chol_a_chunks = rot_chol_a.reshape(nchol, 1, *rot_chol_a.shape[1:])
        # rot_chol_b_chunks = rot_chol_b.reshape(nchol, 1, *rot_chol_b.shape[1:])

        def scanned_fun(carry, x):
            chol_a_c, chol_b_c = x  # (nchol_chunk, nocc, norb) each
            lg_a_c = oe.contract("gpr,qr->gpq", chol_a_c, green_a, backend="jax")
            lg_b_c = oe.contract("gpr,qr->gpq", chol_b_c, green_b, backend="jax")
            trlg_a_c = oe.contract("gpp->g", lg_a_c, backend="jax")
            trlg_b_c = oe.contract("gpp->g", lg_b_c, backend="jax")

            e2aa_c_c = jnp.sum(trlg_a_c ** 2)
            e2aa_e_c = oe.contract("gpq,gqp->", lg_a_c, lg_a_c, backend="jax")
            e2aa_c = e2aa_c_c - e2aa_e_c

            e2ab_c = jnp.sum(trlg_a_c * trlg_b_c) * 2

            e2bb_c_c = jnp.sum(trlg_b_c ** 2)
            e2bb_e_c = oe.contract("gpq,gqp->", lg_b_c, lg_b_c, backend="jax")
            e2bb_c = e2bb_c_c - e2bb_e_c

            carry += (e2aa_c + e2ab_c + e2bb_c) / 2
            return carry, 0.0

        e2_0, _ = lax.scan(
            scanned_fun, 0.0, (rot_chol_a.reshape(nchol, 1, *rot_chol_a.shape[1:]), 
                               rot_chol_b.reshape(nchol, 1, *rot_chol_b.shape[1:])))

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
class uptccsd(uhf):
    """A manual implementation of the Spin-Unrestricted ptCCSD wave function."""

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
class upt2ccsd(uhf):
    """Tensor contraction form of the Spin-Unrestricted pt2CCSD (exact T1) trial wave function."""

    norb: int
    nelec: Tuple[int, int]
    nchol_chunk: int = 100
    mix_precision: bool = False
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_energy_pt(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
        # only do this for two-body energy with T contraction
        if self.mix_precision:
            rtype = jnp.float32
            ctype = jnp.complex64
        else:
            rtype = jnp.float64
            ctype = jnp.complex128

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
        t2g_a = oe.contract("iajb,ia->jb", t2_aa, green_a[:nocc_a,nocc_a:], backend="jax") / 4
        t2g_b = oe.contract("iajb,ia->jb", t2_bb, green_b[:nocc_b,nocc_b:], backend="jax") / 4
        t2g_ab_a = oe.contract("iajb,jb->ia", t2_ab, green_b[:nocc_b,nocc_b:], backend="jax")
        t2g_ab_b = oe.contract("iajb,ia->jb", t2_ab, green_a[:nocc_a,nocc_a:], backend="jax")
        # t_iajb (G_ia G_jb - G_ib G_ja)
        gt2g_a = oe.contract("jb,jb->", t2g_a, green_a[:nocc_a,nocc_a:], backend="jax")
        gt2g_b = oe.contract("jb,jb->", t2g_b, green_b[:nocc_b,nocc_b:], backend="jax")
        gt2g_ab = oe.contract("ia,ia->", t2g_ab_a, green_a[:nocc_a,nocc_a:], backend="jax")
        gt2g = 2 * (gt2g_a + gt2g_b) + gt2g_ab # <exp(T1)HF|T2|walker>/<exp(T1)HF|walker>
        e1_2_1 = hg * gt2g
        
        t2_green_aaa = (greenp_a @ t2g_a.T) @ green_a[:nocc_a,:] # Gp_pb t_iajb G_ia G_jq
        t2_green_aba = (greenp_a @ t2g_ab_a.T) @ green_a[:nocc_a,:]
        t2_green_bbb = (greenp_b @ t2g_b.T) @ green_b[:nocc_b,:]
        t2_green_abb = (greenp_b @ t2g_ab_b.T) @ green_b[:nocc_b,:]
        t2_green_a_a = 4 * t2_green_aaa + t2_green_aba # connect a->a
        t2_green_b_b = 4 * t2_green_bbb + t2_green_abb # connect b->b

        e1_2_2_a = -oe.contract("pq,pq->", h1_a, t2_green_a_a, backend="jax")
        e1_2_2_b = -oe.contract("pq,pq->", h1_b, t2_green_b_b, backend="jax")
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2  # <exp(T1)HF|T2 h1|walker>/<exp(T1)HF|walker>

        # <exp(T1)HF|T2 h2|walker>/<exp(T1)HF|walker>
        # double excitations
        nchol = chol_a.shape[0]
        nchol_chunk = self.nchol_chunk
        nchunks = -(-nchol // nchol_chunk)
        pad = nchunks * nchol_chunk - nchol

        chol_a = jnp.pad(chol_a, ((0, pad), (0, 0), (0, 0)))
        chol_b = jnp.pad(chol_b, ((0, pad), (0, 0), (0, 0)))
        chol_a = chol_a.reshape(nchunks, nchol_chunk, self.norb, self.norb)
        chol_b = chol_b.reshape(nchunks, nchol_chunk, self.norb, self.norb)

        def scanned_fun(carry, x):
            chol_a_c, chol_b_c = x  # each shape (nchol_chunk, norb, norb)

            # e2_0
            gl_a_c = oe.contract("pr,gqr->gpq",
                                green_a.astype(jnp.complex128),
                                chol_a_c.astype(jnp.float64), 
                                backend="jax").astype(jnp.complex128)
            gl_b_c = oe.contract("pr,gqr->gpq", 
                                green_b.astype(jnp.complex128),
                                chol_b_c.astype(jnp.float64), 
                                backend="jax")
            tr_gl_a = oe.contract("gpp->g", 
                                gl_a_c.astype(jnp.complex128), 
                                backend="jax").astype(jnp.complex128)
            tr_gl_b = oe.contract("gpp->g", 
                                gl_b_c.astype(jnp.complex128), 
                                backend="jax").astype(jnp.complex128)
            ex_gl_a = oe.contract("gpq,gqp->g", 
                                gl_a_c.astype(jnp.complex128), 
                                gl_a_c.astype(jnp.complex128), 
                                backend="jax").astype(jnp.complex128)
            ex_gl_b = oe.contract("gpq,gqp->g", 
                                gl_b_c.astype(jnp.complex128), 
                                gl_b_c.astype(jnp.complex128), 
                                backend="jax").astype(jnp.complex128)
            e2_0_1_c = jnp.sum((tr_gl_a + tr_gl_b) ** 2) / 2.0
            e2_0_2_c = -jnp.sum(ex_gl_a + ex_gl_b) / 2.0

            carry[0] += (e2_0_1_c + e2_0_2_c).astype(jnp.complex128)

            # e2_2
            lt2g_a_c = oe.contract("gpr,qr->gpq", 
                                    chol_a_c.astype(jnp.float64), 
                                    (2*t2_green_a_a).astype(jnp.complex128), 
                                    backend="jax")
            lt2g_b_c = oe.contract("gpr,qr->gpq", 
                                    chol_b_c.astype(jnp.float64), 
                                    (2*t2_green_b_b).astype(jnp.complex128), 
                                    backend="jax")
            tr_lt2g_a_c = oe.contract("gqq->g", lt2g_a_c.astype(jnp.complex128), backend="jax")
            tr_lt2g_b_c = oe.contract("gqq->g", lt2g_b_c.astype(jnp.complex128), backend="jax")
            carry[1] += -(((tr_lt2g_a_c.astype(ctype) + tr_lt2g_b_c.astype(ctype)) 
                           @ (tr_gl_a.astype(ctype) + tr_gl_b.astype(ctype))
                           ) / 2.0).astype(jnp.complex128)
            carry[2] += ((oe.contract("gpq,gpq->", 
                                      gl_a_c.astype(ctype), 
                                      lt2g_a_c.astype(ctype), 
                                      backend="jax")
                        + oe.contract("gpq,gpq->", 
                                      gl_b_c.astype(ctype), 
                                      lt2g_b_c.astype(ctype), 
                                      backend="jax")) 
                        / 2).astype(jnp.complex128)

            glgp_a_c = oe.contract("giq,qa->gia",
                                gl_a_c[:,:nocc_a,:].astype(jnp.complex128), 
                                greenp_a.astype(jnp.complex128), 
                                backend="jax")
            glgp_b_c = oe.contract("giq,qa->gia", 
                                gl_b_c[:,:nocc_b,:].astype(jnp.complex128), 
                                greenp_b.astype(jnp.complex128), 
                                backend="jax")

            # l2t2_aa = 0.5 * oe.contract("gia,gjb,iajb->", 
            #                         glgp_a_c.astype(ctype), 
            #                         glgp_a_c.astype(ctype), 
            #                         t2_aa.astype(rtype), 
            #                         backend="jax")
            # l2t2_bb = 0.5 * oe.contract("gia,gjb,iajb->", 
            #                         glgp_b_c.astype(ctype), 
            #                         glgp_b_c.astype(ctype), 
            #                         t2_bb.astype(rtype), 
            #                         backend="jax")
            # l2t2_ab = oe.contract("gia,gjb,iajb->", 
            #                     glgp_a_c.astype(ctype), 
            #                     glgp_b_c.astype(ctype), 
            #                     t2_ab.astype(rtype), 
            #                     backend="jax")
            
            lt2_aa = oe.contract("gia,iajb->gjb", 
                                 glgp_a_c.astype(ctype), 
                                 t2_aa.astype(rtype), 
                                 backend="jax")
            lt2_bb = oe.contract("gia,iajb->gjb", 
                                 glgp_b_c.astype(ctype), 
                                 t2_bb.astype(rtype), 
                                 backend="jax")
            lt2_ab = oe.contract("gia,iajb->gjb", 
                                 glgp_a_c.astype(ctype), 
                                 t2_ab.astype(rtype), 
                                 backend="jax")
            
            l2t2_aa = 0.5 * oe.contract("gjb,gjb->",
                                        lt2_aa.astype(ctype),
                                        glgp_a_c.astype(ctype), 
                                        backend="jax").astype(jnp.complex128)
            l2t2_bb = 0.5 * oe.contract("gjb,gjb->",
                                        lt2_bb.astype(ctype),
                                        glgp_b_c.astype(ctype), 
                                        backend="jax").astype(jnp.complex128)
            l2t2_ab = oe.contract("gjb,gjb->",
                                  lt2_ab.astype(ctype),
                                  glgp_b_c.astype(ctype), 
                                  backend="jax").astype(jnp.complex128)
            
            # l2t2_aa = 0.5 * oe.contract("gjb,gjb->",
            #                             lt2_aa.astype(ctype),
            #                             glgp_a_c.astype(ctype), 
            #                             backend="jax").astype(jnp.complex128)
            # l2t2_bb = 0.5 * oe.contract("gjb,gjb->",
            #                             lt2_bb.astype(ctype),
            #                             glgp_b_c.astype(ctype),
            #                             backend="jax").astype(jnp.complex128)
            # l2t2_ab = oe.contract("gjb,gjb->",
            #                       lt2_ab.astype(ctype),
            #                       glgp_b_c.astype(ctype), 
            #                       backend="jax").astype(jnp.complex128)
            
            carry[3] += (l2t2_aa + l2t2_bb + l2t2_ab).astype(jnp.complex128)
            return carry, 0.0

        [e2_0, e2_2_2_1, e2_2_2_2, e2_2_3], _ = lax.scan(
            scanned_fun, [0.0, 0.0, 0.0, 0.0], (chol_a, chol_b)
        )

        e2_2_1 = e2_0 * gt2g
        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <exp(T1)HF|T2 h2|walker>/<exp(T1)HF|walker>

        # o0 = jnp.linalg.det(walker_up[:nocc_a,:nocc_a]
        #     ) * jnp.linalg.det(walker_dn[:nocc_b,:nocc_b])
        # o0 = self._calc_overlap(walker_up, walker_dn, wave_data)
        # <exp(T1)HF|walker>/<HF|walker>
        ot1 = jnp.linalg.det(wave_data["mo_ta"].T.conj() @ walker_up
            ) * jnp.linalg.det(wave_data["mo_tb"].T.conj() @ walker_dn)
        t2 = gt2g # * t1 # <exp(T1)HF|T2|walker>/<exp(T1)HF|walker>
        e0 = (e1_0 + e2_0) # * t1 # <exp(T1)HF|h1+h2|walker>/<exp(T1)HF|walker>
        e1 = (e1_2 + e2_2) # * t1 # <exp(T1)HF|T2 (h1+h2)|walker>/<exp(T1)HF|walker>

        return ot1, t2, e0, e1
    
    # @partial(jit, static_argnums=0)
    # def _mapped_energy_pt(self, walker_up, walker_dn, ham_data, wave_data):
    #     return vmap(self._calc_energy_pt, in_axes=(0, 0, None, None))(
    #         walker_up, walker_dn, ham_data, wave_data)
    
    # def calc_energy_pt(self, walkers, ham_data, wave_data):
    #     devices = jax.devices()
    #     n_dev = len(devices)

    #     # 1D mesh over all GPUs; name the axis "walkers"
    #     mesh = Mesh(np.array(devices), axis_names=("walkers",))
    #     walker_sharding = NamedSharding(mesh, P("walkers"))  # split axis 0
    #     replicated      = NamedSharding(mesh, P())           # copy to every GPU

    #     walker_up, walker_dn = walkers[0], walkers[1]
    #     n_walkers = walker_up.shape[0]

    #     # The walker axis must be divisible by n_dev. Pad by duplicating
    #     # valid walkers (safe — won't make mo_a.T@walker_up singular), slice off later.
    #     pad = (-n_walkers) % n_dev
    #     if pad:
    #         walker_up = jnp.concatenate([walker_up, walker_up[:pad]], axis=0)
    #         walker_dn = jnp.concatenate([walker_dn, walker_dn[:pad]], axis=0)

    #     # Distribute the data
    #     walker_up = jax.device_put(walker_up, walker_sharding)
    #     walker_dn = jax.device_put(walker_dn, walker_sharding)
    #     ham_data  = jax.device_put(ham_data,  replicated)
    #     wave_data = jax.device_put(wave_data, replicated)

    #     t1, t2, e0, e1 = self._mapped_energy_pt(
    #         walker_up, walker_dn, ham_data, wave_data)

    #     if pad:
    #         t1, t2, e0, e1 = (x[:n_walkers] for x in (t1, t2, e0, e1))
    #     return t1, t2, e0, e1

    def calc_energy_pt(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
        ot1, t2, e0, e1 = vmap(
            self._calc_energy_pt, in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data, wave_data)
        return ot1, t2, e0, e1
    

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
    
@dataclass
class upt2ccsd_bar(upt2ccsd):
    """apply exp(T1) to the right"""

    norb: int
    nelec: Tuple[int, int]
    nchol_chunk: int = 100
    mix_precision: bool = False
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_energy_pt(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
        '''
        calculate terms related to <bra|T2(h1+h2)|ket>/<bra|ket>
        bra is assumed to be an identity in its mo_coeff
        return <bra|ket> <T2> <h1+h2> <T2(h1+h2)> 
        '''
        # only do this for two-body energy with T contraction
        if self.mix_precision:
            rtype = jnp.float32
            ctype = jnp.complex64
        else:
            rtype = jnp.float64
            ctype = jnp.complex128

        norb_a, nocc_a = walker_up.shape
        norb_b, nocc_b = walker_dn.shape

        # o0 = jnp.linalg.det(walker_up[:nocc_a,:]) \
        #     * jnp.linalg.det(walker_dn[:nocc_b,:])
        # o0 = self._calc_overlap(walker_up, walker_dn, wave_data)
        
        t2aa = wave_data["t2aa"]
        t2ab = wave_data["t2ab"]
        t2bb = wave_data["t2bb"]

        chol_a = ham_data["chol_bar"][0]
        chol_b = ham_data["chol_bar"][1]
        h1_a = ham_data["h1_bar"][0]
        h1_b = ham_data["h1_bar"][1]
        
        walker_up = wave_data['exp_t1a'] @ walker_up
        walker_dn = wave_data['exp_t1b'] @ walker_dn

        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a,:]))).T
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[:nocc_b,:]))).T
        greenov_a = green_a[:nocc_a,nocc_a:]
        greenov_b = green_b[:nocc_b,nocc_b:]
        greenp_a = jnp.vstack((greenov_a, -jnp.eye(norb_a-nocc_a)))
        greenp_b = jnp.vstack((greenov_b, -jnp.eye(norb_b-nocc_b)))

        hg_a = oe.contract("pq,pq->", h1_a[:nocc_a,:], green_a, backend="jax")
        hg_b = oe.contract("pq,pq->", h1_b[:nocc_b,:], green_b, backend="jax")
        e1_0 = hg_a + hg_b # <bra|h1|ket>/<bra|ket>

        # <bra|T2 h1|ket>/<bra|ket>
        t2g_a = oe.contract("iajb,ia->jb", t2aa, greenov_a, backend="jax") / 4
        t2g_b = oe.contract("iajb,ia->jb", t2bb, greenov_b, backend="jax") / 4
        t2g_ab_a = oe.contract("iajb,jb->ia", t2ab, greenov_b, backend="jax")
        t2g_ab_b = oe.contract("iajb,ia->jb", t2ab, greenov_a, backend="jax")
        # t_iajb (G_ia G_jb - G_ib G_ja)
        gt2g_a = oe.contract("jb,jb->", t2g_a, greenov_a, backend="jax")
        gt2g_b = oe.contract("jb,jb->", t2g_b, greenov_b, backend="jax")
        gt2g_ab = oe.contract("ia,ia->", t2g_ab_a, greenov_a, backend="jax")
        gt2g = 2 * (gt2g_a + gt2g_b) + gt2g_ab # <bra|T2|ket>/<bra|ket>
        e1_2_1 = e1_0 * gt2g
        
        t2_green_aaa = (greenp_a @ t2g_a.T) @ green_a[:nocc_a,:] # Gp_pb t_iajb G_ia G_jq (-)
        t2_green_aba = (greenp_a @ t2g_ab_a.T) @ green_a[:nocc_a,:]
        t2_green_bbb = (greenp_b @ t2g_b.T) @ green_b[:nocc_b,:]
        t2_green_abb = (greenp_b @ t2g_ab_b.T) @ green_b[:nocc_b,:]
        t2_green_a_a = 4 * t2_green_aaa + t2_green_aba # connect a->a
        t2_green_b_b = 4 * t2_green_bbb + t2_green_abb # connect b->b

        e1_2_2_a = -oe.contract("pq,pq->", h1_a, t2_green_a_a, backend="jax")
        e1_2_2_b = -oe.contract("pq,pq->", h1_b, t2_green_b_b, backend="jax")
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2  # <bra|T2 h1|ket>/<bra|ket>

        # <bra|T2 h2|ket>/<bra|ket>
        nchol = chol_a.shape[0]
        nchunks = -(-nchol // self.nchol_chunk)
        pad = nchunks * self.nchol_chunk - nchol
        chol_a = jnp.pad(chol_a, ((0, pad), (0, 0), (0, 0)))
        chol_b = jnp.pad(chol_b, ((0, pad), (0, 0), (0, 0)))
        chol_a = chol_a.reshape(nchunks, self.nchol_chunk, *chol_a.shape[-2:])
        chol_b = chol_b.reshape(nchunks, self.nchol_chunk, *chol_b.shape[-2:])

        def scanned_fun(carry, x):
            chol_a_c, chol_b_c = x

            # e2_0 = <h2>
            gl_a_c = oe.contract("ir,gpr->gip",
                                green_a.astype(jnp.complex128),
                                chol_a_c.astype(jnp.float64), 
                                backend="jax").astype(jnp.complex128)
            gl_b_c = oe.contract("ir,gpr->gip", 
                                green_b.astype(jnp.complex128),
                                chol_b_c.astype(jnp.float64), 
                                backend="jax")
            tr_gl_a = oe.contract("gii->g", 
                                gl_a_c[:,:nocc_a,:nocc_a].astype(jnp.complex128), 
                                backend="jax").astype(jnp.complex128)
            tr_gl_b = oe.contract("gii->g", 
                                gl_b_c[:,:nocc_b,:nocc_b].astype(jnp.complex128), 
                                backend="jax").astype(jnp.complex128)
            ex_gl_a = oe.contract("gij,gji->g", 
                                gl_a_c[:,:nocc_a,:nocc_a].astype(jnp.complex128), 
                                gl_a_c[:,:nocc_a,:nocc_a].astype(jnp.complex128), 
                                backend="jax").astype(jnp.complex128)
            ex_gl_b = oe.contract("gij,gji->g", 
                                gl_b_c[:,:nocc_b,:nocc_b].astype(jnp.complex128), 
                                gl_b_c[:,:nocc_b,:nocc_b].astype(jnp.complex128), 
                                backend="jax").astype(jnp.complex128)
            e2_0_1_c = jnp.sum((tr_gl_a + tr_gl_b) ** 2) / 2.0
            e2_0_2_c = -jnp.sum(ex_gl_a + ex_gl_b) / 2.0

            carry[0] += (e2_0_1_c + e2_0_2_c).astype(jnp.complex128)

            # e2_2 = <T2 h2>
            lt2g_a_c = oe.contract("gpr,qr->gpq", 
                                    chol_a_c.astype(jnp.float64), 
                                    (2*t2_green_a_a).astype(jnp.complex128), 
                                    backend="jax")
            lt2g_b_c = oe.contract("gpr,qr->gpq", 
                                    chol_b_c.astype(jnp.float64), 
                                    (2*t2_green_b_b).astype(jnp.complex128), 
                                    backend="jax")
            tr_lt2g_a_c = oe.contract("gqq->g", lt2g_a_c.astype(jnp.complex128), backend="jax")
            tr_lt2g_b_c = oe.contract("gqq->g", lt2g_b_c.astype(jnp.complex128), backend="jax")

            carry[1] += -(((tr_lt2g_a_c.astype(ctype) + tr_lt2g_b_c.astype(ctype)) 
                            @ (tr_gl_a.astype(ctype) + tr_gl_b.astype(ctype))
                            ) / 2).astype(jnp.complex128)
            
            carry[2] += ((oe.contract("giq,giq->", 
                                        gl_a_c.astype(ctype), 
                                        lt2g_a_c[:,:nocc_a,:].astype(ctype), 
                                        backend="jax")
                        + oe.contract("giq,giq->", 
                                        gl_b_c.astype(ctype), 
                                        lt2g_b_c[:,:nocc_b,:].astype(ctype), 
                                        backend="jax")) / 2).astype(jnp.complex128)

            glgp_a_c = oe.contract("giq,qa->gia",
                                gl_a_c.astype(jnp.complex128), 
                                greenp_a.astype(jnp.complex128), 
                                backend="jax")
            glgp_b_c = oe.contract("giq,qa->gia", 
                                gl_b_c.astype(jnp.complex128), 
                                greenp_b.astype(jnp.complex128), 
                                backend="jax")
            
            lt2_aa = oe.contract("gia,iajb->gjb", 
                                    glgp_a_c.astype(ctype), 
                                    t2aa.astype(rtype), 
                                    backend="jax")
            lt2_bb = oe.contract("gia,iajb->gjb", 
                                    glgp_b_c.astype(ctype), 
                                    t2bb.astype(rtype), 
                                    backend="jax")
            lt2_ab = oe.contract("gia,iajb->gjb", 
                                    glgp_a_c.astype(ctype), 
                                    t2ab.astype(rtype), 
                                    backend="jax")
            
            l2t2_aa = 0.5 * oe.contract("gjb,gjb->",
                                        lt2_aa.astype(ctype),
                                        glgp_a_c.astype(ctype), 
                                        backend="jax").astype(jnp.complex128)
            l2t2_bb = 0.5 * oe.contract("gjb,gjb->",
                                        lt2_bb.astype(ctype),
                                        glgp_b_c.astype(ctype), 
                                        backend="jax").astype(jnp.complex128)
            l2t2_ab = oe.contract("gjb,gjb->",
                                    lt2_ab.astype(ctype),
                                    glgp_b_c.astype(ctype), 
                                    backend="jax").astype(jnp.complex128)
            
            carry[3] += (l2t2_aa + l2t2_bb + l2t2_ab).astype(jnp.complex128)
            return carry, 0.0

        [e2_0, e2_2_2_1, e2_2_2_2, e2_2_3], _ = lax.scan(
            scanned_fun, [0.0, 0.0, 0.0, 0.0], (chol_a, chol_b)
        )

        e2_2_1 = e2_0 * gt2g
        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <bra|T2 h2|ket>/<bra|ket>

        ot1 = jnp.linalg.det(walker_up[:nocc_a,:]) \
            * jnp.linalg.det(walker_dn[:nocc_b,:]) # <bra|ket_bar>
        t2 = gt2g # * t1o # <bra|T2|ket_bar>/<bra|ket>
        e0 = (e1_0 + e2_0) # * t1o # <bra|h1+h2|ket_bar>/<bra|ket>
        e1 = (e1_2 + e2_2) # * t1o # <bra|T2 (h1+h2)|ket_bar>/<bra|ket>

        return ot1, t2, e0, e1
    
    @partial(jit, static_argnums=(0,))
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        norb = self.norb
        h1a, h1b = ham_data["h1"]
        chola = ham_data["chol"][0].reshape(-1, norb, norb)
        cholb = ham_data["chol"][1].reshape(-1, norb, norb)
        moa, mob = wave_data["mo_coeff"]

        ham_data["rot_h1"] = [moa.T.conj() @ h1a, mob.T.conj() @ h1b]

        ham_data["rot_chol"] = [
            oe.contract("pi,gij->gpj", moa.T.conj(), chola, backend="jax"),
            oe.contract("pi,gij->gpj", mob.T.conj(), cholb, backend="jax")]
        
        h1bar_a = wave_data['exp_t1a'] @ h1a @ wave_data['exp_mt1a']
        h1bar_b = wave_data['exp_t1b'] @ h1b @ wave_data['exp_mt1b']
        ham_data["h1_bar"] = [h1bar_a, h1bar_b]

        chol_bar_a = oe.contract(
            'pr,grs,sq->gpq', 
            wave_data['exp_t1a'], 
            chola, 
            wave_data['exp_mt1a'], 
            backend='jax')
        chol_bar_b = oe.contract(
            'pr,grs,sq->gpq', 
            wave_data['exp_t1b'], 
            cholb, 
            wave_data['exp_mt1b'], 
            backend='jax')
        ham_data["chol_bar"] = [chol_bar_a, chol_bar_b]

        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class upt2ccsd_sto_chol(upt2ccsd_bar):
    """upt2CCSD with a semistochastic Cholesky sum in the T2-contracted energy.

    Unrestricted counterpart of pt2ccsd_sto_chol.  Same split: e2_0 -- and hence
    e2_2_1 = e2_0 * gt2g -- is always exact, because the per-Cholesky two-body
    energies are needed to build the sampling proposal anyway and cost only the
    gl = green.chol contraction.  The three accumulators that contract with T2
    (whose "gia,iajb->gjb" contractions scale as nocc^2 nvir^2 per vector and
    dominate) are split into an exactly summed head and an importance sampled
    tail.  Both spin channels use the same head/tail split and the same draws,
    scored by the combined alpha+beta two-body energy.

    Knobs (dataclass fields, static under jit):
      n_chol_head    : head size; takes precedence over head_chol_ratio.  0
                       defers to the ratio; a positive int sets it explicitly;
                       "full" puts every vector in the head, disabling sampling
                       and reproducing upt2ccsd_bar exactly
      head_chol_ratio: head as a fraction of nchol when n_chol_head == 0
                       (default 0.125 = nchol/8)
      n_chol_samples : tail draws per walker per block
      chol_score_floor / chol_uniform_mix : proposal guards

    Randomness arrives via wave_data["sto_chol_key"], refreshed each block by
    sampler_pt2_sto_chol.  The estimator is unbiased.
    """

    n_chol_head: Union[int, str] = 0
    head_chol_ratio: float = 0.125
    n_chol_samples: int = 128
    chol_score_floor: float = 1.0e-6
    chol_uniform_mix: float = 0.01
    head_from_guide: bool = False

    @partial(jit, static_argnums=0)
    def _prop_chol_in_place(self, e2_g_estimate):
        """pi_g = (1-u) |e2_g| / sum|e2_g| + u/nchol, with a relative floor.

        The uniform component keeps pi_g > 0 for every vector, which bounds the
        importance weights 1/pi_g and is what makes the estimator unbiased.
        """
        e2_g = jnp.abs(e2_g_estimate)
        e2_g = jnp.where(e2_g >= self.chol_score_floor * jnp.max(e2_g), e2_g, 0.0)
        nchol = e2_g.shape[0]
        uniform = jnp.full((nchol,), 1.0 / nchol)
        total = jnp.sum(e2_g)
        guided = jnp.where(total > 0.0, e2_g / jnp.where(total > 0.0, total, 1.0), uniform)
        return (1.0 - self.chol_uniform_mix) * guided + self.chol_uniform_mix * uniform

    @partial(jit, static_argnums=0)
    def _calc_energy_pt(self, walker_up, walker_dn, ham_data, wave_data, key):
        if self.mix_precision:
            rtype, ctype = jnp.float32, jnp.complex64
        else:
            rtype, ctype = jnp.float64, jnp.complex128

        norb_a, nocc_a = walker_up.shape
        norb_b, nocc_b = walker_dn.shape
        nchol_chunk = self.nchol_chunk

        t2aa, t2ab, t2bb = wave_data["t2aa"], wave_data["t2ab"], wave_data["t2bb"]
        chol_a, chol_b = ham_data["chol_bar"][0], ham_data["chol_bar"][1]
        h1_a, h1_b = ham_data["h1_bar"][0], ham_data["h1_bar"][1]
        nchol = chol_a.shape[0]

        walker_up = wave_data['exp_t1a'] @ walker_up
        walker_dn = wave_data['exp_t1b'] @ walker_dn
        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a, :]))).T
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[:nocc_b, :]))).T
        greenov_a = green_a[:nocc_a, nocc_a:]
        greenov_b = green_b[:nocc_b, nocc_b:]
        greenp_a = jnp.vstack((greenov_a, -jnp.eye(norb_a - nocc_a)))
        greenp_b = jnp.vstack((greenov_b, -jnp.eye(norb_b - nocc_b)))

        # ---- Cholesky-independent pieces (identical to upt2ccsd_bar) ----
        hg_a = oe.contract("pq,pq->", h1_a[:nocc_a, :], green_a, backend="jax")
        hg_b = oe.contract("pq,pq->", h1_b[:nocc_b, :], green_b, backend="jax")
        e1_0 = hg_a + hg_b

        t2g_a = oe.contract("iajb,ia->jb", t2aa, greenov_a, backend="jax") / 4
        t2g_b = oe.contract("iajb,ia->jb", t2bb, greenov_b, backend="jax") / 4
        t2g_ab_a = oe.contract("iajb,jb->ia", t2ab, greenov_b, backend="jax")
        t2g_ab_b = oe.contract("iajb,ia->jb", t2ab, greenov_a, backend="jax")
        gt2g_a = oe.contract("jb,jb->", t2g_a, greenov_a, backend="jax")
        gt2g_b = oe.contract("jb,jb->", t2g_b, greenov_b, backend="jax")
        gt2g_ab = oe.contract("ia,ia->", t2g_ab_a, greenov_a, backend="jax")
        gt2g = 2 * (gt2g_a + gt2g_b) + gt2g_ab
        e1_2_1 = e1_0 * gt2g

        t2_green_aaa = (greenp_a @ t2g_a.T) @ green_a[:nocc_a, :]
        t2_green_aba = (greenp_a @ t2g_ab_a.T) @ green_a[:nocc_a, :]
        t2_green_bbb = (greenp_b @ t2g_b.T) @ green_b[:nocc_b, :]
        t2_green_abb = (greenp_b @ t2g_ab_b.T) @ green_b[:nocc_b, :]
        t2_green_a_a = 4 * t2_green_aaa + t2_green_aba
        t2_green_b_b = 4 * t2_green_bbb + t2_green_abb

        e1_2_2_a = -oe.contract("pq,pq->", h1_a, t2_green_a_a, backend="jax")
        e1_2_2_b = -oe.contract("pq,pq->", h1_b, t2_green_b_b, backend="jax")
        e1_2 = e1_2_1 + e1_2_2_a + e1_2_2_b

        # ============ pass 1: e2_0, exact, every gamma, no T2 ============
        # Scan the padded array directly, exactly as upt2ccsd_bar does -- no index
        # gather and no keep mask.  Padded Cholesky vectors are zero, so they
        # contribute nothing on their own.  The padded copy is built once here and
        # reused by pass 2 when the head is the whole set.
        # e2_0 only ever touches the occupied-occupied block of gl, so feed this
        # pass the half-rotated chol[:, :nocc, :] rather than the full
        # (norb, norb) three-index tensor.  gl then comes out (chunk, nocc, nocc)
        # instead of (chunk, nocc, norb) -- under vmap that factor of norb/nocc is
        # paid per walker, so it is the dominant cost of this pass.
        rot_a = chol_a[:, :nocc_a, :]
        rot_b = chol_b[:, :nocc_b, :]
        nchunk, chunk1, npad = _chol_chunking(nchol, nchol_chunk)
        if npad:
            rot_a = jnp.concatenate(
                [rot_a, jnp.zeros((npad, *rot_a.shape[-2:]), rot_a.dtype)], axis=0)
            rot_b = jnp.concatenate(
                [rot_b, jnp.zeros((npad, *rot_b.shape[-2:]), rot_b.dtype)], axis=0)

        def scan_chol_chunk_e2_0(carry, x):
            rot_a_c, rot_b_c = x
            # (chunk, nocc_a, nocc_a) and (chunk, nocc_b, nocc_b)
            gl_a = oe.contract("ir,gpr->gip", green_a.astype(jnp.complex128),
                               rot_a_c.astype(jnp.float64), backend="jax")
            gl_b = oe.contract("ir,gpr->gip", green_b.astype(jnp.complex128),
                               rot_b_c.astype(jnp.float64), backend="jax")
            tr_a = oe.contract("gii->g", gl_a, backend="jax")
            tr_b = oe.contract("gii->g", gl_b, backend="jax")
            ex_a = oe.contract("gij,gji->g", gl_a, gl_a, backend="jax")
            ex_b = oe.contract("gij,gji->g", gl_b, gl_b, backend="jax")
            e2_0_g = (((tr_a + tr_b) ** 2 - (ex_a + ex_b)) / 2.0).astype(ctype)
            return carry + jnp.sum(e2_0_g), e2_0_g

        e2_0, e2_0_chunks = lax.scan(
            scan_chol_chunk_e2_0, jnp.zeros((), dtype=ctype),
            (rot_a.reshape(nchunk, chunk1, *rot_a.shape[-2:]),
             rot_b.reshape(nchunk, chunk1, *rot_b.shape[-2:])))
        e2_0_g = e2_0_chunks.reshape(-1)[:nchol]

        # ---- proposal, head/tail split ----
        if isinstance(self.n_chol_head, str):
            if self.n_chol_head.lower() != "full":
                raise ValueError(
                    f"n_chol_head must be an int or 'full', got {self.n_chol_head!r}.")
            n_head = nchol
        elif self.n_chol_head > 0:
            n_head = self.n_chol_head
        else:
            if not 0.0 <= self.head_chol_ratio <= 1.0:
                raise ValueError(
                    f"head_chol_ratio must lie in [0, 1], got {self.head_chol_ratio}.")
            n_head = min(max(int(round(self.head_chol_ratio * nchol)), 0), nchol)
        n_samples = self.n_chol_samples

        # The head is a contiguous prefix by default, NOT ranked per walker.  That
        # matters for memory, not just tidiness: under vmap a walker-dependent index
        # array turns chol_a[idx] into a *batched* gather, so one head chunk costs
        # n_walkers * n_head * norb^2 instead of n_head * norb^2 shared.  A prefix is
        # a plain slice, shared across the batch.  Cholesky vectors come out of the
        # decomposition in decreasing importance, so the prefix is already a sensible
        # head.  head_from_guide=True restores per-walker ranking at that memory cost.
        head_prefix = None
        if n_head >= nchol:
            head_prefix = nchol
            tail = jnp.zeros((0,), dtype=jnp.int32)
            tail_prob = jnp.zeros((0,))
        else:
            pi_g = self._prop_chol_in_place(e2_0_g)
            if self.head_from_guide:
                order = jnp.argsort(-pi_g)
                head_idx = jnp.sort(order[:n_head])
                tail = jnp.sort(order[n_head:])
            else:
                head_prefix = n_head
                tail = jnp.arange(n_head, nchol, dtype=jnp.int32)
            tail_prob = pi_g[tail]
            tail_prob = tail_prob / jnp.sum(tail_prob)

        # ======== pass 2: only the e2_2 terms that contract with T2 ========
        def scan_chol_chunk_e2_2(carry, x):
            chol_a_c, chol_b_c, w_c = x
            w_c = w_c.astype(ctype)

            gl_a = oe.contract("ir,gpr->gip", green_a.astype(jnp.complex128),
                               chol_a_c.astype(jnp.float64), backend="jax")
            gl_b = oe.contract("ir,gpr->gip", green_b.astype(jnp.complex128),
                               chol_b_c.astype(jnp.float64), backend="jax")
            tr_a = oe.contract("gii->g", gl_a[:, :nocc_a, :nocc_a], backend="jax")
            tr_b = oe.contract("gii->g", gl_b[:, :nocc_b, :nocc_b], backend="jax")

            lt2g_a = oe.contract("gpr,qr->gpq", chol_a_c.astype(jnp.float64),
                                 (2 * t2_green_a_a).astype(jnp.complex128), backend="jax")
            lt2g_b = oe.contract("gpr,qr->gpq", chol_b_c.astype(jnp.float64),
                                 (2 * t2_green_b_b).astype(jnp.complex128), backend="jax")
            tr_lt2g_a = oe.contract("gqq->g", lt2g_a, backend="jax")
            tr_lt2g_b = oe.contract("gqq->g", lt2g_b, backend="jax")

            carry[0] += jnp.sum(w_c * (-((tr_lt2g_a.astype(ctype) + tr_lt2g_b.astype(ctype))
                                         * (tr_a.astype(ctype) + tr_b.astype(ctype))) / 2))

            carry[1] += jnp.sum(w_c * ((oe.contract("giq,giq->g", gl_a.astype(ctype),
                                                    lt2g_a[:, :nocc_a, :].astype(ctype),
                                                    backend="jax")
                                        + oe.contract("giq,giq->g", gl_b.astype(ctype),
                                                      lt2g_b[:, :nocc_b, :].astype(ctype),
                                                      backend="jax")) / 2))

            glgp_a = oe.contract("giq,qa->gia", gl_a, greenp_a.astype(jnp.complex128),
                                 backend="jax")
            glgp_b = oe.contract("giq,qa->gia", gl_b, greenp_b.astype(jnp.complex128),
                                 backend="jax")
            lt2_aa = oe.contract("gia,iajb->gjb", glgp_a.astype(ctype),
                                 t2aa.astype(rtype), backend="jax")
            lt2_bb = oe.contract("gia,iajb->gjb", glgp_b.astype(ctype),
                                 t2bb.astype(rtype), backend="jax")
            lt2_ab = oe.contract("gia,iajb->gjb", glgp_a.astype(ctype),
                                 t2ab.astype(rtype), backend="jax")
            l2t2_aa = 0.5 * oe.contract("gjb,gjb->g", lt2_aa.astype(ctype),
                                        glgp_a.astype(ctype), backend="jax")
            l2t2_bb = 0.5 * oe.contract("gjb,gjb->g", lt2_bb.astype(ctype),
                                        glgp_b.astype(ctype), backend="jax")
            l2t2_ab = oe.contract("gjb,gjb->g", lt2_ab.astype(ctype),
                                  glgp_b.astype(ctype), backend="jax")
            carry[2] += jnp.sum(w_c * (l2t2_aa + l2t2_bb + l2t2_ab).astype(ctype))
            return carry, 0.0

        def _run(chol_a_s, chol_b_s, weights):
            n = weights.shape[0]
            z = jnp.zeros((), dtype=ctype)
            if n == 0:
                return z, z, z
            nch2, chunk2, npad2 = _chol_chunking(n, nchol_chunk)
            if npad2:
                chol_a_s = jnp.concatenate(
                    [chol_a_s, jnp.zeros((npad2, *chol_a_s.shape[-2:]), chol_a_s.dtype)])
                chol_b_s = jnp.concatenate(
                    [chol_b_s, jnp.zeros((npad2, *chol_b_s.shape[-2:]), chol_b_s.dtype)])
                weights = jnp.concatenate([weights, jnp.zeros(npad2, weights.dtype)])
            out, _ = lax.scan(
                scan_chol_chunk_e2_2, [z, z, z],
                (chol_a_s.reshape(nch2, chunk2, *chol_a_s.shape[-2:]),
                 chol_b_s.reshape(nch2, chunk2, *chol_b_s.shape[-2:]),
                 weights.reshape(nch2, chunk2)))
            return out[0], out[1], out[2]

        def accumulate_prefix(n):
            # contiguous slice -- no gather, so this is shared across the vmap batch
            return _run(chol_a[:n], chol_b[:n], jnp.ones(n, dtype=ctype))

        def accumulate_gather(indices, weights):
            # Walker-dependent indices, so these Cholesky vectors cannot be shared
            # across the vmap batch -- each walker holds its own copy.  That cost is
            # unavoidable once the sample is drawn; scanning the gathered vectors in
            # chunks is what we do with them.
            return _run(chol_a[indices], chol_b[indices], weights)

        if head_prefix is not None:
            b_h, c_h, d_h = accumulate_prefix(head_prefix)
        else:
            b_h, c_h, d_h = accumulate_gather(
                head_idx, jnp.ones(head_idx.shape[0], dtype=ctype))

        if tail.shape[0] == 0:
            b_t = c_t = d_t = jnp.zeros((), dtype=ctype)
        else:
            sel = random.choice(key, tail.shape[0], shape=(n_samples,),
                                replace=True, p=tail_prob)
            samp_w = (1.0 / (n_samples * tail_prob[sel])).astype(ctype)
            b_t, c_t, d_t = accumulate_gather(tail[sel], samp_w)

        # e2_2_1 = e2_0 * gt2g is exact, since e2_0 is exact
        e2_2 = e2_0 * gt2g + (b_h + b_t) + (c_h + c_t) + (d_h + d_t)

        ot1 = jnp.linalg.det(walker_up[:nocc_a, :]) * jnp.linalg.det(walker_dn[:nocc_b, :])
        return ot1, gt2g, e1_0 + e2_0, e1_2 + e2_2

    @partial(jit, static_argnums=0)
    def calc_energy_pt(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
        """Map over walkers, giving each its own key split from the block key."""
        n_walkers = walkers[0].shape[0]
        key = wave_data.get("sto_chol_key", random.PRNGKey(0))
        keys = random.split(key, n_walkers)
        ot1, t2, e0, e1 = vmap(
            self._calc_energy_pt, in_axes=(0, 0, None, None, 0))(
            walkers[0], walkers[1], ham_data, wave_data, keys)
        return ot1, t2, e0, e1

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class upt2ccsd_red(upt2ccsd_bar):

    @partial(jit, static_argnums=0)
    def _calc_energy_pt(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
        '''
        <bra|T2(h1+h2)|ket>/<bra|ket> with rank-decomposed T2 (be careful: tau is complex!):
            t2aa = 2 * sum_y tau_a[y,i,a] tau_a[y,j,b]
            t2ab =     sum_y tau_a[y,i,a] tau_b[y,j,b]
            t2bb = 2 * sum_y tau_b[y,i,a] tau_b[y,j,b]
        '''
        if self.mix_precision:
            rtype = jnp.float32
            ctype = jnp.complex64
        else:
            rtype = jnp.float64
            ctype = jnp.complex128

        norb_a, nocc_a = walker_up.shape
        norb_b, nocc_b = walker_dn.shape

        tau_a = wave_data["tau_a"]          # (ny, nocc_a, nvir_a)
        tau_b = wave_data["tau_b"]          # (ny, nocc_b, nvir_b)

        chol_a = ham_data["chol_bar"][0]
        chol_b = ham_data["chol_bar"][1]
        h1_a = ham_data["h1_bar"][0]
        h1_b = ham_data["h1_bar"][1]

        walker_up = wave_data['exp_t1a'] @ walker_up
        walker_dn = wave_data['exp_t1b'] @ walker_dn

        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a, :]))).T
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[:nocc_b, :]))).T
        greenov_a = green_a[:nocc_a, nocc_a:]
        greenov_b = green_b[:nocc_b, nocc_b:]
        greenp_a = jnp.vstack((greenov_a, -jnp.eye(norb_a - nocc_a)))
        greenp_b = jnp.vstack((greenov_b, -jnp.eye(norb_b - nocc_b)))

        hg_a = oe.contract("pq,pq->", h1_a[:nocc_a, :], green_a, backend="jax")
        hg_b = oe.contract("pq,pq->", h1_b[:nocc_b, :], green_b, backend="jax")
        e1_0 = hg_a + hg_b  # <bra|h1|ket>/<bra|ket>

        # ---- <bra|T2 h1|ket>: scan over the decomposition rank y ----
        def scan_tau1(carry, x):
            tau_a_y, tau_b_y = x
            # spin a
            taug_a   = oe.contract("ia,ja->ij", tau_a_y, greenov_a, backend="jax")
            tr_a     = oe.contract("ii->", taug_a, backend="jax")
            taugp_a  = oe.contract("jb,pb->jp", tau_a_y, greenp_a, backend="jax")
            taugpg_a = oe.contract("jp,jq->pq", taugp_a, green_a[:nocc_a, :], backend="jax")
            # spin b
            taug_b   = oe.contract("ia,ja->ij", tau_b_y, greenov_b, backend="jax")
            tr_b     = oe.contract("ii->", taug_b, backend="jax")
            taugp_b  = oe.contract("jb,pb->jp", tau_b_y, greenp_b, backend="jax")
            taugpg_b = oe.contract("jp,jq->pq", taugp_b, green_b[:nocc_b, :], backend="jax")

            carry[0] += tr_a ** 2 + tr_b ** 2 + tr_a * tr_b   # gt2g   = <bra|T2|ket>
            carry[1] += (2 * tr_a + tr_b) * taugpg_a          # t2_green_a_a (4*aaa + aba)
            carry[2] += (2 * tr_b + tr_a) * taugpg_b          # t2_green_b_b (4*bbb + abb)
            return carry, 0.0

        init = [jnp.zeros((), jnp.complex128),                 # gt2g          (scalar)
                jnp.zeros((norb_a, norb_a), jnp.complex128),   # t2_green_a_a  (matrix)
                jnp.zeros((norb_b, norb_b), jnp.complex128)]   # t2_green_b_b  (matrix)
        [gt2g, t2_green_a_a, t2_green_b_b], _ = lax.scan(scan_tau1, init, (tau_a, tau_b))

        e1_2_1 = e1_0 * gt2g
        e1_2_2_a = -oe.contract("pq,pq->", h1_a, t2_green_a_a, backend="jax")
        e1_2_2_b = -oe.contract("pq,pq->", h1_b, t2_green_b_b, backend="jax")
        e1_2 = e1_2_1 + e1_2_2_a + e1_2_2_b  # <bra|T2 h1|ket>/<bra|ket>

        # ---- <bra|T2 h2|ket>: scan over cholesky chunks ----
        nchol = chol_a.shape[0]
        nchunks = -(-nchol // self.nchol_chunk)
        pad = nchunks * self.nchol_chunk - nchol
        chol_a = jnp.pad(chol_a, ((0, pad), (0, 0), (0, 0)))
        chol_b = jnp.pad(chol_b, ((0, pad), (0, 0), (0, 0)))
        chol_a = chol_a.reshape(nchunks, self.nchol_chunk, *chol_a.shape[-2:])
        chol_b = chol_b.reshape(nchunks, self.nchol_chunk, *chol_b.shape[-2:])

        def scanned_fun(carry, x):
            chol_a_c, chol_b_c = x

            # e2_0 = <h2>   (no T2 — identical to the dense function)
            gl_a_c = oe.contract("ir,gpr->gip",
                                    green_a.astype(jnp.complex128),
                                    chol_a_c.astype(jnp.float64),
                                    backend="jax").astype(jnp.complex128)
            gl_b_c = oe.contract("ir,gpr->gip",
                                    green_b.astype(jnp.complex128),
                                    chol_b_c.astype(jnp.float64),
                                    backend="jax")
            tr_gl_a = oe.contract("gii->g", gl_a_c[:, :nocc_a, :nocc_a], backend="jax").astype(jnp.complex128)
            tr_gl_b = oe.contract("gii->g", gl_b_c[:, :nocc_b, :nocc_b], backend="jax").astype(jnp.complex128)
            ex_gl_a = oe.contract("gij,gji->g", gl_a_c[:, :nocc_a, :nocc_a], gl_a_c[:, :nocc_a, :nocc_a], backend="jax").astype(jnp.complex128)
            ex_gl_b = oe.contract("gij,gji->g", gl_b_c[:, :nocc_b, :nocc_b], gl_b_c[:, :nocc_b, :nocc_b], backend="jax").astype(jnp.complex128)
            e2_0_1_c = jnp.sum((tr_gl_a + tr_gl_b) ** 2) / 2.0
            e2_0_2_c = -jnp.sum(ex_gl_a + ex_gl_b) / 2.0
            carry[0] += (e2_0_1_c + e2_0_2_c).astype(jnp.complex128)

            # e2_2_2 = <T2 h2> pieces that only need t2_green (no T2 directly)
            lt2g_a_c = oe.contract("gpr,qr->gpq", chol_a_c.astype(jnp.float64),
                                    (2 * t2_green_a_a).astype(jnp.complex128), backend="jax")
            lt2g_b_c = oe.contract("gpr,qr->gpq", chol_b_c.astype(jnp.float64),
                                    (2 * t2_green_b_b).astype(jnp.complex128), backend="jax")
            tr_lt2g_a_c = oe.contract("gqq->g", lt2g_a_c, backend="jax")
            tr_lt2g_b_c = oe.contract("gqq->g", lt2g_b_c, backend="jax")
            carry[1] += -(((tr_lt2g_a_c.astype(ctype) + tr_lt2g_b_c.astype(ctype))
                            @ (tr_gl_a.astype(ctype) + tr_gl_b.astype(ctype))) / 2).astype(jnp.complex128)
            carry[2] += ((oe.contract("giq,giq->", gl_a_c.astype(ctype), lt2g_a_c[:, :nocc_a, :].astype(ctype), backend="jax")
                        + oe.contract("giq,giq->", gl_b_c.astype(ctype), lt2g_b_c[:, :nocc_b, :].astype(ctype), backend="jax")) / 2).astype(jnp.complex128)

            # e2_2_3 = decomposed T2 two-body — inner scan over y
            glgp_a_c = oe.contract("giq,qa->gia", gl_a_c, greenp_a.astype(jnp.complex128), backend="jax")
            glgp_b_c = oe.contract("giq,qa->gia", gl_b_c, greenp_b.astype(jnp.complex128), backend="jax")

            def scan_tau2(carry, x):
                tau_a_y, tau_b_y = x
                # A_s[g] = sum_ia glgp_s[g,i,a] tau_s[y,i,a]
                A_a = oe.contract("gia,ia->g", glgp_a_c.astype(ctype), tau_a_y.astype(ctype), backend="jax")
                A_b = oe.contract("gia,ia->g", glgp_b_c.astype(ctype), tau_b_y.astype(ctype), backend="jax")
                carry[0] += oe.contract("g,g->", A_a, A_a, backend="jax").astype(jnp.complex128)  # l2t2_aa
                carry[1] += oe.contract("g,g->", A_b, A_b, backend="jax").astype(jnp.complex128)  # l2t2_bb
                carry[2] += oe.contract("g,g->", A_a, A_b, backend="jax").astype(jnp.complex128)  # l2t2_ab
                return carry, 0.0

            init2 = [jnp.zeros((), jnp.complex128),
                    jnp.zeros((), jnp.complex128),
                    jnp.zeros((), jnp.complex128)]
            [l2t2_aa, l2t2_bb, l2t2_ab], _ = lax.scan(scan_tau2, init2, (tau_a, tau_b))
            carry[3] += (l2t2_aa + l2t2_bb + l2t2_ab).astype(jnp.complex128)
            return carry, 0.0

        init_c = [jnp.zeros((), jnp.complex128)] * 4
        [e2_0, e2_2_2_1, e2_2_2_2, e2_2_3], _ = lax.scan(
            scanned_fun, init_c, (chol_a, chol_b)
        )

        e2_2_1 = e2_0 * gt2g
        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3  # <bra|T2 h2|ket>/<bra|ket>

        ot1 = jnp.linalg.det(walker_up[:nocc_a, :]) \
            * jnp.linalg.det(walker_dn[:nocc_b, :])  # <bra|ket_bar>
        t2 = gt2g
        e0 = e1_0 + e2_0
        e1 = e1_2 + e2_2
        return ot1, t2, e0, e1


    def __hash__(self):
        return hash(tuple(self.__dict__.values()))

# @dataclass
# class upt2ccsd_eff(upt2ccsd):
#     """Tensor contraction form of the Spin-Unrestricted CCSD (exact T1) trial wave function."""

#     @partial(jit, static_argnums=0)
#     def _calc_energy_pt(
#         self,
#         walker_up: jax.Array,
#         walker_dn: jax.Array,
#         ham_data: dict,
#         wave_data: dict,
#     ) -> complex:
#         # CISD trial with half Green
#         nocc_a, nocc_b = self.nelec
#         norb = self.norb
#         nvir_a = norb - nocc_a
#         nvir_b = norb - nocc_b
#         t2_aa = wave_data["rot_t2aa"]
#         t2_bb = wave_data["rot_t2bb"]
#         t2_ab = wave_data["rot_t2ab"]
#         mo_a, mo_b = wave_data['mo_ta'], wave_data['mo_tb']
#         h1_a, h1_b = ham_data["rot_h1_ci"]
#         dh1_a, dh1_b = ham_data["d_h1_ci"] # delta_pb h_pq
#         chol_a = ham_data["rot_chol_ci"][0].reshape(-1, nocc_a, norb)
#         chol_b = ham_data["rot_chol_ci"][1].reshape(-1, nocc_b, norb)
#         dchol_a = ham_data["d_chol_ci"][0].reshape(-1, nvir_a, norb) # delta_pb L_gpq
#         dchol_b = ham_data["d_chol_ci"][1].reshape(-1, nvir_b, norb)

#         # half green's function G_pq
#         green_a = (walker_up @ (jnp.linalg.inv(mo_a.T @ walker_up))).T
#         green_b = (walker_dn @ (jnp.linalg.inv(mo_b.T @ walker_dn))).T

#         # ref one-body energy
#         hg_a = oe.contract("pq,rq->pr", h1_a, green_a, backend="jax")
#         hg_b = oe.contract("pq,rq->pr", h1_b, green_b, backend="jax")
#         trhg_a = oe.contract("pp->", hg_a, backend="jax")
#         trhg_b = oe.contract("pp->", hg_b, backend="jax")
#         e1_0 = trhg_a + trhg_b # <psi|h1|walker>/<psi|walker>

#         # <psi|T2 h1|walker>/<exp(T1)HF|walker>
#         # double excitations
#         t2g_aa_a = oe.contract("iajb,ia->jb", t2_aa, green_a[:nocc_a,nocc_a:], backend="jax") / 4
#         t2g_bb_b = oe.contract("iajb,ia->jb", t2_bb, green_b[:nocc_b,nocc_b:], backend="jax") / 4
#         t2g_ab_a = oe.contract("iajb,jb->ia", t2_ab, green_b[:nocc_b,nocc_b:], backend="jax")
#         t2g_ab_b = oe.contract("iajb,ia->jb", t2_ab, green_a[:nocc_a,nocc_a:], backend="jax")

#         t2gg_aa = oe.contract("jb,jq->bq", t2g_aa_a, green_a[:nocc_a,:]) # t_iajb G_ia G_jq
#         t2gg_bb = oe.contract("jb,jq->bq", t2g_bb_b, green_b[:nocc_b,:]) 
#         t2gg_ab = oe.contract("jb,jq->bq", t2g_ab_a, green_a[:nocc_a,:])
#         t2gg_ba = oe.contract("jb,jq->bq", t2g_ab_b, green_b[:nocc_b,:])

#         # t_iajb (G_ia G_jb - G_ib G_ja)
#         gt2g_a = oe.contract("jb,jb->", t2g_aa_a, green_a[:nocc_a,nocc_a:], backend="jax")
#         gt2g_b = oe.contract("jb,jb->", t2g_bb_b, green_b[:nocc_b,nocc_b:], backend="jax")
#         gt2g_ab = oe.contract("ia,ia->", t2g_ab_a, green_a[:nocc_a,nocc_a:], backend="jax")
#         gt2g = 2 * (gt2g_a + gt2g_b) + gt2g_ab # <exp(T1)HF|T2|walker>/<exp(T1)HF|walker>

#         e1_2_1 = e1_0 * gt2g

#         # t2_green_a = (greenp_a @ t2g_a.T) @ green_a[:nocc_a,:] # t_iajb G_ia G_jq Gp_pb
#         t2ggg_aaa = oe.contract('pb,jb,jq->pq', green_a[:,nocc_a:], 
#                                 t2g_aa_a, green_a[:nocc_a,:], backend="jax") # t_iajb G_ia G_jq G_pb
#         t2ggg_aba = oe.contract('pb,jb,jq->pq', green_a[:,nocc_a:], 
#                                 t2g_ab_a, green_a[:nocc_a,:], backend="jax") # (greenp_a @ t2g_ab_a.T) @ green_a[:nocc_a,:]
#         t2ggg_bbb = oe.contract('pb,jb,jq->pq', green_b[:,nocc_b:], 
#                                 t2g_bb_b, green_b[:nocc_b,:], backend="jax")
#         t2ggg_bab = oe.contract('pb,jb,jq->pq', green_b[:,nocc_b:], 
#                                 t2g_ab_b, green_b[:nocc_b,:], backend="jax") # (greenp_b @ t2g_ab_b.T) @ green_b[:nocc_b,:]
#         t2ggg_a_a = 4 * t2ggg_aaa + t2ggg_aba
#         t2ggg_b_b = 4 * t2ggg_bbb + t2ggg_bab
#         t2gg_a = 4 * t2gg_aa + t2gg_ab
#         t2gg_b = 4 * t2gg_bb + t2gg_ba

#         e1_2_2_a = -(oe.contract("pq,pq->", t2ggg_a_a, h1_a, backend="jax")
#                     -oe.contract('bq,bq->', t2gg_a, dh1_a))
#         e1_2_2_b = -(oe.contract("pq,pq->", t2ggg_b_b, h1_b, backend="jax")
#                     -oe.contract('bq,bq->', t2gg_b, dh1_b))
#         e1_2_2 = e1_2_2_a + e1_2_2_b
#         e1_2 = e1_2_1 + e1_2_2  # <exp(T1)HF|T2 h1|walker>/<exp(T1)HF|walker>

#         # <exp(T1)HF|T2 h2|walker>/<exp(T1)HF|walker>
#         # double excitations
#         nchol = chol_a.shape[0]
#         nchol_chunk = self.nchol_chunk
#         nchunks = -(-nchol // nchol_chunk)
#         pad = nchunks * nchol_chunk - nchol

#         chol_a  = jnp.pad(chol_a,  ((0, pad), (0, 0), (0, 0))).reshape(nchunks, nchol_chunk, nocc_a, norb)
#         chol_b  = jnp.pad(chol_b,  ((0, pad), (0, 0), (0, 0))).reshape(nchunks, nchol_chunk, nocc_b, norb)
#         dchol_a = jnp.pad(dchol_a, ((0, pad), (0, 0), (0, 0))).reshape(nchunks, nchol_chunk, nvir_a, norb)
#         dchol_b = jnp.pad(dchol_b, ((0, pad), (0, 0), (0, 0))).reshape(nchunks, nchol_chunk, nvir_b, norb)

#         def scanned_fun(carry, x):
#             chol_a_i, chol_b_i, dchol_a_i, dchol_b_i = x
#             # e2_0
#             gl_a_i = oe.contract("rp,grq->gpq", green_a, chol_a_i, backend="jax")
#             gl_b_i = oe.contract("rp,grq->gpq", green_b, chol_b_i, backend="jax")
#             tr_gl_a_i = oe.contract("gpp->g", gl_a_i, backend="jax")
#             tr_gl_b_i = oe.contract("gpp->g", gl_b_i, backend="jax")
#             e2_0_1_i = jnp.sum((tr_gl_a_i + tr_gl_b_i)**2) / 2.0
#             e2_0_2_i = -(oe.contract('gpq,gqp->', gl_a_i, gl_a_i, backend="jax") 
#                         + oe.contract('gpq,gqp->', gl_b_i, gl_b_i, backend="jax")
#                         ) / 2.0
#             carry[0] += e2_0_1_i + e2_0_2_i

#             # e2_2
#             # gl_a_i = oe.contract("ps,pr->sr", green_a, chol_a_i, backend="jax")
#             # gl_b_i = oe.contract("ps,pr->sr", green_b, chol_b_i, backend="jax")
#             # lt2_green_a_i = (oe.contract("qs,qr->sr", chol_a_i, 2 * t2ggg_a_a, backend="jax")
#             #                 -oe.contract("bs,br->sr", dchol_a_i, 2 * t2gg_a, backend="jax"))
#             # lt2_green_b_i = (oe.contract("qs,qr->sr", chol_b_i, 2 * t2ggg_b_b, backend="jax")
#             #                 -oe.contract("bs,br->sr", dchol_b_i, 2 * t2gg_b, backend="jax"))

#             lt2ggg_a_i = (oe.contract("gqs,qr->gsr", chol_a_i, 2 * t2ggg_a_a, backend="jax")
#                             -oe.contract("gbs,br->gsr", dchol_a_i, 2 * t2gg_a, backend="jax"))
#             lt2ggg_b_i = (oe.contract("gqs,qr->gsr", chol_b_i, 2 * t2ggg_b_b, backend="jax")
#                             -oe.contract("gbs,br->gsr", dchol_b_i, 2 * t2gg_b, backend="jax"))
#             tr_lt2ggg_a_i = oe.contract("gpp->g", lt2ggg_a_i, backend="jax")
#             tr_lt2ggg_b_i = oe.contract("gpp->g", lt2ggg_b_i, backend="jax")
#             carry[1] += -jnp.sum((tr_lt2ggg_a_i + tr_lt2ggg_b_i) * (tr_gl_a_i + tr_gl_b_i)) / 2.0
#             carry[2] += (oe.contract("gsr,gsr->", gl_a_i, lt2ggg_a_i, backend="jax")
#                         + oe.contract("gsr,gsr->", gl_b_i, lt2ggg_b_i, backend="jax")) / 2.0
            
#             gl_a_i = oe.contract("ir,gpr->gip", green_a[:nocc_a,:], chol_a_i, backend="jax")
#             gl_b_i = oe.contract("ir,gpr->gip", green_b[:nocc_a,:], chol_b_i, backend="jax")
#             glgp_a_i = (oe.contract("gip,pa->gia", gl_a_i[:,:nocc_a,:], green_a[:,nocc_a:], backend="jax")
#                         -oe.contract("ir,gar->gia", green_a[:nocc_a,:], dchol_a_i, backend="jax"))
#             glgp_b_i = (oe.contract("gip,pa->gia", gl_b_i[:,:nocc_b,:], green_b[:,nocc_b:], backend="jax")
#                         -oe.contract("ir,gar->gia", green_b[:nocc_b,:], dchol_b_i, backend="jax"))
#             l2t2_a = 0.5 * oe.contract("gia,gjb,iajb->", glgp_a_i, glgp_a_i, t2_aa, backend="jax")
#             l2t2_b = 0.5 * oe.contract("gia,gjb,iajb->", glgp_b_i, glgp_b_i, t2_bb, backend="jax")
#             l2t2_ab = oe.contract("gia,gjb,iajb->", glgp_a_i, glgp_b_i, t2_ab, backend="jax")
#             carry[3] += l2t2_a + l2t2_b + l2t2_ab
#             return carry, 0.0

#         [e2_0, e2_2_2_1, e2_2_2_2, e2_2_3], _ = lax.scan(
#             scanned_fun, [0.0, 0.0, 0.0, 0.0], (chol_a, chol_b, dchol_a, dchol_b))
#         e2_2_1 = e2_0 * gt2g
#         e2_2_2 = e2_2_2_1 + e2_2_2_2
#         e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <exp(T1)HF|T2 h2|walker>/<exp(T1)HF|walker>

#         o0 = jnp.linalg.det(walker_up[:nocc_a,:nocc_a]
#             ) * jnp.linalg.det(walker_dn[:nocc_b,:nocc_b])
#         # <exp(T1)HF|walker>/<HF|walker>
#         t1 = jnp.linalg.det(wave_data["mo_ta"].T.conj() @ walker_up
#             ) * jnp.linalg.det(wave_data["mo_tb"].T.conj() @ walker_dn) / o0
#         t2 = gt2g * t1 # <exp(T1)HF|T2|walker>/<HF|walker>
#         e0 = (e1_0 + e2_0) * t1 # <exp(T1)HF|h1+h2|walker>/<HF|walker>
#         e1 = (e1_2 + e2_2) * t1 # <exp(T1)HF|T2 (h1+h2)|walker>/<HF|walker>

#         return t1, t2, e0, e1 
    
#     @partial(jit, static_argnums=(0,))
#     def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
#         nocc_a, nocc_b = self.nelec
#         norb = self.norb
#         nvir_a = norb - nocc_a
#         nvir_b = norb - nocc_b
#         ham_data["h1"] = (
#             ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
#         )
#         ham_data["h1"] = (
#             ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
#         )

#         ham_data["rot_h1"] = [
#             wave_data["mo_coeff"][0].T.conj() @ ham_data["h1"][0],
#             wave_data["mo_coeff"][1].T.conj() @ ham_data["h1"][1],
#         ]
#         ham_data["rot_chol"] = [
#             oe.contract(
#                 "pi,gij->gpj",
#                 wave_data["mo_coeff"][0].T.conj(),
#                 ham_data["chol"][0].reshape(-1, norb, norb), 
#                 backend="jax"),
#             oe.contract(
#                 "pi,gij->gpj",
#                 wave_data["mo_coeff"][1].T.conj(),
#                 ham_data["chol"][1].reshape(-1, norb, norb), 
#                 backend="jax")]
        
#         ham_data["rot_h1_ci"] = [
#             wave_data["mo_ta"].T.conj() @ ham_data["h1"][0],
#             wave_data["mo_tb"].T.conj() @ ham_data["h1"][1],
#         ]
#         ham_data["rot_chol_ci"] = [
#             oe.contract(
#                 "ip,gpq->giq",
#                 wave_data["mo_ta"].T.conj(),
#                 ham_data["chol"][0].reshape(-1, norb, norb), 
#                 backend="jax"),
#             oe.contract(
#                 "ip,gpq->giq",
#                 wave_data["mo_tb"].T.conj(),
#                 ham_data["chol"][1].reshape(-1, norb, norb), 
#                 backend="jax")]
        
#         ham_data['d_h1_ci'] = [ham_data['h1'][0][nocc_a:,:],
#                                ham_data['h1'][1][nocc_b:,:]]
#         ham_data['d_chol_ci'] = [ham_data['chol'][0].reshape(-1, norb, norb)[:,nocc_a:,:],
#                                  ham_data['chol'][1].reshape(-1, norb, norb)[:,nocc_b:,:]]
        
#         return ham_data

#     def __hash__(self):
#         return hash(tuple(self.__dict__.values()))

class upt2ccsd_cisd(ucisd):
    norb: int
    nelec: Tuple[int, int]
    nchol_chunk: int = 100
    mix_precision: bool = False
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_energy_pt(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
        '''
        calculate terms related to <bra|T2(h1+h2)|ket>/<bra|ket>
        bra is assumed to be an identity in its mo_coeff
        return <bra|ket> <T2> <h1+h2> <T2(h1+h2)> 
        '''
        # only do this for two-body energy with T contraction
        if self.mix_precision:
            rtype = jnp.float32
            ctype = jnp.complex64
        else:
            rtype = jnp.float64
            ctype = jnp.complex128

        norb_a, nocc_a = walker_up.shape
        norb_b, nocc_b = walker_dn.shape

        # o0 = jnp.linalg.det(walker_up[:nocc_a,:]) \
        # * jnp.linalg.det(walker_dn[:nocc_b,:])
        
        t2aa = wave_data["t2aa"]
        t2ab = wave_data["t2ab"]
        t2bb = wave_data["t2bb"]

        chol_a = ham_data["chol_bar"][0]
        chol_b = ham_data["chol_bar"][1]
        h1_a = ham_data["h1_bar"][0]
        h1_b = ham_data["h1_bar"][1]
        
        walker_up = wave_data['exp_t1a'] @ walker_up
        walker_dn = wave_data['exp_t1b'] @ walker_dn

        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a,:]))).T
        green_b = (walker_dn.dot(jnp.linalg.inv(walker_dn[:nocc_b,:]))).T
        greenov_a = green_a[:nocc_a,nocc_a:]
        greenov_b = green_b[:nocc_b,nocc_b:]
        greenp_a = jnp.vstack((greenov_a, -jnp.eye(norb_a-nocc_a)))
        greenp_b = jnp.vstack((greenov_b, -jnp.eye(norb_b-nocc_b)))

        hg_a = oe.contract("pq,pq->", h1_a[:nocc_a,:], green_a, backend="jax")
        hg_b = oe.contract("pq,pq->", h1_b[:nocc_b,:], green_b, backend="jax")
        e1_0 = hg_a + hg_b # <bra|h1|ket>/<bra|ket>

        # <bra|T2 h1|ket>/<bra|ket>
        t2g_a = oe.contract("iajb,ia->jb", t2aa, greenov_a, backend="jax") / 4
        t2g_b = oe.contract("iajb,ia->jb", t2bb, greenov_b, backend="jax") / 4
        t2g_ab_a = oe.contract("iajb,jb->ia", t2ab, greenov_b, backend="jax")
        t2g_ab_b = oe.contract("iajb,ia->jb", t2ab, greenov_a, backend="jax")
        # t_iajb (G_ia G_jb - G_ib G_ja)
        gt2g_a = oe.contract("jb,jb->", t2g_a, greenov_a, backend="jax")
        gt2g_b = oe.contract("jb,jb->", t2g_b, greenov_b, backend="jax")
        gt2g_ab = oe.contract("ia,ia->", t2g_ab_a, greenov_a, backend="jax")
        gt2g = 2 * (gt2g_a + gt2g_b) + gt2g_ab # <bra|T2|ket>/<bra|ket>
        e1_2_1 = e1_0 * gt2g
        
        t2_green_aaa = (greenp_a @ t2g_a.T) @ green_a[:nocc_a,:] # Gp_pb t_iajb G_ia G_jq (-)
        t2_green_aba = (greenp_a @ t2g_ab_a.T) @ green_a[:nocc_a,:]
        t2_green_bbb = (greenp_b @ t2g_b.T) @ green_b[:nocc_b,:]
        t2_green_abb = (greenp_b @ t2g_ab_b.T) @ green_b[:nocc_b,:]
        t2_green_a_a = 4 * t2_green_aaa + t2_green_aba # connect a->a
        t2_green_b_b = 4 * t2_green_bbb + t2_green_abb # connect b->b

        e1_2_2_a = -oe.contract("pq,pq->", h1_a, t2_green_a_a, backend="jax")
        e1_2_2_b = -oe.contract("pq,pq->", h1_b, t2_green_b_b, backend="jax")
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2  # <bra|T2 h1|ket>/<bra|ket>

        # <bra|T2 h2|ket>/<bra|ket>
        nchol = chol_a.shape[0]
        nchunks = -(-nchol // self.nchol_chunk)
        pad = nchunks * self.nchol_chunk - nchol
        chol_a = jnp.pad(chol_a, ((0, pad), (0, 0), (0, 0)))
        chol_b = jnp.pad(chol_b, ((0, pad), (0, 0), (0, 0)))
        chol_a = chol_a.reshape(nchunks, self.nchol_chunk, *chol_a.shape[-2:])
        chol_b = chol_b.reshape(nchunks, self.nchol_chunk, *chol_b.shape[-2:])

        def scanned_fun(carry, x):
            chol_a_c, chol_b_c = x

            # e2_0 = <h2>
            gl_a_c = oe.contract("ir,gpr->gip",
                                green_a.astype(jnp.complex128),
                                chol_a_c.astype(jnp.float64), 
                                backend="jax").astype(jnp.complex128)
            gl_b_c = oe.contract("ir,gpr->gip", 
                                green_b.astype(jnp.complex128),
                                chol_b_c.astype(jnp.float64), 
                                backend="jax")
            tr_gl_a = oe.contract("gii->g", 
                                gl_a_c[:,:nocc_a,:nocc_a].astype(jnp.complex128), 
                                backend="jax").astype(jnp.complex128)
            tr_gl_b = oe.contract("gii->g", 
                                gl_b_c[:,:nocc_b,:nocc_b].astype(jnp.complex128), 
                                backend="jax").astype(jnp.complex128)
            ex_gl_a = oe.contract("gij,gji->g", 
                                gl_a_c[:,:nocc_a,:nocc_a].astype(jnp.complex128), 
                                gl_a_c[:,:nocc_a,:nocc_a].astype(jnp.complex128), 
                                backend="jax").astype(jnp.complex128)
            ex_gl_b = oe.contract("gij,gji->g", 
                                gl_b_c[:,:nocc_b,:nocc_b].astype(jnp.complex128), 
                                gl_b_c[:,:nocc_b,:nocc_b].astype(jnp.complex128), 
                                backend="jax").astype(jnp.complex128)
            e2_0_1_c = jnp.sum((tr_gl_a + tr_gl_b) ** 2) / 2.0
            e2_0_2_c = -jnp.sum(ex_gl_a + ex_gl_b) / 2.0

            carry[0] += (e2_0_1_c + e2_0_2_c).astype(jnp.complex128)

            # e2_2 = <T2 h2>
            lt2g_a_c = oe.contract("gpr,qr->gpq", 
                                    chol_a_c.astype(jnp.float64), 
                                    (2*t2_green_a_a).astype(jnp.complex128), 
                                    backend="jax")
            lt2g_b_c = oe.contract("gpr,qr->gpq", 
                                    chol_b_c.astype(jnp.float64), 
                                    (2*t2_green_b_b).astype(jnp.complex128), 
                                    backend="jax")
            tr_lt2g_a_c = oe.contract("gqq->g", lt2g_a_c.astype(jnp.complex128), backend="jax")
            tr_lt2g_b_c = oe.contract("gqq->g", lt2g_b_c.astype(jnp.complex128), backend="jax")
            carry[1] += -(((tr_lt2g_a_c.astype(ctype) + tr_lt2g_b_c.astype(ctype)) 
                            @ (tr_gl_a.astype(ctype) + tr_gl_b.astype(ctype))
                            ) / 2).astype(jnp.complex128)
            carry[2] += ((oe.contract("giq,giq->", 
                                        gl_a_c.astype(ctype), 
                                        lt2g_a_c[:,:nocc_a,:].astype(ctype), 
                                        backend="jax")
                        + oe.contract("giq,giq->", 
                                        gl_b_c.astype(ctype), 
                                        lt2g_b_c[:,:nocc_b,:].astype(ctype), 
                                        backend="jax")) / 2).astype(jnp.complex128)

            glgp_a_c = oe.contract("giq,qa->gia",
                                gl_a_c.astype(jnp.complex128), 
                                greenp_a.astype(jnp.complex128), 
                                backend="jax")
            glgp_b_c = oe.contract("giq,qa->gia", 
                                gl_b_c.astype(jnp.complex128), 
                                greenp_b.astype(jnp.complex128), 
                                backend="jax")
            
            lt2_aa = oe.contract("gia,iajb->gjb", 
                                    glgp_a_c.astype(ctype), 
                                    t2aa.astype(rtype), 
                                    backend="jax")
            lt2_bb = oe.contract("gia,iajb->gjb", 
                                    glgp_b_c.astype(ctype), 
                                    t2bb.astype(rtype), 
                                    backend="jax")
            lt2_ab = oe.contract("gia,iajb->gjb", 
                                    glgp_a_c.astype(ctype), 
                                    t2ab.astype(rtype), 
                                    backend="jax")
            
            l2t2_aa = 0.5 * oe.contract("gjb,gjb->",
                                        lt2_aa.astype(ctype),
                                        glgp_a_c.astype(ctype), 
                                        backend="jax").astype(jnp.complex128)
            l2t2_bb = 0.5 * oe.contract("gjb,gjb->",
                                        lt2_bb.astype(ctype),
                                        glgp_b_c.astype(ctype), 
                                        backend="jax").astype(jnp.complex128)
            l2t2_ab = oe.contract("gjb,gjb->",
                                    lt2_ab.astype(ctype),
                                    glgp_b_c.astype(ctype), 
                                    backend="jax").astype(jnp.complex128)
            
            carry[3] += (l2t2_aa + l2t2_bb + l2t2_ab).astype(jnp.complex128)
            return carry, 0.0

        [e2_0, e2_2_2_1, e2_2_2_2, e2_2_3], _ = lax.scan(
            scanned_fun, [0.0, 0.0, 0.0, 0.0], (chol_a, chol_b)
        )

        e2_2_1 = e2_0 * gt2g
        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <bra|T2 h2|ket>/<bra|ket>

        ot1 = jnp.linalg.det(walker_up[:nocc_a,:]) \
            * jnp.linalg.det(walker_dn[:nocc_b,:]) # <bra|ket_bar>
        t2 = gt2g # * t1o # <bra|T2|ket_bar>/<bra|ket>
        e0 = (e1_0 + e2_0) # * t1o # <bra|h1+h2|ket_bar>/<bra|ket>
        e1 = (e1_2 + e2_2) # * t1o # <bra|T2 (h1+h2)|ket_bar>/<bra|ket>

        return ot1, t2, e0, e1
    
    def calc_energy_pt(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
        ot1, t2, e0, e1 = vmap(
            self._calc_energy_pt, in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data, wave_data)
        return ot1, t2, e0, e1

    @partial(jit, static_argnums=(0,))
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        norb = self.norb
        h1a, h1b = ham_data["h1"]
        chola = ham_data["chol"][0].reshape(-1, norb, norb)
        cholb = ham_data["chol"][1].reshape(-1, norb, norb)
        moa, mob = wave_data["mo_coeff"]

        ham_data["rot_h1"] = [moa.T.conj() @ h1a, mob.T.conj() @ h1b]

        ham_data["rot_chol"] = [
            oe.contract("pi,gij->gpj", moa.T.conj(), chola, backend="jax"),
            oe.contract("pi,gij->gpj", mob.T.conj(), cholb, backend="jax")]
        
        h1bar_a = wave_data['exp_t1a'] @ h1a @ wave_data['exp_mt1a']
        h1bar_b = wave_data['exp_t1b'] @ h1b @ wave_data['exp_mt1b']
        ham_data["h1_bar"] = [h1bar_a, h1bar_b]

        chol_bar_a = oe.contract(
            'pr,grs,sq->gpq', 
            wave_data['exp_t1a'], 
            chola, 
            wave_data['exp_mt1a'], 
            backend='jax')
        chol_bar_b = oe.contract(
            'pr,grs,sq->gpq', 
            wave_data['exp_t1b'], 
            cholb, 
            wave_data['exp_mt1b'], 
            backend='jax')
        ham_data["chol_bar"] = [chol_bar_a, chol_bar_b]

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
class uptccsd_ad(uhf):
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
class upt2ccsd_ad(uhf):
    """differential form of the CCSD_PT2 (exact T1) wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1
    
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

        e1t1 = self._tls_olp(walker_up_1x, walker_dn_1x, wave_data)
        ot1 = self._tls_olp(walker_up, walker_dn, wave_data)

        return e1t1/ot1

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

        e2t1 = self._tls_olp(walker_up_2x,walker_dn_2x,wave_data)
        ot1 = self._tls_olp(walker_up, walker_dn, wave_data)
        
        return e2t1/ot1
    
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
        
        e1t2 = self._ut2_walker_olp(walker_up_1x, walker_dn_1x, wave_data)
        ot1 = self._tls_olp(walker_up, walker_dn, wave_data)

        return e1t2/ot1

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
        
        e2t2 = self._ut2_walker_olp(walker_up_2x,walker_dn_2x,wave_data)
        ot1 = self._tls_olp(walker_up, walker_dn, wave_data)

        return e2t2/ot1
    
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
        # chol = (ham_data["chol"][0].reshape(-1, norb, norb),
        #         ham_data["chol"][1].reshape(-1, norb, norb))
        d2_exp2_batch = jax.vmap(self._d2_tls_exp2_i, in_axes=(0,None,None,None))
        d2_exp2s = d2_exp2_batch(chol,walker_up,walker_dn,wave_data)
        h2 = jnp.sum(d2_exp2s)/2
        return h2

    @partial(jit, static_argnums=0)
    def _d2_ut2_exp2(self,walker_up,walker_dn,ham_data,wave_data):
        norb = self.norb
        chol = ham_data["chol"].reshape(2, -1, norb, norb)
        chol = chol.transpose(1,0,2,3)
        # chol = (ham_data["chol"][0].reshape(-1, norb, norb),
        #         ham_data["chol"][1].reshape(-1, norb, norb))
        d2_exp2_batch = jax.vmap(self._d2_ut2_exp2_i, in_axes=(0,None,None,None))
        d2_exp2s = d2_exp2_batch(chol,walker_up,walker_dn,wave_data)
        h2 = jnp.sum(d2_exp2s)/2
        return h2

    @partial(jit, static_argnums=0)
    def _calc_energy_pt(self, walker_up, walker_dn, ham_data, wave_data):
        '''
        ot1 = <exp(T1)HF|walker>
        t2 = <exp(T1)HF|T1+T2|walker>/<exp(T1)HF|walker>
        e0 = <exp(T1)HF|h1+h2|walker>/<exp(T1)HF|walker>
        e1 = <exp(T1)HF|(T1+T2)(h1+h2)|walker>/<exp(T1)HF|walker>
        '''

        norb = self.norb
        h1_mod = ham_data['h1_mod']

        ot1 = self._tls_olp(walker_up, walker_dn, wave_data)

        # e0 = <exp(T1)HF|h1+h2|walker>/<HF|walker> #
        # one body
        x = 0.0
        f1 = lambda a: self._tls_exp1(a,h1_mod,walker_up,walker_dn,wave_data)
        _, d_exp1_0 = jvp(f1, [x], [1.0])

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
        
        return ot1, t2, e0, e1

    def calc_energy_pt(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
        t1, t2, e0, e1 = vmap(
            self._calc_energy_pt, in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data, wave_data)
        return t1, t2, e0, e1
    
    @partial(jit, static_argnums=(0,))
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        norb = self.norb
        h1a, h1b = ham_data["h1"]
        ham_data["chol"] = jnp.array(ham_data["chol"])
        chola = ham_data["chol"][0].reshape(-1, norb, norb)
        cholb = ham_data["chol"][1].reshape(-1, norb, norb)
        moa, mob = wave_data["mo_coeff"]
        ham_data["rot_h1"] = (moa.T.conj() @ h1a,
                              mob.T.conj() @ h1b)
        ham_data["rot_chol"] = (oe.contract("pi,gij->gpj", moa.T.conj(), chola, backend="jax"),
                                oe.contract("pi,gij->gpj", mob.T.conj(), cholb, backend="jax"))
        v0a = 0.5 * jnp.einsum("gpr,grq->pq", chola, chola, optimize="optimal")
        v0b = 0.5 * jnp.einsum("gpr,grq->pq", cholb, cholb, optimize="optimal")
        h1mod_a = h1a - v0a
        h1mod_b = h1b - v0b
        ham_data['h1_mod'] = (h1mod_a,h1mod_b)

        return ham_data

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