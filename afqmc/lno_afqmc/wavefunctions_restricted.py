# from abc import ABC
from dataclasses import dataclass
from functools import partial
from typing import Union
# from typing import  Sequence, Tuple, Union

import jax
import jax.numpy as jnp
# import numpy as np
from jax import jit, jvp, lax, random, vmap
import opt_einsum as oe

from afqmc.wavefunctions.wavefunctions_restricted import rhf as rwfn
from afqmc import integral
from afqmc.wavefunctions.wavefunctions_restricted import (
    _chol_chunking, _resolve_chol_budget)


# class rwfn(ABC):
#     """Base class for wave functions. Contains methods for wave function measurements.

#     The measurement methods support two types of walker batches:

#     1) unrestricted: walkers is a list ([up, down]). up and down are jax.Arrays of shapes
#     (nwalkers, norb, nelec[sigma]). In this case the _calc_<property> method is mapped over.

#     2) restricted (up and down dets are assumed to be the same): walkers is a jax.Array of shape
#     (nwalkers, max(nelec[0], nelec[1])). In this case the _calc_<property>_restricted method is mapped over. By default
#     this method is defined to call _calc_<property>. For certain trial states, one can override
#     it for computational efficiency.

#     A minimal implementation of a wave function should define the _calc_<property> methods for
#     property = overlap, force_bias, energy.

#     The wave function data is stored in a separate wave_data dictionary. Its structure depends on the
#     wave function type and is described in the corresponding class. It may contain "rdm1" which is a
#     one-body spin RDM (2, norb, norb). If it is not provided, wave function specific methods are called.

#     Attributes:
#         norb: Number of orbitals.
#         nelec: Number of electrons of each spin.
#         n_batch: Number of batches used in scan.
#     """

#     norb: Tuple[int, int]
#     nelec: Tuple[int, int]
#     n_batch: int = 1


#     def calc_overlap(self, walkers: jax.Array, wave_data: dict) -> jax.Array:
#         n_walkers = walkers.shape[0]
#         batch_size = n_walkers // self.n_batch

#         def scanned_fun(carry, walker_batch):
#             overlap_batch = vmap(self._calc_overlap_restricted, in_axes=(0, None))(
#                 walker_batch, wave_data
#             )
#             return carry, overlap_batch

#         _, overlaps = lax.scan(
#             scanned_fun, None, walkers.reshape(self.n_batch, batch_size, self.norb, -1)
#         )
#         return overlaps.reshape(n_walkers)


#     def calc_force_bias(self, walkers: jax.Array, ham_data: dict, wave_data: dict) -> jax.Array:
#         n_walkers = walkers.shape[0]
#         batch_size = n_walkers // self.n_batch

#         def scanned_fun(carry, walker_batch):
#             fb_batch = vmap(self._calc_force_bias_restricted, in_axes=(0, None, None))(
#                 walker_batch, ham_data, wave_data
#             )
#             return carry, fb_batch

#         _, fbs = lax.scan(
#             scanned_fun, None, walkers.reshape(self.n_batch, batch_size, self.norb, -1)
#         )
#         return fbs.reshape(n_walkers, -1)


#     def calc_energy(self, walkers: jax.Array, ham_data: dict, wave_data: dict) -> jax.Array:
#         n_walkers = walkers.shape[0]
#         batch_size = n_walkers // self.n_batch

#         def scanned_fun(carry, walker_batch):
#             energy_batch = vmap(self._calc_energy_restricted, in_axes=(0, None, None))(
#                 walker_batch, ham_data, wave_data
#             )
#             return carry, energy_batch

#         _, energies = lax.scan(
#             scanned_fun,
#             None,
#             walkers.reshape(self.n_batch, batch_size, self.norb, -1),
#         )
#         return energies.reshape(n_walkers)


#     def get_rdm1(self, wave_data: dict) -> jax.Array:
#         """Returns the one-body spin reduced density matrix of the trial.
#         Used for calculating mean-field shift and as a default value in cases of large
#         deviations in observable samples. If wave_data contains "rdm1" this value is used,
#         calls otherwise _calc_rdm1.

#         Args:
#             wave_data : The trial wave function data.

#         Returns:
#             rdm1: The one-body spin reduced density matrix (2, norb, norb).
#         """
#         if "rdm1" in wave_data:
#             return wave_data["rdm1"]
#         else:
#             return self._calc_rdm1(wave_data)

#     # def get_init_walkers(
#     #     self, wave_data: dict, n_walkers: int, restricted: bool = False
#     # ) -> Union[Sequence, jax.Array]:
#     #     """Get the initial walkers. Uses the rdm1 natural orbitals.

#     #     Args:
#     #         wave_data: The trial wave function data.
#     #         n_walkers: The number of walkers.
#     #         restricted: Whether the walkers should be restricted.

#     #     Returns:
#     #         walkers: The initial walkers.
#     #             If restricted, a single jax.Array of shape (nwalkers, norb, nelec[0]).
#     #             If unrestricted, a list of two jax.Arrays each of shape (nwalkers, norb, nelec[sigma]).
#     #     """
#     #     rdm1 = self.get_rdm1(wave_data)
#     #     natorbs_up = jnp.linalg.eigh(rdm1[0])[1][:, ::-1][:, : self.nelec[0]]
#     #     natorbs_dn = jnp.linalg.eigh(rdm1[1])[1][:, ::-1][:, : self.nelec[1]]
#     #     if restricted:
#     #         if self.nelec[0] == self.nelec[1]:
#     #             det_overlap = np.linalg.det(
#     #                 natorbs_up[:, : self.nelec[0]].T @ natorbs_dn[:, : self.nelec[1]]
#     #             )
#     #             if (
#     #                 np.abs(det_overlap) > 1e-3
#     #             ):  # probably should scale this threshold with number of electrons
#     #                 return jnp.array([natorbs_up + 0.0j] * n_walkers)
#     #             else:
#     #                 overlaps = np.array(
#     #                     [
#     #                         natorbs_up[:, i].T @ natorbs_dn[:, i]
#     #                         for i in range(self.nelec[0])
#     #                     ]
#     #                 )
#     #                 new_vecs = natorbs_up[:, : self.nelec[0]] + np.einsum(
#     #                     "ij,j->ij", natorbs_dn[:, : self.nelec[1]], np.sign(overlaps)
#     #                 )
#     #                 new_vecs = np.linalg.qr(new_vecs)[0]
#     #                 det_overlap = np.linalg.det(
#     #                     new_vecs.T @ natorbs_up[:, : self.nelec[0]]
#     #                 ) * np.linalg.det(new_vecs.T @ natorbs_dn[:, : self.nelec[1]])
#     #                 if np.abs(det_overlap) > 1e-3:
#     #                     return jnp.array([new_vecs + 0.0j] * n_walkers)
#     #                 else:
#     #                     raise ValueError(
#     #                         "Cannot find a set of RHF orbitals with good trial overlap."
#     #                     )
#     #         else:
#     #             # bring the dn orbital projection onto up space to the front
#     #             dn_proj = natorbs_up.T.conj() @ natorbs_dn
#     #             proj_orbs = jnp.linalg.qr(dn_proj, mode="complete")[0]
#     #             orbs = natorbs_up @ proj_orbs
#     #             return jnp.array([orbs + 0.0j] * n_walkers)
#     #     else:
#     #         return [
#     #             jnp.array([natorbs_up + 0.0j] * n_walkers),
#     #             jnp.array([natorbs_dn + 0.0j] * n_walkers),
#     #         ]

#     def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
#         """Build intermediates for measurements in ham_data. This method is called by the hamiltonian class.

#         Args:
#             ham_data: The hamiltonian data.
#             wave_data: The trial wave function data.

#         Returns:
#             ham_data: The updated Hamiltonian data.
#         """
#         return ham_data



#     def __hash__(self) -> int:
#         return hash(tuple(self.__dict__.values()))



# we assume afqmc is performed in the rhf orbital basis
# @dataclass
# class rhf(rwfn):

#     norb: int
#     nelec: int
#     n_batch: int = 1
#     nchol_chunk: int = 100
@dataclass
class rhf(rwfn):

    def __post_init__(self):
        assert (
            self.nelec[0] == self.nelec[1]
        ), "RHF requires equal number of up and down electrons."

    # def _calc_rdm1(self, wave_data: dict) -> jax.Array:
    #     rdm1 = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T] * 2)
    #     return rdm1

    # @partial(jit, static_argnums=0)
    # def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> jax.Array:
    #     nocc = self.nelec[0]
    #     return jnp.linalg.det(walker[:nocc,:nocc]) ** 2

    # @partial(jit, static_argnums=0)
    # def _calc_green(self, walker: jax.Array, wave_data: dict) -> jax.Array:
    #     green = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
    #     return green
    
    # @partial(jit, static_argnums=0)
    # def _calc_force_bias_restricted(
    #     self, walker: Sequence, ham_data: dict, wave_data: dict
    # ) -> jax.Array:
    #     nocc, norb = self.nelec[0], self.norb
    #     rot_chol = ham_data["chol"].reshape(-1,norb,norb)[:,:nocc,:]
    #     green = self._calc_green(walker, wave_data)
    #     fb = 2.0 * oe.contract("gij,ij->g", rot_chol, green, backend="jax")
    #     return fb

    # @partial(jit, static_argnums=0)
    # def _calc_energy_restricted(
    #     self, 
    #     walker: jax.Array, 
    #     ham_data: dict, 
    #     wave_data: dict
    #     ):
    #     nocc, norb = self.nelec[0], self.norb
    #     h0 = ham_data["h0"]
    #     rot_h1 = ham_data["h1"][0][:nocc,:]
    #     rot_chol = ham_data["chol"].reshape(-1,norb,norb)[:,:nocc,:]
    #     green = self._calc_green(walker, wave_data)
    #     hg = oe.contract("pq,pq->", rot_h1, green, backend="jax")
    #     e1 = 2 * hg

    #     nchol = rot_chol.shape[0]
    #     nchol_chunk = self.nchol_chunk
    #     nchunks = -(-nchol // nchol_chunk)
    #     pad = nchunks * nchol_chunk - nchol
    #     rot_chol = jnp.pad(rot_chol, ((0, pad), (0, 0), (0, 0)))
    #     rot_chol = rot_chol.reshape(nchunks, nchol_chunk, nocc, norb)

    #     def scanned_fun(carry, x):
    #         chol_c = x  # (nchol_chunk, nocc, norb)
    #         lg_c = oe.contract("gpr,qr->gpq", chol_c, green, backend="jax")
    #         tr_c = oe.contract("gpp->g", lg_c, backend="jax")
    #         e2_1_c = 2 * jnp.sum(tr_c ** 2)
    #         e2_2_c = -oe.contract("gpq,gqp->", lg_c, lg_c, backend="jax")
    #         carry += e2_1_c + e2_2_c
    #         return carry, 0.0

    #     e2, _ = lax.scan(scanned_fun, 0.0, rot_chol)

    #     return h0 + e1 + e2
    
    # @partial(jit, static_argnums=0)
    # def _calc_ecorr(self, walker: jax.Array, ham_data: dict, wave_data: dict):
    #     '''hf correlation energy'''
    #     # <HF|H-E0|walker>/<HF|walker>
    #     rot_h1 = ham_data['rot_h1']
    #     nocc = rot_h1.shape[0]
    #     rot_chol_ov = ham_data['rot_chol'][:, :nocc, nocc:]
    #     green_ov = self._calc_green(walker, wave_data)[:nocc, nocc:]

    #     nchol = rot_chol_ov.shape[0]
    #     nchol_chunk = self.nchol_chunk
    #     nchunks = -(-nchol // nchol_chunk)
    #     pad = nchunks * nchol_chunk - nchol
    #     rot_chol_ov = jnp.pad(rot_chol_ov, ((0, pad), (0, 0), (0, 0)))
    #     rot_chol_ov = rot_chol_ov.reshape(nchunks, nchol_chunk, nocc, -1)

    #     def scanned_fun(carry, x):
    #         chol_c = x  # (nchol_chunk, nocc, nvir)
    #         lg_c = oe.contract('gia,ja->gij', chol_c, green_ov, backend="jax")
    #         trlg_c = oe.contract('gii->g', lg_c, backend="jax")
    #         e1_c = oe.contract('g,g->', trlg_c, trlg_c, backend="jax") * 2
    #         e2_c = oe.contract('gij,gji->', lg_c, lg_c, backend="jax")
    #         carry += e1_c - e2_c
    #         return carry, 0.0

    #     e_corr, _ = lax.scan(scanned_fun, 0.0, rot_chol_ov)
    #     return jnp.real(e_corr)
    
    @partial(jit, static_argnums=0)
    def _calc_eorb(self, walker: jax.Array, ham_data: dict, wave_data: dict):
        '''hf orbital correlation energy'''
        # <HF|H_i|walker>/<HF|walker>
        rot_h1 = ham_data['rot_h1']
        prj = wave_data["prjlo"]
        nocc = rot_h1.shape[0]
        rot_chol_ov = ham_data['rot_chol'][:, :nocc, nocc:]
        green_ov = self._calc_green(walker, wave_data)[:nocc, nocc:]

        nchol = rot_chol_ov.shape[0]
        nchol_chunk = self.nchol_chunk
        nchunks = -(-nchol // nchol_chunk)
        pad = nchunks * nchol_chunk - nchol
        rot_chol_ov = jnp.pad(rot_chol_ov, ((0, pad), (0, 0), (0, 0)))
        rot_chol_ov = rot_chol_ov.reshape(nchunks, nchol_chunk, nocc, -1)

        def scanned_fun(carry, x):
            chol_c = x  # (nchol_chunk, nocc, nvir)
            lg_c = oe.contract('gia,ja->gij', chol_c, green_ov, backend="jax")
            trlg_c = oe.contract('gii->g', lg_c, backend="jax")
            e1_c = oe.contract('gik,ik,g->', lg_c, prj, trlg_c, backend="jax") * 2
            e2_c = oe.contract('gij,gjk,ik->', lg_c, lg_c, prj, backend="jax")
            carry += e1_c - e2_c
            return carry, 0.0

        eorb, _ = lax.scan(scanned_fun, 0.0, rot_chol_ov)

        return jnp.real(eorb)

    # @partial(jit, static_argnums=0)
    # def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
    #     """Builds half rotated integrals for efficient force bias and energy calculations."""

    #     ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ ham_data["h1"][0]
    #     ham_data["rot_chol"] = oe.contract(
    #         "pi,gij->gpj",
    #         wave_data["mo_coeff"].T.conj(),
    #         ham_data["chol"].reshape(-1, self.norb, self.norb), 
    #         backend="jax")
    #     return ham_data
    
    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class ptccsd(rhf):

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

        # ham_data["h1"] = (
        #     ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        # )
        # ham_data["h1"] = (
        #     ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        # )
        ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ ham_data["h1"][0]
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
class ptccsd_ad(rhf):

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
class pt2ccsd_ad(rhf):

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

        return t2_orb, t2e_orb

    @partial(jit, static_argnums=0)
    def _calc_ept2_frag(self, walker: jax.Array, ham_data: dict, wave_data: dict):
        
        walker_bar = wave_data['exp_t1'] @ walker
        o0 = jnp.linalg.det(walker[:walker.shape[1], :]) ** 2
        o_bar = jnp.linalg.det(walker_bar[:walker_bar.shape[1], :]) ** 2
        t1 = o_bar/o0 # <exp(T1)HF|walker>/<HF|walker>
        eg = self._calc_energy_restricted(walker, ham_data, wave_data)
        e0frag = self._calc_eorb_bar(walker_bar, ham_data, wave_data)
        t2frag, e1frag = self._t2e_orb_ad(walker_bar, ham_data, wave_data)
        e0 = self._calc_energy_bar(walker_bar, ham_data, wave_data)

        return eg, t1, t2frag, e0frag, e1frag, e0

    @partial(jit, static_argnums=(0)) 
    def calc_ept2_frag(self,walkers,ham_data,wave_data):
        eg, t1, t2frag, e0frag, e1frag, e0 = vmap(
            self._calc_ept2_frag,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        return eg, t1, t2frag, e0frag, e1frag, e0
    
    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        """Builds half rotated integrals for efficient force bias and energy calculations."""
        norb, nocc = self.norb, self.nelec[0]
        chol = ham_data["chol"].reshape(-1, norb, norb)

        ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ ham_data["h1"][0]
        ham_data["rot_chol"] = oe.contract(
            "pi,gij->gpj",
            wave_data["mo_coeff"].T.conj(),
            ham_data["chol"].reshape(-1, self.norb, self.norb), 
            backend="jax")

        # exp(T1^dagger) H exp(-T1^dagger)
        h1_bar = wave_data['exp_t1'] @ ham_data['h1'][0] @ wave_data['exp_mt1']
        ham_data['h1_bar'] = h1_bar
        
        chol_bar = oe.contract(
            'pr,grs,sq->gpq', wave_data['exp_t1'], chol, wave_data['exp_mt1'], backend='jax')
        ham_data["chol_bar"] = chol_bar
        
        v0_bar = 0.5 * oe.contract("gpr,grq->pq", chol_bar, chol_bar, backend="jax")
        ham_data['h1_mod_bar'] = h1_bar - v0_bar

        # exp(T1^dagger) Fock exp(-T1^dagger)
        jeff = oe.contract('gpq,gjj->pq', chol_bar, chol_bar[:,:nocc,:nocc], backend="jax")
        keff = oe.contract('gpj,gjq->pq', chol_bar[:,:,:nocc],
                        chol_bar[:,:nocc,:], backend="jax")
        fock_bar = h1_bar + 2 * jeff - keff
        ham_data['fock_bar'] = oe.contract(
            'ip,ik->kp', fock_bar[:nocc, :], wave_data['prjlo'], backend="jax")
        
        lt1 = oe.contract('ia,gja->gij', wave_data["t1"], chol[:, :nocc, nocc:], backend='jax')
        ham_data['e0t1orb'] = 2 * oe.contract('gik,ik,gjj->',lt1, wave_data['prjlo'], lt1, backend='jax') \
                    - oe.contract('gij,gjk,ik->',lt1, lt1, wave_data['prjlo'], backend='jax')
        
        del h1_bar, chol_bar, chol, jeff, keff, fock_bar, lt1
        
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))

from afqmc import slater_tools

@dataclass
class pt2ccsd(rhf):
    mix_precision: bool = True

    # @partial(jit, static_argnums=0)
    # def _calc_e0bar_frag(self, walker, ham_data, wave_data):
    #     '''
    #     calculate the correlation energy of the Hamiltonian
    #     transformed by exp(T1^dagger):
    #     ecorr_bar = <psi_0|H_bar|walker_bar>/<psi_0|walker_bar>
    #     |walker_bar> = exp(T1^dagger) |walker>
    #     H_bar = exp(T1^dagger) H exp(-T1^dagger)
    #     |psi_0> is the mean-field solution of H
    #     '''
    #     nocc, norb = self.nelec[0], self.norb
    #     prjlo = wave_data['prjlo']
    #     e0 = ham_data['e0t1orb'] # <psi_0|H_bar|psi_0>
    #     # rot_fock_ov = ham_data['fock_bar'][:nocc,nocc:]
    #     # rot_chol_ov = ham_data['chol_bar'].reshape(-1,norb,norb)[:, :nocc, nocc:]
    #     # gf_ov = self._calc_green(walker, wave_data)[:nocc, nocc:]

    #     # e1 = oe.contract('ia,ia->', gf_ov, rot_fock_ov, backend="jax") * 2

    #     # Pad along the auxiliary axis so every chunk has the same size
    #     # nchol = rot_chol_ov.shape[0]
    #     # nchol_chunk = self.nchol_chunk
    #     # nchunks = -(-nchol // nchol_chunk)
    #     # pad = nchunks * nchol_chunk - nchol
    #     # rot_chol_ov = jnp.pad(rot_chol_ov, ((0, pad), (0, 0), (0, 0)))
    #     # rot_chol_ov = rot_chol_ov.reshape(nchunks, nchol_chunk, nocc, -1)

    #     # this fock is not projected!!
    #     e_corr12 = slater_tools.r_energy_corr_frag(
    #         walker, walker, 
    #         ham_data['fock_bar'], ham_data['chol_bar'], 
    #         wave_data['prjlo'])

    #     e_corr = e0 + e_corr12

    #     return e_corr

    @partial(jit, static_argnums=0)
    def _calc_e0bar_frag(self, walker, ham_data, wave_data):
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
        rot_fock_ov = ham_data['fock_bar'][:nocc,nocc:] # this fock is projected!
        rot_chol_ov = ham_data['chol_bar'].reshape(-1,norb,norb)[:, :nocc, nocc:]
        gf_ov = self._calc_green(walker, wave_data)[:nocc, nocc:]

        e1 = oe.contract('ia,ik,ka->', gf_ov, prjlo, rot_fock_ov, backend="jax") * 2

        # Pad along the auxiliary axis so every chunk has the same size
        nchol = rot_chol_ov.shape[0]
        nchol_chunk = self.nchol_chunk
        nchunks = -(-nchol // nchol_chunk)
        pad = nchunks * nchol_chunk - nchol
        rot_chol_ov = jnp.pad(rot_chol_ov, ((0, pad), (0, 0), (0, 0)))
        rot_chol_ov = rot_chol_ov.reshape(nchunks, nchol_chunk, nocc, -1)

        def scanned_fun(carry, x):
            chol_c = x  # (nchol_chunk, nocc, nvir)
            lg_c = oe.contract('gia,ka->gik', chol_c, gf_ov, backend="jax")
            e2_1_c = oe.contract('gik,ik,gjj->', lg_c, prjlo, lg_c, backend="jax") * 2
            e2_2_c = oe.contract('gij,gjk,ik->', lg_c, lg_c, prjlo, backend="jax")
            carry += e2_1_c - e2_2_c
            return carry, 0.0

        e2, _ = lax.scan(scanned_fun, 0.0, rot_chol_ov)

        e_corr = e0 + e1 + e2

        return e_corr


    @partial(jit, static_argnums=(0, 4))
    def _t2eorb_tc(self, walker, ham_data, wave_data, frozen_vir=None):
        if self.mix_precision:
            rtype = jnp.float32
            ctype = jnp.complex64
        else:
            rtype = jnp.float64
            ctype = jnp.complex128
        
        nocc, norb = self.nelec[0], self.norb
        nchol_chunk = self.nchol_chunk  # nchol per chunk
        t2 = wave_data["t2"]
        chol = ham_data["chol_bar"]
        h1 = ham_data["h1_bar"]

        if frozen_vir is not None:
            # Drop the last frozen_vir virtuals from the T2 terms.  The virtuals
            # are ordered by decreasing occupation (natural orbitals), so this is
            # a small, quickly converging perturbation.
            fv = frozen_vir
            n_keep = norb - fv               # total kept orbitals
            nv_keep = (norb - nocc) - fv     # kept virtuals
            assert nv_keep > 0, "frozen_vir exceeds number of virtuals"

            norb = n_keep
            walker = walker[:n_keep, :]
            # one-body: both axes are orbital axes
            h1 = h1[:n_keep, :n_keep]
            # cholesky: slice the two ORBITAL axes, never axis 0 (the chol index)
            chol = chol[:, :n_keep, :n_keep]
            # amplitudes: slice the two VIRTUAL axes only
            t2 = t2[:, :nv_keep, :, :nv_keep]
            # _calc_green contracts the walker against mo_coeff, so that has to
            # lose the same rows or the shapes no longer match
            wave_data = {**wave_data, "mo_coeff": wave_data["mo_coeff"][:n_keep, :]}

        green = self._calc_green(walker, wave_data)
        green_occ = green[:, nocc:]
        greenp = jnp.vstack((green_occ, -jnp.eye(norb - nocc)))
        rot_chol = chol[:, :nocc, :]
        nchol = chol.shape[0]
        # chunk_size = naux // nchol_chunk

        # 1 body energy
        hg = oe.contract("pi,pi->", h1[:nocc, :], green, backend="jax")
        e1_0 = 2 * hg

        # double excitations (unchanged)
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

        # pad with zero cholesky vectors — contributes nothing to any contraction
        npad = (-nchol) % nchol_chunk
        chol = jnp.concatenate([chol, jnp.zeros((npad, norb, norb))], axis=0)
        rot_chol = jnp.concatenate([rot_chol, jnp.zeros((npad, nocc, norb))], axis=0)

        # reshape into chunks: (n_chunks, chunk_size, ...)
        nchunk = (nchol + npad) // nchol_chunk
        chol = chol.reshape(nchunk, nchol_chunk, norb, norb)
        rot_chol = rot_chol.reshape(nchunk, nchol_chunk, nocc, norb)

        # two body — scan over chunks, explicit contractions within a chunk
        def scan_chunk(carry, x):
            chol_c, rot_chol_c = x  # (chunk_size, norb, norb), (chunk_size, nocc, norb)

            gl = oe.contract("ir,gqr->giq", green, chol_c, backend="jax")
            gl_c = oe.contract("gii->g", gl[:, :, :nocc], backend="jax")
            e2_0_c = oe.contract("g,g->", gl_c, gl_c, backend="jax") * 2
            e2_0_e = -oe.contract("gij,gji->", gl[:, :, :nocc], gl[:, :, :nocc], backend="jax")
            carry[0] += e2_0_c + e2_0_e

            lt2g = oe.contract("gpr,pr->g", 
                               chol_c.astype(rtype), 
                               t2_green.astype(ctype), 
                               backend="jax")
            carry[1] += -oe.contract("g,g->", 
                                     lt2g.astype(ctype), 
                                     gl_c.astype(ctype), 
                                     backend="jax")

            lt2_green = oe.contract("gir,qr->giq", 
                                    rot_chol_c.astype(rtype), 
                                    t2_green.astype(ctype), 
                                    backend="jax")
            # t_iajb |G_ia G_js Gp_pb| G_qr L_pr L_qs
            carry[2] += 0.5 * oe.contract("giq,giq->", 
                                          gl.astype(ctype), 
                                          lt2_green.astype(ctype), 
                                          backend="jax")

            # t_iajb G_ir G_js Gp_pa Gp_qb L_pr L_qs type
            glgp = oe.contract("gir,rb->gib", 
                               gl.astype(ctype), 
                               greenp.astype(ctype), 
                               backend="jax")
            lt2_c = oe.contract("gia,iajb->gjb", 
                                glgp.astype(ctype), 
                                t2.astype(rtype), 
                                backend="jax")
            lt2_e = oe.contract("gib,iajb->gja", 
                                glgp.astype(ctype), 
                                t2.astype(rtype), 
                                backend="jax")
            
            l2t2_c = oe.contract("gjb,gjb->", 
                                 lt2_c.astype(ctype), 
                                 glgp.astype(ctype), 
                                 backend="jax").astype(jnp.complex128)
            l2t2_e = oe.contract("gja,gja->", 
                                 lt2_e.astype(ctype), 
                                 glgp.astype(ctype), 
                                 backend="jax").astype(jnp.complex128)
            carry[3] += (2*l2t2_c - l2t2_e).astype(jnp.complex128)

            return carry, 0.0

        [e2_0, e2_2_2_1, e2_2_2_2, e2_2_3], _ = lax.scan(
            scan_chunk, [0.0, 0.0, 0.0, 0.0], (chol, rot_chol)
        )

        e2_2_1 = e2_0 * gt2g
        e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3

        e0 = e1_0 + e2_0  # <psi|(h1+h2)|phi>/<psi|phi>
        e1frag = e1_2 + e2_2  # <psi|t2(h1+h2)|phi>/<psi|phi>
        t2frag = gt2g          # <psi|t2|phi>/<psi|phi>
        return t2frag, e1frag, e0


    @partial(jit, static_argnums=(0, 4))
    def _calc_ept2_frag(self, walker: jax.Array, ham_data: dict, wave_data: dict,
                        frozen_vir=None):

        eg = self._calc_energy_restricted(walker, ham_data, wave_data)

        walker_bar = wave_data['exp_t1'] @ walker
        o0 = jnp.linalg.det(walker[:walker.shape[1], :]) ** 2
        obar = jnp.linalg.det(walker_bar[:walker_bar.shape[1], :]) ** 2
        t1 = obar/o0 # <exp(T1)HF|walker>/<HF|walker>
        e0frag = self._calc_e0bar_frag(walker_bar, ham_data, wave_data)
        t2frag, e1frag, e0 = self._t2eorb_tc(walker_bar, ham_data, wave_data, frozen_vir)

        return eg, t1, t2frag, e0frag, e1frag, e0

    @partial(jit, static_argnums=(0, 4))
    def calc_ept2_frag(self, walkers: jax.Array, ham_data: dict, wave_data: dict,
                       frozen_vir=None) -> jax.Array:

        n_walkers = walkers.shape[0]
        batch_size = n_walkers // self.n_batch
        
        def scan_batch(carry, walker_batch):
            eg, t1, t2frag, e0frag, e1frag, e0 \
                = vmap(self._calc_ept2_frag, in_axes=(0, None, None, None))(
                walker_batch, ham_data, wave_data, frozen_vir
            )
            return carry, (eg, t1, t2frag, e0frag, e1frag, e0)
        
        _, (eg, t1, t2frag, e0frag, e1frag, e0) \
            = lax.scan(scan_batch, None, walkers.reshape(self.n_batch, batch_size, self.norb,-1))

        eg = eg.reshape(n_walkers)
        t1 = t1.reshape(n_walkers)
        t2frag = t2frag.reshape(n_walkers)
        e0frag = e0frag.reshape(n_walkers)
        e1frag = e1frag.reshape(n_walkers)
        e0 = e0.reshape(n_walkers)

        return eg, t1, t2frag, e0frag, e1frag, e0
    
    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        """Builds half rotated integrals for efficient force bias and energy calculations."""
        norb, nocc = self.norb, self.nelec[0]
        chol = ham_data["chol"].reshape(-1, norb, norb)

        ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ ham_data["h1"][0]
        ham_data["rot_chol"] = oe.contract(
            "pi,gij->gpj",
            wave_data["mo_coeff"].T.conj(),
            ham_data["chol"].reshape(-1, self.norb, self.norb), 
            backend="jax")

        # exp(T1^dagger) H exp(-T1^dagger)
        ham_data["h1_bar"] = wave_data['exp_t1'] @ ham_data['h1'][0] @ wave_data['exp_mt1']
        # ham_data["h1_bar"] = h1_bar
        ham_data["chol_bar"] = oe.contract('pr,grs,sq->gpq', 
                               wave_data['exp_t1'], chol, 
                               wave_data['exp_mt1'], backend="jax")
        # ham_data["chol_bar"] = chol_bar
        # exp(T1^dagger) Fock exp(-T1^dagger)
        # jeff = oe.contract('gpq,gjj->pq', chol_bar, chol_bar[:,:nocc,:nocc], backend="jax")
        # keff = oe.contract('gpj,gjq->pq', chol_bar[:,:,:nocc],
        #                 chol_bar[:,:nocc,:], backend="jax")
        # fock_bar = h1_bar + 2 * jeff - keff
        # ham_data['fock_bar'] = fock_bar
        ham_data['fock_bar'] = integral.get_rfock(nocc, ham_data["h1_bar"], ham_data["chol_bar"])
        # ham_data['fock_bar'] = oe.contract('ip,ik->kp', ham_data['fock_bar'][:nocc, :], wave_data['prjlo'], backend="jax")
        
        lt1 = oe.contract('ia,gja->gij', wave_data["t1"], chol[:, :nocc, nocc:], backend='jax')
        ham_data['e0t1orb'] = 2 * oe.contract('gik,ik,gjj->',lt1, wave_data['prjlo'], lt1, backend='jax') \
                    - oe.contract('gij,gjk,ik->',lt1, lt1, wave_data['prjlo'], backend='jax')
        
        del chol, lt1
        
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class pt2ccsd_sto_chol(pt2ccsd):
    """LNO pt2CCSD with a semistochastic Cholesky sum in the T2*h2 term.

    Restricted counterpart of the unrestricted lno pt2ccsd_sto_chol.  Only the
    fragment T2*h2 contributions are sampled -- e2_2_2_1, e2_2_2_2 and e2_2_3 of
    `_t2eorb_tc`'s scan.  e2_0 stays exact, and therefore so does
    e2_2_1 = e2_0 * gt2g: it needs only the cheap gl = green.chol contraction and
    is required to build the proposal anyway, while the sampled terms carry the
    "gia,iajb->gjb" contractions that scale as nocc^2 nvir^2 per vector.

    The proposal comes from `_calc_e0bar_frag`, which `_calc_ept2_frag` already
    evaluates on the same walker: its per-Cholesky fragment two-body energies are
    returned alongside the scalar and turned into pi_g.  That makes the score a
    surrogate -- it ranks vectors by the fragment two-body energy rather than by
    the T2 terms actually sampled -- which costs variance, never bias.

    Budget knobs (dataclass fields, static under jit).  `chol_cost_ratio` sets the
    per-walker fraction of nchol and splits it head : samples = 3 : 1; an explicit
    `head_chol_ratio` or `n_chol_samples` overrides its half.  `n_chol_head`
    beats both, and "full" disables sampling and reproduces pt2ccsd exactly.
    """

    n_chol_head: Union[int, str] = 0
    head_chol_ratio: Union[float, None] = None
    n_chol_samples: Union[int, None] = None
    chol_cost_ratio: Union[float, None] = None
    head_sample_ratio: float = 3.0
    chol_score_floor: float = 1.0e-6
    chol_uniform_mix: float = 0.01
    head_from_guide: bool = False

    @partial(jit, static_argnums=0)
    def _prop_chol_in_place(self, e2_g_estimate):
        """pi_g = (1-u)|e2_g|/sum|e2_g| + u/nchol, with a relative floor.

        The uniform part keeps pi_g > 0 everywhere, which bounds the importance
        weights 1/pi_g; a vector with pi_g = 0 would never be drawn yet still
        contributes, and that would be a bias rather than extra variance.
        """
        e2_g = jnp.abs(e2_g_estimate)
        e2_g = jnp.where(e2_g >= self.chol_score_floor * jnp.max(e2_g), e2_g, 0.0)
        nchol = e2_g.shape[0]
        uniform = jnp.full((nchol,), 1.0 / nchol)
        total = jnp.sum(e2_g)
        guided = jnp.where(total > 0.0, e2_g / jnp.where(total > 0.0, total, 1.0), uniform)
        return (1.0 - self.chol_uniform_mix) * guided + self.chol_uniform_mix * uniform

    @partial(jit, static_argnums=0)
    def _calc_e0bar_frag_scored(self, walker, ham_data, wave_data):
        """`_calc_e0bar_frag`, also returning the per-Cholesky two-body energies.

        Identical arithmetic to the parent; the chunk scan just emits its per-gamma
        contributions instead of only accumulating them.
        """
        nocc, norb = self.nelec[0], self.norb
        prjlo = wave_data['prjlo']
        e0 = ham_data['e0t1orb']
        rot_fock_ov = ham_data['fock_bar'][:nocc, nocc:]
        rot_chol_ov = ham_data['chol_bar'].reshape(-1, norb, norb)[:, :nocc, nocc:]
        gf_ov = self._calc_green(walker, wave_data)[:nocc, nocc:]

        e1 = oe.contract('ia,ik,ka->', gf_ov, prjlo, rot_fock_ov, backend="jax") * 2

        nchol = rot_chol_ov.shape[0]
        nchunks, chunk, npad = _chol_chunking(nchol, self.nchol_chunk)
        if npad:
            rot_chol_ov = jnp.pad(rot_chol_ov, ((0, npad), (0, 0), (0, 0)))
        rot_chol_ov = rot_chol_ov.reshape(nchunks, chunk, nocc, -1)

        def scanned_fun(carry, x):
            chol_c = x
            lg_c = oe.contract('gia,ka->gik', chol_c, gf_ov, backend="jax")
            # per-gamma pieces of the parent's e2_1_c - e2_2_c
            p_g = oe.contract('gik,ik->g', lg_c, prjlo, backend="jax")
            t_g = oe.contract('gjj->g', lg_c, backend="jax")
            x_g = oe.contract('gij,gjk,ik->g', lg_c, lg_c, prjlo, backend="jax")
            e2_g = 2.0 * p_g * t_g - x_g
            return carry + jnp.sum(e2_g), e2_g

        e2, e2_chunks = lax.scan(scanned_fun, 0.0 + 0.0j, rot_chol_ov)
        return e0 + e1 + e2, e2_chunks.reshape(-1)[:nchol]

    @partial(jit, static_argnums=(0, 6))
    def _t2eorb_tc_sto(self, walker, ham_data, wave_data, pi_g, key, frozen_vir=None):
        """`_t2eorb_tc` with the T2*h2 Cholesky sum split head/tail.

        Pass 1 sums e2_0 exactly over every gamma with no T2 work; pass 2 does the
        three T2-contracted accumulators on the head exactly and on an importance
        sampled tail.
        """
        if self.mix_precision:
            rtype = jnp.float32
            ctype = jnp.complex64
        else:
            rtype = jnp.float64
            ctype = jnp.complex128

        nocc, norb = self.nelec[0], self.norb
        nchol_chunk = self.nchol_chunk
        t2 = wave_data["t2"]
        chol = ham_data["chol_bar"]
        h1 = ham_data["h1_bar"]

        if frozen_vir is not None:
            # Same slicing as the parent.  nchol is untouched, so pi_g -- built by
            # _calc_e0bar_frag_scored on the full Cholesky set -- still lines up.
            fv = frozen_vir
            n_keep = norb - fv
            nv_keep = (norb - nocc) - fv
            assert nv_keep > 0, "frozen_vir exceeds number of virtuals"
            norb = n_keep
            walker = walker[:n_keep, :]
            h1 = h1[:n_keep, :n_keep]
            chol = chol[:, :n_keep, :n_keep]
            t2 = t2[:, :nv_keep, :, :nv_keep]
            wave_data = {**wave_data, "mo_coeff": wave_data["mo_coeff"][:n_keep, :]}

        green = self._calc_green(walker, wave_data)
        green_occ = green[:, nocc:]
        greenp = jnp.vstack((green_occ, -jnp.eye(norb - nocc)))
        nchol = chol.shape[0]

        # ---- Cholesky-independent pieces (identical to _t2eorb_tc) ----
        hg = oe.contract("pi,pi->", h1[:nocc, :], green, backend="jax")
        e1_0 = 2 * hg

        t2g_c_1 = oe.contract("iajb,ia->jb", t2, green_occ, backend="jax")
        t2g_c_2 = oe.contract("iajb,jb->ia", t2, green_occ, backend="jax")
        t2g_e_1 = oe.contract("iajb,ib->ja", t2, green_occ, backend="jax")
        t2g_e_2 = oe.contract("iajb,ja->ib", t2, green_occ, backend="jax")
        t2_green_c_1 = oe.contract("pb,jb,jq->pq", greenp, t2g_c_1, green, backend="jax")
        t2_green_c_2 = oe.contract("pa,ia,iq->pq", greenp, t2g_c_2, green, backend="jax")
        t2_green_e_1 = oe.contract("pa,ja,jq->pq", greenp, t2g_e_1, green, backend="jax")
        t2_green_e_2 = oe.contract("pb,ib,iq->pq", greenp, t2g_e_2, green, backend="jax")
        t2g_c = t2g_c_1 + t2g_c_2
        t2g_e = t2g_e_1 + t2g_e_2
        t2_green_c = t2_green_c_1 + t2_green_c_2
        t2_green_e = t2_green_e_1 + t2_green_e_2
        t2_green = t2_green_c - t2_green_e * 0.5
        t2g = t2g_c - t2g_e * 0.5
        gt2g = oe.contract("ia,ia->", t2g, green_occ, backend="jax")
        e1_2 = 2 * hg * gt2g - 2 * oe.contract("pq,pq->", h1, t2_green, backend="jax")

        # ============ pass 1: e2_0, exact, every gamma, no T2 ============
        # Only the occupied-occupied block of gl is used, so feed this pass the
        # half-rotated chol[:, :nocc, :]: gl comes out (chunk, nocc, nocc) rather
        # than (chunk, nocc, norb), and that factor is paid per walker under vmap.
        def scan_chunk_e2_0(carry, x):
            rot_c = x
            gl_occ = oe.contract("ir,gqr->giq", green, rot_c, backend="jax")
            gl_c = oe.contract("gii->g", gl_occ, backend="jax")
            e2_0_g = (2 * oe.contract("g,g->g", gl_c, gl_c, backend="jax")
                      - oe.contract("gij,gji->g", gl_occ, gl_occ, backend="jax"))
            return carry + jnp.sum(e2_0_g.astype(ctype)), 0.0

        nchunk, chunk1, npad = _chol_chunking(nchol, nchol_chunk)
        rot_all = chol[:, :nocc, :]
        if npad:
            rot_all = jnp.concatenate(
                [rot_all, jnp.zeros((npad, *rot_all.shape[-2:]), rot_all.dtype)], axis=0)
        e2_0, _ = lax.scan(scan_chunk_e2_0, jnp.zeros((), dtype=ctype),
                           rot_all.reshape(nchunk, chunk1, *rot_all.shape[-2:]))

        # ---- head / tail split from the supplied proposal ----
        n_head, n_samples = _resolve_chol_budget(
            nchol, self.n_chol_head, self.head_chol_ratio, self.n_chol_samples,
            self.chol_cost_ratio, self.head_sample_ratio)

        # Contiguous prefix head by default, NOT ranked per walker: under vmap a
        # walker-dependent index array makes chol[idx] a *batched* gather, costing
        # n_walkers * n_head * norb^2 instead of the shared n_head * norb^2.  A
        # prefix is a plain slice, and the Cholesky ordering already runs from most
        # to least important.  head_from_guide=True restores per-walker ranking.
        head_prefix = None
        if n_head >= nchol:
            head_prefix = nchol
            tail = jnp.zeros((0,), dtype=jnp.int32)
            tail_prob = jnp.zeros((0,))
        else:
            if self.head_from_guide:
                order = jnp.argsort(-pi_g)
                head_idx = jnp.sort(order[:n_head])
                tail = jnp.sort(order[n_head:])
            else:
                head_prefix = n_head
                tail = jnp.arange(n_head, nchol, dtype=jnp.int32)
            tail_prob = pi_g[tail]
            tail_prob = tail_prob / jnp.sum(tail_prob)

        # ======== pass 2: only the T2-contracted accumulators ========
        def scan_chunk_e2_2(carry, x):
            chol_c, w_c = x
            rot_chol_c = chol_c[:, :nocc, :]
            w_c = w_c.astype(ctype)

            gl = oe.contract("ir,gqr->giq", green, chol_c, backend="jax")
            gl_c = oe.contract("gii->g", gl[:, :, :nocc], backend="jax")

            lt2g = oe.contract("gpr,pr->g", chol_c.astype(rtype),
                               t2_green.astype(ctype), backend="jax")
            carry[0] += jnp.sum(w_c * (-lt2g.astype(ctype) * gl_c.astype(ctype)))

            lt2_green = oe.contract("gir,qr->giq", rot_chol_c.astype(rtype),
                                    t2_green.astype(ctype), backend="jax")
            carry[1] += jnp.sum(w_c * 0.5 * oe.contract("giq,giq->g", gl.astype(ctype),
                                lt2_green.astype(ctype), backend="jax"))

            glgp = oe.contract("gir,rb->gib", gl.astype(ctype),
                               greenp.astype(ctype), backend="jax")
            lt2_c = oe.contract("gia,iajb->gjb", glgp.astype(ctype),
                                t2.astype(rtype), backend="jax")
            lt2_e = oe.contract("gib,iajb->gja", glgp.astype(ctype),
                                t2.astype(rtype), backend="jax")
            l2t2_c = oe.contract("gjb,gjb->g", lt2_c.astype(ctype),
                                 glgp.astype(ctype), backend="jax")
            l2t2_e = oe.contract("gja,gja->g", lt2_e.astype(ctype),
                                 glgp.astype(ctype), backend="jax")
            carry[2] += jnp.sum(w_c * (2 * l2t2_c - l2t2_e).astype(ctype))
            return carry, 0.0

        def _run(chol_s, weights):
            n = weights.shape[0]
            z = jnp.zeros((), dtype=ctype)
            if n == 0:
                return z, z, z
            nch2, chunk2, npad2 = _chol_chunking(n, nchol_chunk)
            if npad2:
                chol_s = jnp.concatenate(
                    [chol_s, jnp.zeros((npad2, *chol_s.shape[-2:]), chol_s.dtype)])
                weights = jnp.concatenate([weights, jnp.zeros(npad2, weights.dtype)])
            out, _ = lax.scan(scan_chunk_e2_2, [z, z, z],
                              (chol_s.reshape(nch2, chunk2, *chol_s.shape[-2:]),
                               weights.reshape(nch2, chunk2)))
            return out[0], out[1], out[2]

        # prefix -> plain slice, shared across the vmap batch
        if head_prefix is not None:
            b_h, c_h, d_h = _run(chol[:head_prefix], jnp.ones(head_prefix, dtype=ctype))
        else:
            b_h, c_h, d_h = _run(chol[head_idx],
                                 jnp.ones(head_idx.shape[0], dtype=ctype))
        # tail -> walker-dependent indices, but only n_chol_samples vectors wide
        if tail.shape[0] == 0:
            b_t = c_t = d_t = jnp.zeros((), dtype=ctype)
        else:
            sel = random.choice(key, tail.shape[0], shape=(n_samples,),
                                replace=True, p=tail_prob)
            samp_w = (1.0 / (n_samples * tail_prob[sel])).astype(ctype)
            b_t, c_t, d_t = _run(chol[tail[sel]], samp_w)

        # e2_2_1 = e2_0 * gt2g is exact, since e2_0 is
        e2_2 = e2_0 * gt2g + 4 * ((b_h + b_t) + (c_h + c_t)) + (d_h + d_t)

        e0 = e1_0 + e2_0
        e1frag = e1_2 + e2_2
        t2frag = gt2g
        return t2frag, e1frag, e0

    @partial(jit, static_argnums=(0, 5))
    def _calc_ept2_frag(self, walker: jax.Array, ham_data: dict, wave_data: dict, key,
                        frozen_vir=None):

        eg = self._calc_energy_restricted(walker, ham_data, wave_data)

        walker_bar = wave_data['exp_t1'] @ walker
        o0 = jnp.linalg.det(walker[:walker.shape[1], :]) ** 2
        obar = jnp.linalg.det(walker_bar[:walker_bar.shape[1], :]) ** 2
        t1 = obar / o0

        # fragment reference energy, and the per-gamma scores it produces for free
        e0frag, e2_g = self._calc_e0bar_frag_scored(walker_bar, ham_data, wave_data)
        pi_g = self._prop_chol_in_place(e2_g)

        t2frag, e1frag, e0 = self._t2eorb_tc_sto(
            walker_bar, ham_data, wave_data, pi_g, key, frozen_vir)

        return eg, t1, t2frag, e0frag, e1frag, e0

    @partial(jit, static_argnums=(0, 4))
    def calc_ept2_frag(self, walkers: jax.Array, ham_data: dict, wave_data: dict,
                       frozen_vir=None) -> jax.Array:
        """Map over walkers, giving each its own key split from the block key."""
        n_walkers = walkers.shape[0]
        batch_size = n_walkers // self.n_batch
        key = wave_data.get("sto_chol_key", random.PRNGKey(0))
        keys = random.split(key, n_walkers)

        def scan_batch(carry, x):
            walker_batch, batch_keys = x
            eg, t1, t2frag, e0frag, e1frag, e0 \
                = vmap(self._calc_ept2_frag, in_axes=(0, None, None, 0, None))(
                walker_batch, ham_data, wave_data, batch_keys, frozen_vir)
            return carry, (eg, t1, t2frag, e0frag, e1frag, e0)

        _, (eg, t1, t2frag, e0frag, e1frag, e0) = lax.scan(
            scan_batch, None,
            (walkers.reshape(self.n_batch, batch_size, self.norb, -1),
             keys.reshape(self.n_batch, batch_size, -1)))

        return (eg.reshape(n_walkers), t1.reshape(n_walkers), t2frag.reshape(n_walkers),
                e0frag.reshape(n_walkers), e1frag.reshape(n_walkers), e0.reshape(n_walkers))

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
