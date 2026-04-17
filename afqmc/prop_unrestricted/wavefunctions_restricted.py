from abc import ABC
from dataclasses import dataclass
from functools import partial
from typing import Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, jit, jvp, lax, vmap
import opt_einsum as oe


class wave_function_restricted(ABC):
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

    # @calc_overlap.register
    def calc_overlap(self, walkers: jax.Array, wave_data: dict) -> jax.Array:
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

    def calc_force_bias(self, walkers: jax.Array, ham_data: dict, wave_data: dict) -> jax.Array:
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

    def calc_energy(self, walkers: jax.Array, ham_data: dict, wave_data: dict) -> jax.Array:
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
        
    def decompose_t2(self, t2, thresh: float = 1e-8):
        # adapted from Yann

        nO = self.nelec[0]
        nV = self.norb - nO
        nex = nO * nV

        assert t2.shape == (nO, nV, nO, nV)
        
        t2 = t2.reshape(nex, nex)
        e_val, e_vec = jnp.linalg.eigh(t2)

        # Keep only important modes
        mask = jnp.abs(e_val) > thresh
        e_val_trunc = e_val[mask]
        e_vec_trunc = e_vec[:, mask]

        L = e_vec_trunc @ jnp.diag(jnp.sqrt(e_val_trunc + 0.0j))
        
        err = jnp.linalg.norm(t2 - L @ L.T)
        print(f'# Throw {len(e_val)-len(e_val_trunc)} vectors in T2 deomposition')
        print(f'# cutoff = {thresh:.2e} | error = {err:.2e}')
        print(f'# number of T2 decomposition vectors {len(e_val_trunc)}')

        # L = e_vec @ jnp.diag(jnp.sqrt(e_val + 0.0j))
        # assert jnp.abs(jnp.linalg.norm(t2 - L @ L.T)) < 1e-12

        # Summation on the left
        L = L.T.reshape(-1, nO, nV)
        # t2_rec = jnp.einsum('gia,gjb->iajb', L, L)
        # assert jnp.abs(wave_data['t2'] - t2_rec).max() < 1e-12

        return L
    
    @partial(jit, static_argnums=0)
    def _thouless(self, init_slater, t):
        # calculate |psi'> = exp(t_ia a+ i)|psi>
        norb, nocc = self.norb, self.nelec[0]
        nvir = norb - nocc
        assert t.shape == (nocc, nvir)
        t_full = jnp.eye(norb, dtype=jnp.complex128)
        exp_t = t_full.at[:nocc, nocc:].set(t)
        # exp_tau = jsp.linalg.expm(t_full)
        return exp_t.T @ init_slater
    
    @partial(jit, static_argnums=0)
    def _thouless_full(self, init_slater, t):
        # calculate |psi'> = exp(t_pq p+ q)|psi>
        from jax import scipy as jsp
        norb, nocc = self.norb, self.nelec[0]
        assert t.shape == (norb, norb)
        t_full = jnp.zeros((norb, norb), dtype=jnp.complex128)
        t_full = t_full.at[:nocc, nocc:].set(t)
        exp_t = jsp.linalg.expm(t_full)
        return exp_t.T @ init_slater

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))



# we assume afqmc is performed in the rhf orbital basis
@dataclass
class rhf(wave_function_restricted):
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

    # @partial(jit, static_argnums=0)
    # def _calc_overlap(
    #     self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    # ) -> jax.Array:
    #     return jnp.linalg.det(
    #         wave_data["mo_coeff"].T.conj() @ walker_up
    #     ) * jnp.linalg.det(wave_data["mo_coeff"].T.conj() @ walker_dn)

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
        # green_walker = self._calc_green(walker, wave_data)
        # ene1 = 2.0 * jnp.sum(green_walker * rot_h1)
        # f = oe.contract("gij,jk->gik", rot_chol, green_walker.T, backend="jax")
        # c = vmap(jnp.trace)(f)
        # exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        # ene2 = 2.0 * jnp.sum(c * c) - exc
        # return h0 + ene1 + ene2
        green = self._calc_green(walker, wave_data)
        hg = oe.contract("pq,pq->", rot_h1, green, backend="jax")
        e1 = 2 * hg
        lg = oe.contract("gpr,qr->gpq", rot_chol, green, backend="jax")
        e2_1 = 2 * jnp.sum(oe.contract('gpp->g', lg, backend="jax")**2)
        e2_2 = -oe.contract('gpq,gqp->',lg,lg, backend="jax")
        e2 = e2_1 + e2_2
        return h0 + e1 + e2

    # @partial(jit, static_argnums=0)
    # def _calc_energy(
    #     self,
    #     walker_up: jax.Array,
    #     walker_dn: jax.Array,
    #     ham_data: dict,
    #     wave_data: dict,
    # ) -> jax.Array:
    #     h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
    #     ene0 = h0
    #     green_walker_up = self._calc_green(walker_up, wave_data)
    #     green_walker_dn = self._calc_green(walker_dn, wave_data)
    #     green_walker = green_walker_up + green_walker_dn
    #     ene1 = jnp.sum(green_walker * rot_h1)
    #     f = oe.contract("gij,jk->gik", rot_chol, green_walker.T, backend="jax")
    #     c = vmap(jnp.trace)(f)
    #     exc = jnp.sum(vmap(lambda x: x * x.T)(f))
    #     ene2 = jnp.sum(c * c) - exc
    #     return ene2 + ene1 + ene0

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        rdm1 = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T] * 2)
        return rdm1

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
    
    
    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))
    

@dataclass
class cisd_hf(rhf):
    """
    use CISD Trial and HF Guide, the 2nd way: 
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

    # @partial(jit, static_argnums=0)
    def get_xtau(self, wave_data: dict, prop_data: dict):
        nslater = self.nslater

        # t_iajb = tau_gia tau_gjb
        tau, _ = self.decompose_t2(wave_data)

        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                nslater,
                tau.shape[0],
            ),
        )

        xtau =jnp.einsum("sg,gia->sia", fields, tau) # (nslater,nocc,nvir)

        return xtau

    # sto-CCSD correction part
    @partial(jit, static_argnums=0)
    def xtau_exp_energy_one_walker(
            self,
            walker: jax.Array,
            h0: float,
            h1: jax.Array,
            chol: jax.Array,
            xtau: jax.Array,
            )-> complex:
        # <exp(xtau)HF|H|walker>

        norb = self.norb
        nocc = self.nelec[0]
        one = jnp.eye(norb, dtype=jnp.complex128)
        exp_xtau = one.at[:nocc, nocc:].set(xtau)
        slater = jnp.eye(norb, dtype=jnp.complex128)[:,:nocc]
        slater = exp_xtau.T @ slater

        green = (walker @ (
                jnp.linalg.inv(slater.T.conj() @ walker)
                    ) @ slater.T.conj()
                    ).T
        
        hg = oe.contract("pq,pq->", h1, green, backend="jax")
        e1 = 2 * hg
        lg = oe.contract("gpr,qr->gpq", chol, green, backend="jax")
        e2_1 = 2 * jnp.sum(oe.contract('gpp->g', lg, backend="jax")**2)
        e2_2 = oe.contract('gpq,gqp->',lg,lg, backend="jax")
        e2 = e2_1 - e2_2

        olp = jnp.linalg.det(slater.T.conj() @ walker) ** 2
        
        return (h0 + e1 + e2)*olp, olp

    # sto-CCSD correction part
    @partial(jit, static_argnums=0)
    def xtau_cisd_energy_one_walker(
        self,
        walker: jax.Array,
        h0: float,
        h1: jax.Array,
        chol: jax.Array,
        xtau: jax.Array,
    ) -> complex:
        # <[1 + xtau + (xtau)^2]HF|H|walker>
        
        ci1 = xtau
        ci2 = oe.contract('ia,jb->iajb', xtau, xtau, backend='jax')
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:]
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        rot_chol = chol[:, : self.nelec[0], :]
        hg = oe.contract("pj,pj->", h1[:nocc, :], green, backend="jax")

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
        e1 = e1_0 + e1_1 + e1_2

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
        lci1 = oe.contract(
                "git,pt->gip",
                chol[:, :, self.nelec[0] :],
                ci1,
                backend="jax"
            )
        lci1g = oe.contract("gip,qi->gpq", lci1, green, backend="jax")
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
        e2 = e2_0 + e2_1 + e2_2

        # overlap
        olp_hf = jnp.linalg.det(walker[:nocc,:nocc]) ** 2
        o1 = 2 * ci1g
        o2 = gci2g
        olp_cisd = (1.0 + o1 + o2) * olp_hf

        return h0*olp_cisd + (e1 + e2)*olp_hf, olp_cisd
    
    @partial(jit, static_argnums=0)
    def xtau_exp_cisd_e_olp_one_walker(
        self,
        walker: jax.Array,
        h0: float,
        h1: jax.Array,
        chol: jax.Array,
        xtau: jax.Array,
        ):

        e_exp, o_exp = self.xtau_exp_energy_one_walker(walker, h0, h1, chol, xtau)
        e_cisd, o_cisd = self.xtau_cisd_energy_one_walker(walker, h0, h1, chol, xtau)

        de = e_exp - e_cisd
        do = o_exp - o_cisd
        
        return (de, do)
    
    @partial(jit, static_argnums=0)
    def get_stoccsd_cr_one_walker(self, walker, ham_data, wave_data):
        def scan_xtau_one_walker(carry, xtau: jax.Array):
            (de, do) = self.xtau_exp_cisd_e_olp_one_walker(
                walker,
                ham_data['h0'],
                (ham_data['h1'][0]+ham_data['h1'][0])/2,
                ham_data['chol'].reshape(-1,self.norb,self.norb),
                xtau,
                )
            return carry, (de, do)

        init_carry = 0.0
        _, (des, dos) = lax.scan(scan_xtau_one_walker, init_carry, wave_data['xtau'])

        de = jnp.sum(des)
        do = jnp.sum(dos)

        return de, do
    
    @partial(jit, static_argnums=(0))
    def calc_stoccsd_cr(self, walkers, ham_data, wave_data):
        de, do = vmap(
            self.get_stoccsd_cr_one_walker,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        
        return de, do

    @partial(jit, static_argnums=0)
    def _calc_energy_cisd_hf(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
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
        # e1 = e1_0 + e1_1 + e1_2

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

        # lci2g1 = jnp.einsum("gij,jk->gik", chol, ci2_green, optimize="optimal")
        # lci2_green = jnp.einsum("gpi,ji->gpj", rot_chol, ci2_green, optimize="optimal")
        # e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lci2_green, optimize="optimal")
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

        olp = 1 + 2*ci1g + gci2g # <CISD|walker>/HF|walker>
        e_hf = h0 + e1_0 + e2_0 # <HF|(h1+h2)|phi>/<HF|phi>
        # h0 + {<HF|(1+C1+C2)(h1+h2)|walker>/<HF|walker>} / {<CISD|walker>/<HF|walker>}
        e_ci = h0 + (e1_0 + e2_0 + e1_1 + e1_2 + e2_1 + e2_2) / olp

        return olp, e_hf, e_ci

    @partial(jit, static_argnums=(0))
    def calc_energy_mixed(self,walkers,ham_data,wave_data):
        olp, ehf, eci = vmap(
            self._calc_energy_cisd_hf,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        return olp, ehf, eci

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        norb = self.norb
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )

        ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ (
            (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        )
        ham_data["rot_chol"] = oe.contract(
            "pi,gij->gpj",
            wave_data["mo_coeff"].T.conj(),
            ham_data["chol"].reshape(-1, norb, norb), backend="jax"
        )
        ham_data["lci1"] = oe.contract(
            "git,pt->gip",
            ham_data["chol"].reshape(-1, self.norb, self.norb)[:, :, self.nelec[0] :],
            wave_data["ci1"],
            optimize="optimal", backend="jax"
        )
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


class mixed(wave_function_restricted):

    def __init__(self, guide, trial):
        self._guide = guide
        self._trial = trial

    def _calc_rdm1(self, wave_data):
        return self._guide._calc_rdm1(wave_data)

    def _calc_force_bias_restricted(self, walker, ham_data, wave_data):
        return self._guide._calc_force_bias_restricted(walker, ham_data, wave_data)

    def _calc_overlap_restricted(self, walker, wave_data):
        return self._guide._calc_overlap_restricted(walker, wave_data)
    
    def _calc_energy_restricted(self, walker, ham_data, wave_data):
        return self._guide._calc_energy_restricted(walker, ham_data, wave_data)
    
    # def _calc_energy_restricted_trial(self, walker, ham_data, wave_data):
    #     return self._trial._calc_energy_restricted(walker, ham_data, wave_data)
    
    def _calc_energy_mixed(self, walker, ham_data, wave_data):
        eg = self._guide._calc_energy_restricted(walker, ham_data, wave_data)
        et = self._trial._calc_energy_restricted(walker, ham_data, wave_data)
        og = self._guide._calc_overlap_restricted(walker, wave_data)
        ot = self._trial._calc_overlap_restricted(walker, wave_data)
        otg = ot/og
        return ot, eg, et
    
    @partial(jit, static_argnums=(0))
    def calc_energy_mixed(self,walkers,ham_data,wave_data):
        otg, eg, et = vmap(
            self._calc_energy_mixed,in_axes=(0, None, None))(
                walkers, ham_data, wave_data)
        return otg, eg, et
    
    def _build_measurement_intermediates(self, ham_data, wave_data):
        ham_data = self._guide._build_measurement_intermediates(ham_data, wave_data)
        ham_data = self._trial._build_measurement_intermediates(ham_data, wave_data)
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))

@dataclass
class cisd_pt(wave_function_restricted):
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

        # lci2g1 = jnp.einsum("gij,jk->gik", chol, ci2_green, optimize="optimal")
        # lci2_green = jnp.einsum("gpi,ji->gpj", rot_chol, ci2_green, optimize="optimal")
        # e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lci2_green, optimize="optimal")
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
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
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
class cisd(wave_function_restricted):
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
        e0 = ham_data["h0"]

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
        e1 = e1_0 + e1_1 + e1_2

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

        # lci2g1 = jnp.einsum("gij,jk->gik", chol, ci2_green, optimize="optimal")
        # lci2_green = jnp.einsum("gpi,ji->gpj", rot_chol, ci2_green, optimize="optimal")
        # e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lci2_green, optimize="optimal")
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

        e2 = e2_0 + e2_1 + e2_2

        # overlap
        overlap_1 = 2 * ci1g
        overlap_2 = gci2g
        overlap = 1.0 + overlap_1 + overlap_2
        return (e1 + e2) / overlap + e0


    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
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
class cid(wave_function_restricted):
    """A manual implementation of the CISD wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        rdm1 = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T] * 2)
        return rdm1

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci2 = walker.shape[1], wave_data["ci2"]
        GF = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        o2 = 2 * oe.contract(
            "iajb, ia, jb", ci2, GF[:, nocc:], GF[:, nocc:], backend="jax"
        ) - oe.contract("iajb, ib, ja", ci2, GF[:, nocc:], GF[:, nocc:], backend="jax")
        return (1.0 + o2) * o0


    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
        ci2 = wave_data["ci2"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]
        lg = oe.contract("gpj,pj->g", rot_chol, green, backend="jax")

        # ref
        fb_0 = 2 * lg

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
        overlap_2 = gci2g / 2.0
        overlap = 1.0 + overlap_2

        return (fb_0 + fb_2) / overlap
    
    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        ci2 = wave_data["ci2"]
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
        e0 = ham_data["h0"]

        # 1 body energy
        # ref
        e1_0 = 2 * hg

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
        e1 = e1_0 + e1_2

        # two body energy
        # ref
        lg = oe.contract("gpj,pj->g", rot_chol, green, backend="jax")
        # lg1 = jnp.einsum("gpj,pk->gjk", rot_chol, green, optimize="optimal")
        lg1 = oe.contract("gpj,qj->gpq", rot_chol, green, backend="jax")
        e2_0_1 = 2 * lg @ lg
        e2_0_2 = -jnp.sum(vmap(lambda x: x * x.T)(lg1))
        e2_0 = e2_0_1 + e2_0_2

        # double excitations
        e2_2_1 = e2_0 * gci2g
        lci2g = oe.contract("gij,ij->g", chol, ci2_green, backend="jax")
        e2_2_2_1 = -lci2g @ lg

        # lci2g1 = jnp.einsum("gij,jk->gik", chol, ci2_green, optimize="optimal")
        # lci2_green = jnp.einsum("gpi,ji->gpj", rot_chol, ci2_green, optimize="optimal")
        # e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lci2_green, optimize="optimal")
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

        e2 = e2_0 + e2_2

        # overlap
        overlap_2 = gci2g
        overlap = 1.0 + overlap_2
        return (e1 + e2) / overlap + e0
    
    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:

        return ham_data
    
    def __hash__(self):
        return hash(tuple(self.__dict__.values()))

@dataclass
class cisd_faster(cisd):
    """A manual implementation of the CISD wave function.

    Faster than cisd, but the energy function builds some large intermediates, O(XMN),
    so memory usage is high.
    """

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

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
        rot_chol = chol[:, : self.nelec[0], :]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        hg = oe.contract("pj,pj->", h1[:nocc, :], green, backend="jax")

        # 0 body energy
        e0 = ham_data["h0"]

        # 1 body energy
        # ref
        e1_0 = 2 * hg

        # single excitations
        ci1g = oe.contract("pt,pt->", ci1, green_occ, backend="jax")
        e1_1_1 = 4 * ci1g * hg
        gpci1 = greenp @ ci1.T
        ci1_green = gpci1 @ green
        e1_1_2 = -2 *oe.contract("ij,ij->", h1, ci1_green, backend="jax")
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
        e1 = e1_0 + e1_1 + e1_2

        # two body energy
        # ref
        lg = oe.contract("gpj,pj->g", rot_chol, green, backend="jax")
        lg1 = oe.contract("gpj,qj->gpq", rot_chol, green, backend="jax")
        e2_0_1 = 2 * lg @ lg
        e2_0_2 = -jnp.sum(vmap(lambda x: x * x.T)(lg1))
        e2_0 = e2_0_1 + e2_0_2

        # single excitations
        e2_1_1 = 2 * e2_0 * ci1g
        lci1g = oe.contract("gij,ij->g", chol, ci1_green, backend="jax")
        e2_1_2 = -2 * (lci1g @ lg)
        gl = oe.contract("pj,gji->gpi", green, chol, backend="jax")
        ci1g1 = ci1 @ green_occ.T
        e2_1_3_1 = oe.contract("gpq,gqr,rp->", lg1, lg1, ci1g1, backend="jax")
        lci1g = oe.contract("gip,qi->gpq", ham_data["lci1"], green, backend="jax")
        e2_1_3_2 = -oe.contract("gpq,gqp->", lci1g, lg1, backend="jax")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3)

        # double excitations
        e2_2_1 = e2_0 * gci2g
        lci2g = oe.contract("gij,ij->g", chol, ci2_green, backend="jax")
        e2_2_2_1 = -lci2g @ lg
        lci2_green = oe.contract("gpi,ji->gpj", rot_chol, ci2_green, backend="jax")
        e2_2_2_2 = 0.5 * oe.contract("gpi,gpi->", gl, lci2_green, backend="jax")
        e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
        glgp = oe.contract("gpi,it->gpt", gl, greenp, backend="jax")
        l2ci2_1 = oe.contract("gpt,gqu,ptqu->g", glgp, glgp, ci2, backend="jax")
        l2ci2_2 = oe.contract("gpu,gqt,ptqu->g", glgp, glgp, ci2, backend="jax")
        e2_2_3 = 2 * l2ci2_1.sum() - l2ci2_2.sum()
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3
        e2 = e2_0 + e2_1 + e2_2

        # overlap
        overlap_1 = 2 * ci1g
        overlap_2 = gci2g
        overlap = 1.0 + overlap_1 + overlap_2
        return (e1 + e2) / overlap + e0
     

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class ccsd_pt(rhf):
    """A manual implementation of the CCSD_PT wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_green(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        nocc = self.nelec[0]
        gf = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        return gf

    @partial(jit, static_argnums=0)
    def _calc_energy_pt(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        t1, t2 = wave_data["t1"], wave_data["t2"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
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
        e1 = e1_1 + e1_2 + e2_1 + e2_2 # <psi|(t1+t2)(h1+h2)|phi>/<psi|phi>

        t = 2 * t1g + gt2g # <psi|(t1+t2)|phi>/<psi|phi>

        return t, e0, e1
    
    @partial(jit, static_argnums=(0)) 
    def calc_energy_pt(self, walkers:jax.Array, ham_data: dict, wave_data: dict):
        
        n_walkers = walkers.shape[0]
        batch_size = n_walkers // self.n_batch
        
        def scan_batch(carry, walker_batch):
            t, e0, e1 \
                = vmap(self._calc_energy_pt, in_axes=(0, None, None))(
                    walker_batch, ham_data, wave_data)
            return carry, (t, e0, e1)
        
        _, (t, e0, e1) \
            = lax.scan(
                scan_batch, None, 
                walkers.reshape(self.n_batch, batch_size, self.norb, self.nelec[0]))
        
        t = t.reshape(n_walkers)
        e0 = e0.reshape(n_walkers)
        e1 = e1.reshape(n_walkers)
        
        return t, e0, e1


    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        
        # nocc = wave_data['t1'].shape[0]
        norb = self.norb
        # wave_data["mo_coeff"] = np.eye(norb)[:,:nocc]
        # t1, t2 = wave_data['t1'],wave_data['t2']

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
class ccd_pt(rhf):

    @partial(jit, static_argnums=0)
    def _calc_green(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        nocc = self.nelec[0]
        gf = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        return gf

    @partial(jit, static_argnums=0)
    def _calc_energy_pt(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        nocc, norb = self.nelec[0], self.norb
        t2 = wave_data["t2"]
        chol = ham_data["chol"].reshape(-1, norb, norb)

        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:]
        greenp = jnp.vstack((green_occ, -jnp.eye(norb - nocc)))

        rot_chol = chol[:, :nocc, :]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        hg = oe.contract("pj,pj->", h1[:nocc, :], green, backend="jax")

        # 0 body energy
        h0 = ham_data["h0"]

        # 1 body energy
        # ref
        e1_0 = 2 * hg

        # double excitations
        t2g_c = oe.contract("iajb,ia->jb", t2, green_occ, backend="jax")
        t2g_e = oe.contract("iajb,ib->ja", t2, green_occ, backend="jax")
        t2_green_c = (greenp @ t2g_c.T) @ green
        t2_green_e = (greenp @ t2g_e.T) @ green
        t2_green = 2 * t2_green_c - t2_green_e
        t2g = 2 * t2g_c - t2g_e
        gt2g = oe.contract("ia,ia->", t2g, green_occ, backend="jax")
        e1_2_1 = 2 * hg * gt2g
        e1_2_2 = -2 * oe.contract("ij,ij->", h1, t2_green, backend="jax")
        e1_2 = e1_2_1 + e1_2_2

        # <HF|h2|walker>/<HF|walker>
        gl = oe.contract("ir,gpr->gip", green, chol, backend="jax")
        tr_gl = oe.contract("gii->g", gl[:,:nocc,:nocc], backend="jax")
        e2_0_1 = oe.contract('g,g->', tr_gl, tr_gl) * 2
        e2_0_2 = -oe.contract("gij,gji->", gl[:,:nocc,:nocc], gl[:,:nocc,:nocc], backend="jax")
        e2_0 = e2_0_1 + e2_0_2

        e2_2_1 = e2_0 * gt2g
        lt2g = oe.contract("gij,ij->g", chol, t2_green, backend="jax")
        e2_2_2_1 = -oe.contract('g,g->', lt2g, tr_gl, backend="jax")
        # e2_2_2_1 = -lt2g @ lg

        lt2_green = oe.contract("gir,pr->gip", rot_chol, t2_green, backend="jax")
        e2_2_2_2 = 0.5 * oe.contract("gip,gip->", gl, lt2_green, backend="jax")
        
        glgp = oe.contract("gip,pa->gia", gl, greenp, backend="jax")
        glgp_t_1 = oe.contract("gia,iajb->gjb", glgp, t2, backend="jax")
        glgp_t_2 = oe.contract("gib,iajb->gja", glgp, t2, backend="jax")
        l2t2_1 = oe.contract("gjb,gjb->", glgp_t_1, glgp, backend="jax")
        l2t2_2 = oe.contract("gja,gja->", glgp_t_2, glgp, backend="jax")

        e2_2_3 = 2 * l2t2_1 - l2t2_2

        e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)

        e2_2 = e2_2_1 + e2_2_2 + e2_2_3

        e0 = h0 + e1_0 + e2_0 # h0 + <psi|(h1+h2)|phi>/<psi|phi>
        e1 = e1_2 + e2_2 # <psi|t2(h1+h2)|phi>/<psi|phi>

        t = gt2g # <psi|(t1+t2)|phi>/<psi|phi>

        return t, e0, e1
    
    @partial(jit, static_argnums=(0)) 
    def calc_energy_pt(self, walkers:jax.Array, ham_data: dict, wave_data: dict):
        
        n_walkers = walkers.shape[0]
        batch_size = n_walkers // self.n_batch
        
        def scan_batch(carry, walker_batch):
            t, e0, e1 \
                = vmap(self._calc_energy_pt, in_axes=(0, None, None))(
                    walker_batch, ham_data, wave_data)
            return carry, (t, e0, e1)
        
        _, (t, e0, e1) \
            = lax.scan(scan_batch, None, 
                walkers.reshape(self.n_batch, batch_size, self.norb, self.nelec[0]))
        
        t = t.reshape(n_walkers)
        e0 = e0.reshape(n_walkers)
        e1 = e1.reshape(n_walkers)
        
        return t, e0, e1


    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
    
@dataclass
class ccsd_pt2(rhf):
    """Tensor contraction form of the CCSD_PT2 (exact T1) trial wave function."""

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
        # this sgn is a problem when
        # turn on mol point group symmetry
        # sgn = jnp.sign((mo_t).diagonal())
        # choose the mo_t s.t it has positive olp with the original mo
        # <psi'_i|psi_i>
        # mo_t = jnp.einsum("ij,j->ij", mo_t, sgn)
        return mo_t
    
    @partial(jit, static_argnums=0)
    def _calc_energy_pt(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ):
        nocc = self.nelec[0]
        mo_t, t2 = wave_data["mo_t"], wave_data["t2"]
        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        green = (walker @ (jnp.linalg.inv(mo_t.T @ walker)) @ mo_t.T).T
        greenp = (green - jnp.eye(self.norb))[:,nocc:]

        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        hg = oe.contract("pq,pq->", h1, green, backend="jax")
        e1_0 = 2 * hg

        # double excitations
        t2g_c = oe.contract("iajb,ia->jb", t2, green[:nocc,nocc:], backend="jax")
        t2g_e = oe.contract("iajb,ib->ja", t2, green[:nocc,nocc:], backend="jax")
        t2_green_c = (greenp @ t2g_c.T) @ green[:nocc,:]
        t2_green_e = (greenp @ t2g_e.T) @ green[:nocc,:]
        t2_green = 2 * t2_green_c - t2_green_e
        t2g = 2 * t2g_c - t2g_e
        gt2g = oe.contract("ia,ia->", t2g, green[:nocc,nocc:], backend="jax")
        e1_2_1 = 2 * hg * gt2g
        e1_2_2 = -2 * oe.contract("ij,ij->", h1, t2_green, backend="jax")
        e1_2 = e1_2_1 + e1_2_2 # <exp(T1)HF|T2 h1|walker>/<exp(T1)HF|walker>

        # two body energy
        lg = oe.contract("gpq,pq->g", chol, green, backend="jax")

        # double excitations
        lt2g = oe.contract("gij,ij->g", chol, t2_green, backend="jax")
        e2_2_2_1 = -lt2g @ lg

        def scanned_fun(carry, x):
            chol_i = x
            # e2_0
            lg_i = oe.contract("pr,qr->pq", chol_i, green, backend="jax")
            e2_0_1_i = (2*jnp.trace(lg_i))**2 / 2.0
            e2_0_2_i = - oe.contract('pq,qp->',lg_i,lg_i, backend="jax")
            carry[0] += e2_0_1_i + e2_0_2_i
            # e2_2
            gl_i = oe.contract("pr,rq->pq",green,chol_i,backend="jax")
            lt2_green_i = oe.contract("pr,qr->pq",chol_i,t2_green,backend="jax")
            carry[1] += 0.5 * oe.contract("pq,pq->",gl_i,lt2_green_i,backend="jax")
            glgp_i = oe.contract("iq,qa->ia", gl_i[:nocc,:],greenp,backend="jax")
            l2t2_1 = oe.contract("ia,jb,iajb->",glgp_i,glgp_i,t2,backend="jax")
            l2t2_2 = oe.contract("ib,ja,iajb->",glgp_i,glgp_i,t2,backend="jax")
            carry[2] += 2 * l2t2_1 - l2t2_2
            return carry, 0.0

        [e2_0, e2_2_2_2, e2_2_3], _ = lax.scan(scanned_fun, [0.0,0.0,0.0], chol)
        e2_2_1 = e2_0 * gt2g
        e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3

        # <HF|walker>
        o0 = jnp.linalg.det(walker[:nocc,:nocc]) ** 2
        # <exp(T1)HF|walker>/<HF|walker>
        t1 = jnp.linalg.det(wave_data["mo_t"].T.conj() @ walker)**2 / o0
        t2 = gt2g * t1 # <exp(T1)HF|T2|walker>/<HF|walker>
        e0 = (e1_0 + e2_0) * t1 # <exp(T1)HF|h1+h2|walker>/<HF|walker>
        e1 = (e1_2 + e2_2) * t1 # <exp(T1)HF|T2 (h1+h2)|walker>/<HF|walker>

        return t1, t2, e0, e1


    @partial(jit, static_argnums=0)
    def calc_energy_pt(self,walkers,ham_data,wave_data):
        t1, t2, e0, e1 = vmap(
            self._calc_energy_pt,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        return t1, t2, e0, e1
    
    def __hash__(self):
        return hash(tuple(self.__dict__.values()))

@dataclass
class ccsd_pt_ad(rhf):
    """differential form of the CCSD_PT wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _t1t2_walker_olp(self,walker,wave_data):
        ''' <psi_0(t1+t2)|phi> '''
        t1, t2 = wave_data['t1'], wave_data['t2']
        nocc = walker.shape[1]
        # GF = (walker.dot(jnp.linalg.inv(walker[: nocc, :]))).T
        GF = self._calc_green(walker,wave_data)
        o0 = self._calc_overlap_restricted(walker, wave_data)
        o1 = oe.contract("ia,ia", t1, GF[:, nocc:], backend="jax")
        o2 = 2 * oe.contract("iajb, ia, jb", t2, GF[:, nocc:], GF[:, nocc:], backend="jax"
        ) - oe.contract("iajb, ib, ja", t2, GF[:, nocc:], GF[:, nocc:], backend="jax")
        return (2*o1 + o2) * o0

    @partial(jit, static_argnums=0)
    def _t1t2_olp_exp1(self, x: float, h1_mod: jax.Array, walker: jax.Array,
                    wave_data: dict) -> complex:
        '''
        t_ia <psi_i^a|exp(x*h1_mod)|walker>
        '''
        t = x * h1_mod
        walker_1x = walker + t.dot(walker)
        olp = self._t1t2_walker_olp(walker_1x,wave_data)
        return olp

    @partial(jit, static_argnums=0)
    def _t1t2_olp_exp2(self, x: float, chol_i: jax.Array, walker: jax.Array,
                    wave_data: dict) -> complex:
        '''
        t_ia <psi_i^a|exp(x*h2_mod)|walker>
        '''
        walker_2x = (
                walker
                + x * chol_i.dot(walker)
                + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
            )
        olp = self._t1t2_walker_olp(walker_2x,wave_data)
        return olp

    @partial(jit, static_argnums=0)
    def _calc_energy_pt(self, walker, ham_data, wave_data):
        ''' <psi_0|(t1+t2)(h1+h2)|phi>/<psi_0|phi> '''

        eps=3e-4

        norb = self.norb
        h1_mod = ham_data['h1_mod']
        chol = ham_data["chol"].reshape(-1, norb, norb)

        # one body
        x = 0.0
        f1 = lambda a: self._t1t2_olp_exp1(a,h1_mod,walker,wave_data)
        olp_t, d_olp = jvp(f1, [x], [1.0])

        # two body
        # c_ij^ab <psi_ij^ab|phi_2x>/<psi_0|phi>
        def scanned_fun(carry, c):
            eps, walker, wave_data = carry
            return carry, self._t1t2_olp_exp2(eps,c,walker,wave_data)

        _, olp_p = lax.scan(scanned_fun, (eps, walker, wave_data), chol)
        _, olp_0 = lax.scan(scanned_fun, (0.0, walker, wave_data), chol)
        _, olp_m = lax.scan(scanned_fun, (-1.0 * eps, walker, wave_data), chol)
        d_2_olp = (olp_p - 2.0 * olp_0 + olp_m) / eps / eps
        
        o0 = self._calc_overlap_restricted(walker, wave_data)
        t = olp_t/o0
        e0 = self._calc_energy_restricted(walker,ham_data,wave_data)
        e1 = (d_olp + jnp.sum(d_2_olp) / 2.0 ) / o0

        return t, e0, e1

    @partial(jit, static_argnums=(0)) 
    def calc_energy_pt(self,walkers,ham_data,wave_data):
        t, e0, e1 = vmap(
            self._calc_energy_pt,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        return t, e0, e1
    
    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        """Builds half rotated integrals for efficient force bias and energy calculations."""

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

        chol = ham_data["chol"].reshape(-1, norb, norb)
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        v0 = 0.5 * oe.contract("gik,gjk->ij",
                                chol.reshape(-1, norb, norb),
                                chol.reshape(-1, norb, norb),
                                backend="jax")
        h1_mod = h1 - v0 
        ham_data['h1_mod'] = h1_mod
        return ham_data
    
    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class ccsd_pt2_ad(rhf):

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
        u_occ = jnp.vstack((u_ji,u_ai))
        mo_t, _ = jnp.linalg.qr(u_occ,mode='complete')
        return mo_t

    @partial(jit, static_argnums=0)
    def _tls_green(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        '''<exp(T1)HF|a_p^dagger a_q|walker>/<exp(T1)HF|walker>'''
        tls_gf = (walker.dot(
                jnp.linalg.inv(wave_data["mo_t"].T.conj() @ walker))
                ).T
        return tls_gf

    @partial(jit, static_argnums=0)
    def _tls_walker_olp(self, walker, wave_data):
        ''' 
        <exp(T1)HF|walker>
        '''
        o_t = jnp.linalg.det(wave_data["mo_t"].T.conj() @ walker) ** 2
        return o_t

    @partial(jit, static_argnums=0)
    def _tls_exp1(self, x, h1_mod, walker, wave_data) -> complex:
        '''
        <exp(T1)HF|exp(x*h1_mod)|walker>/<HF|walker>
        '''
        t = x * h1_mod
        walker_1x = walker + t.dot(walker)
        t1 = self._tls_walker_olp(walker_1x,wave_data)
        o0 = self._calc_overlap_restricted(walker, wave_data)
        return t1/o0

    @partial(jit, static_argnums=0)
    def _tls_exp2(self, x, chol_i, walker, wave_data) -> complex:
        '''
        <exp(T1)HF|exp(x*h2_mod)|walker>/<HF|walker>
        '''
        walker_2x = (
                walker
                + x * chol_i.dot(walker)
                + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
            )
        t2 = self._tls_walker_olp(walker_2x,wave_data)
        o0 = self._calc_overlap_restricted(walker, wave_data)
        return t2/o0

    @partial(jit, static_argnums=0)
    def _t2_tls_walker_olp(self, walker, wave_data):
        ''' 
        <exp(T1)HF|T2|walker>
        = t_iajb <exp(T1)HF|ijab|phi>/<exp(T1)HF|phi> * <exp(T1)HF|phi>
        '''
        rot_t2 = wave_data['rot_t2']
        nocc = walker.shape[1]
        GF = self._tls_green(walker, wave_data)
        o_t = self._tls_walker_olp(walker, wave_data)
        t2 = 2 * oe.contract(
            "iajb, ia, jb", rot_t2, GF[:, nocc:], GF[:, nocc:], backend="jax"
        ) - oe.contract("iajb, ib, ja", rot_t2, GF[:, nocc:], GF[:, nocc:], backend="jax")
        return t2 * o_t

    @partial(jit, static_argnums=0)
    def _t2_tls_exp1(self, x, h1_mod, walker, wave_data) -> complex:
        '''
        <exp(T1)HF|T2 exp(x*h1_mod)|walker>/<HF|walker>
        '''
        t = x * h1_mod
        walker_1x = walker + t.dot(walker)
        t2 = self._t2_tls_walker_olp(walker_1x,wave_data)
        o0 = self._calc_overlap_restricted(walker, wave_data)
        return t2/o0

    @partial(jit, static_argnums=0)
    def _t2_tls_exp2(self, x, chol_i, walker, wave_data) -> complex:
        '''
        t_iajb <psi|ijab exp(x*h2_mod)|walker>/<psi_0|walker>
        '''
        walker_2x = (
                walker
                + x * chol_i.dot(walker)
                + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
            )
        t2 = self._t2_tls_walker_olp(walker_2x,wave_data)
        o0 = self._calc_overlap_restricted(walker, wave_data)
        return t2/o0

    @partial(jit, static_argnums=0)
    def _calc_energy_pt_restricted(self, walker, ham_data, wave_data):
        ''' 
        t1 = <exp(T1)HF|walker>/<HF|walker>
        t2 = <exp(T1)HF|T2|walker>/<HF|walker>
        e0 = <exp(T1)HF|H|walker>/<HF|walker>
        e1 = <exp(T1)HF|T2(h1+h2)|walker>/<HF|walker>
        '''

        eps = 1e-4
        norb = self.norb
        h1_mod = ham_data['h1_mod']
        chol = ham_data["chol"].reshape(-1, norb, norb)

        # <exp(T1)HF|h1+h2|walker>/<HF|walker>
        # one body
        # <exp(T1)HF|walker_1x>/<HF|walker>
        x = 0.0
        f1 = lambda a: self._tls_exp1(a,h1_mod,walker,wave_data)
        t1, d_exp1 = jvp(f1, [x], [1.0])

        # two body
        # <exp(T1)HF|walker_2x>/<HF|walker>
        def scanned_fun(carry, c):
            eps, walker, wave_data = carry
            return carry, self._tls_exp2(eps,c,walker,wave_data)

        _, exp2_p = lax.scan(scanned_fun, (eps, walker, wave_data), chol)
        _, exp2_0 = lax.scan(scanned_fun, (0.0, walker, wave_data), chol)
        _, exp2_m = lax.scan(scanned_fun, (-1.0 * eps, walker, wave_data), chol)
        d2_exp2 = (exp2_p - 2.0 * exp2_0 + exp2_m) / eps / eps

        e0 = (d_exp1 + jnp.sum(d2_exp2) / 2.0 )

        d_exp1, d2_exp2 = None, None
        exp2_p, exp2_0, exp2_m = None, None, None

        # <exp(T1)HF|T2(h1+h2)|walker>/<HF|walker>
        # one body
        # <exp(T1)HF|T2|walker_1x>/<HF|walker>
        x = 0.0
        f1 = lambda a: self._t2_tls_exp1(a,h1_mod,walker,wave_data)
        t2, d_exp1 = jvp(f1, [x], [1.0])

        # two body
        # <exp(T1)HF|T2|walker_2x>/<HF|walker>
        def scanned_fun(carry, c):
            eps, walker, wave_data = carry
            return carry, self._t2_tls_exp2(eps,c,walker,wave_data)

        _, exp2_p = lax.scan(scanned_fun, (eps, walker, wave_data), chol)
        _, exp2_0 = lax.scan(scanned_fun, (0.0, walker, wave_data), chol)
        _, exp2_m = lax.scan(scanned_fun, (-1.0 * eps, walker, wave_data), chol)
        d2_exp2 = (exp2_p - 2.0 * exp2_0 + exp2_m) / eps / eps
    
        e1 = (d_exp1 + jnp.sum(d2_exp2) / 2.0 )

        return t1, t2, e0, e1
    
    @partial(jit, static_argnums=0)
    def calc_energy_pt(self,walkers,ham_data,wave_data):
        t1, t2, e0, e1 = vmap(
            self._calc_energy_pt_restricted,in_axes=(0, None, None))(
            walkers, ham_data, wave_data)
        return t1, t2, e0, e1

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
    

class stoccsd(rhf):
    '''
    Trial = Stochastically sampled CCSD wavefunction
    Guide = RHF
    '''

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1
    nslater: int = 1000

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
        return (eloc, oloc) 
    
    @partial(jit, static_argnums=0)
    def calc_energy_overlap_stocc(
            self, walkers: jax.Array, ham_data: jax.Array, wave_data: dict
            ):

        (energies, overlaps) =  vmap(
            lambda walker: self.get_eloc_oloc_stocc(walker, ham_data, wave_data
            ))(walkers)
        
        return (energies, overlaps)


    def __hash__(self):
        return hash(tuple(self.__dict__.values()))