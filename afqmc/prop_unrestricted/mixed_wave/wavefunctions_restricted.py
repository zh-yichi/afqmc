# from abc import ABC
from dataclasses import dataclass
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random, jit, lax, vmap
import opt_einsum as oe

from afqmc.prop_unrestricted.wavefunctions_restricted import rhf


@dataclass
class stoccsd(rhf):
    '''
    Trial = Stochastically sampled CCSD wavefunction
    Guide = RHF
    '''
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
                wave_data['tau'].shape[0],
            ),
        )
        # xtaus shape (nwalker, nslater, nocc, nvir)
        xtaus = oe.contract("wsg,gia->wsia", fieldx, wave_data['tau'], backend='jax')

        return xtaus, prop_data
    
    @partial(jit, static_argnums=(0))
    def get_same_xtaus(self, prop_data, wave_data):
        prop_data["key"], subkey = random.split(prop_data["key"])
        
        fieldx = random.normal(
            subkey,
            shape=(
                self.nslater,
                wave_data['tau'].shape[0],
            ),
        )
        # xtaus shape (nslater, nocc, nvir)
        xtaus = oe.contract("sg,gia->sia", fieldx, wave_data['tau'], backend='jax')

        return xtaus, prop_data
    

    @partial(jit, static_argnums=0)
    def _green(self, trial_slater: jax.Array, walker: jax.Array) -> jax.Array:
        
        green = (walker @ (
                jnp.linalg.inv(trial_slater.T.conj() @ walker)
                    ) @ trial_slater.T.conj()).T
        
        return green
    

    @partial(jit, static_argnums=0)
    def _slater_olp(self, trial_slater: jax.Array, walker: jax.Array):
        ''' 
        <psi|walker>
        '''
        olp = jnp.linalg.det(trial_slater.T.conj() @ walker) ** 2
        return olp
    

    @partial(jit, static_argnums=0)
    def _calc_energy_slater(self, slater: jax.Array, walker: jax.Array, ham_data: dict) -> jax.Array:
        norb = self.norb

        h0, chol = ham_data["h0"], ham_data["chol"]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        chol = chol.reshape(-1, norb, norb)

        green = self._green(slater, walker)
        gh = oe.contract("pq,pq->", green, h1, backend="jax")
        e1 = 2 * gh

        # lg = oe.contract("gpr,qr->gpq", chol, green, backend="jax")
        # e2_1 = 2 * jnp.sum(oe.contract('gpp->g', lg, backend="jax")**2)
        # e2_2 = -oe.contract('gpq,gqp->',lg,lg, backend="jax")
        # e2 = e2_1 + e2_2

        def scan_chol(carry, x):
            chol_i = x
            gl_i = oe.contract("pr,qr->pq", green, chol_i, backend="jax")
            e2_c_i = 2 * oe.contract('pp->', gl_i, backend="jax")**2
            e2_e_i = -oe.contract('pq,qp->', gl_i, gl_i, backend="jax")
            carry += e2_c_i + e2_e_i
            return carry, 0
        
        e2, _ = lax.scan(scan_chol, 0.0, chol)
        overlap = self._slater_olp(slater, walker)
        energy = h0 + e1 + e2

        return overlap, energy
    

    @partial(jit, static_argnums=0)
    def _calc_energy_exp_xtau(self, 
                              xtau: jax.Array,
                              walker: jax.Array, 
                              ham_data: dict, 
                              wave_data: dict, 
                              ) -> jax.Array :
        # transform exp(y*tau)|psi> = |psi(y)>
        
        slater = self._thouless(wave_data['mo_t'], xtau)
        overlap, energy = self._calc_energy_slater(slater, walker, ham_data)

        return overlap, energy 

    
    @partial(jit, static_argnums=0)
    def _calc_energy_exp_xtaus(self, walker, xtaus, ham_data, wave_data):
        # scan over taus (nslaters) for one walker

        nslater = self.nslater
        nocc, norb = self.nelec[0], self.norb
        nvir = norb - nocc
        assert xtaus.shape == (nslater, nocc, nvir)

        def _scan_xtaus(carry, xtau: jax.Array):
            overlap, energy = self._calc_energy_exp_xtau(xtau, walker, ham_data, wave_data)
            return carry, (overlap, energy)

        init_carry = 0.0
        _, (overlaps, energies) = lax.scan(_scan_xtaus, init_carry, xtaus)

        # intermediately normalize stocc
        overlap_cc = jnp.sum(overlaps) # / nslater
        energy_cc = jnp.sum(overlaps * energies) / overlap_cc # / nslater

        return overlap_cc, energy_cc
    

    @partial(jit, static_argnums=0)
    def _calc_energy_get_xtaus(self, walker, ham_data, wave_data, prop_data):
        # first get xtaus
        # scan over xtaus (nslaters) for one walker

        nslater = self.nslater
        nocc, norb = self.nelec[0], self.norb
        nvir = norb - nocc
        xtaus, prop_data = self.get_same_xtaus(prop_data, wave_data)

        assert xtaus.shape == (nslater, nocc, nvir)

        def _scan_xtaus(carry, xtau: jax.Array):
            overlap, energy = self._calc_energy_exp_xtau(xtau, walker, ham_data, wave_data)
            return carry, (overlap, energy)

        init_carry = 0.0
        _, (overlaps, energies) = lax.scan(_scan_xtaus, init_carry, xtaus)

        # intermediately normalize stocc
        overlap_cc = jnp.sum(overlaps) # / nslater
        energy_cc = jnp.sum(overlaps * energies) / overlap_cc # / nslater

        return overlap_cc, energy_cc
    

    @partial(jit, static_argnums=(0))
    def calc_energy_stoccsd(self, walkers, xtaus, ham_data, wave_data):
        # scan over walkers
        # xtaus shape (nwalker, nslater, nocc, nvir)
        nocc = self.nelec[0]
        norb = self.norb
        nvir = norb - nocc
        nwalker = walkers.shape[0]
        batch_size = nwalker // self.n_batch
        nslater = self.nslater

        assert xtaus.shape == (nwalker, nslater, nocc, nvir)

        def scan_batch(carry, xs):
            walker_batch, xtaus_batch = xs
            overlap, energy = vmap(self._calc_energy_exp_xtaus, in_axes=(0, 0, None, None))(
                walker_batch, xtaus_batch, ham_data, wave_data
            )
            return carry, (overlap, energy)

        _, (overlaps, energies) = lax.scan(
            scan_batch, None,
            (walkers.reshape(self.n_batch, batch_size, norb, nocc),
             xtaus.reshape(self.n_batch, batch_size, nslater, nocc, nvir))
            )
        
        overlaps = overlaps.reshape(nwalker)
        energies = energies.reshape(nwalker)
        
        return overlaps, energies
    

    @partial(jit, static_argnums=(0))
    def calc_energy_same_stoccsd(self, walkers, xtaus, ham_data, wave_data):
        # scan over walkers
        # all walkers feel the same sto-sample of the CCSD trial during a step
        # xtaus shape (nslater, nocc, nvir)

        nocc = self.nelec[0]
        norb = self.norb
        nvir = norb - nocc
        nwalker = walkers.shape[0]
        batch_size = nwalker // self.n_batch
        nslater = self.nslater

        assert xtaus.shape == (nslater, nocc, nvir)

        def scan_batch(carry, xs):
            walker_batch = xs
            overlap, energy = vmap(self._calc_energy_exp_xtaus, in_axes=(0, None, None, None))(
                walker_batch, xtaus, ham_data, wave_data
            )
            return carry, (overlap, energy)

        _, (overlaps, energies) = lax.scan(
            scan_batch, None, walkers.reshape(self.n_batch, batch_size, norb, nocc))
        
        overlaps = overlaps.reshape(nwalker)
        energies = energies.reshape(nwalker)
        
        return overlaps, energies
    

    @partial(jit, static_argnums=(0))
    def calc_energy_get_stoccsd(self, walkers, ham_data, wave_data, prop_data):
        # scan over walkers
        # stocc sample generated inside
        # all walkers feel the same sto-sample of the CCSD trial during a step
        # xtaus shape (nslater, nocc, nvir)

        nocc = self.nelec[0]
        norb = self.norb
        # nvir = norb - nocc
        nwalker = walkers.shape[0]
        batch_size = nwalker // self.n_batch
        # nslater = self.nslater

        # assert xtaus.shape == (nslater, nocc, nvir)

        def scan_batch(carry, xs):
            walker_batch = xs
            overlap, energy = vmap(self._calc_energy_get_xtaus, in_axes=(0, None, None, None))(
                walker_batch, ham_data, wave_data, prop_data
            )
            return carry, (overlap, energy)

        _, (overlaps, energies) = lax.scan(
            scan_batch, None, walkers.reshape(self.n_batch, batch_size, norb, nocc))
        
        overlaps = overlaps.reshape(nwalker)
        energies = energies.reshape(nwalker)
        
        return overlaps, energies


    def __hash__(self):
        return hash(tuple(self.__dict__.values()))



@dataclass
class stoccsd2(rhf):
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
                wave_data['tau'].shape[0],
            ),
        )
        # xtaus shape (nwalker, nslater, nocc, nvir)
        xtaus = oe.contract("wsg,gia->wsia", fieldx, wave_data['tau'], backend='jax')

        return xtaus, prop_data
    
    @partial(jit, static_argnums=0)
    def _green(self, walker: jax.Array, slater: jax.Array) -> jax.Array:
        '''
        full green's function 
        <psi|a_p^dagger a_q|walker>/<exp(T1)HF|walker>
        '''
        green = (walker @ (
            jnp.linalg.inv(slater.T.conj() @ walker)
            ) @ slater.T.conj()).T
        return green

    @partial(jit, static_argnums=0)
    def _slater_olp(self, walker: jax.Array, slater: jax.Array):
        ''' 
        <psi|walker>
        '''
        olp = jnp.linalg.det(slater.T.conj() @ walker) ** 2
        return olp

    @partial(jit, static_argnums=0)
    def _ci_walker_olp(self, walker: jax.Array, slater: jax.Array, ci1: jax.Array, ci2: jax.Array) -> complex:
        ''' 
        <(1+ci1+ci2)psi|walker>
        = c_ia* <psi|ia|walker> + 1/2 c_iajb* <psi|ijab|walker>
        '''
        ci1 = ci1.conj()
        ci2 = ci2.conj()
        nocc = walker.shape[1]
        green_ov = self._green(walker, slater)[:nocc, nocc:]
        o0 = self._slater_olp(walker, slater)
        o1 = 2 * oe.contract("ia,ia-> ", ci1, green_ov, backend="jax")
        o2 = 2 * oe.contract("iajb,ia,jb->", ci2, green_ov, green_ov, backend="jax") \
            - oe.contract("iajb,ib,ja->", ci2, green_ov, green_ov, backend="jax")
        return (1.0 + o1 + o2) * o0

    # @partial(jit, static_argnums=0)
    # def _calc_energy_cisd(
    #     self, 
    #     walker: jax.Array, 
    #     ham_data: dict, 
    #     wave_data: dict,
    #     ci1:  jax.Array,
    #     ci2:  jax.Array,
    #     ):

    #     '''
    #     A local energy evaluator for <psi(ci1+ci2)HF|H|walker> / <psi(ci1+ci2)|walker>
    #     all operators and the walkers and psi are in the same basis (normally MO)
    #     |psi> is not necesarily diagonal
        
    #     all green's function and the chol and ci coeff are as their original definition
    #     no half rotation performed
    #     '''

    #     ci1 = ci1.conj() # applied to the bra
    #     ci2 = ci2.conj() # applied to the bra

    #     nocc, norb = self.nelec[0], self.norb
    #     h0  = ham_data['h0']
    #     trial_slater = wave_data["mo_t"]
    #     h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
    #     chol = ham_data["chol"].reshape(-1, norb, norb)
    #     green = self._green(walker, trial_slater) # full green
    #     green_ov = green[:nocc, nocc:]
    #     greenp = (green - jnp.eye(norb))[:, nocc:]
        
    #     ##################### ref terms #########################
    #     # one-body 
    #     hg = oe.contract("pq,pq->", h1, green, backend="jax")
    #     e1_0 = 2 * hg

    #     # two-body 
    #     gl = oe.contract("pr,gqr->gpq", green, chol, backend="jax")
    #     trgl = oe.contract('gpp->g', gl, backend="jax")
    #     e2_0_1 = 2 * jnp.sum(trgl**2)
    #     e2_0_2 = -oe.contract('gpq,gqp->',gl,gl, backend="jax")
    #     e2_0 = e2_0_1 + e2_0_2
    #     ##########################################################

    #     ######################### ci terms #######################
    #     # one-body single excitations 
    #     ci1g = oe.contract("ia,ia->", ci1, green_ov, backend="jax")
    #     e1_1_1 = 4 * ci1g * hg # c_ia G_ia G_pq h_pq
    #     gpci1 = oe.contract("pa,ia->pi", greenp, ci1, backend="jax")
    #     ci1_green = oe.contract("pi,iq->pq", gpci1, green[:nocc,:], backend="jax")
    #     e1_1_2 = -2 * oe.contract("pq,pq->", h1, ci1_green, backend="jax") # c_ia Gp_pa G_iq h_pq
    #     e1_1 = e1_1_1 + e1_1_2 # <psi|ci1 h1|walker> / <psi|walker>

    #     # one-body double excitations
    #     t2g_c = oe.contract("iajb,ia->jb", ci2, green_ov, backend="jax")
    #     t2g_e = oe.contract("iajb,ib->ja", ci2, green_ov, backend="jax")
    #     t2_green_c = (greenp @ t2g_c.T) @ green[:nocc,:]
    #     t2_green_e = (greenp @ t2g_e.T) @ green[:nocc,:]
    #     t2_green = 2 * t2_green_c - t2_green_e
    #     t2g = 2 * t2g_c - t2g_e
    #     gt2g = oe.contract("ia,ia->", t2g, green_ov, backend="jax")
    #     e1_2_1 = 2 * hg * gt2g
    #     e1_2_2 = -2 * oe.contract("ij,ij->", h1, t2_green, backend="jax")
    #     e1_2 = e1_2_1 + e1_2_2 # <exp(T1)HF|T2 h1|walker> / <exp(T1)HF|walker>

    #     # two-body single excitations
    #     e2_1_1 = 2 * e2_0 * ci1g # c_ia G_ia G_pr G_ps L_pr L_ps
    #     lci1g = oe.contract("gpq,pq->g", chol, ci1_green, backend="jax")
    #     e2_1_2 = -2 * oe.contract("g,g->",lci1g, trgl, backend="jax") # c_ia Gp_pa G_ir G_qs L_pr L_qs

    #     lci1 = oe.contract("gpa,ia->gpi", chol[:, :, nocc:], ci1, backend="jax")
    #     lg1 = oe.contract("gpr,qr->gpq", chol, green[:nocc,:], backend="jax")
    #     lci1g = oe.contract("gri,pr->gip", lci1, green, backend="jax")
    #     glgpci1 = jnp.einsum("gpr,ri->gpi", gl, gpci1, optimize="optimal")
    #     e2_1_3 = jnp.einsum("gpi,gpi->", glgpci1, lg1, optimize="optimal")
    #     e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3) # <exp(T1)HF|ci1 h2|walker> / <exp(T1)HF|walker>

    #     # two-body double excitations
    #     e2_2_1 = e2_0 * gt2g
    #     lt2g = oe.contract("gij,ij->g", chol, t2_green, backend="jax")
    #     e2_2_2_1 = -oe.contract("g,g->", lt2g, trgl, backend="jax")

    #     def scan_aux(carry, x):
    #         chol_i, gl_i = x
    #         lt2_green_i = oe.contract("pr,qr->pq",chol_i,t2_green,backend="jax")
    #         carry[0] += 0.5 * oe.contract("pq,pq->",gl_i,lt2_green_i,backend="jax")
    #         glgp_i = oe.contract("iq,qa->ia", gl_i[:nocc,:],greenp,backend="jax")
    #         l2t2_1 = oe.contract("ia,jb,iajb->",glgp_i,glgp_i,ci2,backend="jax")
    #         l2t2_2 = oe.contract("ib,ja,iajb->",glgp_i,glgp_i,ci2,backend="jax")
    #         carry[1] += 2 * l2t2_1 - l2t2_2
    #         return carry, 0.0

    #     [e2_2_2_2, e2_2_3], _ = lax.scan(scan_aux, [0.0, 0.0], (chol, gl))

    #     e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
    #     e2_2 = e2_2_1 + e2_2_2 + e2_2_3

    #     o0 = self._slater_olp(walker, trial_slater)
    #     overlap = (1.0 + 2*ci1g +  gt2g) * o0 # <(1+c1+c2)psi|walker>
    #     # <psi|h0+h1+h2|walker> + <ci1 psi|h0+h1+h2|walker> + <ci2 pai|h0+h1+h2|walker>
    #     energy = h0 + (e1_0 + e2_0 + e1_1 + e2_1 + e1_2 + e2_2) / (1.0 + 2*ci1g +  gt2g)

    #     return overlap, energy
    
    @partial(jit, static_argnums=0)
    def _ci_walker_olp_disconnected(self, walker: jax.Array, slater: jax.Array, ci1: jax.Array) -> complex:
        ''' 
        disconnected ci2
        <(1+ci1+ci2)psi|walker>
        = c_ia* <psi|ia|walker> + 1/2 c_iajb* <psi|ijab|walker>
        '''
        ci1 = ci1.conj()

        nocc = walker.shape[1]
        green_ov = self._green(walker, slater)[:nocc, nocc:]
        cig = oe.contract("ia,ja->ij", ci1, green_ov, backend="jax")
        o0 = self._slater_olp(walker, slater)
        o1 = oe.contract("ii->", cig, backend="jax") # c_ia G_ia
        o2_1 = o1**2
        o2_2 = -oe.contract("ij,ji->", cig, cig, backend="jax") # c_ia G_ja c_jb G_ib
        o2 = 2*o2_1 + o2_2

        return (1.0 + 2*o1 + o2) * o0

    @partial(jit, static_argnums=0)
    def _calc_energy_cisd_disconnected(
        self,
        walker: jax.Array, 
        ham_data: dict, 
        wave_data: dict,
        ci1:  jax.Array,
        ):

        '''
        Disconnected Doubles!!! c_iajb = c_ia c_jb
        A local energy evaluator for <psi(ci1+ci2)HF|H|walker> / <psi(ci1+ci2)|walker>
        all operators and the walkers and psi are in the same basis (normally MO)
        |psi> is not necesarily diagonal
        
        all green's function and the chol and ci coeff are as their original definition
        no half rotation performed
        '''

        nocc, norb = self.nelec[0], self.norb
        h0  = ham_data['h0']
        # trial_slater = wave_data["mo_t"]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        chol = ham_data["chol"].reshape(-1, norb, norb)
        green = self._green(walker, wave_data["mo_t"]) # full green
        green_ov = green[:nocc, nocc:]
        greenp = (green - jnp.eye(norb))[:, nocc:]
        
        ci1 = ci1.conj() # applied to the bra
        
        ##################### ref terms #########################
        # one-body 
        gh = oe.contract("pr,qr->pq", h1, green, backend="jax")
        tr_gh = oe.contract("pp->", gh, backend="jax")
        e1_0 = 2 * tr_gh

        # two-body 
        # gl = oe.contract("pr,gqr->gpq", green, chol, backend="jax")
        # trgl = oe.contract('gpp->g', gl, backend="jax")
        # e2_0_1 = 2 * jnp.sum(trgl**2)
        # e2_0_2 = -oe.contract('gpq,gqp->',gl,gl, backend="jax")
        # e2_0 = e2_0_1 + e2_0_2
        ##########################################################

        ######################### ci terms #######################
        # universal terms #
        cig = oe.contract("ia,ja->ij", ci1, green_ov, backend="jax")
        cigp = oe.contract("ia,pa->ip", ci1, greenp, backend="jax")
        
        o0 = self._slater_olp(walker, wave_data["mo_t"])
        o1 = oe.contract("ii->", cig, backend="jax") # c_ia G_ia
        o2_1 = o1**2
        o2_2 = -oe.contract("ij,ji->", cig, cig, backend="jax") # c_ia G_ja c_jb G_ib
        o2 = 2*o2_1 + o2_2

        olp = (1.0 + 2*o1 + o2) * o0
        ###################

        # one-body single excitations 
        e1_1_1 = 4 * o1 * tr_gh # c_ia G_ia G_pq h_pq
        cigpg = oe.contract("ip,iq->pq", cigp, green[:nocc,:], backend="jax") # c_ia Gp_pa G_ir
        e1_1_2 = -2 * oe.contract("pq,pq->", cigpg, h1, backend="jax") # c_ia Gp_pa G_iq h_pq
        e1_1 = e1_1_1 + e1_1_2 # <psi|ci1 h1|walker> / <psi|walker>

        # one-body double excitations

        t2_green_c = o1 * oe.contract('jp,jq->pq', cigp, green[:nocc,:], backend='jax')
        t2_green_e = oe.contract('ji,ip,jq->pq', cig, cigp, green[:nocc,:], backend='jax')
        t2_green = 2 * t2_green_c - t2_green_e
        e1_2_1 = 2 * o2 * tr_gh
        e1_2_2 = -2 * oe.contract("ij,ij->", h1, t2_green, backend="jax")
        e1_2 = e1_2_1 + e1_2_2 # <exp(T1)HF|T2 h1|walker> / <exp(T1)HF|walker>

        # two-body single excitations
        # e2_1_1 = 2 * o1 * e2_0 # c_ia G_ia G_pr G_ps L_pr L_ps

        # lci1g = oe.contract("gpq,pq->g", chol, cigp_g, backend="jax") # c_ia Gp_pa G_ir L_pr
        # e2_1_2 = -2 * oe.contract("g,g->", lci1g, trgl, backend="jax") # c_ia Gp_pa G_ir G_qs L_pr L_qs

        # lci1 = oe.contract("gpa,ia->gpi", chol[:, :, nocc:], ci1, backend="jax")
        # lg1 = oe.contract("gpr,qr->gpq", chol, green[:nocc,:], backend="jax")
        # # lci1g = oe.contract("gri,pr->gip", lci1, green, backend="jax")
        # glgpci1 = jnp.einsum("gpr,ir->gpi", gl, cigp, optimize="optimal")
        # e2_1_3 = jnp.einsum("gpi,gpi->", glgpci1, lg1, optimize="optimal")
        # e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3) # <exp(T1)HF|ci1 h2|walker> / <exp(T1)HF|walker>

        # two-body double excitations
        # e2_2_1 = o2 * e2_0

        def scan_chol(carry, x):
            chol_i = x

            gl_i = oe.contract("pr,qr->pq", green, chol_i, backend="jax")
            trgl_i = oe.contract("pp->", gl_i, backend="jax")
            e2_0_c_i = 2 * trgl_i**2
            e2_0_e_i = -oe.contract('pq,qp->', gl_i, gl_i, backend="jax")
            carry[0] += e2_0_c_i + e2_0_e_i

            c1gpgl_i = oe.contract("pr,qr->pq", cigpg, chol_i, backend="jax") # c_ia Gp_pa G_iq
            trc1gpgl_i = oe.contract("pp->", c1gpgl_i, backend="jax")
            e2_1_2_c_i = -2 * trc1gpgl_i * trgl_i
            e2_1_2_e_i = oe.contract("pq,qp->", c1gpgl_i, gl_i, backend="jax")
            carry[1] += 2 * (e2_1_2_c_i + e2_1_2_e_i)

            lt2g_i = oe.contract("pr,qr->pq", chol_i, t2_green, backend="jax")
            trlt2g_i = oe.contract("pp->", lt2g_i, backend="jax")
            e2_2_2_c_i = -trlt2g_i * trgl_i
            e2_2_2_e_i = 0.5 * oe.contract("pq,pq->", lt2g_i, gl_i, backend="jax")
            carry[2] += 4*(e2_2_2_c_i + e2_2_2_e_i)

            glgp_i = oe.contract("iq,qa->ia", gl_i[:nocc,:], greenp, backend="jax")
            glgpc_i = oe.contract("ia,ja->ij", glgp_i, ci1, backend="jax")
            l2c2_1_i = oe.contract("ii->", glgpc_i, backend="jax")**2
            l2c2_2_i = oe.contract("ij,ji->", glgpc_i, glgpc_i, backend="jax")
            e2_2_3_i = 2 * l2c2_1_i - l2c2_2_i
            carry[3] += e2_2_3_i
            return carry, 0.0

        [e2_0, e2_1_2, e2_2_2, e2_2_3], _ = lax.scan(scan_chol, [0.0, 0.0, 0.0, 0.0], (chol))
        
        e2_1_1 = 2 * o1 * e2_0
        e2_1 = e2_1_1 + e2_1_2
        
        # lt2g = oe.contract("gij,ij->g", chol, t2_green, backend="jax")
        # e2_2_2_1 = -oe.contract("g,g->", lt2g, trgl, backend="jax")

        # lt2_green = oe.contract("gpr,qr->gpq", chol, t2_green, backend="jax")
        # e2_2_2_2 = 0.5 * oe.contract("gpq,gpq->", gl, lt2_green, backend="jax")

        # glgp = oe.contract("giq,qa->gia", gl[:,:nocc,:], greenp, backend="jax")
        # glgp_ci = oe.contract("gia,ja->gij", glgp, ci1, backend="jax")
        # # tr_glgp_ci = oe.contract("gii->g", glgp_ci, backend="jax")
        # l2t2_1 = jnp.sum(oe.contract("gii->g", glgp_ci, backend="jax")**2)
        # l2t2_2 = oe.contract("gij,gji->", glgp_ci, glgp_ci, backend="jax")
        # e2_2_3 = 2 * l2t2_1 - l2t2_2

        e2_2_1 = o2 * e2_0
        # e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3

        energy = h0 + (e1_0 + e2_0 + e1_1 + e2_1 + e1_2 + e2_2) / (1.0 + 2*o1 + o2)

        return olp, energy
    
    @partial(jit, static_argnums=0)
    def _calc_energy_ci_xtau(self, walker: jax.Array, ham_data: dict, wave_data: dict, xtau: jax.Array) -> jax.Array:
        
        overlap, energy = self._calc_energy_cisd_disconnected(walker, ham_data, wave_data, xtau)
        # overlap, energy = self._calc_energy_cisd(walker, ham_data, wave_data, xtau)

        return overlap, energy

    @partial(jit, static_argnums=0)
    def _calc_energy_slater(self, slater: jax.Array, walker: jax.Array, ham_data: dict) -> jax.Array:
        norb = self.norb

        h0, chol = ham_data["h0"], ham_data["chol"]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        chol = chol.reshape(-1, norb, norb)

        green = self._green(walker, slater)
        gh = oe.contract("pq,pq->", green, h1, backend="jax")
        e1 = 2 * gh

        # lg = oe.contract("gpr,qr->gpq", chol, green, backend="jax")
        # e2_1 = 2 * jnp.sum(oe.contract('gpp->g', lg, backend="jax")**2)
        # e2_2 = -oe.contract('gpq,gqp->',lg,lg, backend="jax")
        # e2 = e2_1 + e2_2

        def scan_chol(carry, x):
            chol_i = x
            gl_i = oe.contract("pr,qr->pq", green, chol_i, backend="jax")
            e2_c_i = 2 * oe.contract('pp->', gl_i, backend="jax")**2
            e2_e_i = -oe.contract('pq,qp->', gl_i, gl_i, backend="jax")
            carry += e2_c_i + e2_e_i
            return carry, 0
        
        e2, _ = lax.scan(scan_chol, 0.0, chol)
        overlap = self._slater_olp(walker, slater)
        energy = h0 + e1 + e2

        return overlap, energy

    @partial(jit, static_argnums=0)
    def _calc_energy_exp_xtau(self, walker: jax.Array, ham_data: dict, wave_data: dict, xtau: jax.Array) -> jax.Array:
        
        slater = self._thouless(wave_data['mo_t'], xtau)
        # overlap = self._slater_olp(walker, slater)
        overlap, energy = self._calc_energy_slater(slater, walker, ham_data)

        return overlap, energy
    
    @partial(jit, static_argnums=0)
    def _calc_overlap_exp_xtau(self, walker: jax.Array, wave_data: dict, xtau: jax.Array) -> jax.Array:
        
        slater = self._thouless(wave_data['mo_t'], xtau)
        overlap = self._slater_olp(walker, slater)

        return overlap
    
    @partial(jit, static_argnums=0)
    def _calc_correction_xtau(self, walker, xtau, ham_data, wave_data):
        # num = <exp(xtau)|H|walker> - <1+xtau+(xtau)^2|H|walker>
        # den = <exp(xtau)|walker> - <1+xtau+(xtau)^2|walker>

        o_exp, e_exp = self._calc_energy_exp_xtau(walker, ham_data, wave_data, xtau)
        o_ci, e_ci =  self._calc_energy_ci_xtau(walker, ham_data, wave_data, xtau)

        numerator = o_exp*e_exp - o_ci*e_ci 
        denominator = o_exp - o_ci

        return numerator, denominator

    @partial(jit, static_argnums=0)
    def _calc_correction_xtaus(self, walker, xtaus, ham_data, wave_data):
        nslater = self.nslater
        nocc, norb = self.nelec[0], self.norb
        nvir = norb - nocc
        assert xtaus.shape == (nslater, nocc, nvir)

        def _scan_xtaus(carry, xtau: jax.Array):
            num, den = self._calc_correction_xtau(walker, xtau, ham_data, wave_data)
            return carry, (num, den)

        init_carry = 0.0
        _, (num, den) = lax.scan(_scan_xtaus, init_carry, xtaus)

        # intermediately normalize stocc
        numerator = jnp.sum(num) / nslater
        denominator = jnp.sum(den) / nslater

        return numerator, denominator
    
    @partial(jit, static_argnums=(0))
    def calc_correction(self, walkers, xtaus, ham_data, wave_data):
        # xtaus shape (nwalker, nslater, nocc, nvir)
        nocc = self.nelec[0]
        norb = self.norb
        nvir = norb - nocc
        nwalker = walkers.shape[0]
        batch_size = nwalker // self.n_batch
        nslater = self.nslater

        assert xtaus.shape == (nwalker, nslater, nocc, nvir)

        def scan_batch(carry, xs):
            walker_batch, xtaus_batch = xs
            num, den = vmap(self._calc_correction_xtaus, in_axes=(0, 0, None, None))(
                walker_batch, xtaus_batch, ham_data, wave_data
            )
            return carry, (num, den)

        _, (num, den) = lax.scan(
            scan_batch, None,
            (walkers.reshape(self.n_batch, batch_size, norb, nocc),
             xtaus.reshape(self.n_batch, batch_size, nslater, nocc, nvir))
            )
        
        num = num.reshape(nwalker)
        den = den.reshape(nwalker)
        
        return num, den

    @partial(jit, static_argnums=0)
    def _calc_energy_cid(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ):
        nocc = self.nelec[0]
        mo_t, t2 = wave_data["mo_t"], wave_data["t2"]
        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        green = (walker @ (jnp.linalg.inv(mo_t.T @ walker)) @ mo_t.T).T
        greenp = (green - jnp.eye(self.norb))[:,nocc:]

        ################## ref ##################
        h0 = ham_data['h0']
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        hg = oe.contract("pq,pq->", h1, green, backend="jax")
        e1_0 = 2 * hg

        # build two body terms in scan_chol
        # gl = oe.contract("pr,gqr->gpq", green, chol, backend="jax")
        # trgl = oe.contract('gpp->g', gl, backend="jax")
        # e2_0_1 = 2 * jnp.sum(trgl**2)
        # e2_0_2 = -oe.contract('gpq,gqp->',gl,gl, backend="jax")
        # e2_0 = e2_0_1 + e2_0_2
        ############################################

        # one-body double excitations #############
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

        # two body double excitations #############
        # e2_2_1 = e2_0 * gt2g
        # lt2g = oe.contract("gij,ij->g", chol, t2_green, backend="jax")
        # e2_2_2_1 = -oe.contract("g,g->", lt2g, trgl, backend="jax")

        def scan_chol(carry, x):
            chol_i = x
            gl_i = oe.contract("pr,qr->pq", green, chol_i, backend="jax")
            trgl_i = oe.contract('pp->', gl_i, backend="jax")
            e2_0_c_i = 2 * trgl_i**2
            e2_0_e_i = -oe.contract('pq,qp->', gl_i, gl_i, backend="jax")
            e2_0_i = e2_0_c_i + e2_0_e_i
            carry[0] += e2_0_i

            lt2_green_i = oe.contract("pr,qr->pq", chol_i, t2_green, backend="jax")
            trlt2_green_i = oe.contract("pp->", lt2_green_i, backend="jax")
            e2_2_2_c_i = - trlt2_green_i * trgl_i
            e2_2_2_e_i = 0.5 * oe.contract("pq,pq->", gl_i, lt2_green_i, backend="jax")
            e2_2_2_i = e2_2_2_c_i + e2_2_2_e_i
            carry[1] += 4 * e2_2_2_i
            # carry[0] += 0.5 * oe.contract("pq,pq->", gl_i, lt2_green_i, backend="jax")

            glgp_i = oe.contract("iq,qa->ia", gl_i[:nocc,:], greenp, backend="jax")
            l2t2_1 = oe.contract("ia,jb,iajb->", glgp_i, glgp_i, t2, backend="jax")
            l2t2_2 = oe.contract("ib,ja,iajb->", glgp_i, glgp_i, t2, backend="jax")
            e2_2_3_i = 2 * l2t2_1 - l2t2_2
            carry[2] += e2_2_3_i
            return carry, 0.0

        [e2_0, e2_2_2, e2_2_3], _ = lax.scan(scan_chol, [0.0, 0.0, 0.0], (chol))

        e2_2_1 = e2_0 * gt2g
        # e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3

        o0 = self._slater_olp(walker, wave_data["mo_t"])
        overlap = (1.0 + gt2g) * o0 # denominator
        
        energy = h0 + (e1_0 + e2_0 + e1_2 + e2_2) / (1.0 + gt2g)

        return overlap, energy

    @partial(jit, static_argnums=(0))
    def calc_energy_cid(self,walkers,ham_data,wave_data):
        nwalker = walkers.shape[0]
        batch_size = nwalker // self.n_batch

        def scan_batch(carry, walker_batch):
            overlap, energy = vmap(self._calc_energy_cid, in_axes=(0, None, None))(
                walker_batch, ham_data, wave_data
            )
            return carry, (overlap, energy)

        _, (overlaps, energies) = lax.scan(
            scan_batch,
            None, walkers.reshape(self.n_batch, batch_size, self.norb, -1),
            )

        overlaps = overlaps.reshape(nwalker)
        energies = energies.reshape(nwalker)
        
        return overlaps, energies
    

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


class mixed_wave:

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
    
    def _calc_energy_mixed(self, walker, ham_data, wave_data):
        eg = self._guide._calc_energy_restricted(walker, ham_data, wave_data)
        et = self._trial._calc_energy_restricted(walker, ham_data, wave_data)
        og = self._guide._calc_overlap_restricted(walker, wave_data)
        ot = self._trial._calc_overlap_restricted(walker, wave_data)
        otg = ot/og
        return jnp.real(otg), jnp.real(eg), jnp.real(et)
    
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
    