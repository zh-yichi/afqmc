# from abc import ABC
from dataclasses import dataclass
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random, jit, lax, vmap, jvp
import opt_einsum as oe

from afqmc.prop_unrestricted.wavefunctions_unrestricted import uhf


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
    
        # gl_a = oe.contract("pr,gqr->gpq", green_a, chol_a, backend="jax")
        # gl_b = oe.contract("pr,gqr->gpq", green_b, chol_b, backend="jax")
        # trgl_a = oe.contract('gpp->g', gl_a, backend="jax")
        # trgl_b = oe.contract('gpp->g', gl_b, backend="jax")
        # e2_0_1 = jnp.sum((trgl_a + trgl_b)**2) / 2
        # e2_0_2 = -(oe.contract('gpq,gqp->', gl_a, gl_a, backend="jax")
        #         + oe.contract('gpq,gqp->', gl_b, gl_b, backend="jax")) / 2
        # e2_0 = e2_0_1 + e2_0_2

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

        ###### two-body single excitations ######
        # e2_1_1 = o1 * e2_0

        # c_ia Gp_pa G_ir L_pr G_qs L_qs
        # lc1g_a = oe.contract("gpq,pq->g", chol_a, c1gpg_a, backend="jax")
        # lc1g_b = oe.contract("gpq,pq->g", chol_b, c1gpg_b, backend="jax")
        # e2_1_2 = -jnp.sum((lc1g_a + lc1g_b) * (trgl_a + trgl_b))

        # glc1gp_a = jnp.einsum("gpq,iq->gpi", gl_a, c1gp_a, optimize="optimal") # c_ia Gp_qa G_pr L_qr 
        # glc1gp_b = jnp.einsum("gpq,iq->gpi", gl_b, c1gp_b, optimize="optimal")
        # e2_1_3 = jnp.einsum("gpi,gip->", glc1gp_a, gl_a[:,:nocc_a,:], optimize="optimal") \
        #         + jnp.einsum("gpi,gip->", glc1gp_b, gl_b[:,:nocc_b,:], optimize="optimal") # t_ia Gp_pa G_qr L_pr G_is L_qs
        
        # e2_1 = e2_1_1 + e2_1_2 + e2_1_3 # <C1 psi|h2|walker> / <psi|walker>

        ###### two-body double excitations ######
        # e2_2_1 = o2 * e2_0

        # ccGGGp G_qs L_ps
        # lc2ggg_a = oe.contract("gpr,qr->gpq", chol_a, 2*(c2ggg_aaa + c2ggg_aba), backend="jax")
        # lc2ggg_b = oe.contract("gpr,qr->gpq", chol_b, 2*(c2ggg_bbb + c2ggg_bab), backend="jax")
        # trlc2ggg_a = oe.contract("gpp->g", lc2ggg_a, backend="jax")
        # trlc2ggg_b = oe.contract("gpp->g", lc2ggg_b, backend="jax")
        # e2_2_2_c = -jnp.sum((trlc2ggg_a + trlc2ggg_b)*(trgl_a + trgl_b)) / 4
        # e2_2_2_e = (oe.contract("gpq,gpq->", gl_a, lc2ggg_a, backend="jax")
        #             + oe.contract("gpq,gpq->", gl_b, lc2ggg_b, backend="jax")) / 4
        # e2_2_2 = e2_2_2_c + e2_2_2_e
        
        # c1glgp_a = oe.contract("ip,gjp->gij", c1gp_a, gl_a[:,:nocc_a,:], backend="jax")
        # c1glgp_b = oe.contract("ip,gjp->gij", c1gp_b, gl_b[:,:nocc_b,:], backend="jax")
        # trc1glgp_a = oe.contract("gii->g", c1glgp_a, backend="jax")
        # trc1glgp_b = oe.contract("gii->g", c1glgp_b, backend="jax")
        # e2_2_3_c = jnp.sum((trc1glgp_a + trc1glgp_b)**2) / 2
        # e2_2_3_e = (oe.contract("gij,gji->", c1glgp_a, c1glgp_a, backend="jax")
        #             + oe.contract("gij,gji->", c1glgp_b, c1glgp_b, backend="jax")) / 2
        # e2_2_3 = e2_2_3_c - e2_2_3_e

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
    