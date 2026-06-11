import jax
from jax import lax
import jax.numpy as jnp
import opt_einsum as oe
from .. import slater_tools
from . import rpt2ccsd_wfn

from jax import jit
from functools import partial

energy_formula = rpt2ccsd_wfn.energy_formula

@partial(jit, static_argnums=0)
def calc_overlap(wave, walker, wave_data):
    return slater_tools.u_overlap(wave_data["mo_coeff"], walker)

@partial(jit, static_argnums=0)
def calc_energy(
        wave,
        walker: tuple,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
    
    if wave.mix_precision: # only do this for two-body energy with T contraction
        rtype = jnp.float32
        ctype = jnp.complex64
    else:
        rtype = jnp.float64
        ctype = jnp.complex128

    (nocc_a, nocc_b), norb = wave.nelec, wave.norb
    mo_a, mo_b = wave_data['mo_ta'], wave_data['mo_tb']
    t2_aa, t2_ab, t2_bb = wave_data["t2aa"], wave_data["t2ab"], wave_data["t2bb"]
    chol_a = ham_data["chol"][0].reshape(-1, norb, norb)
    chol_b = ham_data["chol"][1].reshape(-1, norb, norb)
    h1_a = ham_data["h1"][0]
    h1_b = ham_data["h1"][1]
    # walker_up, walker_dn = walker

    # full green's function G_pq
    green_a, green_b = slater_tools.u_green((mo_a, mo_b), walker)
    greenp_a = (green_a - jnp.eye(norb))[:,nocc_a:]
    greenp_b = (green_b - jnp.eye(norb))[:,nocc_b:]

    hg_a = oe.contract("pq,pq->", h1_a, green_a, backend="jax")
    hg_b = oe.contract("pq,pq->", h1_b, green_b, backend="jax")
    e1_0 = hg_a + hg_b # <exp(T1)HF|h1|walker>/<exp(T1)HF|walker>

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
    e1_2_1 = e1_0 * gt2g
    
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
    nchol, nchol_chunk = wave.nchol, wave.nchol_chunk
    nchunks = -(-nchol // nchol_chunk)
    pad = nchunks * nchol_chunk - nchol

    chol_a = jnp.pad(chol_a, ((0, pad), (0, 0), (0, 0)))
    chol_b = jnp.pad(chol_b, ((0, pad), (0, 0), (0, 0)))
    chol_a = chol_a.reshape(nchunks, nchol_chunk, norb, norb)
    chol_b = chol_b.reshape(nchunks, nchol_chunk, norb, norb)

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
        
        carry[3] += (l2t2_aa + l2t2_bb + l2t2_ab).astype(jnp.complex128)
        return carry, 0.0

    [e2_0, e2_2_2_1, e2_2_2_2, e2_2_3], _ = lax.scan(
        scanned_fun, [0.0, 0.0, 0.0, 0.0], (chol_a, chol_b)
    )

    e2_2_1 = e2_0 * gt2g
    e2_2_2 = e2_2_2_1 + e2_2_2_2
    e2_2 = e2_2_1 + e2_2_2 + e2_2_3 # <exp(T1)HF|T2 h2|walker>/<exp(T1)HF|walker>

    o0 = slater_tools.u_delta_overlap((mo_a, mo_b), walker) # <HF|walker>/<HF|walker>
    t1 = slater_tools.u_overlap((mo_a, mo_b), walker) / o0  # <exp(T1)HF|walker>/<HF|walker>
    t2 = gt2g * t1                                          # <exp(T1)HF|T2|walker>/<HF|walker>
    e0 = (e1_0 + e2_0) * t1                                 # <exp(T1)HF|h1+h2|walker>/<HF|walker>
    e1 = (e1_2 + e2_2) * t1                                 # <exp(T1)HF|T2 (h1+h2)|walker>/<HF|walker>

    return jnp.array((t1, t2, e0, e1))

@partial(jit, static_argnums=0)
def calc_intermediate(trial, ham_data: dict, wave_data: dict):

    return ham_data
