import jax
import jax.numpy as jnp
from jax import jit, lax, jvp

import opt_einsum as oe
import numpy as np

from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD

from functools import partial
from afqmc import slater_tools

def decompose_rt2(t2, thresh=1e-8):
    nocc, nvir, _, _ = t2.shape
    npair = nocc * nvir

    assert t2.shape == (nocc, nvir, nocc, nvir)
    
    t2 = t2.reshape(npair, npair)
    e_val, e_vec = jnp.linalg.eigh(t2)

    # Keep only important modes
    mask = jnp.abs(e_val) > thresh
    e_val_trunc = e_val[mask]
    e_vec_trunc = e_vec[:, mask]

    tau = e_vec_trunc @ jnp.diag(jnp.sqrt(e_val_trunc + 0.0j))
    
    err = jnp.linalg.norm(t2 - tau @ tau.T)
    assert err < 10 * thresh

    tau = tau.T.reshape(-1, nocc, nvir)

    return tau

def decompose_ut2(t2, thresh=1e-8):
    t2aa, t2ab, t2bb = t2
    nocca, nvira, noccb, nvirb = t2ab.shape

    npaira = nocca * nvira
    npairb = noccb * nvirb

    assert t2aa.shape == (nocca, nvira, nocca, nvira)
    assert t2bb.shape == (noccb, nvirb, noccb, nvirb)

    t2aa = t2aa.reshape(npaira, npaira)
    t2ab = t2ab.reshape(npaira, npairb)
    t2bb = t2bb.reshape(npairb, npairb)

    # Symmetric full t2 
    # [[ t2aa/2  t2ab   ]]
    # [[ t2ab^T  t2bb/2 ]]
    # t2full = np.zeros((npaira + npairb, npaira + npairb))
    # t2full[:npaira, :npaira] = 0.5 * t2aa
    # t2full[npaira:, :npaira] = t2ab.T
    # t2full[:npaira, npaira:] = t2ab
    # t2full[npaira:, npaira:] = 0.5 * t2bb
    # t2full = jnp.array(t2full)
    t2full = jnp.block([[0.5 * t2aa, t2ab],
                        [t2ab.T, 0.5 * t2bb]])
    # t2 = LL^T
    e_val, e_vec = jnp.linalg.eigh(t2full)

    # Keep only important modes
    mask = jnp.abs(e_val) > thresh
    e_val_trunc = e_val[mask]
    e_vec_trunc = e_vec[:, mask]
    
    tau = e_vec_trunc @ jnp.diag(jnp.sqrt(e_val_trunc + 0.0j))
    err = jnp.linalg.norm(t2full - tau @ tau.T)
    assert err < 10 * thresh

    # alpha/beta operators for HS
    # Summation on the left to have a list of operators
    taua = tau.T[:,:npaira]
    taub = tau.T[:, npaira:]
    taua = taua.reshape(-1, nocca, nvira)
    taub = taub.reshape(-1, noccb, nvirb)

    return (taua, taub)

# def decompose_t2(t2, thresh=1e-8):
#     if isinstance(t2, jax.array) and len(t2.shape) == 4:
#         return decompose_rt2(t2, thresh)
#     elif isinstance(t2, tuple) and len(t2) == 3:
#         return decompose_ut2(t2, thresh)
#     else:
#         raise TypeError(f"T2 amplitude should either be a rank-4 tensor"
#                         f"or a tuple of 3 (aa,ab,bb) rank-4 tensors.")

def decompose_t2(t2, thresh=1e-8):
    if isinstance(t2, jax.Array) and len(t2.shape) == 4:
        return decompose_rt2(t2, thresh)
    elif isinstance(t2, tuple) and len(t2) == 3:
        return decompose_ut2(t2, thresh)
    else:
        raise TypeError(f"T2 amplitude should either be a rank-4 tensor "
                        f"or a tuple of 3 (aa,ab,bb) rank-4 tensors.")


def get_cc_amps(cc, save2disk, amp_file):

    if isinstance(cc, UCCSD):
        t1a = np.array(cc.t1[0])
        t1b = np.array(cc.t1[1])
        t2aa, t2ab, t2bb = cc.t2
        t2aa = (t2aa - t2aa.transpose(0, 1, 3, 2)) / 2
        t2bb = (t2bb - t2bb.transpose(0, 1, 3, 2)) / 2
        t2aa = t2aa.transpose(0, 2, 1, 3)
        t2bb = t2bb.transpose(0, 2, 1, 3)
        t2ab = t2ab.transpose(0, 2, 1, 3)
        if save2disk:
            np.savez(
                amp_file,
                t1a=t1a,
                t1b=t1b,
                t2aa=t2aa,
                t2ab=t2ab,
                t2bb=t2bb,
            )
        t1 = (jnp.array(t1a), 
              jnp.array(t1b))
        t2 = (jnp.array(t2aa), 
              jnp.array(t2ab), 
              jnp.array(t2bb))

    elif isinstance(cc, CCSD):
        t1 = np.array(cc.t1)
        t2 = cc.t2
        t2 = t2.transpose(0, 2, 1, 3)
        if save2disk:
            np.savez(amp_file, t1=t1, t2=t2)
        t1 = jnp.array(t1) 
        t2 = jnp.array(t2)

    return t1, t2


def read_cc_amps(amp_file):
    data = np.load(amp_file)

    if "t1a" in data:  # UCCSD
        t1 = (jnp.array(data["t1a"]), 
              jnp.array(data["t1b"]))
        t2 = (jnp.array(data["t2aa"]), 
              jnp.array(data["t2ab"]), 
              jnp.array(data["t2bb"]))
    else:              # RCCSD
        t1 = jnp.array(data["t1"])
        t2 = jnp.array(data["t2"])

    return t1, t2

def rcc2ci(t1, t2):
    ci1 = t1
    ci2 = t2 + oe.contract("ia,jb->iajb", t1, t1, backend='jax')
    return ci1, ci2

def ucc2ci(t1, t2):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    ci1 = (t1a, t1b)
    
    ci2aa = t2aa + 2 * oe.contract("ia,jb->iajb", t1a, t1a, backend='jax')
    ci2ab = t2ab + oe.contract("ia,jb->iajb", t1a, t1b, backend='jax')
    ci2bb = t2bb + 2 * oe.contract("ia,jb->iajb", t1b, t1b, backend='jax')
    ci2aa = (ci2aa - ci2aa.transpose(0, 3, 2, 1)) / 2
    ci2bb = (ci2bb - ci2bb.transpose(0, 3, 2, 1)) / 2
    ci2 = (ci2aa, ci2ab, ci2bb)
    return ci1, ci2

def cc2ci(t1, t2):
    if isinstance(t2, jax.Array) and len(t2.shape) == 4:
        return rcc2ci(t1, t2)
    elif isinstance(t2, tuple) and len(t2) == 3:
        return ucc2ci(t1, t2)
    else:
        raise TypeError(f"restricted T1 T2 should be jax.array "
                        f"and tuple for unrestricted case")
    
@partial(jit, static_argnums=(5,6))
def ut2h12(bra, ket, t2, h1, chol, mix_precision=False, nchol_chunk=1):
    '''
    calculate <bra|T2(h1+h2)|ket>/<bra|ket>
    general form of the unrestricted bra state
    return <bra|ket> <T2> <h1+h2> <T2(h1+h2)> 
    '''
    if mix_precision:
        rtype = jnp.float32
        ctype = jnp.complex64
    else:
        rtype = jnp.float64
        ctype = jnp.complex128

    ket_up, ket_dn = ket
    norb_a, nocc_a = ket_up.shape
    norb_b, nocc_b = ket_dn.shape
    t2aa, t2ab, t2bb = t2
    h1_a, h1_b = h1
    chol_a, chol_b = chol

    green_a, green_b = slater_tools.u_green(bra, ket)
    greenp_a = (green_a - jnp.eye(norb_a))[:,nocc_a:]
    greenp_b = (green_b - jnp.eye(norb_b))[:,nocc_b:]

    hg_a = oe.contract("pq,pq->", h1_a, green_a, backend="jax")
    hg_b = oe.contract("pq,pq->", h1_b, green_b, backend="jax")
    e1_0 = hg_a + hg_b # <bra|h1|ket>/<bra|ket>

    # <bra|T2 h1|ket>/<bra|ket>
    # double excitations
    t2g_a = oe.contract("iajb,ia->jb", t2aa, green_a[:nocc_a,nocc_a:], backend="jax") / 4
    t2g_b = oe.contract("iajb,ia->jb", t2bb, green_b[:nocc_b,nocc_b:], backend="jax") / 4
    t2g_ab_a = oe.contract("iajb,jb->ia", t2ab, green_b[:nocc_b,nocc_b:], backend="jax")
    t2g_ab_b = oe.contract("iajb,ia->jb", t2ab, green_a[:nocc_a,nocc_a:], backend="jax")
    # t_iajb (G_ia G_jb - G_ib G_ja)
    gt2g_a = oe.contract("jb,jb->", t2g_a, green_a[:nocc_a,nocc_a:], backend="jax")
    gt2g_b = oe.contract("jb,jb->", t2g_b, green_b[:nocc_b,nocc_b:], backend="jax")
    gt2g_ab = oe.contract("ia,ia->", t2g_ab_a, green_a[:nocc_a,nocc_a:], backend="jax")
    gt2g = 2 * (gt2g_a + gt2g_b) + gt2g_ab # <bra|T2|ket>/<bra|ket>
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
    e1_2 = e1_2_1 + e1_2_2  # <bra|T2 h1|ket>/<bra|ket>

    # <bra|T2 h2|ket>/<bra|ket>
    nchol = chol_a.shape[0]
    nchunks = -(-nchol // nchol_chunk)
    pad = nchunks * nchol_chunk - nchol
    chol_a = jnp.pad(chol_a, ((0, pad), (0, 0), (0, 0)))
    chol_b = jnp.pad(chol_b, ((0, pad), (0, 0), (0, 0)))
    chol_a = chol_a.reshape(nchunks, nchol_chunk, *chol_a.shape[-2:])
    chol_b = chol_b.reshape(nchunks, nchol_chunk, *chol_b.shape[-2:])

    def scanned_fun(carry, x):
        chol_a_c, chol_b_c = x

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

    # <bra|ket>
    # olp = slater_tools.u_overlap(bra, ket) 
    e10 = gt2g # <bra|T2|ket>/<bra|ket>
    e01 = (e1_0 + e2_0) # <bra|h1+h2|ket>/<bra|ket>
    e11 = (e1_2 + e2_2) # <bra|T2 (h1+h2)|ket>/<bra|ket>

    return jnp.array([e10, e01, e11])

@partial(jit, static_argnums=(5,6))
def ut2h12_delta(bra, ket, t2, h1, chol, mix_precision=False, nchol_chunk=1):
    '''
    calculate terms related to <bra|T2(h1+h2)|ket>/<bra|ket>
    bra is assumed to be an identity in its mo_coeff
    return <bra|ket> <T2> <h1+h2> <T2(h1+h2)> 
    '''
    if mix_precision:
        rtype = jnp.float32
        ctype = jnp.complex64
    else:
        rtype = jnp.float64
        ctype = jnp.complex128

    ket_up, ket_dn = ket
    norb_a, nocc_a = ket_up.shape
    norb_b, nocc_b = ket_dn.shape
    t2aa, t2ab, t2bb = t2
    h1_a, h1_b = h1
    chol_a, chol_b = chol

    green_a, green_b = slater_tools.u_delta_green(bra, ket)
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
    nchunks = -(-nchol // nchol_chunk)
    pad = nchunks * nchol_chunk - nchol
    chol_a = jnp.pad(chol_a, ((0, pad), (0, 0), (0, 0)))
    chol_b = jnp.pad(chol_b, ((0, pad), (0, 0), (0, 0)))
    chol_a = chol_a.reshape(nchunks, nchol_chunk, *chol_a.shape[-2:])
    chol_b = chol_b.reshape(nchunks, nchol_chunk, *chol_b.shape[-2:])

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

    # olp = slater_tools.u_delta_overlap(bra, ket) # <bra|ket>
    e10 = gt2g # <bra|T2|ket>/<bra|ket>
    e01 = (e1_0 + e2_0) # <bra|h1+h2|ket>/<bra|ket>
    e11 = (e1_2 + e2_2) # <bra|T2 (h1+h2)|ket>/<bra|ket>

    return jnp.array([e10, e01, e11])

@jit
def ut2_overlap(bra: tuple, ket: tuple, t2: tuple):
    '''<bra|T2|ket>'''
    bra_up, bra_dn = bra
    ket_up, ket_dn = ket
    norb_a, nocc_a = ket_up.shape
    norb_b, nocc_b = ket_dn.shape
    t2aa, t2ab, t2bb = t2
    green_a, green_b = slater_tools.u_green(bra, ket)
    green_a, green_b = green_a[:nocc_a, nocc_a:], green_b[:nocc_b, nocc_b:]
    o0 = slater_tools.u_overlap(bra, ket)
    o2 = (0.5 * oe.contract("iajb,ia,jb->", t2aa, green_a, green_a, backend="jax")
        + 0.5 * oe.contract("iajb,ia,jb->", t2bb, green_b, green_b, backend="jax")
        + oe.contract("iajb,ia,jb->", t2ab, green_a, green_b, backend="jax"))
    return o2 * o0

@jit
def ut2_olp_exp1(x: float, h1_mod: tuple, bra:tuple, ket:tuple, t2:tuple):
    '''
    <bra|T2 exp(x*h1_mod)|ket>/<bra|ket>
    this function has to be differentiable
    '''
    ket_up_1x = ket[0] + x * h1_mod[0].dot(ket[0])
    ket_dn_1x = ket[1] + x * h1_mod[1].dot(ket[1])
    ket1x = (ket_up_1x, ket_dn_1x)
    o1x = ut2_overlap(bra, ket1x, t2)
    o0 = slater_tools.u_overlap(bra, ket)
    return o1x / o0

@jit
def ut2_olp_exp2_i(x: float, chol_i: tuple, bra: tuple, ket: tuple, t2:tuple):
    '''
    <bra|T2 exp(x*chol_i)|ket>/<bra|ket>
    '''
    ket_up_2x = (
        ket[0] + x * chol_i[0].dot(ket[0]) 
        + x**2 / 2.0 * chol_i[0].dot(chol_i[0].dot(ket[0]))
        )
    ket_dn_2x = (
        ket[1] + x * chol_i[1].dot(ket[1])
        + x**2 / 2.0 * chol_i[1].dot(chol_i[1].dot(ket[1]))
        )
    
    ket2x = (ket_up_2x, ket_dn_2x)
    o2x = ut2_overlap(bra, ket2x, t2)
    o0 = slater_tools.u_overlap(bra, ket)
    return o2x / o0

@jit
def d2_ut2_olp_exp2_i(chol_i, bra, ket, t2):
    x = 0.0
    f = lambda a: ut2_olp_exp2_i(a, chol_i, bra, ket, t2)
    _, d2f = jax.jvp(lambda x: jax.jvp(f, [x], [1.0])[1], [x], [1.0])
    return d2f

@jit
def d2_ut2_olp_exp2(chol, bra, ket, t2):

    def scan_chol(carry, chol_i):
        d2_exp2_i = d2_ut2_olp_exp2_i(chol_i, bra, ket, t2)
        return carry + d2_exp2_i, None
    
    init = 0.0
    e2_sum, _ = jax.lax.scan(scan_chol, init, chol)
    return e2_sum / 2 

@jit
def ut2h12_ad(bra, ket, t2, h1mod, chol):
    '''
    t2olp = <bra|T2|ket>
    energy = e1 + e2
    e1 = partial_x <bra|T2 exp(xh1mod)|ket>/<bra|ket>
    e2 = 1/2 partial_x2 <bra|T2 exp(xchol)|ket>/<bra|ket>
    h1mod = h1 - 1/2 v_gpr v_gqr from commutator
    '''
    # AD chol can't be chunked in the current implementation
    chola, cholb = chol
    if len(chola.shape) == 4:
        nc, nchol_c, _, _ = chola.shape
        chola = chola.reshape(nc*nchol_c,*chola.shape[-2:])
    if len(cholb.shape) == 4:
        nc, nchol_c, _, _ = cholb.shape
        cholb = cholb.reshape(nc*nchol_c,*cholb.shape[-2:])

    # one body
    x = 0.0
    f1 = lambda a: ut2_olp_exp1(a, h1mod, bra, ket, t2)
    t2olp, e1 = jvp(f1, [x], [1.0])

    # two body
    e2 = d2_ut2_olp_exp2(chol, bra, ket, t2)

    return jnp.array([t2olp, e1 + e2])