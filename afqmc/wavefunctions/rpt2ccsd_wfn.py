import jax
from jax import lax
import jax.numpy as jnp
import opt_einsum as oe
from .. import slater_tools, sampling
from jax import jit
from functools import partial

@partial(jit, static_argnums=0)
def overlap(wave, walker, wave_data):
    return slater_tools.u_overlap(wave_data['mo_t'], walker)

@partial(jit, static_argnums=0)
def energy(
        wave, 
        walker: jax.Array, 
        ham_data: dict, 
        wave_data: dict
        ):
    
    if wave.mix_precision:
        rtype = jnp.float32
        ctype = jnp.complex64
    else:
        rtype = jnp.float64
        ctype = jnp.complex128

    nocc, norb = wave.nelec[0], wave.norb
    nchol, nchol_chunk = wave.nchol, wave.nchol_chunk
    mo_t, t2 = wave_data["mo_t"], wave_data["t2"]
    chol = ham_data["chol"].reshape(-1, wave.norb, wave.norb)
    green = slater_tools.r_green(mo_t, walker)
    greenp = (green - jnp.eye(norb))[:,nocc:]

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
    nchunks = -(-nchol // nchol_chunk)
    pad = nchunks * nchol_chunk - nchol
    chol = jnp.pad(chol, ((0, pad), (0, 0), (0, 0)))
    chol = chol.reshape(nchunks, nchol_chunk, norb, norb) 

    def scanned_fun(carry, x):
        chol_c = x  # (nchol_chunk, norb, norb)
        # e2_0
        gl_c = oe.contract("pr,gqr->gpq", green, chol_c, backend="jax")
        tr_gl_c = oe.contract("gpp->g", gl_c, backend="jax")
        e2_0_1_c = jnp.sum((2 * tr_gl_c) ** 2) / 2.0
        e2_0_2_c = -oe.contract("gpq,gqp->", gl_c, gl_c, backend="jax")
        carry[0] += e2_0_1_c + e2_0_2_c

        # e2_2
        lt2g_c = oe.contract("gpr,qr->gpq", chol_c, t2_green, backend="jax")
        tr_lt2g_c = oe.contract("gpp->g", lt2g_c, backend="jax")
        carry[1] += -oe.contract("g,g->", 
                                    tr_lt2g_c.astype(ctype), 
                                    tr_gl_c.astype(ctype), 
                                    backend="jax").astype(jnp.complex128)
        carry[2] += 0.5 * oe.contract("gpq,gpq->", 
                                        gl_c.astype(ctype), 
                                        lt2g_c.astype(ctype), 
                                        backend="jax").astype(jnp.complex128)
        
        glgp_c = oe.contract("giq,qa->gia", gl_c[:,:nocc,:], greenp, backend="jax")

        lt2_1 = oe.contract("gia,iajb->gjb", 
                                glgp_c.astype(ctype),
                                t2.astype(rtype), 
                                backend="jax")
        lt2_2 = oe.contract("gib,iajb->gja", 
                                glgp_c.astype(ctype),
                                t2.astype(rtype), 
                                backend="jax")
        l2t2_1 = oe.contract("gjb,gjb->", 
                                lt2_1.astype(ctype),
                                glgp_c.astype(ctype),
                                backend="jax").astype(jnp.complex128)
        l2t2_2 = oe.contract("gja,gja->", 
                                lt2_2.astype(ctype), 
                                glgp_c.astype(ctype), 
                                backend="jax").astype(jnp.complex128)
        
        carry[3] += (2*l2t2_1 - l2t2_2).astype(jnp.complex128)
        return carry, 0.0

    [e2_0, e2_2_2_1, e2_2_2_2, e2_2_3], _ = lax.scan(
        scanned_fun, [0.0, 0.0, 0.0, 0.0], chol)

    e2_2_1 = e2_0 * gt2g
    e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
    e2_2 = e2_2_1 + e2_2_2 + e2_2_3

    o0 = slater_tools.r_delta_overlap(mo_t, walker) # <HF|walker>
    t1 = slater_tools.r_overlap(mo_t, walker) / o0  # <exp(T1)HF|walker>/<HF|walker>
    t2 = gt2g * t1                                  # <exp(T1)HF|T2|walker>/<HF|walker>
    e0 = (e1_0 + e2_0) * t1                         # <exp(T1)HF|h1+h2|walker>/<HF|walker>
    e1 = (e1_2 + e2_2) * t1                         # <exp(T1)HF|T2(h1+h2)|walker>/<HF|walker>

    return jnp.array((t1, t2, e0, e1))

@partial(jit, static_argnums=0)
def build_intermediate(trial, ham_data: dict, wave_data: dict):
    wave_data["mo_t"] = slater_tools.thouless(wave_data["mo_coeff"], wave_data["t1"])
    return ham_data, wave_data

def energy_formula(weights, samples, ham_data):
    # samples shape: (nsamples, nterms)
    h0 = ham_data["h0"]
    nsamples = len(weights)

    weight_mean, sample_mean, _ = sampling.weighted_average(weights, samples)
    weight = weight_mean.real
    t2_mean, e0_mean, e1_mean = sample_mean

    energy = (
        h0 + e0_mean + e1_mean - (t2_mean * e0_mean)
    ).real

    if nsamples < 2:
        # one block: covariance is undefined, so no delta-method error bar
        energy_err = jnp.nan
        return weight, energy, energy_err

    else:
        dE = jnp.array([-e0_mean, 1-t2_mean, 1])
        cov_te0e1 = jnp.cov(samples.T)
        energy_err = (jnp.sqrt(dE @ cov_te0e1 @ dE) / jnp.sqrt(nsamples)).real
        return weight, energy, energy_err
