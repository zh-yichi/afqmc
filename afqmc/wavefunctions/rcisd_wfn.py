import jax
from jax import lax, jit
from jax import numpy as jnp, vmap
import opt_einsum as oe
from functools import partial
from .. import slater_tools
from . import rhf_wfn

energy_formula = rhf_wfn.energy_formula

@partial(jit, static_argnums=0)
def overlap(trial, walker: jax.Array, wave_data: dict) -> complex:
    nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
    green = slater_tools.r_delta_green(wave_data["mo_coeff"], walker)
    o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
    o1 = oe.contract("ia,ia", ci1, green[:,nocc:], backend="jax")
    o2 = 2 * oe.contract("iajb, ia, jb", ci2, green[:,nocc:], green[:,nocc:], backend="jax") \
        - oe.contract("iajb, ib, ja", ci2, green[:,nocc:], green[:,nocc:], backend="jax")
    return (1.0 + 2 * o1 + o2) * o0

@partial(jit, static_argnums=0)
def force_bias(trial, walker: jax.Array, ham_data: dict, wave_data: dict) -> jax.Array:
    """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
    nocc, norb, nchol = trial.nelec[0], trial.norb, trial.nchol
    ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
    green = slater_tools.r_delta_green(wave_data["mo_coeff"], walker)
    green_occ = green[:, nocc:].copy()
    greenp = jnp.vstack((green_occ, -jnp.eye(norb - nocc)))

    chol = ham_data["chol"].reshape(nchol, norb, norb)
    rot_chol = chol[:,:nocc,:]
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
def energy(trial, walker: jax.Array, ham_data: dict, wave_data: dict) -> complex:
    nocc, norb, nchol = trial.nelec[0], trial.norb, trial.nchol
    ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
    green = slater_tools.r_delta_green(wave_data["mo_coeff"], walker)
    green_occ = green[:, nocc:].copy()
    greenp = jnp.vstack((green_occ, -jnp.eye(norb - nocc)))

    chol = ham_data["chol"].reshape(-1,norb,norb)
    rot_chol = chol[:,:nocc,:]
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
def build_intermediate(trial, ham_data: dict, wave_data: dict):
    ham_data["lci1"] = oe.contract(
        "git,pt->gip",
        ham_data["chol"].reshape(-1, trial.norb, trial.norb)[:, :, trial.nelec[0] :],
        wave_data["ci1"],
        backend="jax")
    
    return ham_data, wave_data
