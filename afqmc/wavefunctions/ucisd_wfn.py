import jax
from jax import lax, jit
from jax import numpy as jnp, vmap
import opt_einsum as oe
from functools import partial
from afqmc.wavefunctions import uhf_wfn

@partial(jit, static_argnums=0)
def calc_overlap(
    trial, 
    walker: tuple,
    wave_data: dict
) -> complex:
    (walker_a, walker_b) = walker
    noccA, ci1A, ci2AA = trial.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
    noccB, ci1B, ci2BB = trial.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
    ci2AB = wave_data["ci2AB"]
    (green_a, green_b) = uhf_wfn.u_slater_delta_green(wave_data["mo_coeff"], walker)
    green_a, green_b = green_a[:, noccA:], green_b[:, noccB:]
    o0 = jnp.linalg.det(walker_a[:noccA, :]) * jnp.linalg.det(walker_b[:noccB, :])
    o1 = oe.contract("ia,ia", ci1A, green_a, backend="jax") \
        + oe.contract("ia,ia", ci1B, green_b, backend="jax")
    o2 = 0.5 * oe.contract("iajb, ia, jb", ci2AA, green_a, green_a, backend="jax")\
        + 0.5 * oe.contract("iajb, ia, jb", ci2BB, green_b, green_b, backend="jax")\
        + oe.contract("iajb, ia, jb", ci2AB, green_a, green_b, backend="jax")
    return (1.0 + o1 + o2) * o0

@partial(jit, static_argnums=0)
def calc_force_bias(
    trial,
    walker: tuple,
    ham_data: dict,
    wave_data: dict,
) -> jax.Array:
    """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
    nocc_a, ci1_a, ci2_aa = trial.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
    nocc_b, ci1_b, ci2_bb = trial.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
    ci2_ab = wave_data["ci2AB"]
    norb = trial.norb
    (green_a, green_b) = uhf_wfn.u_slater_delta_green(wave_data["mo_coeff"], walker)
    green_occ_a = green_a[:, nocc_a:].copy()
    green_occ_b = green_b[:, nocc_b:].copy()
    greenp_a = jnp.vstack((green_occ_a, -jnp.eye(norb - nocc_a)))
    greenp_b = jnp.vstack((green_occ_b, -jnp.eye(norb - nocc_b)))

    chol_a = ham_data["chol"][0].reshape(-1, norb, norb)
    chol_b = ham_data["chol"][1].reshape(-1, norb, norb)
    rot_chol_a = chol_a[:,:nocc_a,:]
    rot_chol_b = chol_b[:,:nocc_b,:]
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
def calc_energy(
    trial,
    walker: tuple,
    ham_data: dict,
    wave_data: dict,
) -> complex:
    nocc_a, ci1_a, ci2_aa = trial.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
    nocc_b, ci1_b, ci2_bb = trial.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
    ci2_ab = wave_data["ci2AB"]
    norb = trial.norb
    (green_a, green_b) = uhf_wfn.u_slater_delta_green(wave_data["mo_coeff"], walker)
    green_occ_a = green_a[:, nocc_a:].copy()
    green_occ_b = green_b[:, nocc_b:].copy()
    greenp_a = jnp.vstack((green_occ_a, -jnp.eye(norb - nocc_a)))
    greenp_b = jnp.vstack((green_occ_b, -jnp.eye(norb - nocc_b)))

    chol_a = ham_data["chol"][0].reshape(-1, norb, norb)
    chol_b = ham_data["chol"][1].reshape(-1, norb, norb)
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