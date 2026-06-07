import jax
from jax import jit, lax
from jax import numpy as jnp

import opt_einsum as oe

@jit
def u_slater_overlap(bra: tuple, ket: tuple):
    olp = jnp.linalg.det(bra[0].T.conj() @ ket[0]) \
        * jnp.linalg.det(bra[1].T.conj() @ ket[1])
    return olp

@jit
def u_slater_half_green(bra: tuple, ket: tuple):
    green_a = (ket[0] @ (jnp.linalg.inv(bra[0].T.conj() @ ket[0]))).T
    green_b = (ket[1] @ (jnp.linalg.inv(bra[1].T.conj() @ ket[1]))).T
    return (green_a, green_b)

@jit
def u_slater_delta_green(bra: tuple, ket: tuple):
    green_a = (ket[0].dot(jnp.linalg.inv(ket[0][:ket[0].shape[1],:]))).T
    green_b = (ket[1].dot(jnp.linalg.inv(ket[1][:ket[1].shape[1],:]))).T
    return (green_a, green_b)

@jit
def u_slater_green(bra: tuple, ket: tuple):
    green_a = (ket[0] @ (jnp.linalg.inv(bra[0].T.conj() @ ket[0])) @ bra[0].T.conj()).T
    green_b = (ket[1] @ (jnp.linalg.inv(bra[1].T.conj() @ ket[1])) @ bra[0].T.conj()).T
    return (green_a, green_b)

@jit
def u_rot_slater_force_bias(bra: tuple, ket: tuple, rot_chol: tuple):
    green = u_slater_half_green(bra, ket)
    fb_a = oe.contract("gij,ij->g", rot_chol[0], green[0], backend="jax")
    fb_b = oe.contract("gij,ij->g", rot_chol[1], green[1], backend="jax")
    return fb_a + fb_b

@jit
def u_slater_force_bias(bra: tuple, ket: tuple, chol: tuple):
    green = u_slater_green(bra, ket)
    fb_a = oe.contract("gij,ij->g", chol[0], green[0], backend="jax")
    fb_b = oe.contract("gij,ij->g", chol[1], green[1], backend="jax")
    return fb_a + fb_b

@jit
def u_rot_slater_energy(
    bra: tuple, 
    ket: tuple, 
    h0: tuple, 
    rot_h1: tuple, 
    rot_chol: tuple
    ):
    # rot_chol has to be chunked before calling this function

    green = u_slater_half_green(bra, ket)
    e1 = oe.contract("pq,pq->", rot_h1[0], green[0]) \
        + oe.contract("pq,pq->", rot_h1[1], green[1])

    def scanned_fun(carry, x):
        chol_a_c, chol_b_c = x  # (nchol_chunk, nocc, norb) each
        lg_a_c = oe.contract("gpr,qr->gpq", chol_a_c, green[0], backend="jax")
        lg_b_c = oe.contract("gpr,qr->gpq", chol_b_c, green[1], backend="jax")
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

    e2, _ = lax.scan(scanned_fun, 0.0, (rot_chol[0], rot_chol[1]))

    return h0 + e1 + e2

@jit
def u_slater_energy(
    bra: tuple, 
    ket: tuple, 
    h0: tuple, 
    h1: tuple, 
    chol: tuple
    ):
    # chol_a and chol_b has to be chunked 
    # into shape (nchunk, nchol_chunk, nocc, norb)
    # before calling this function

    green = u_slater_green(bra, ket)
    e1 = oe.contract("pq,pq->", h1[0], green[0]) \
        + oe.contract("pq,pq->", h1[1], green[1])

    def scanned_fun(carry, x):
        chol_a_c, chol_b_c = x  # (nchol_chunk, nocc, norb) each
        lg_a_c = oe.contract("gpr,qr->gpq", chol_a_c, green[0], backend="jax")
        lg_b_c = oe.contract("gpr,qr->gpq", chol_b_c, green[1], backend="jax")
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

    e2, _ = lax.scan(scanned_fun, 0.0, (chol[0], chol[1]))

    return h0 + e1 + e2


# implementation of above functions in QMC sampling
def u_overlap(trial, walker, wave_data):
    return u_slater_overlap(wave_data["mo_coeff"], walker)

def u_force_bias(trial, walker, ham_data, wave_data):
    chol_a = ham_data["chol"][0].reshape(trial.nchol, trial.norb, trial.norb)
    chol_b = ham_data["chol"][1].reshape(trial.nchol, trial.norb, trial.norb)
    chol = (chol_a, chol_b)
    return u_slater_force_bias(wave_data["mo_coeff"], walker, chol)

def u_energy(trial, walker, ham_data, wave_data):
    h0 = ham_data["h0"]
    h1 = ham_data["h1"]
    chol_a = ham_data["chol"][0].reshape(trial.nchol, trial.norb, trial.norb)
    chol_b = ham_data["chol"][1].reshape(trial.nchol, trial.norb, trial.norb)
    nchol = trial.nchol
    nchol_chunk = trial.nchol_chunk
    nchunks = -(-nchol // nchol_chunk)
    pad = nchunks * nchol_chunk - nchol
    chol_a = jnp.pad(chol_a, ((0, pad), (0, 0), (0, 0)))
    chol_b = jnp.pad(chol_b, ((0, pad), (0, 0), (0, 0)))
    chol_a = chol_a.reshape(nchunks, nchol_chunk, *chol_a.shape[1:])
    chol_b = chol_b.reshape(nchunks, nchol_chunk, *chol_b.shape[1:])
    chol = (chol_a, chol_b)
    return u_slater_energy(wave_data["mo_coeff"], walker, h0, h1, chol)

def u_rot_force_bias(trial, walker, ham_data, wave_data):
    rot_chol_a = ham_data["rot_chol"][0].reshape(trial.nchol, trial.norb, trial.norb)
    rot_chol_b = ham_data["rot_chol"][1].reshape(trial.nchol, trial.norb, trial.norb)
    rot_chol = (rot_chol_a, rot_chol_b)
    return u_rot_slater_force_bias(wave_data["mo_coeff"], walker, rot_chol)

def u_rot_energy(trial, walker, ham_data, wave_data):
    h0 = ham_data["h0"]
    rot_h1 = ham_data["rot_h1"]
    rot_chol_a = ham_data["rot_chol"][0].reshape(trial.nchol, trial.norb, trial.norb)
    rot_chol_b = ham_data["rot_chol"][1].reshape(trial.nchol, trial.norb, trial.norb)
    nchol = rot_chol_a.shape[0]
    nchol_chunk = trial.nchol_chunk
    nchunks = -(-nchol // nchol_chunk)
    pad = nchunks * nchol_chunk - nchol
    rot_chol_a = jnp.pad(rot_chol_a, ((0, pad), (0, 0), (0, 0)))
    rot_chol_b = jnp.pad(rot_chol_b, ((0, pad), (0, 0), (0, 0)))
    rot_chol_a = rot_chol_a.reshape(nchunks, nchol_chunk, *rot_chol_a.shape[1:])
    rot_chol_b = rot_chol_b.reshape(nchunks, nchol_chunk, *rot_chol_b.shape[1:])
    rot_chol = (rot_chol_a, rot_chol_b)
    return u_rot_slater_energy(wave_data["mo_coeff"], walker, h0, rot_h1, rot_chol)