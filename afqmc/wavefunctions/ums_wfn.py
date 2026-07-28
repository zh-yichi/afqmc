from jax import lax
from jax import numpy as jnp
import opt_einsum as oe
from .. import slater_tools
from . import rhf_wfn
from jax import jit
from functools import partial

energy_formula = rhf_wfn.energy_formula


def ums_overlap(slaters, walker):

    def scan_slaters(carry, slater):
        olp = slater_tools.u_overlap(slater, walker)
        return carry, olp

    init_carry = 0.0
    _, olps = lax.scan(scan_slaters, init_carry, slaters)

    olp = jnp.sum(olps) / slaters.shape[0]

    return olp


def ums_force_bias(slaters, walker, chol):

    def scan_slaters(carry, slater):
        olp = slater_tools.u_overlap(slater, walker)
        fb = slater_tools.u_force_bias(slater, walker, chol)
        return carry, (olp, fb)

    init_carry = 0.0
    _, (olps, fbs) = lax.scan(scan_slaters, init_carry, slaters)

    olp = jnp.sum(olps)
    fb = oe.contract("s,sg->g", olps,fbs, backend="jax") / olp

    return fb


def ums_rot_force_bias(slaters, walker, rot_chol):

    def scan_slaters(carry, slater):
        olp = slater_tools.u_overlap(slater, walker)
        fb = slater_tools.u_rot_force_bias(slater, walker, rot_chol)
        return carry, (olp, fb)

    init_carry = 0.0
    _, (olps, fbs) = lax.scan(scan_slaters, init_carry, slaters)

    olp = jnp.sum(olps)
    fb = oe.contract("s,sg->g", olps,fbs, backend="jax") / olp

    return fb


def ums_energy(slaters, walker, h0, h1, chol):

    def scan_slaters(carry, slater):
        olp = slater_tools.u_overlap(slater, walker)
        energy = slater_tools.u_energy(slater, walker, h0, h1, chol)
        return carry, (olp, energy)

    init_carry = 0.0
    _, (olps, energies) = lax.scan(scan_slaters, init_carry, slaters)

    olp = jnp.sum(olps)
    energy = jnp.sum(olps*energies) / olp

    return energy

def ums_rot_energy(slaters, walker, h0, rot_h1, rot_chol):

    def scan_slaters(carry, slater):
        olp = slater_tools.u_overlap(slater, walker)
        energy = slater_tools.u_rot_energy(slater, walker, h0, rot_h1, rot_chol)
        return carry, (olp, energy)

    init_carry = 0.0
    _, (olps, energies) = lax.scan(scan_slaters, init_carry, slaters)

    olp = jnp.sum(olps)
    energy = jnp.sum(olps*energies) / olp

    return energy

# implementation of above functions in QMC sampling
@partial(jit, static_argnums=0)
def overlap(trial, walker, wave_data):
    return ums_overlap(wave_data["slaters"], walker)

@partial(jit, static_argnums=0)
def force_bias(trial, walker, ham_data, wave_data):
    chol_a = ham_data["chol"][0].reshape(trial.nchol, trial.norb, trial.norb)
    chol_b = ham_data["chol"][1].reshape(trial.nchol, trial.norb, trial.norb)
    chol = (chol_a, chol_b)
    return ums_force_bias(wave_data["slaters"], walker, chol)

@partial(jit, static_argnums=0)
def energy(trial, walker, ham_data, wave_data):
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
    return ums_energy(wave_data["slaters"], walker, h0, h1, chol)

@partial(jit, static_argnums=0)
def rot_force_bias(trial, walker, ham_data, wave_data):
    rot_chol_a = ham_data["rot_chol"][0]
    rot_chol_b = ham_data["rot_chol"][1]
    nchol = trial.nchol
    nchol_chunk = trial.nchol_chunk
    nchunks = -(-nchol // nchol_chunk)
    pad = nchunks * nchol_chunk - nchol
    rot_chol_a = jnp.pad(rot_chol_a, ((0, pad), (0, 0), (0, 0)))
    rot_chol_b = jnp.pad(rot_chol_b, ((0, pad), (0, 0), (0, 0)))
    rot_chol_a = rot_chol_a.reshape(nchunks, nchol_chunk, *rot_chol_a.shape[1:])
    rot_chol_b = rot_chol_b.reshape(nchunks, nchol_chunk, *rot_chol_b.shape[1:])
    rot_chol = (rot_chol_a, rot_chol_b)
    return ums_rot_force_bias(wave_data["slaters"], walker, rot_chol)

@partial(jit, static_argnums=0)
def rot_energy(trial, walker, ham_data, wave_data):
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
    return ums_rot_energy(wave_data["slaters"], walker, h0, rot_h1, rot_chol)
