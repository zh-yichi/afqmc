from jax import lax
from jax import numpy as jnp
import opt_einsum as oe
from afqmc.wavefunctions import rhf_wfn

def rms_overlap(slaters, walker):

    def scan_slaters(carry, slater):
        olp = rhf_wfn.r_overlap(slater, walker)
        return carry, olp

    init_carry = 0.0
    _, olps = lax.scan(scan_slaters, init_carry, slaters)

    olp = jnp.sum(olps) / slaters.shape[0]

    return olp

def rms_force_bias(slaters, walker, chol):

    def scan_slaters(carry, slater):
        olp = rhf_wfn.r_overlap(slater, walker)
        fb = rhf_wfn.r_force_bias(slater, walker, chol)
        return carry, (olp, fb)

    init_carry = 0.0
    _, (olps, fbs) = lax.scan(scan_slaters, init_carry, slaters)

    olp = jnp.sum(olps)
    fb = oe.contract("s,sg->g", olps,fbs, backend="jax") / olp

    return fb

def rms_rot_force_bias(slaters, walker, rot_chol):

    def scan_slaters(carry, slater):
        olp = rhf_wfn.r_overlap(slater, walker)
        fb = rhf_wfn.r_rot_force_bias(slater, walker, rot_chol)
        return carry, (olp, fb)

    init_carry = 0.0
    _, (olps, fbs) = lax.scan(scan_slaters, init_carry, slaters)

    olp = jnp.sum(olps)
    fb = oe.contract("s,sg->g", olps,fbs, backend="jax") / olp

    return fb

def rms_energy(slaters, walker, h0, h1, chol):

    def scan_slaters(carry, slater):
        olp = rhf_wfn.r_overlap(slater, walker)
        energy = rhf_wfn.r_energy(slater, walker, h0, h1, chol)
        return carry, (olp, energy)

    init_carry = 0.0
    _, (olps, energies) = lax.scan(scan_slaters, init_carry, slaters)

    olp = jnp.sum(olps)
    energy = jnp.sum(olps*energies) / olp

    return energy

def rms_rot_energy(slaters, walker, h0, rot_h1, rot_chol):

    def scan_slaters(carry, slater):
        olp = rhf_wfn.r_overlap(slater, walker)
        energy = rhf_wfn.r_rot_energy(slater, walker, h0, rot_h1, rot_chol)
        return carry, (olp, energy)

    init_carry = 0.0
    _, (olps, energies) = lax.scan(scan_slaters, init_carry, slaters)

    olp = jnp.sum(olps)
    energy = jnp.sum(olps*energies) / olp

    return energy

# implementation of above functions in QMC sampling
def r_overlap(trial, walker, wave_data):
    return rms_overlap(wave_data["slaters"], walker)

def r_force_bias(trial, walker, ham_data, wave_data):
    chol = ham_data["chol"].reshape(trial.nchol, trial.norb, trial.norb)
    return rms_force_bias(wave_data["slaters"], walker, chol)

def r_energy(trial, walker, ham_data, wave_data):
    h0 = ham_data["h0"]
    h1 = ((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
    chol = ham_data["chol"].reshape(trial.nchol, trial.norb, trial.norb)
    return rms_energy(wave_data["slaters"], walker, h0, h1, chol)

def r_rot_force_bias(trial, walker, ham_data, wave_data):
    rot_chol = ham_data["rot_chol"].reshape(trial.nchol, trial.norb, trial.norb)
    return rms_rot_force_bias(wave_data["slaters"], walker, rot_chol)

def r_rot_energy(trial, walker, ham_data, wave_data):
    h0 = ham_data["h0"]
    rot_h1 = ham_data["rot_h1"]
    rot_chol = ham_data["rot_chol"]
    return rms_rot_energy(wave_data["slaters"], walker, h0, rot_h1, rot_chol)