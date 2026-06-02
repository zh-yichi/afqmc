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