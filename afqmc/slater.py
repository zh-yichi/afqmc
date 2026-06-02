"""
Tool kit for single slater determinant operations
Including: 
    Overlap, 
    Green's Function, 
    Force Bias,
    Energy
For both restricted and unrestricted determinant
"""

import jax
from jax import jit, lax
from jax import numpy as jnp

import opt_einsum as oe

@jit
def r_overlap(bra: jax.Array, ket: jax.Array):
    ''' 
    <bra|ket>
    '''
    olp = jnp.linalg.det(bra.T.conj() @ ket) ** 2
    return olp

@jit
def r_green(bra: jax.Array, ket: jax.Array) -> jax.Array:
    '''<bra|a^dagger_p a_q|ket>/<bra|ket>'''
    green = (ket @ (jnp.linalg.inv(bra.T.conj() @ ket)) @ bra.T.conj()).T
    return green

@jit
def r_half_green(bra: jax.Array, ket: jax.Array) -> jax.Array:
    '''half Green's function - the ket coefficient
       is contracted with the observable tensors'''
    green = (ket @ (jnp.linalg.inv(bra.T.conj() @ ket)) @ bra.T.conj()).T
    return green

@jit
def r_delta_green(bra: jax.Array, ket:jax.Array) -> jax.Array:
    '''hald Green's function when bra is identity'''
    green = (ket.dot(jnp.linalg.inv(ket[:ket.shape[1], :]))).T
    return green

@jit
def r_force_bias(bra, ket, chol):
    green = r_green(bra, ket)
    fb = 2.0 * oe.contract("gpq,pq->g", chol, green, backend="jax")
    return fb

@jit
def r_rot_force_bias(bra, ket, rot_chol):
    green = r_half_green(bra, ket)
    fb = 2.0 * oe.contract("gpq,pq->g", rot_chol, green, backend="jax")
    return fb

@jit
def r_energy(
    bra: jax.Array, 
    ket: jax.Array, 
    h0: float, 
    h1:jax.Array, 
    chol: jax.Array
    ) -> jax.Array:
    '''
    h0 + h_pq <bra|a^dagger_p a_q|ket>/<bra|ket> 
    + 1/2 v_pqrs <bra|a^dagger_p a^dagger_q a_s a_r|ket>/<bra|ket>
    '''

    green = r_green(bra, ket)
    e1 = 2* oe.contract("pq,pq->", green, h1, backend="jax")

    # lg = oe.contract("gpr,qr->gpq", chol, green, backend="jax")
    # e2_1 = 2 * jnp.sum(oe.contract('gpp->g', lg, backend="jax")**2)
    # e2_2 = -oe.contract('gpq,gqp->',lg,lg, backend="jax")
    # e2 = e2_1 + e2_2

    def scan_chol(carry, x):
        chol_i = x
        gl_i = oe.contract("pr,qr->pq", green, chol_i, backend="jax")
        e2_c_i = 2 * oe.contract('pp->', gl_i, backend="jax")**2
        e2_e_i = -oe.contract('pq,qp->', gl_i, gl_i, backend="jax")
        carry += e2_c_i + e2_e_i
        return carry, 0
    
    e2, _ = lax.scan(scan_chol, 0.0, chol)
    energy = h0 + e1 + e2

    return energy

@jit
def r_rot_energy(
    bra: jax.Array, 
    ket: jax.Array, 
    h0: float, 
    rot_h1:jax.Array, 
    rot_chol: jax.Array
    ) -> jax.Array:
    '''
    h0 + h_pq <bra|a^dagger_p a_q|ket>/<bra|ket> 
    + 1/2 v_pqrs <bra|a^dagger_p a^dagger_q a_s a_r|ket>/<bra|ket>
    '''

    green = r_half_green(bra, ket)
    e1 = 2* oe.contract("pq,pq->", green, rot_h1, backend="jax")

    def scan_chol(carry, x):
        chol_i = x
        gl_i = oe.contract("pr,qr->pq", green, chol_i, backend="jax")
        e2_c_i = 2 * oe.contract('pp->', gl_i, backend="jax")**2
        e2_e_i = -oe.contract('pq,qp->', gl_i, gl_i, backend="jax")
        carry += e2_c_i + e2_e_i
        return carry, 0
    
    e2, _ = lax.scan(scan_chol, 0.0, rot_chol)
    energy = h0 + e1 + e2

    return energy