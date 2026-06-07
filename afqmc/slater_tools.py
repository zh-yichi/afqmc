import h5py
import pickle

import numpy as np

import opt_einsum as oe

import jax
import jax.numpy as jnp
from jax import scipy as jsp
from jax import jit, lax, random, vmap

@jit
def rthouless(init_slater, t):
    '''
    restricted thouless transformation
    |psi'> = exp(t_ia a+ i)|psi>
    use the block form of t, no need to apply full exp(t)
    equivalent to the function below
    '''
    # norb, nocc = self.norb, self.nelec[0]
    # nvir = norb - nocc
    nocc, nvir = t.shape
    norb = nocc + nvir
    assert init_slater.shape == (norb, nocc)
    t_full = jnp.eye(norb, dtype=jnp.complex128)
    exp_t = t_full.at[:nocc, nocc:].set(t)
    return exp_t.T @ init_slater

@jit
def rthouless_full(init_slater, t):
    '''
    restricted thouless transformation
    |psi'> = exp(t_ia a+ i)|psi>
    apply full exp(t)
    equivalent to the function above
    '''
    nocc, nvir = t.shape
    norb = nocc + nvir
    assert init_slater.shape == (norb, nocc)
    t_full = jnp.zeros((norb, norb), dtype=jnp.complex128)
    t_full = t_full.at[:nocc, nocc:].set(t)
    exp_t = jsp.linalg.expm(t_full)
    return exp_t.T @ init_slater

@jit
def uthouless(slater, tau):
    # calculate |psi'> = exp(t_ia a+ i)|psi>
    
    slater_a, slater_b = slater
    ta, tb = tau
    nocc_a, nvir_a = ta.shape
    nocc_b, nvir_b = tb.shape
    
    norb_a = nocc_a + nvir_a
    norb_b = nocc_b + nvir_b
    
    assert norb_a == norb_b
    norb = norb_a
    
    assert slater_a.shape == (norb, nocc_a)
    assert slater_b.shape == (norb, nocc_b)

    ta_full = jnp.eye(norb, dtype=jnp.complex128)
    tb_full = jnp.eye(norb, dtype=jnp.complex128)
    exp_ta = ta_full.at[:nocc_a, nocc_a:].set(ta)
    exp_tb = tb_full.at[:nocc_b, nocc_b:].set(tb)

    slater_ta = exp_ta.T @ slater_a
    slater_tb = exp_tb.T @ slater_b

    return [slater_ta, slater_tb]

def thouless(slater, tau):
    if isinstance(slater, jax.Array) and len(slater.shape) == 2:
        return rthouless(slater, tau)
    elif isinstance(slater, (tuple, list)) and isinstance(tau, (tuple, list)):
        return uthouless(slater, tau)