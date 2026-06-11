"""
Tool kit for single slater determinant operations
Including:
    Thouless transformation
    Overlap, 
    Green's Function, 
    Force Bias,
    Energy
For both restricted and unrestricted determinant
"""

import jax
import jax.numpy as jnp
from jax import scipy as jsp
from jax import jit, lax
import opt_einsum as oe

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
    

@jit
def r_overlap(bra: jax.Array, ket: jax.Array):
    ''' 
    <bra|ket>
    '''
    olp = jnp.linalg.det(bra.T.conj() @ ket) ** 2
    return olp

@jit
def r_delta_overlap(bra: jax.Array, ket: jax.Array):
    ''' 
    <bra|ket> when bra is the identity
    '''
    nocc = ket.shape[1]
    olp = jnp.linalg.det(ket[:nocc,:nocc]) ** 2
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
    green = (ket @ (jnp.linalg.inv(bra.T.conj() @ ket))).T
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
    ):
    '''
    h0 + h_pq <bra|a^dagger_p a_q|ket>/<bra|ket> 
    + 1/2 v_pqrs <bra|a^dagger_p a^dagger_q a_s a_r|ket>/<bra|ket>
    chunk the cholesky to (nchunk, nchol_per_chunk, norb, norb)
    before calling this function
    '''

    green = r_green(bra, ket)
    e1 = 2* oe.contract("pq,pq->", green, h1, backend="jax")

    def scan_chol(carry, x):
        chol_c = x
        gl_c = oe.contract("pr,gqr->gpq", green, chol_c, backend="jax")
        e2_c_c = 2 * jnp.sum(oe.contract('gpp->g', gl_c, backend="jax")**2)
        e2_e_c = -oe.contract('gpq,gqp->', gl_c, gl_c, backend="jax")
        carry += e2_c_c + e2_e_c
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
    ):
    '''
    h0 + h_pq <bra|a^dagger_p a_q|ket>/<bra|ket> 
    + 1/2 v_pqrs <bra|a^dagger_p a^dagger_q a_s a_r|ket>/<bra|ket>
    chunk the rot_cholesky to (nchunk, nchol_per_chunk, nocc, norb)
    before calling this function
    '''

    green = r_half_green(bra, ket)
    e1 = 2* oe.contract("pq,pq->", green, rot_h1, backend="jax")

    def scan_chol(carry, x):
        chol_c = x
        gl_c = oe.contract("pr,gqr->gpq", green, chol_c, backend="jax")
        e2_c_i = 2 * jnp.sum(oe.contract('gpp->g', gl_c, backend="jax")**2)
        e2_e_i = -oe.contract('gpq,gqp->', gl_c, gl_c, backend="jax")
        carry += e2_c_i + e2_e_i
        return carry, 0
    
    e2, _ = lax.scan(scan_chol, 0.0, rot_chol)
    energy = h0 + e1 + e2

    return energy

# unrestricted

@jit
def u_overlap(bra: tuple, ket: tuple):
    olp = jnp.linalg.det(bra[0].T.conj() @ ket[0]) \
        * jnp.linalg.det(bra[1].T.conj() @ ket[1])
    return olp

@jit
def u_delta_overlap(bra: tuple, ket: tuple):
    # when bra is identity
    nocc_a, nocc_b = ket[0].shape[1], ket[1].shape[1]
    olp = jnp.linalg.det(ket[0][:nocc_a,:nocc_a]) \
        * jnp.linalg.det(ket[1][:nocc_b,:nocc_b])
    return olp

@jit
def u_half_green(bra: tuple, ket: tuple):
    green_a = (ket[0] @ (jnp.linalg.inv(bra[0].T.conj() @ ket[0]))).T
    green_b = (ket[1] @ (jnp.linalg.inv(bra[1].T.conj() @ ket[1]))).T
    return (green_a, green_b)

@jit
def u_delta_green(bra: tuple, ket: tuple):
    green_a = (ket[0].dot(jnp.linalg.inv(ket[0][:ket[0].shape[1],:]))).T
    green_b = (ket[1].dot(jnp.linalg.inv(ket[1][:ket[1].shape[1],:]))).T
    return (green_a, green_b)

@jit
def u_green(bra: tuple, ket: tuple):
    green_a = (ket[0] @ (jnp.linalg.inv(bra[0].T.conj() @ ket[0])) @ bra[0].T.conj()).T
    green_b = (ket[1] @ (jnp.linalg.inv(bra[1].T.conj() @ ket[1])) @ bra[1].T.conj()).T
    return (green_a, green_b)

@jit
def u_rot_force_bias(bra: tuple, ket: tuple, rot_chol: tuple):
    green = u_half_green(bra, ket)
    fb_a = oe.contract("gij,ij->g", rot_chol[0], green[0], backend="jax")
    fb_b = oe.contract("gij,ij->g", rot_chol[1], green[1], backend="jax")
    return fb_a + fb_b

@jit
def u_force_bias(bra: tuple, ket: tuple, chol: tuple):
    green = u_green(bra, ket)
    fb_a = oe.contract("gij,ij->g", chol[0], green[0], backend="jax")
    fb_b = oe.contract("gij,ij->g", chol[1], green[1], backend="jax")
    return fb_a + fb_b

@jit
def u_rot_energy(
    bra: tuple, 
    ket: tuple, 
    h0: tuple, 
    rot_h1: tuple, 
    rot_chol: tuple
    ):
    # rot_chol has to be chunked before calling this function

    green = u_half_green(bra, ket)
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
def u_energy(
    bra: tuple, 
    ket: tuple, 
    h0: tuple, 
    h1: tuple, 
    chol: tuple
    ):
    # chol_a and chol_b has to be chunked 
    # into shape (nchunk, nchol_chunk, nocc, norb)
    # before calling this function

    green = u_green(bra, ket)
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

# multislater bra implementation
def rms_overlap(bras, ket):

    def scan_slaters(carry, bra):
        olp = r_overlap(bra, ket)
        return carry, olp

    init_carry = 0.0
    _, olps = lax.scan(scan_slaters, init_carry, bras)

    olp = jnp.sum(olps) / bras.shape[0]

    return olp

def rms_force_bias(bras, ket, chol):

    def scan_slaters(carry, bra):
        olp = r_overlap(bra, ket)
        fb = r_force_bias(bra, ket, chol)
        return carry, (olp, fb)

    init_carry = 0.0
    _, (olps, fbs) = lax.scan(scan_slaters, init_carry, bras)

    olp = jnp.sum(olps)
    fb = oe.contract("s,sg->g", olps, fbs, backend="jax") / olp

    return fb

def rms_rot_force_bias(bras, ket, rot_chol):

    def scan_slaters(carry, bra):
        olp = r_overlap(bra, ket)
        fb = r_rot_force_bias(bra, ket, rot_chol)
        return carry, (olp, fb)

    init_carry = 0.0
    _, (olps, fbs) = lax.scan(scan_slaters, init_carry, bras)

    olp = jnp.sum(olps)
    fb = oe.contract("s,sg->g", olps,fbs, backend="jax") / olp

    return fb

def rms_energy(bras, ket, h0, h1, chol):

    def scan_slaters(carry, bra):
        olp = r_overlap(bra, ket)
        energy = r_energy(bra, ket, h0, h1, chol)
        return carry, (olp, energy)

    init_carry = 0.0
    _, (olps, energies) = lax.scan(scan_slaters, init_carry, bras)

    olp = jnp.sum(olps)
    energy = jnp.sum(olps*energies) / olp

    return energy

def rms_rot_energy(bras, ket, h0, rot_h1, rot_chol):

    def scan_slaters(carry, bra):
        olp = r_overlap(bra, ket)
        energy = r_rot_energy(bra, ket, h0, rot_h1, rot_chol)
        return carry, (olp, energy)

    init_carry = 0.0
    _, (olps, energies) = lax.scan(scan_slaters, init_carry, bras)

    olp = jnp.sum(olps)
    energy = jnp.sum(olps*energies) / olp

    return energy