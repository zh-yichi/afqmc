import numpy as np
from jax import numpy as jnp
import jax

def df2chol(dferi, max_error=1e-6):
    """Modified Cholesky decomposition by Density Fit Tensor.

    Args:
        dferi (Array): Flattened Density Fitting 3-index integral of shape (N_aux, N_pair)
        max_error (float, optional): Maximum error allowed. Defaults to 1e-6.

    Returns:
        Array: Cholesky vectors of shape (N_chol, N_orb, N_orb).
    """
    n_aux, n_pair = dferi.shape
    diag = (dferi**2).sum(axis=0) 
    norb = int(((-1 + (1 + 8 * n_pair) ** 0.5) / 2))
    nchol_max = n_aux
    chol_vecs = np.zeros((nchol_max, n_pair))
    Mapprox = np.zeros(n_pair)
    diag_residual = diag.copy()
    
    nchol = 0
    while nchol < nchol_max:
        # Find the max error in the remaining diagonal
        nu = np.argmax(diag_residual)
        delta_max = diag_residual[nu]
        if delta_max < max_error:
            break
            
        # Compute the specific row of the full ERI matrix: V_{\nu P}
        # Matrix-vector multiply is faster than einsum here
        row_nu = dferi.T @ dferi[:, nu]
        
        if nchol == 0:
            chol_vecs[nchol] = row_nu / delta_max**0.5
        else:
            # R = sum of previous Cholesky contributions
            R = chol_vecs[:nchol, nu] @ chol_vecs[:nchol, :]
            chol_vecs[nchol] = (row_nu - R) / delta_max**0.5
            
        # Update the residual diagonal for the next iteration
        Mapprox += chol_vecs[nchol]**2
        diag_residual = np.abs(diag - Mapprox)
        nchol += 1

    chol0 = chol_vecs[:nchol]

    chol = np.zeros((nchol, norb, norb))
    
    row_idx, col_idx = np.tril_indices(norb)
    chol[:, row_idx, col_idx] = chol0
    chol[:, col_idx, row_idx] = chol0
    
    return chol

@jax.jit
def df2chol_gpu(dferi, max_error=1e-6):
    """
    JAX-compiled Cholesky decomposition.
    Note: Returns a zero-padded array of max size, plus the valid vector count.
    """
    n_aux, n_pair = dferi.shape
    diag = jnp.sum(dferi**2, axis=0)
    norb = int(((-1 + (1 + 8 * n_pair) ** 0.5) / 2))
    
    chol_vecs = jnp.zeros((n_aux, n_pair))
    Mapprox = jnp.zeros(n_pair)
    diag_residual = diag
    initial_max_val = jnp.max(diag_residual)

    # State tuple for the while loop:
    # (nchol, chol_vecs, Mapprox, diag_residual, max_error_val)
    init_state = (0, chol_vecs, Mapprox, diag_residual, initial_max_val)

    def cond_fun(state):
        nchol, _, _, _, max_val = state
        return jnp.logical_and(nchol < n_aux, max_val >= max_error)

    def body_fun(state):
        nchol, chol_vecs_loop, Mapprox_loop, diag_res_loop, _ = state

        # Find the next pivot
        nu = jnp.argmax(diag_res_loop)
        delta_max = diag_res_loop[nu]

        # Compute specific row
        row_nu = jnp.dot(dferi.T, dferi[:, nu])
        R = jnp.dot(chol_vecs_loop[:, nu], chol_vecs_loop)
        new_vec = (row_nu - R) / jnp.sqrt(jnp.maximum(delta_max, 1e-12))

        chol_vecs_loop = chol_vecs_loop.at[nchol].set(new_vec)
        Mapprox_loop = Mapprox_loop + new_vec**2
        diag_res_loop = jnp.abs(diag - Mapprox_loop)

        new_max_val = jnp.max(diag_res_loop)

        return (nchol + 1, chol_vecs_loop, Mapprox_loop, diag_res_loop, new_max_val)

    # Run the compiled loop
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    final_nchol, final_chol_vecs, _, _, _ = final_state

    # Reshape back to (N_max, N_orb, N_orb)
    chol_out = jnp.zeros((n_aux, norb, norb))
    row_idx, col_idx = jnp.tril_indices(norb)
    
    chol_out = chol_out.at[:, row_idx, col_idx].set(final_chol_vecs)
    chol_out = chol_out.at[:, col_idx, row_idx].set(final_chol_vecs)

    # We must return the integer nchol so you can slice it outside the JIT function
    return chol_out, final_nchol