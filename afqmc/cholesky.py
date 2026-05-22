import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # CUDA async allocator

import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
import opt_einsum as oe

from pyscf import lib
import numpy as np

def modified_cholesky(mat: np.ndarray, max_error: float = 1e-5) -> np.ndarray:
    """Modified cholesky decomposition for a given matrix.

    Args:
        mat (np.ndarray): Matrix to decompose.
        max_error (float, optional): Maximum error allowed. Defaults to 1e-6.

    Returns:
        np.ndarray: Cholesky vectors.
    """
    diag = mat.diagonal()
    size = mat.shape[0]
    nchol_max = size
    chol_vecs = np.zeros((nchol_max, nchol_max))
    # ndiag = 0
    nu = np.argmax(diag)
    delta_max = diag[nu]
    Mapprox = np.zeros(size)
    chol_vecs[0] = np.copy(mat[nu]) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error and (nchol + 1) < nchol_max:
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        R = np.dot(chol_vecs[: nchol + 1, nu], chol_vecs[: nchol + 1, :])
        chol_vecs[nchol + 1] = (mat[nu] - R) / (delta_max + 1e-10) ** 0.5
        nchol += 1

    return chol_vecs[:nchol]

def cholesky_by_mol(mol, max_error=1e-5, cmax=10):
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

    Only works for molecular systems.

    Parameters
    ----------
    mol : :class:`pyscf.mol`
        pyscf mol object.
    orthoAO: :class:`numpy.ndarray`
        Orthogonalising matrix for AOs. (e.g., mo_coeff).
    delta : float
        Accuracy desired.
    cmax : int
        nchol = cmax * M, where M is the number of basis functions.
        Controls buffer size for cholesky vectors.

    Returns
    -------
    chol_vecs : :class:`numpy.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    nao = mol.nao_nr()
    diag = np.zeros(nao * nao)
    nchol_max = cmax * nao
    # This shape is more convenient for pauxy.
    chol_vecs = np.zeros((nchol_max, nao * nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0, mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2 * l + 1) * nc
        dims.append(nao_per_i)
    # print (dims)
    for i in range(0, mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        buf = mol.intor("int2e_sph", shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag : ndiag + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
        ndiag += di * nao
    nu = np.argmax(diag)
    delta_max = diag[nu]
    # if verbose:
    #     print("# Generating Cholesky decomposition of ERIs." % nchol_max)
    #     print("# max number of cholesky vectors = %d" % nchol_max)
    #     print("# iteration %5d: delta_max = %f" % (0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = np.searchsorted(dims, j)
    sl = np.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = np.zeros(nao * nao)
    # ERI[:,jl]
    eri_col = mol.intor(
        "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
    )
    cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
    chol_vecs[0] = np.copy(eri_col[:, :, cj, cl].reshape(nao * nao)) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = np.searchsorted(dims, j)
        sl = np.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor(
            "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
        )
        # Select correct ERI chunk from shell.
        cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
        Munu0 = eri_col[:, :, cj, cl].reshape(nao * nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = np.dot(chol_vecs[: nchol + 1, nu], chol_vecs[: nchol + 1, :])
        chol_vecs[nchol + 1] = (Munu0 - R) / (delta_max) ** 0.5
        nchol += 1

    return chol_vecs[:nchol]

# common stuff 
def pack_symmetric(L):
    """
    L: shape (g, n, n), symmetric in last two indices
    returns: shape (g, n*(n+1)//2)
    """
    n = L.shape[-1]
    iu = jnp.tril_indices(n)
    return L[:, iu[0], iu[1]]

def unpack_symmetric(L_packed, n):
    """
    L_packed: shape (g, n*(n+1)//2), lower-tri packed with jnp.tril_indices order
    returns: shape (g, n, n), symmetric
    """
    g = L_packed.shape[0]
    iu = jnp.tril_indices(n)
    L = jnp.zeros((g, n, n), dtype=L_packed.dtype)
    L = L.at[:, iu[0], iu[1]].set(L_packed)        # lower triangle
    L = L.at[:, iu[1], iu[0]].set(L_packed)        # mirror to upper (diag rewritten with same values, fine)
    return L

@jax.jit
def cderi2mo_gpu(cderi, mo_coeff):
    cderi_mo = oe.contract('pr,grs,sq->gpq', mo_coeff.T, cderi, mo_coeff, backend='jax')
    return pack_symmetric(cderi_mo)

def cderi2mo_cpu(cderi, mo_coeff):
    cderi_mo = lib.einsum('pr,grs,sq->gpq', mo_coeff.T, cderi, mo_coeff, optimize='optimal')
    return pack_symmetric(cderi_mo)

def compress_cderi_cpu(cderi, thresh=1e-6):
    """
    Perform SVD on cderi (CPU) and keep components with s^2 > thresh.

    Parameters
    ----------
    cderi : np.ndarray (naux, npair)
    thresh : float |Threshold on squared singular values sqaure

    Returns
    -------
    compressed cderi: np.ndarray
    """
    _, s, Vh = np.linalg.svd(cderi, full_matrices=False)
    mask = s**2 > thresh

    s = s[mask]
    Vh = Vh[mask, :]
    cp_cderi = s[:, None] * Vh

    return cp_cderi

@jax.jit
def _svd_gpu(cderi):
    return jnp.linalg.svd(cderi, full_matrices=False)

# @jax.jit
def compress_cderi_gpu(cderi, thresh=1e-6):
    """
    Perform SVD on cderi (GPU via JAX) and keep components with s^2 > thresh.

    Parameters
    ----------
    cderi : jnp.ndarray
        Input matrix (m, n) already on GPU
    thresh : float
        Threshold on squared singular values

    Returns
    -------
    cp_cderi : jnp.ndarray
        Compressed cderi (k, n)
    """

    # SVD on GPU
    _, s, Vh = _svd_gpu(cderi)

    # singular values are sorted descending
    mask = s**2 > thresh
    s = s[mask]

    Vh = Vh[mask, :]
    cp_cderi = s[:, None] * Vh

    return cp_cderi

def df2chol_cpu(dferi, max_error=1e-6):
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

def chunk_chol(chol, nchol_chunk_init = None, memory = None):
    '''
    chunk the cholesky vectors (nchol, norb, norb). 
    The size of nchunk is determined by allowed memory. 
    nchunk * nchol_chunk not necessarily = nchol. 
    The Cholesky vectors maybe padded minimumly s.t. 
    nchol_pad / nchunk <= nchol per chunk allowed

    `memory` takes precedence when both arguments are given.

    Input
        chol:             Cholesky vectors, shape (nchol, norb, norb)
                          or (2, nchol, norb, norb) for unrestricted spin
        nchol_chunk_init: fallback chunk size when `memory` is not set
        memory:           allowed memory per walker in MB (takes precedence
                          over nchol_chunk_init when both are set)

    Return
        nchunk:      number of chunks
        nchol_chunk: number of cholesky vectors per chunk
    '''
    if nchol_chunk_init is None and memory is None:
        raise ValueError("Specify at least one of `nchol_chunk` or `memory`.")
    
    if chol.ndim == 3: # (nchol, norb, norb)
        spin_factor = 1
        nchol, m1, m2 = chol.shape
    elif chol.ndim == 4: # (2, nchol, norb, norb)
        spin_factor, nchol, m1, m2 = chol.shape
    else:
        raise ValueError(
            f"chol must be 3D or 4D, got shape {chol.shape}."
        )

    if memory is not None:
        print(f"Maximum memory per walker:            {memory:.2f}") # (MB)
        bytes_per_element = 16  # complex128
        bytes_per_vector = m1 * m2 * bytes_per_element
        max_chunk_size = int(memory * 1024**2 // bytes_per_vector)
    else:
        max_chunk_size = nchol_chunk_init

    max_chunk_size = max_chunk_size // spin_factor

    if max_chunk_size < 1:
        raise ValueError(
            f"Chunk size after spin_factor={spin_factor} division is < 1. "
            f"Increase memory budget or nchol_chunk "
            f"(norb1={m1}, norb2={m2}, bytes_per_vector="
            f"{m1 * m2 * 16 / 1024**2:.3f} MB)."
        )

    nchunk = -(-nchol // max_chunk_size)   # ceil(Ng / max_chunk_size)
    nchol_chunk = -(-nchol // nchunk)           # ceil(Ng / Nc)
    npad = nchunk * nchol_chunk - nchol

    print(f"Maximum number of Cholesky per chunk: {max_chunk_size}")
    print(f"Number of Cholesky chunks:            {nchunk}")
    print(f"Number of Cholesky per chunk:         {nchol_chunk}")
    print(f"Number of padding Cholesky:           {npad}")

    return nchol_chunk