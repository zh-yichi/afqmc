import jax
jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp

import h5py
import numpy as np

from pyscf import lib, scf

from afqmc import cholesky, integral

from functools import partial
print = partial(print, flush=True)

def write_integral(nelec, norb, h0, h1, chol, filename):
    with h5py.File(filename, "w") as fh5:
        fh5["nelec"] = nelec
        fh5["norb"] = norb
        fh5["h0"] = h0

        if isinstance(h1, (tuple, list)):
            fh5["h1a"] = h1[0]
            fh5["h1b"] = h1[1]
            fh5["chola"] = chol[0]
            fh5["cholb"] = chol[1]
        else:
            fh5["h1"] = h1
            fh5["chol"] = chol

def read_integral(filename):
    with h5py.File(filename, "r") as fh5:
        nelec_arr = fh5["nelec"][()]
        nelec = tuple(int(x) for x in nelec_arr)

        norb_arr = fh5["norb"][()]
        norb = int(norb_arr) if norb_arr.ndim == 0 else tuple(int(x) for x in norb_arr)

        h0 = float(fh5["h0"][()])

        if "h1a" in fh5:
            h1 = (jnp.asarray(fh5["h1a"][()]), jnp.asarray(fh5["h1b"][()]))
        else:
            h1 = jnp.asarray(fh5["h1"][()])

        if "chola" in fh5:
            chol = (jnp.asarray(fh5["chola"][()]), jnp.asarray(fh5["cholb"][()]))
        else:
            chol = jnp.asarray(fh5["chol"][()])

    return nelec, norb, h0, h1, chol

def get_hamiltonian(mf, 
                    norb_frozen=0, 
                    chol_cut=1e-5, 
                    basis_coeff=None, 
                    save2disk=False,
                    ham_file="FCIDUMP_chol"
                    ):

    mol = mf.mol
    nao = mf.mol.nao

    if basis_coeff is None:
        basis_coeff = mf.mo_coeff
    
    if getattr(mf, "with_df", None) is not None:
        print('Find Density Fit Teonsers in Mean-Field object')
        print('Hamltonian will be construncted with Density Fit')
        useDF = True
    else:
        useDF = False

    if isinstance(mf, scf.rhf.RHF):
        nbasis = nao - norb_frozen
        nocc = int(np.count_nonzero(mf.mo_occ))
        nelec = [nocc - norb_frozen, nocc - norb_frozen]
        h1e, h0 = integral.h1e_ras(mf, basis_coeff, nbasis, norb_frozen, useDF)
        chol_ao = cholesky.cholesky_by_mol(mol, max_error=chol_cut, cmax=10)
        chol_ao = jnp.array(chol_ao.reshape((-1, nao, nao)))
        chol = cholesky.cderi2mo_gpu(chol_ao, basis_coeff)
        chol = cholesky.unpack_symmetric(chol, nao)
        chol = chol[:, norb_frozen:, norb_frozen:]
            
    elif isinstance(mf, scf.uhf.UHF):
        ncore = np.array([norb_frozen, norb_frozen], dtype = np.int32)
        nocc = np.array([np.count_nonzero(mf.mo_occ[0]),
                         np.count_nonzero(mf.mo_occ[1])],
                         dtype = np.int32)
        nelec = nocc - norb_frozen
        ncas = nao - ncore
        nbasis = ncas
        h1e, h0 = integral.h1e_uas(mf, basis_coeff, ncas, ncore, useDF)

        chol_ao = cholesky.cholesky_by_mol(mol, max_error=chol_cut, cmax=10)
        chol_ao = jnp.array(chol_ao.reshape((-1, nao, nao)))
        chol_a = cholesky.cderi2mo_gpu(chol_ao, basis_coeff[0])
        chol_b = cholesky.cderi2mo_gpu(chol_ao, basis_coeff[1])
        chol_a = cholesky.unpack_symmetric(chol_a, nao)
        chol_b = cholesky.unpack_symmetric(chol_b, nao)
        chol_a = chol_a[:, ncore[0]:, ncore[0]:]
        chol_b = chol_b[:, ncore[1]:, ncore[1]:]
        chol = (chol_a, chol_b)
    
    if save2disk:
        write_integral(nelec, nbasis, h0, h1e, chol, ham_file)

    return nelec, nbasis, h0, h1e, chol