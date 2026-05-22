import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
import opt_einsum as oe

import h5py
import numpy as np
from typing import Optional, Union

from pyscf import lib, scf
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD

from afqmc import cholesky

from functools import partial
print = partial(print, flush=True)

# restricted below #
def get_rveff(mf, dm):
    '''restricted'''
    mol = mf.mol
    vj, vk = mf.get_jk(mol, dm, hermi=1)
    return 2*vj - vk

@jax.jit
def rjk_from_cderi(cderi, dm):
    '''restricted'''
    cderi_dm = oe.contract('gik,kj->gij', cderi, dm, backend='jax')
    vj = oe.contract('gkk,gij->ij', cderi_dm, cderi, backend='jax')
    vk = oe.contract('gik,gkj->ij', cderi_dm, cderi, backend='jax')
    return vj, vk

def get_rveff_df(mf, dm):
    '''restricted GPU accelerated'''
    dm = jnp.array(dm)
    vj = jnp.zeros(dm.shape)
    vk = jnp.zeros(dm.shape)
    for cderi in mf.with_df.loop():
        cderi = jnp.array(lib.unpack_tril(cderi, axis=-1))
        dvj, dvk = rjk_from_cderi(cderi, dm)
        vj += dvj
        vk += dvk
    return 2*vj - vk

def h1e_ras(mf, mo_coeff, ncas, ncore, useDF=False):
    '''
    effective one-electron integral for restricted active space
    ncas = nact_electron/2
    ncore = ncore_electrons/2
    '''
    # note casci undo DF

    mo_core = jnp.array(mo_coeff[:,:ncore])
    mo_cas = jnp.array(mo_coeff[:,ncore:ncore+ncas])

    hcore = jnp.array(mf.get_hcore())
    energy_core = mf.energy_nuc()

    if mo_core.size == 0:
        corevhf = 0.
    else:
        core_dm = mo_core @ mo_core.T

        if useDF:
            corevhf = get_rveff_df(mf, core_dm) # GPU Accelerated
        else:
            corevhf = get_rveff(mf, core_dm)
        
        energy_core += 2 * oe.contract('ij,ji', core_dm, hcore, backend='jax')
        energy_core += oe.contract('ij,ji', core_dm, corevhf, backend='jax')

    h1eff = mo_cas.T @ (hcore+corevhf) @ mo_cas

    return h1eff, energy_core

# unrestricted below #

def get_uveff(mf, dm):
    # dm = np.array(dm)
    mol = mf.mol
    vj, vk = mf.get_jk(mol, dm, hermi=1)
    return vj[0]+vj[1] - vk

@jax.jit
def ujk_from_cderi(cderi, dm_a, dm_b):
    """
    cderi : (g, nao, nao)
    dm_a  : (nao, nao)
    dm_b  : (nao, nao)
    """
    # dm_a, dm_b = dm
    dm_tot = dm_a + dm_b # Coulomb uses total density

    cderi_dm_tot = oe.contract('gik,kj->gij', cderi, dm_tot, backend='jax')
    vj = oe.contract('gkk,gij->ij', cderi_dm_tot, cderi, backend='jax')

    cderi_dm_a = oe.contract('gik,kj->gij', cderi, dm_a, backend='jax')
    cderi_dm_b = oe.contract('gik,kj->gij', cderi, dm_b, backend='jax')

    vk_a = oe.contract('gik,gkj->ij', cderi_dm_a, cderi, backend='jax')
    vk_b = oe.contract('gik,gkj->ij', cderi_dm_b, cderi, backend='jax')

    return vj, vk_a, vk_b

def get_uveff_df(mf, dm):
    '''unrestricted GPU accelerated'''
    dm_a, dm_b = dm
    dm_a = jnp.array(dm_a)
    dm_b = jnp.array(dm_b)
    
    vj = jnp.zeros_like(dm_a)
    vk_a = jnp.zeros_like(dm_a)
    vk_b = jnp.zeros_like(dm_b)

    print('Building JK matrix')
    for cderi in mf.with_df.loop():
        cderi = jnp.array(lib.unpack_tril(cderi, axis=-1))
        dvj, dvk_a, dvk_b = ujk_from_cderi(cderi, dm_a, dm_b)
        vj += dvj
        vk_a += dvk_a
        vk_b += dvk_b

    return jnp.array([vj - vk_a, vj - vk_b])

def h1e_uas(mf, mo_coeff, ncas, ncore, useDF=False):
    '''
    effective one-electron integral for unrestricted active space
    ncas = (ncas_a, ncas_b) size of active space
    ncore = (ncore_a, ncore_b) number of core electrons
    '''
    # mf = mf.undo_df() ucasci undo DF

    mo_core = [jnp.array(mo_coeff[0][:,:ncore[0]]),
               jnp.array(mo_coeff[1][:,:ncore[1]])]
    mo_cas = [jnp.array(mo_coeff[0][:,ncore[0]:ncore[0]+ncas[0]]),
              jnp.array(mo_coeff[1][:,ncore[1]:ncore[1]+ncas[1]])]

    hcore = mf.get_hcore()
    hcore = [jnp.array(hcore), jnp.array(hcore)]
    energy_core = mf.energy_nuc()
    if mo_core[0].size == 0 and mo_core[1].size == 0:
        corevhf = (0,0)
    else:
        core_dm = jnp.array([mo_core[0] @ mo_core[0].T, 
                            mo_core[1] @ mo_core[1].T])
        
        if useDF:
            corevhf = get_uveff_df(mf, core_dm) # GPU Accelerated
        else:
            corevhf = get_uveff(mf, core_dm)

        energy_core += oe.contract('ij,ji', core_dm[0], hcore[0], backend='jax')
        energy_core += oe.contract('ij,ji', core_dm[1], hcore[1], backend='jax')
        energy_core += oe.contract('ij,ji', core_dm[0], corevhf[0], backend='jax') * .5
        energy_core += oe.contract('ij,ji', core_dm[1], corevhf[1], backend='jax') * .5
        # time2 = time.perf_counter()
    h1eff = [jnp.array(mo_cas[0].T @ (hcore[0]+corevhf[0]) @ mo_cas[0]),
             jnp.array(mo_cas[1].T @ (hcore[1]+corevhf[1]) @ mo_cas[1])]
    # time3 = time.perf_counter()
    # print(f"build JK time: {time1 - time0:.6f} s")
    # print(f"build ecore time: {time2 - time1:.6f} s")
    # print(f"build h1eff time: {time3 - time0:.6f} s")
    return h1eff, energy_core

def prjmo(prj, s1e, mo):
    # prj and reconstruct mo
    # e.g. |B_p> = |A_q><A_q|B_p>
    #            = C^A_mq C^A(T)_qn|m><n|s> C^B_sp
    mo_rec = prj @ prj.T @ s1e @ mo
    return mo_rec

def common_as(mf, mo_coeff, ncas, ncore, torr=1e-10):
    print("Constracting common Active Space (coAS) that span both Alpha and Beta active space")
    # time0 = time.perf_counter()
    s1e = mf.get_ovlp()
    mo_acta = mo_coeff[0][:,ncore[0]:ncore[0]+ncas[0]]
    mo_actb = mo_coeff[1][:,ncore[1]:ncore[1]+ncas[1]]
    mo_actaa = mo_coeff[0].T @ s1e @ mo_acta # proj to the complete
    mo_actba = mo_coeff[0].T @ s1e @ mo_actb # alpha basis for orthogonal
    cmo_act = np.hstack([mo_actaa, mo_actba]) # common active lno
    print('Naive coAS Shape: ', cmo_act.shape)
    # full_matrices = False gives u that just span the space of clno_act
    u, s, _ = np.linalg.svd(cmo_act, full_matrices=False)
    print(f'Orthonormalize coAS shape: {u.shape}')
    print(f'Smallest coAS SVD Singular values: {s[-1]}')
    print(f"coAS projection torr: {torr}")
    for idx in range(mo_acta.shape[1],u.shape[1]+1):
        prj = mo_coeff[0] @ u[:,:idx]
        prj_acta = prjmo(prj,s1e,mo_actb)
        prj_actb = prjmo(prj,s1e,mo_acta)
        losa = abs(prj_acta - mo_actb).max()
        losb = abs(prj_actb - mo_acta).max()
        if losa < torr and losb < torr:
            break
    print(f"Minimum size of coAS to span both Alpha and Beta LAS: {idx}")
    print(f"cLAS projection loss: ({losa:.2e}, {losb:.2e})")
    # span{|C>} = span{|A>} U span{|B>}
    cas_coeff = mo_coeff[0] @ u[:,:idx] # in ao
    print('True Common coAS Shape: ', cas_coeff.shape)
    a2c = cas_coeff.T @ s1e @ mo_acta # <C|A>
    b2c = cas_coeff.T @ s1e @ mo_actb # <C|B>

    return cas_coeff, a2c, b2c


def prep_integral(
    mf_or_cc: Union[scf.rhf.RHF, scf.uhf.UHF, CCSD, UCCSD],
    basis_coeff: Optional[np.ndarray] = None,
    norb_frozen: int = 0,
    chol_cut: float = 1e-5,
    amp_file = "amplitudes.npz",
    chol_file = "FCIDUMP_chol"
):

    print("\nPreparing AFQMC calculation")

    if isinstance(mf_or_cc, (CCSD, UCCSD)):
        mf = mf_or_cc._scf
        cc = mf_or_cc
        if cc.frozen is not None:
            norb_frozen = cc.frozen
        if isinstance(cc, UCCSD):
            # spin_type = 'unrestricted'
            t1a = np.array(cc.t1[0])
            t1b = np.array(cc.t1[1])
            t2aa, t2ab, t2bb = cc.t2
            t2aa = (t2aa - t2aa.transpose(0, 1, 3, 2)) / 2
            t2bb = (t2bb - t2bb.transpose(0, 1, 3, 2)) / 2
            t2aa = t2aa.transpose(0, 2, 1, 3)
            t2bb = t2bb.transpose(0, 2, 1, 3)
            t2ab = t2ab.transpose(0, 2, 1, 3)
            np.savez(
                amp_file,
                t1a=t1a,
                t1b=t1b,
                t2aa=t2aa,
                t2ab=t2ab,
                t2bb=t2bb,
            )
        elif isinstance(cc, CCSD):
            # spin_type = 'restricted'
            t2 = cc.t2
            t2 = t2.transpose(0, 2, 1, 3)
            t1 = np.array(cc.t1)
            np.savez(amp_file, t1=t1, t2=t2)
    else:
        mf = mf_or_cc

    if isinstance(mf, scf.rhf.RHF):
        spin_type = 'restricted'
    elif isinstance(mf, scf.uhf.UHF):
        spin_type = 'unrestricted'

    mol = mf.mol
    nao = mf.mol.nao

    if basis_coeff is None:
        basis_coeff = mf.mo_coeff

    print("Calculating Cholesky integrals")
    
    if getattr(mf, "with_df", None) is not None:
        print('Find Density Fit Teonsers in MF object')
        print('Integrals will be built by DF Tensors')
        useDF = True
    else:
        useDF = False

    if spin_type == 'restricted':

        nbasis = nao - norb_frozen
        nocc = int(np.count_nonzero(mf.mo_occ))
        nelec = [nocc - norb_frozen, nocc - norb_frozen]
        h1e, enuc = h1e_ras(mf, basis_coeff, nbasis, norb_frozen, useDF)
        chol_ao = cholesky.cholesky_by_mol(mol, max_error=chol_cut, cmax=10)
        chol_ao = jnp.array(chol_ao.reshape((-1, nao, nao)))
        chol = cholesky.cderi2mo_gpu(chol_ao, basis_coeff)
        chol = cholesky.unpack_symmetric(chol, nao)
        chol = chol[:, norb_frozen:, norb_frozen:]

        v0 = 0.5 * oe.contract("gpr,gqr->pq", chol, chol, backend="jax")
        h1e_mod = h1e - v0
        chol = chol.reshape((chol.shape[0], -1))
            
    elif spin_type == 'unrestricted':

        ncore = np.array([norb_frozen, norb_frozen], dtype = np.int32)
        nocc = np.array([np.count_nonzero(mf.mo_occ[0]),
                         np.count_nonzero(mf.mo_occ[1])],
                         dtype = np.int32)
        nelec = nocc - norb_frozen
        ncas = nao - ncore
        nbasis = ncas[0]
        h1e, enuc = h1e_uas(mf, basis_coeff, ncas, ncore, useDF)

        chol_ao = cholesky.cholesky_by_mol(mol, max_error=chol_cut, cmax=10)
        chol_ao = jnp.array(chol_ao.reshape((-1, nao, nao)))
        chol_a = cholesky.cderi2mo_gpu(chol_ao, basis_coeff[0])
        chol_b = cholesky.cderi2mo_gpu(chol_ao, basis_coeff[1])
        chol_a = cholesky.unpack_symmetric(chol_a, nao)
        chol_b = cholesky.unpack_symmetric(chol_b, nao)
        print(f"Alpha Cholesky shape: {chol_a.shape} ")
        print(f" Beta Cholesky shape: {chol_b.shape} ")

        chol_a = chol_a[:, ncore[0]:, ncore[0]:]
        chol_b = chol_b[:, ncore[1]:, ncore[1]:]
        v0_a = 0.5 * oe.contract("gpr,gqr->pq", chol_a, chol_a, backend="jax")
        v0_b = 0.5 * oe.contract("gpr,gqr->pq", chol_b, chol_b, backend="jax")
        h1e = jnp.array(h1e)
        h1e_mod = jnp.array(h1e - jnp.array([v0_a,v0_b]))
        chol = jnp.array([chol_a.reshape(chol_a.shape[0], -1), chol_b.reshape(chol_b.shape[0], -1)])

    print("Finished calculating Cholesky integrals")
    print("Size of the correlation space:")
    print(f"Number of electrons:        {nelec}")
    print(f"Number of basis functions:  {nbasis}")
    print(f"Number of Cholesky vectors: {chol.shape[-2]}")

    write_integral(
        enuc=enuc,
        hcore=h1e,
        hcore_mod=h1e_mod,
        chol=chol,
        nelec=sum(nelec),
        nmo=nbasis,
        ms=mol.spin,
        spin_type=spin_type,
        filename=chol_file,
    )

def write_integral(enuc, hcore, hcore_mod, chol,
                   nelec, nmo, ms, spin_type,
                   filename="FCIDUMP_chol",):
    
    with h5py.File(filename, "w") as fh5:
        fh5["header"] = np.array([nelec, nmo, ms])
        fh5["spin_type"] = spin_type
        fh5["hcore"] = hcore.flatten()
        fh5["hcore_mod"] = hcore_mod.flatten()
        fh5["chol"] = chol.flatten()
        fh5["energy_core"] = enuc