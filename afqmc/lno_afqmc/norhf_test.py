from jax import numpy as jnp
from jax import vmap, jit, lax, random
import jax
from ad_afqmc.lno.base import lno
from ad_afqmc.lno_afqmc import lno_maker, afqmc_maker, lnoafqmc_runner, data_maker
import numpy as np
from pyscf import lib, ao2mo
from functools import partial
from ad_afqmc import wavefunctions, propagation, sampling
from typing import Tuple
import os

@jax.jit
def _walker_rhf_efock(walker,ham_data,wave_data):
    '''C_ia <psi|H|psi_i^a}>'''
    trial_mo = wave_data['mo_coeff']
    f_ia = ham_data['fock_ia']
    nocc = f_ia.shape[0]
    gw = (walker.dot(jnp.linalg.inv(trial_mo.T.conj() @ walker))).T
    gw_ia = gw[:nocc,nocc:]
    e_fock = 2*jnp.einsum('ia,ia->',gw_ia,f_ia)
    return e_fock

@jax.jit
def _walker_rhf_coexc(walker,ham_data,wave_data):
    '''C_ijab <psi|H|psi_{ia}^{jb}>'''
    rot_h1,rot_chol = ham_data['rot_h1'],ham_data['rot_chol']
    trial_mo = wave_data['mo_coeff']
    nocc = rot_h1.shape[0]
    gw = (walker.dot(jnp.linalg.inv(trial_mo.T.conj() @ walker))).T
    f = jnp.einsum('gja,ai->gji',rot_chol[:,:nocc,nocc:],
                    gw.T[nocc:,:nocc],optimize='optimal')
    tr_f = vmap(jnp.trace)(f)
    e_col = 2*jnp.einsum('g,g->',tr_f,tr_f)
    e_exc = -jnp.einsum('gij,gji->',f,f)
    return e_col+e_exc

@jax.jit
def _walker_norhf_energy(walker,ham_data,wave_data):
    '''
    C_ia <psi|H|psi_i^a}> + C_ijab <psi|H|psi_{ia}^{jb}> 
    '''
    trial_mo = wave_data['mo_coeff']
    f_ia = ham_data['fock_ia']
    rot_h1,rot_chol = ham_data['rot_h1'],ham_data['rot_chol']
    nocc = rot_h1.shape[0]
    gw = (walker.dot(jnp.linalg.inv(trial_mo.T.conj() @ walker))).T
    gw_ia = gw[:nocc,nocc:]
    cg = jnp.einsum('gja,ai->gji',rot_chol[:,:nocc,nocc:],
                    gw_ia.T,optimize='optimal')
    tr_cg = vmap(jnp.trace)(cg)
    e_fock = 2*jnp.einsum('ia,ia->',gw_ia,f_ia)
    e_col = 2*jnp.einsum('g,g->',tr_cg,tr_cg)
    e_exc = -jnp.einsum('gij,gji->',cg,cg)
    e_corr = e_fock+e_col+e_exc
    return e_corr

@jax.jit
def _walker_rhf_efock_orb(walker,ham_data,wave_data):
    '''C_ia <psi|H|psi_i^a}> don't sum over i'''
    m = jnp.dot(wave_data["prjlo"].T,wave_data["prjlo"])
    f_ia = ham_data['fock_ia']
    nocc = f_ia.shape[0]
    trial_mo = wave_data['mo_coeff']
    gw = (walker.dot(jnp.linalg.inv(trial_mo.T.conj() @ walker))).T
    gw_ia = gw[:nocc,nocc:]
    e_fock = 2*jnp.einsum('ia,ka,ik->',gw_ia,f_ia,m)
    return e_fock

@jax.jit
def _walker_rhf_coexc_orb(walker,ham_data,wave_data):
    '''C_ijab <psi|H|psi_{ia}^{jb}> don't sum over i'''
    m = jnp.dot(wave_data["prjlo"].T,wave_data["prjlo"])
    trial_mo = wave_data['mo_coeff']
    rot_h1,rot_chol = ham_data['rot_h1'],ham_data['rot_chol']
    nocc = rot_h1.shape[0]
    gw = (walker.dot(jnp.linalg.inv(trial_mo.T.conj() @ walker))).T
    f = jnp.einsum('gja,ai->gji',rot_chol[:,:nocc,nocc:],
                    gw.T[nocc:,:nocc],optimize='optimal')
    tr_f = vmap(jnp.trace)(f)
    e_col = 2*jnp.einsum('gik,ik,g->',f,m,tr_f)
    e_exc = -jnp.einsum('gij,gjk,ik->',f,f,m)
    return e_col+e_exc

@jax.jit
def _walker_norhf_orbenergy(walker,ham_data,wave_data):
    '''
    C_ia <psi|H|psi_i^a}> + C_ijab <psi|H|psi_{ia}^{jb}> 
    don't sum over i
    '''
    m = jnp.dot(wave_data["prjlo"].T,wave_data["prjlo"])
    trial_mo = wave_data['mo_coeff']
    f_ia = ham_data['fock_ia']
    rot_h1,rot_chol = ham_data['rot_h1'],ham_data['rot_chol']
    nocc = rot_h1.shape[0]
    gw = (walker.dot(jnp.linalg.inv(trial_mo.T.conj() @ walker))).T
    gw_ia = gw[:nocc,nocc:]
    cg = jnp.einsum('gja,ai->gji',rot_chol[:,:nocc,nocc:],
                    gw_ia.T,optimize='optimal')
    tr_cg = vmap(jnp.trace)(cg)
    e_fock = jnp.real(2*jnp.einsum('ia,ka,ik->',gw_ia,f_ia,m))
    e_col = jnp.real(2*jnp.einsum('gik,ik,g->',cg,m,tr_cg))
    e_exc = jnp.real(-jnp.einsum('gij,gjk,ik->',cg,cg,m))
    e_corr = e_fock+e_col+e_exc
    return e_corr

@partial(jit, static_argnums=(2,))
def _walker_rhf_orbenergy(walker,ham_data,trial,wave_data):
    '''hf orbital correlation energy'''
    rot_h1, rot_chol = ham_data['rot_h1'],ham_data['rot_chol']
    eorb = wavefunctions.rhf(
        trial.norb, trial.nelec,n_batch=trial.n_batch
            )._calc_orbenergy(
                0,rot_h1,rot_chol,walker,wave_data,orbE=0
                )
    return jnp.real(eorb)

@partial(jit, static_argnums=(3,))
def walker_rhf_orbenergy(walkers,ham_data,wave_data,trial):
    orb_en = vmap(_walker_rhf_orbenergy,
                     in_axes=(0,None,None,None))(
                         walkers,ham_data, trial, wave_data
                         )
    return orb_en

@jax.jit
def walker_norhf_orbenergy(walkers,ham_data,wave_data):
    orb_en = vmap(_walker_norhf_orbenergy, in_axes=(0, None, None))(
        walkers, ham_data, wave_data)
    return orb_en

#### test norhf #######
@jax.jit
def _walker_norhf_orbenergy_test(walker,ham_data,wave_data):
    '''
    C_ia <psi|H|psi_i^a}> + C_ijab <psi|H|psi_{ia}^{jb}> 
    don't sum over i
    '''
    m = jnp.dot(wave_data["prjlo"].T,wave_data["prjlo"])
    trial_mo = wave_data['mo_coeff']
    f_ia = ham_data['fock_ia']
    rot_h1,rot_chol = ham_data['rot_h1'],ham_data['rot_chol']
    nocc = rot_h1.shape[0]
    gw = (walker.dot(jnp.linalg.inv(trial_mo.T.conj() @ walker))).T
    gw_ia = gw[:nocc,nocc:]
    cg = jnp.einsum('gja,ai->gji',rot_chol[:,:nocc,nocc:],
                    gw_ia.T,optimize='optimal')
    tr_cg = vmap(jnp.trace)(cg)
    e_fock = jnp.real(2*jnp.einsum('ia,ka,ik->',gw_ia,f_ia,m))
    e_col = jnp.real(2*jnp.einsum('gik,ik,g->',cg,m,tr_cg))
    e_exc = jnp.real(-jnp.einsum('gij,gjk,ik->',cg,cg,m))
    e_corr = e_fock+e_col+e_exc
    return e_corr,e_fock,e_col+e_exc

@jax.jit
def walker_norhf_orbenergy_test(walkers,ham_data,wave_data):
    orb_en,orb1,orb2 = vmap(_walker_norhf_orbenergy_test, in_axes=(0, None, None))(
        walkers, ham_data, wave_data)
    return orb_en,orb1,orb2

@partial(jit, static_argnums=(2,3,5))
def block_orb_norhf_test(
    prop_data: dict,
    ham_data: dict,
    prop: propagation.propagator,
    trial: wavefunctions,
    wave_data: dict,
    sampler: sampling.sampler,
    ):
    """Block scan function. Propagation and orbital_i energy calculation."""
    prop_data["key"], subkey = random.split(prop_data["key"])
    fields = random.normal(
        subkey,
        shape=(
            sampler.n_prop_steps,
            prop.n_walkers,
            ham_data["chol"].shape[0],
        ),
    )

    _step_scan_wrapper = lambda x, y: sampler._step_scan(
        x, y, ham_data, prop, trial, wave_data
    )
    prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
    prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
        prop_data["weights"]
    )
    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
    
    orb_en,orb_fk,orb_cx \
        = walker_norhf_orbenergy_test(
        prop_data["walkers"],ham_data,wave_data)

    energy_samples = jnp.real(
        trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
    )
    energy_samples = jnp.where(
        jnp.abs(energy_samples - prop_data["e_estimate"]) > jnp.sqrt(2.0 / prop.dt),
        prop_data["e_estimate"],
        energy_samples,
    )

    blk_wt = jnp.sum(prop_data["weights"])
    blk_en = jnp.sum(energy_samples * prop_data["weights"]) / blk_wt
    blk_eorb= jnp.sum(orb_en * prop_data["weights"]) / blk_wt
    blk_orbfk = jnp.sum(orb_fk * prop_data["weights"]) / blk_wt
    blk_orbcx = jnp.sum(orb_cx * prop_data["weights"]) / blk_wt
    
    prop_data["pop_control_ene_shift"] = (
        0.9 * prop_data["pop_control_ene_shift"] + 0.1 * blk_en
    )

    return prop_data,(blk_wt,blk_en,blk_eorb,blk_orbfk,blk_orbcx)

@partial(jit, static_argnums=(2,3,5))
def _sr_block_scan_orb_norhf_test(
    prop_data: dict,
    ham_data: dict,
    prop: propagation.propagator,
    trial: wavefunctions,
    wave_data: dict,
    sampler: sampling.sampler,
) -> Tuple[dict, Tuple[jax.Array, jax.Array]]:
    def _block_scan_wrapper(x,_):
        return block_orb_norhf_test(x,ham_data,prop,trial,wave_data,sampler)
    
    # propagate n_ene_blocks then do sr
    prop_data, (blk_wt,blk_en,blk_eorb,blk_orbfk,blk_orbcx) \
        = lax.scan(
        _block_scan_wrapper, prop_data, xs=None, length=sampler.n_ene_blocks
    )
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

    return prop_data, (blk_wt,blk_en,blk_eorb,blk_orbfk,blk_orbcx)

@partial(jit, static_argnums=(1,3,5))
def propagate_phaseless_orb_norhf_test(
    ham_data: dict,
    prop: propagation.propagator,
    prop_data: dict,
    trial: wavefunctions,
    wave_data: dict,
    sampler: sampling.sampler,
) -> Tuple[jax.Array, dict]:
    def _sr_block_scan_wrapper(x,_):
        return _sr_block_scan_orb_norhf_test(x,ham_data,prop,trial,wave_data,sampler)

    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
    prop_data["n_killed_walkers"] = 0
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
    prop_data,(blk_wt,blk_en,blk_eorb,blk_orbfk,blk_orbcx) \
        = lax.scan(
        _sr_block_scan_wrapper, prop_data, xs=None, length=sampler.n_sr_blocks
    )
    prop_data["n_killed_walkers"] /= (
        sampler.n_sr_blocks * sampler.n_ene_blocks * prop.n_walkers
    )
    wt = jnp.sum(blk_wt)
    en = jnp.sum(blk_en * blk_wt) / wt
    orb_en = jnp.sum(blk_eorb * blk_wt) / wt
    orbfk = jnp.sum(blk_orbfk * blk_wt) / wt
    orbcx = jnp.sum(blk_orbcx * blk_wt) / wt

    return prop_data, (wt,en,orb_en,orbfk,orbcx)
########################################

from pyscf.ci.cisd import CISD
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD
from ad_afqmc.lno.cc import LNOCCSD
def run_lno_afqmc(mfcc,thresh,frozen=None,options=None,
                  lo_type='boys',chol_cut=1e-6,nproc=None,
                  run_frg_list=None,use_df_vecs=False,mp2=True,
                  debug=False,t2_0=False):
    '''
    mfcc: pyscf mean-field object
    thresh: lno thresh
    frozen: frozen orbitals
    options: afqmc options
    chol_cut: Cholesky Decomposition cutoff
    nproc: number of processors
    run_frg_list: list of the fragments to run
    '''

    if isinstance(mfcc, (CCSD, CISD)):
        full_cisd = True
        mf = mfcc._scf
    else:
        full_cisd = False
        mf = mfcc

    no_type = 'ie' # cim

    if isinstance(thresh, list):
        thresh_occ, thresh_vir = thresh
    else:
        thresh_occ = thresh*10
        thresh_vir = thresh

    lno_cc = LNOCCSD(mf, thresh=thresh, frozen=frozen)
    lno_cc.thresh_occ = thresh_occ
    lno_cc.thresh_vir = thresh_vir
    lno_cc.lo_type = lo_type
    lno_cc.no_type = no_type
    lno_cc.force_outcore_ao2mo = True

    s1e = mf.get_ovlp()
    lococc = lno_cc.get_lo(lo_type=lo_type) # localized active occ orbitals
    # lococc,locvir = lno_maker.get_lo(lno_cc,lo_type) ### fix this for DF

    frag_lolist = [[i] for i in range(lococc.shape[1])]
    nfrag = len(frag_lolist)

    frozen_mask = lno_cc.get_frozen_mask()
    thresh_pno = [thresh_occ,thresh_vir]
    print(f'# lno thresh {thresh_pno}')
    
    if run_frg_list is None:
        run_frg_list = range(nfrag)
    
    seeds = random.randint(random.PRNGKey(options["seed"]),
                        shape=(len(run_frg_list),), minval=0, maxval=100*nfrag)
    
    for ifrag in run_frg_list:
        print(f'\n########### running fragment {ifrag+1} ##########')

        fraglo = frag_lolist[ifrag]
        orbfragloc = lococc[:,fraglo]
        THRESH_INTERNAL = 1e-10
        frzfrag, orbfrag, can_orbfrag \
            = lno_maker.make_lno(
                lno_cc,orbfragloc,THRESH_INTERNAL,thresh_pno
                )
        
        mol = mf.mol
        nocc = mol.nelectron // 2 
        nao = mol.nao
        actfrag = np.array([i for i in range(nao) if i not in frzfrag])
        # frzocc = np.array([i for i in range(nocc) if i in frzfrag])
        actocc = np.array([i for i in range(nocc) if i in actfrag])
        actvir = np.array([i for i in range(nocc,nao) if i in actfrag])
        nactocc = len(actocc)
        nactocc = len(actocc)
        nactvir = len(actvir)
        prjlo = orbfragloc.T @ s1e @ orbfrag[:,actocc]

        print(f'# active orbitals: {actfrag}')
        print(f'# active occupied orbitals: {actocc}')
        print(f'# active virtual orbitals: {actvir}')
        print(f'# frozen orbitals: {frzfrag}')

        if options["trial"] == "rhf":
            print('# Using RHF trial wavefunction')
            ci1 = []
            ci2 = []
            ecorr_cc = '  None  '
        if options["trial"] == "cisd":
            if full_cisd:
                # print('# This method is not size-extensive')
                frz_mo_idx = np.where(np.array(frozen_mask) == False)[0]
                act_mo_occ = np.array([i for i in range(nocc) if i not in frz_mo_idx])
                act_mo_vir = np.array([i for i in range(nocc,nao) if i not in frz_mo_idx])
                prj_no2mo = afqmc_maker.no2mo(mf.mo_coeff,s1e,orbfrag)
                prj_oo_act = prj_no2mo[np.ix_(act_mo_occ,actocc)]
                prj_vv_act = prj_no2mo[np.ix_(act_mo_vir,actvir)]
                if isinstance(mfcc, CCSD):
                    print('# Use full CCSD wavefunction')
                    print('# Project CC amplitudes from MO to NO')
                    t1 = mfcc.t1
                    t2 = mfcc.t2
                    if t2_0:
                        t2 = np.zeros(t2.shape)
                    # project to active no
                    t1 = lib.einsum("ij,ia,ba->jb",prj_oo_act,t1,prj_vv_act.T)
                    t2 = lib.einsum("ik,jl,ijab,db,ca->klcd",
                            prj_oo_act,prj_oo_act,t2,prj_vv_act.T,prj_vv_act.T)
                    ci1 = np.array(t1)
                    ci2 = t2 + lib.einsum("ia,jb->ijab",ci1,ci1)
                    ci2 = ci2.transpose(0, 2, 1, 3)
                if isinstance(mfcc, CISD):
                    print('# Use full CISD wavefunction')
                    print('# Project CI coefficients from MO to NO')
                    v_ci = mfcc.ci
                    ci0,ci1,ci2 = mfcc.cisdvec_to_amplitudes(v_ci)
                    ci1 = ci1/ci0
                    ci2 = ci2/ci0
                    ci1 = lib.einsum("ij,ia,ba->jb",prj_oo_act,ci1,prj_vv_act.T)
                    ci2 = lib.einsum("ik,jl,ijab,db,ca->klcd",
                            prj_oo_act,prj_oo_act,ci2,prj_vv_act.T,prj_vv_act.T)
                    ci2 = ci2.transpose(0, 2, 1, 3)
                print('# Finished MO to NO projection')
                ecorr_cc = '  None  '
            else:
                print('# Solving LNO-CCSD')
                ecorr_cc,t1,t2 = lno_maker.lno_cc_solver(
                        mf,orbfrag,orbfragloc,frozen=frzfrag,eris=None
                        )
                if t2_0:
                    t2 = np.zeros(t2.shape)
                ci1 = np.array(t1)
                ci2 = t2 + lib.einsum("ia,jb->ijab",ci1,ci1)
                ci2 = ci2.transpose(0, 2, 1, 3)
                ecorr_cc = f'{ecorr_cc:.8f}'
                print(f'# lno-ccsd fragment correlation energy: {ecorr_cc}')
                from mpi4py import MPI
                if not MPI.Is_finalized():
                    MPI.Finalize() # CCSD initializes MPI
        #MP2 correction 
        if mp2:
            ## mp2 is not invariant to lno transformation
            ## needs to be done in canoical HF orbitals
            ## which the globel mp2 is calculated in
            print('# running fragment MP2')
            ecorr_p2 = \
                lno_maker.lno_mp2_frg_e(mf,frzfrag,orbfragloc,can_orbfrag)
            ecorr_p2 = f'{ecorr_p2:.8f}'
            print(f'# lno-mp2 fragment correlation energy: {ecorr_p2}')
        else:
            ecorr_p2 = '  None  '

        nelec_act = nactocc*2
        norb_act = nactocc+nactvir
        
        print(f'# number of active electrons: {nelec_act}')
        print(f'# number of active orbitals: {norb_act}')
        print(f'# number of frozen orbitals: {len(frzfrag)}')

        options["seed"] = seeds[ifrag]
        lnoafqmc_runner.prep_lnoafqmc_file(
            mf,orbfrag,options,
            norb_act=norb_act,nelec_act=nelec_act,
            prjlo=prjlo,norb_frozen=frzfrag,
            ci1=ci1,ci2=ci2,use_df_vecs=use_df_vecs,
            chol_cut=chol_cut,
            )
        
        # Run AFQMC
        use_gpu = options["use_gpu"]
        if use_gpu:
            print(f'# running AFQMC on GPU')
            gpu_flag = "--use_gpu"
            mpi_prefix = ""
            nproc = None
        else:
            print(f'# running AFQMC on CPU')
            gpu_flag = ""
            mpi_prefix = "mpirun "
            if nproc is not None:
                mpi_prefix += f"-np {nproc} "
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)   
        # if debug:
        #     script = f"{dir_path}/run_lnocc_frg_dbg.py"
        # else:
        #     script = f"{dir_path}/run_lnocc_frg.py"
        script = f"{dir_path}/run_lnocc_frg_test.py"
        os.system(
            f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1;"
            f"{mpi_prefix} python {script} {gpu_flag} |tee frg_{ifrag+1}.out"
        )

        with open(f'frg_{ifrag+1}.out', 'a') as out_file:
            print(f"lno-mp2 orb_corr: {ecorr_p2}",file=out_file)
            print(f"lno-ccsd orb_corr: {ecorr_cc}",file=out_file)
            print(f"number of active electrons: {nelec_act}",file=out_file)
            print(f"number of active orbitals: {norb_act}",file=out_file)

    from pyscf import mp
    mmp = mp.MP2(mf, frozen=frozen)
    e_mp2 = mmp.kernel()[0]

    # if debug:
    #     data_maker.frg2result_dbg(thresh_pno,len(run_frg_list),mf.e_tot,e_mp2=0)
    # else:
    frg2result(thresh_pno,len(run_frg_list),mf.e_tot,e_mp2)

    return None


def frg2result(lno_thresh,nfrag,e_mf,e_mp2):
    with open('results.out', 'w') as out_file:
        print('# frag  mp2_orb_corr  ccsd_orb_corr' \
              '  afqmc_orb_en  err' \
              '  nelec  norb  time',file=out_file)
        for ifrag in range(nfrag):
            with open(f"frg_{ifrag+1}.out","r") as read_file:
                for line in read_file:
                    if "use rhf trial" in line:
                        trial = 'rhf'
                    if "use norhf trial" in line:
                        trial = 'norhf'
                    if "use cisd trial" in line:
                        trial = 'cisd'
                    if "lno-mp2 orb_corr" in line:
                        e_mp2_orb_en = line.split()[2]
                    if "lno-ccsd orb_corr" in line:
                        e_ccsd_orb_en = line.split()[2]
                    if "lno-afqmc orbital energy" in line:
                        afqmc_orb_en = line.split()[-3]
                        afqmc_orb_en_err = line.split()[-1]
                    if "number of active electrons" in line:
                        nelec = line.split()[-1]
                    if "number of active orbitals" in line:
                        norb = line.split()[-1]
                    if "total run time" in line:
                        tot_time = line.split()[3]
                print(f'{ifrag+1:3d}  '
                      f'{e_mp2_orb_en}  {e_ccsd_orb_en}  '
                      f'{afqmc_orb_en}  {afqmc_orb_en_err}  '
                      f'{nelec}  {norb}  {tot_time}', file=out_file)

    data = []
    with open('results.out', 'r') as read_file:
        for line in read_file:
            if not line.startswith("#"):
                data = np.hstack((data,line.split()))
                data[data == 'None'] = '0.000000' 

    data = np.array(data.reshape(nfrag,8))
    e_mp2_orb_en = np.array(data[:,1],dtype='float32')
    e_ccsd_orb_en = np.array(data[:,2],dtype='float32')
    afqmc_orb_en = np.array(data[:,3],dtype='float32')
    afqmc_orb_en_err = np.array(data[:,4],dtype='float32')
    nelec = np.array(data[:,5],dtype='int32')
    norb = np.array(data[:,6],dtype='int32')
    tot_time = np.array(data[:,7],dtype='float32')

    e_mp2_corr = sum(e_mp2_orb_en)
    mp2_cr = e_mp2 - e_mp2_corr
    e_ccsd_corr = sum(e_ccsd_orb_en)
    afqmc_corr = sum(afqmc_orb_en)
    afqmc_corr_err = np.sqrt(sum(afqmc_orb_en_err**2))
    nelec_avg = np.mean(nelec)
    norb_avg = np.mean(norb)
    nelec_max = max(nelec)
    norb_max = max(norb)
    tot_time = sum(tot_time)

    e_mp2_corr = f'{e_mp2_corr:.6f}'
    e_ccsd_corr = f'{e_ccsd_corr:.6f}'
    afqmc_corr = f'{afqmc_corr:.6f}'
    afqmc_corr_err = f'{afqmc_corr_err:.6f}'

    with open('results.out', 'a') as out_file:
        out_file.write(f'# final results \n')
        out_file.write(f'# use {trial} trial wavefunction\n')
        out_file.write(f'# mean-field energy: {e_mf:.8f}\n')
        out_file.write(f'# lno-thresh {lno_thresh}\n')
        out_file.write(f'# e_mp2_corr: {e_mp2_corr}\n')
        out_file.write(f'# e_ccsd_corr: {e_ccsd_corr}\n')
        out_file.write(f'# afqmc_corr: {afqmc_corr} +/- {afqmc_corr_err}\n')
        out_file.write(f'# mp2_correction: {mp2_cr:.8f}\n')
        out_file.write(f'# number of electrons: average {nelec_avg:.2f} maxium {nelec_max}\n')
        out_file.write(f'# number of orbitals: average {norb_avg:.2f} maxium {norb_max}\n')
        out_file.write(f'# total run time: {tot_time:.2f}\n')
    
    return None

def sum_results(n_results):
    with open('sum_results.out', 'w') as out_file:
        print("# lno-thresh(occ,vir) "
              "  mp2_corr  ccsd_corr"
              "  afqmc_corr   err "
              "  mp2_cr nelec_avg   nelec_max  "
              "  norb_avg   norb_max  "
              "  run_time",file=out_file)
        for i in range(n_results):
            with open(f"results.out{i+1}","r") as read_file:
                for line in read_file:
                    if 'wavefunction' in line:
                        trial = line.split()[-3]
                    if "lno-thresh" in line:
                        thresh_occ = line.split()[-2]
                        thresh_vir = line.split()[-1]
                        thresh_occ = float(thresh_occ.strip('()[],'))
                        thresh_vir = float(thresh_vir.strip('()[],'))
                    if "e_mp2_corr:" in line:
                        mp2_corr = line.split()[-1]
                    if "e_ccsd_corr:" in line:
                        ccsd_corr = line.split()[-1]
                    if "afqmc_corr:" in line:
                        afqmc_corr = line.split()[-3]
                        afqmc_corr_err = line.split()[-1]
                    if "mp2_correction:" in line:
                        mp2_cr = line.split()[-1]
                    if "electrons:" in line:
                        nelec_avg = line.split()[-3]
                        nelec_max = line.split()[-1]
                    if "orbitals:" in line:
                        norb_avg = line.split()[-3]
                        norb_max = line.split()[-1]
                    if "time:" in line:
                        run_time = line.split()[-1]
            print(f" ({thresh_occ:.2e},{thresh_vir:.2e}) \t"
                  f" {mp2_corr} \t {ccsd_corr} \t"
                  f" {afqmc_corr} \t {afqmc_corr_err} \t"
                  f" {mp2_cr}  {nelec_avg} \t {nelec_max} \t"
                  f" {norb_avg}  \t {norb_max} \t {run_time}",file=out_file)
        print(f'# use {trial} trial wavefunction',file=out_file)
    return None

def run_lno_afqmc_norhf_test(mfcc,thresh,frozen=None,options=None,
                  lo_type='boys',chol_cut=1e-6,nproc=None,
                  run_frg_list=None,use_df_vecs=False,mp2=True,
                  debug=False,t2_0=False):
    '''
    mfcc: pyscf mean-field object
    thresh: lno thresh
    frozen: frozen orbitals
    options: afqmc options
    chol_cut: Cholesky Decomposition cutoff
    nproc: number of processors
    run_frg_list: list of the fragments to run
    '''

    if isinstance(mfcc, (CCSD, CISD)):
        full_cisd = True
        mf = mfcc._scf
    else:
        full_cisd = False
        mf = mfcc

    no_type = 'ie' # cim

    if isinstance(thresh, list):
        thresh_occ, thresh_vir = thresh
    else:
        thresh_occ = thresh*10
        thresh_vir = thresh

    lno_cc = LNOCCSD(mf, thresh=thresh, frozen=frozen)
    lno_cc.thresh_occ = thresh_occ
    lno_cc.thresh_vir = thresh_vir
    lno_cc.lo_type = lo_type
    lno_cc.no_type = no_type
    lno_cc.force_outcore_ao2mo = True
    lno_cc.frag_lolist = '1o'

    eris = lno_cc.ao2mo()
    s1e = mf.get_ovlp()
    lococc = lno_cc.get_lo(lo_type=lo_type) # localized active occ orbitals
    # lococc,locvir = lno_maker.get_lo(lno_cc,lo_type) ### fix this for DF

    frag_lolist = [[i] for i in range(lococc.shape[1])]
    nfrag = len(frag_lolist)

    frozen_mask = lno_cc.get_frozen_mask()
    thresh_pno = [thresh_occ,thresh_vir]
    print(f'# lno thresh: {thresh_pno}')
    print(f'# fragment list: {frag_lolist}')
    print(f'# number of fragments: {nfrag}')
    
    if run_frg_list is None:
        run_frg_list = range(nfrag)
    
    seeds = random.randint(random.PRNGKey(options["seed"]),
                        shape=(len(run_frg_list),), minval=0, maxval=100*nfrag)

    for ifrag in run_frg_list:
        print(f'\n########### running fragment {ifrag+1} ##########')
        fraglo = frag_lolist[ifrag]
        orbfragloc = lococc[:,fraglo]
        THRESH_INTERNAL = 1e-10
        frzfrag, orbfrag, can_orbfrag \
            = lno.make_fpno1(lno_cc, eris, orbfragloc, no_type,
                            THRESH_INTERNAL, thresh_pno,
                            frozen_mask=frozen_mask,
                            frag_target_nocc=None,
                            frag_target_nvir=None,
                            canonicalize=True)
        eorb_cc, _, _ \
            = lno_maker.lno_cc_solver(mf,orbfrag,orbfragloc,frozen=frzfrag)
        eorb_cc = f'{eorb_cc:.8f}'

        mol = mf.mol
        nocc = mol.nelectron // 2 
        nao = mol.nao
        actfrag = np.array([i for i in range(nao) if i not in frzfrag])
        # frzocc = np.array([i for i in range(nocc) if i in frzfrag])
        actocc = np.array([i for i in range(nocc) if i in actfrag])
        actvir = np.array([i for i in range(nocc,nao) if i in actfrag])
        nactocc = len(actocc)
        nactocc = len(actocc)
        nactvir = len(actvir)
        prjlo = orbfragloc.T @ s1e @ orbfrag[:,actocc]

        print(f'# active orbitals: {actfrag}')
        print(f'# active occupied orbitals: {actocc}')
        print(f'# active virtual orbitals: {actvir}')
        print(f'# frozen orbitals: {frzfrag}')

        if options["trial"] == "rhf":
            print('# Using RHF trial wavefunction')
            ci1 = []
            ci2 = []
            # ecorr_cc = '  None  '
        #MP2 correction 
        if mp2:
            ## mp2 is not invariant to lno transformation
            ## needs to be done in canoical HF orbitals
            ## which the globel mp2 is calculated in
            print('# running fragment MP2')
            eorb_p2 = \
                lno_maker.lno_mp2_frg_e(mf,frzfrag,orbfragloc,can_orbfrag)
            eorb_p2 = f'{eorb_p2:.8f}'
            # print(f'# lno-mp2 fragment correlation energy: {eorb_p2}')
        else:
            eorb_p2 = '  None  '

        nelec_act = nactocc*2
        norb_act = nactocc+nactvir
        
        print(f'# number of active electrons: {nelec_act}')
        print(f'# number of active orbitals: {norb_act}')
        print(f'# number of frozen orbitals: {len(frzfrag)}')
        print(f'# lno-mp2 ref orb energy: {eorb_p2}')
        print(f'# lno-ccsd ref orb energy: {eorb_cc}')

        options["seed"] = seeds[ifrag]
        lnoafqmc_runner.prep_lnoafqmc_file(
            mf,orbfrag,options,
            norb_act=norb_act,nelec_act=nelec_act,
            prjlo=prjlo,norb_frozen=frzfrag,
            ci1=ci1,ci2=ci2,use_df_vecs=use_df_vecs,
            chol_cut=chol_cut,
            )

        # Run AFQMC
        use_gpu = options["use_gpu"]
        if use_gpu:
            print(f'# running AFQMC on GPU')
            gpu_flag = "--use_gpu"
            mpi_prefix = ""
            nproc = None
        else:
            print(f'# running AFQMC on CPU')
            gpu_flag = ""
            mpi_prefix = "mpirun "
            if nproc is not None:
                mpi_prefix += f"-np {nproc} "
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)   
        script = f"{dir_path}/run_lnoafqmc_test_norhf.py"
        os.system(
            f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1;"
            f"{mpi_prefix} python {script} {gpu_flag} |tee frg_{ifrag+1}.out"
        )

        with open(f'frg_{ifrag+1}.out', 'a') as out_file:
            print(f"lno-mp2 orb_corr: {eorb_p2}",file=out_file)
            print(f"lno-ccsd orb_corr: {eorb_cc}",file=out_file)
            print(f"number of active electrons: {nelec_act}",file=out_file)
            print(f"number of active orbitals: {norb_act}",file=out_file)

    from pyscf import mp
    mmp = mp.MP2(mf, frozen=frozen)
    e_mp2 = mmp.kernel()[0]

    frg2result_norhf_test(thresh_pno,len(run_frg_list),mf.e_tot,e_mp2)
    return None


def frg2result_norhf_test(lno_thresh,nfrag,e_mf,e_mp2):
    with open('results.out', 'w') as out_file:
        print('# frag  mp2_orb_corr  ccsd_orb_corr' \
              '  afqmc_orb_en  err' \
              '  afqmc_orb_fk  err' \
              '  afqmc_orb_cx  err' \
              '  nelec  norb  time',file=out_file)
        for ifrag in range(nfrag):
            with open(f"frg_{ifrag+1}.out","r") as read_file:
                for line in read_file:
                    if "lno-mp2 orb_corr" in line:
                        e_mp2_orb_en = line.split()[2]
                    if "lno-ccsd orb_corr" in line:
                        e_ccsd_orb_en = line.split()[2]
                    if "lno-afqmc orbital energy" in line:
                        afqmc_orb_en = line.split()[-3]
                        afqmc_orb_en_err = line.split()[-1]
                    if "lno-afqmc orbital fock energy" in line:
                        afqmc_orb_fk = line.split()[-3]
                        afqmc_orb_fk_err = line.split()[-1]
                    if "lno-afqmc orbital colexc energy" in line:
                        afqmc_orb_cx = line.split()[-3]
                        afqmc_orb_cx_err = line.split()[-1]
                    if "number of active electrons" in line:
                        nelec = line.split()[-1]
                    if "number of active orbitals" in line:
                        norb = line.split()[-1]
                    if "total run time" in line:
                        tot_time = line.split()[3]
                print(f'{ifrag+1:3d}  '
                      f'{e_mp2_orb_en}  {e_ccsd_orb_en}  '
                      f'{afqmc_orb_en}  {afqmc_orb_en_err}  '
                      f'{afqmc_orb_fk}  {afqmc_orb_fk_err}  '
                      f'{afqmc_orb_cx}  {afqmc_orb_cx_err}  '
                      f'{nelec}  {norb}  {tot_time}', file=out_file)

    data = []
    with open('results.out', 'r') as read_file:
        for line in read_file:
            if not line.startswith("#"):
                data = np.hstack((data,line.split()))
                data[data == 'None'] = '0.000000' 

    data = np.array(data.reshape(nfrag,12))
    e_mp2_orb_en = np.array(data[:,1],dtype='float32')
    e_ccsd_orb_en = np.array(data[:,2],dtype='float32')
    afqmc_orb_en = np.array(data[:,3],dtype='float32')
    afqmc_orb_en_err = np.array(data[:,4],dtype='float32')
    afqmc_orb_fk = np.array(data[:,5],dtype='float32')
    afqmc_orb_fk_err = np.array(data[:,6],dtype='float32')
    afqmc_orb_cx = np.array(data[:,7],dtype='float32')
    afqmc_orb_cx_err = np.array(data[:,8],dtype='float32')
    nelec = np.array(data[:,9],dtype='int32')
    norb = np.array(data[:,10],dtype='int32')
    tot_time = np.array(data[:,11],dtype='float32')

    e_mp2_corr = sum(e_mp2_orb_en)
    mp2_cr = e_mp2 - e_mp2_corr
    e_ccsd_corr = sum(e_ccsd_orb_en)
    afqmc_corr = sum(afqmc_orb_en)
    afqmc_corr_err = np.sqrt(sum(afqmc_orb_en_err**2))
    afqmc_fk = sum(afqmc_orb_fk)
    afqmc_fk_err = np.sqrt(sum(afqmc_orb_fk_err**2))
    afqmc_cx = sum(afqmc_orb_cx)
    afqmc_cx_err = np.sqrt(sum(afqmc_orb_cx_err**2))
    nelec_avg = np.mean(nelec)
    norb_avg = np.mean(norb)
    nelec_max = max(nelec)
    norb_max = max(norb)
    tot_time = sum(tot_time)

    e_mp2_corr = f'{e_mp2_corr:.8f}'
    e_ccsd_corr = f'{e_ccsd_corr:.8f}'
    afqmc_corr = f'{afqmc_corr:.6f}'
    afqmc_corr_err = f'{afqmc_corr_err:.6f}'
    afqmc_fk = f'{afqmc_fk:.6f}'
    afqmc_fk_err = f'{afqmc_fk_err:.6f}'
    afqmc_cx = f'{afqmc_cx:.6f}'
    afqmc_cx_err = f'{afqmc_cx_err:.6f}'

    with open('results.out', 'a') as out_file:
        out_file.write(f'# final results \n')
        out_file.write(f'# use norhf trial wavefunction\n')
        out_file.write(f'# mean-field energy: {e_mf:.8f}\n')
        out_file.write(f'# lno-thresh {lno_thresh}\n')
        out_file.write(f'# e_mp2_corr: {e_mp2_corr}\n')
        out_file.write(f'# e_ccsd_corr: {e_ccsd_corr}\n')
        out_file.write(f'# afqmc_corr: {afqmc_corr} +/- {afqmc_corr_err}\n')
        out_file.write(f'# afqmc_fk: {afqmc_fk} +/- {afqmc_fk_err}\n')
        out_file.write(f'# afqmc_cx: {afqmc_cx} +/- {afqmc_cx_err}\n')
        out_file.write(f'# mp2_correction: {mp2_cr:.8f}\n')
        out_file.write(f'# number of electrons: average {nelec_avg:.2f} maxium {nelec_max}\n')
        out_file.write(f'# number of orbitals: average {norb_avg:.2f} maxium {norb_max}\n')
        out_file.write(f'# total run time: {tot_time:.2f}\n')
    
    return None

def sum_results_norhf_test(n_results):
    with open('sum_results.out', 'w') as out_file:
        print("# lno-thresh(occ,vir) "
              "  mp2_corr  ccsd_corr"
              "  afqmc_corr   err "
              "  afqmc_fk   err "
              "  afqmc_cx   err "
              "  mp2_cr nelec_avg   nelec_max  "
              "  norb_avg   norb_max  "
              "  run_time",file=out_file)
        for i in range(n_results):
            with open(f"results.out{i+1}","r") as read_file:
                for line in read_file:
                    if 'wavefunction' in line:
                        trial = line.split()[-3]
                    if "lno-thresh" in line:
                        thresh_occ = line.split()[-2]
                        thresh_vir = line.split()[-1]
                        thresh_occ = float(thresh_occ.strip('()[],'))
                        thresh_vir = float(thresh_vir.strip('()[],'))
                    if "e_mp2_corr:" in line:
                        mp2_corr = line.split()[-1]
                    if "e_ccsd_corr:" in line:
                        ccsd_corr = line.split()[-1]
                    if "afqmc_corr:" in line:
                        afqmc_corr = line.split()[-3]
                        afqmc_corr_err = line.split()[-1]
                    if "afqmc_fk:" in line:
                        afqmc_fk = line.split()[-3]
                        afqmc_fk_err = line.split()[-1]
                    if "afqmc_cx:" in line:
                        afqmc_cx = line.split()[-3]
                        afqmc_cx_err = line.split()[-1]
                    if "mp2_correction:" in line:
                        mp2_cr = line.split()[-1]
                    if "electrons:" in line:
                        nelec_avg = line.split()[-3]
                        nelec_max = line.split()[-1]
                    if "orbitals:" in line:
                        norb_avg = line.split()[-3]
                        norb_max = line.split()[-1]
                    if "time:" in line:
                        run_time = line.split()[-1]
            print(f" ({thresh_occ:.2e},{thresh_vir:.2e}) \t"
                  f" {mp2_corr} \t {ccsd_corr} \t"
                  f" {afqmc_corr} \t {afqmc_corr_err} \t"
                  f" {afqmc_fk} \t {afqmc_fk_err} \t"
                  f" {afqmc_cx} \t {afqmc_cx_err} \t"
                  f" {mp2_cr}  {nelec_avg} \t {nelec_max} \t"
                  f" {norb_avg}  \t {norb_max} \t {run_time}",file=out_file)
        print(f'# use {trial} trial wavefunction',file=out_file)
    return None