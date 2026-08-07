from jax import numpy as jnp
from .. import slater_tools, t2_tools, cc_tools
from . import rhf_wfn
from jax import jit, random
from functools import partial

import opt_einsum as oe

energy_formula = rhf_wfn.energy_formula

# implementation of above functions in QMC sampling
@partial(jit, static_argnums=0)
def overlap(trial, walker, wave_data):
    return slater_tools.rms_overlap(wave_data["slaters"], walker)

@partial(jit, static_argnums=0)
def force_bias(trial, walker, ham_data, wave_data):
    chol = ham_data["chol"].reshape(trial.nchol, trial.norb, trial.norb)
    return slater_tools.rms_force_bias(wave_data["slaters"], walker, chol)

@partial(jit, static_argnums=0)
def rot_force_bias(trial, walker, ham_data, wave_data):
    rot_chol = ham_data["rot_chol"].reshape(trial.nchol, trial.norb, trial.norb)
    return slater_tools.rms_rot_force_bias(wave_data["slaters"], walker, rot_chol)

@partial(jit, static_argnums=0)
def energy(trial, walker, ham_data, wave_data):
    h0 = ham_data["h0"]
    h1 = ((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
    chol = ham_data["chol"].reshape(trial.nchol, trial.norb, trial.norb)
    nchol = chol.shape[0]
    nchol_chunk = trial.nchol_chunk
    nchunks = -(-nchol // nchol_chunk)
    pad = nchunks * nchol_chunk - nchol
    chol = jnp.pad(chol, ((0, pad), (0, 0), (0, 0)))
    chol = chol.reshape(nchunks, nchol_chunk, *chol.shape[1:])
    return slater_tools.rms_energy(wave_data["slaters"], walker, h0, h1, chol)

@partial(jit, static_argnums=0)
def rot_energy(trial, walker, ham_data, wave_data):
    h0 = ham_data["h0"]
    rot_h1 = ham_data["rot_h1"]
    rot_chol = ham_data["rot_chol"]
    nchol = rot_chol.shape[0]
    nchol_chunk = trial.nchol_chunk
    nchunks = -(-nchol // nchol_chunk)
    pad = nchunks * nchol_chunk - nchol
    rot_chol = jnp.pad(rot_chol, ((0, pad), (0, 0), (0, 0)))
    rot_chol = rot_chol.reshape(nchunks, nchol_chunk, *rot_chol.shape[1:])
    return slater_tools.rms_rot_energy(wave_data["slaters"], walker, h0, rot_h1, rot_chol)

# @partial(jit, static_argnums=0)
def build_intermediate(trial, ham_data, wave_data):
    """Builds half rotated integrals for efficient force bias and energy calculations."""

    ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ ((ham_data["h1"][0] + ham_data["h1"][1]) / 2.0)

    ham_data["rot_chol"] = oe.contract("ip,gpq->giq",
                                       wave_data["mo_coeff"].T.conj(),
                                       ham_data["chol"].reshape(-1, trial.norb, trial.norb), 
                                       backend="jax")
    
    wave_data["mo_t"] = slater_tools.thouless(wave_data["mo_coeff"], wave_data["t1"])

    wave_data['tau'] = t2_tools.decompose_t2(wave_data['t2'], jnp.array(wave_data['t2_thresh']))
    print(f"Rank Decomposed T2 (t_iajb -> tau_yia tau_yjb) shape {wave_data['tau'].shape}")

    wave_data['slaters'], _ \
        = cc_tools.get_stoccsd(
            wave_data["mo_t"], 
            wave_data['tau'], 
            wave_data['n_slater'], 
            random.PRNGKey(wave_data["seed"])
            )
    
    return ham_data, wave_data

