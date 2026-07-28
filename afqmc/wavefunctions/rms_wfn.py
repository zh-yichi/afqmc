from jax import numpy as jnp
from .. import slater_tools
from . import rhf_wfn
from jax import jit
from functools import partial

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
