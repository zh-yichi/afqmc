from jax import numpy as jnp
import opt_einsum as oe
from .. import slater_tools
from . import rhf_wfn

from jax import jit
from functools import partial

energy_formula = rhf_wfn.energy_formula

# implementation of above functions in QMC sampling
@partial(jit, static_argnums=0)
def overlap(trial, walker, wave_data):
    return slater_tools.u_overlap(wave_data["mo_coeff"], walker)

@partial(jit, static_argnums=0)
def force_bias(trial, walker, ham_data, wave_data):
    chol_a = ham_data["chol"][0].reshape(trial.nchol, trial.norb, trial.norb)
    chol_b = ham_data["chol"][1].reshape(trial.nchol, trial.norb, trial.norb)
    chol = (chol_a, chol_b)
    return slater_tools.u_force_bias(wave_data["mo_coeff"], walker, chol)

@partial(jit, static_argnums=0)
def energy(trial, walker, ham_data, wave_data):
    h0 = ham_data["h0"]
    h1 = ham_data["h1"]
    chol_a = ham_data["chol"][0].reshape(trial.nchol, trial.norb, trial.norb)
    chol_b = ham_data["chol"][1].reshape(trial.nchol, trial.norb, trial.norb)
    nchol = trial.nchol
    nchol_chunk = trial.nchol_chunk
    nchunks = -(-nchol // nchol_chunk)
    pad = nchunks * nchol_chunk - nchol
    chol_a = jnp.pad(chol_a, ((0, pad), (0, 0), (0, 0)))
    chol_b = jnp.pad(chol_b, ((0, pad), (0, 0), (0, 0)))
    chol_a = chol_a.reshape(nchunks, nchol_chunk, *chol_a.shape[1:])
    chol_b = chol_b.reshape(nchunks, nchol_chunk, *chol_b.shape[1:])
    chol = (chol_a, chol_b)
    return slater_tools.u_energy(wave_data["mo_coeff"], walker, h0, h1, chol)

@partial(jit, static_argnums=0)
def rot_force_bias(trial, walker, ham_data, wave_data):
    rot_chol_a = ham_data["rot_chol"][0]
    rot_chol_b = ham_data["rot_chol"][1]
    rot_chol = (rot_chol_a, rot_chol_b)
    return slater_tools.u_rot_force_bias(wave_data["mo_coeff"], walker, rot_chol)

@partial(jit, static_argnums=0)
def rot_energy(trial, walker, ham_data, wave_data):
    h0 = ham_data["h0"]
    rot_h1 = ham_data["rot_h1"]
    rot_chol_a = ham_data["rot_chol"][0]
    rot_chol_b = ham_data["rot_chol"][1]
    nchol = rot_chol_a.shape[0]
    nchol_chunk = trial.nchol_chunk
    nchunks = -(-nchol // nchol_chunk)
    pad = nchunks * nchol_chunk - nchol
    rot_chol_a = jnp.pad(rot_chol_a, ((0, pad), (0, 0), (0, 0)))
    rot_chol_b = jnp.pad(rot_chol_b, ((0, pad), (0, 0), (0, 0)))
    rot_chol_a = rot_chol_a.reshape(nchunks, nchol_chunk, *rot_chol_a.shape[1:])
    rot_chol_b = rot_chol_b.reshape(nchunks, nchol_chunk, *rot_chol_b.shape[1:])
    rot_chol = (rot_chol_a, rot_chol_b)
    return slater_tools.u_rot_energy(wave_data["mo_coeff"], walker, h0, rot_h1, rot_chol)

@partial(jit, static_argnums=0)
def build_intermediate(trial, ham_data: dict, wave_data: dict) -> dict:
    ham_data["rot_h1"] = (wave_data["mo_coeff"][0].T.conj() @ ham_data["h1"][0],
                          wave_data["mo_coeff"][1].T.conj() @ ham_data["h1"][1])
    ham_data["rot_chol"] = (
        oe.contract(
            "ip,gpq->giq",
            wave_data["mo_coeff"][0].T.conj(),
            ham_data["chol"][0].reshape(-1, trial.norb, trial.norb), 
            backend="jax"
            ),
        oe.contract(
            "ip,gpq->giq",
            wave_data["mo_coeff"][1].T.conj(),
            ham_data["chol"][1].reshape(-1, trial.norb, trial.norb), 
            backend="jax"
            )
        )
    return ham_data, wave_data