from jax import numpy as jnp
from jax import jit
import opt_einsum as oe
from functools import partial
from .. import slater_tools, sampling_exp

# implementation in QMC sampling
@partial(jit, static_argnums=0)
def overlap(trial, walker, wave_data):
    return slater_tools.r_overlap(wave_data["mo_coeff"], walker)

@partial(jit, static_argnums=0)
def force_bias(trial, walker, ham_data, wave_data):
    chol = ham_data["chol"].reshape(trial.nchol, trial.norb, trial.norb)
    return slater_tools.r_force_bias(wave_data["mo_coeff"], walker, chol)

@partial(jit, static_argnums=0)
def rot_force_bias(trial, walker, ham_data, wave_data):
    rot_chol = ham_data["rot_chol"]
    return slater_tools.r_rot_force_bias(wave_data["mo_coeff"], walker, rot_chol)

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
    return slater_tools.r_energy(wave_data["mo_coeff"], walker, h0, h1, chol)

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
    return slater_tools.r_rot_energy(wave_data["mo_coeff"], walker, h0, rot_h1, rot_chol)

@partial(jit, static_argnums=0)
def build_intermediate(trial, ham_data: dict, wave_data: dict) -> dict:
    """Builds half rotated integrals for efficient force bias and energy calculations."""
    ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ (
        (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0)
    ham_data["rot_chol"] = oe.contract("ip,gpq->giq",
                                       wave_data["mo_coeff"].T.conj(),
                                       ham_data["chol"].reshape(-1, trial.norb, trial.norb), 
                                       backend="jax")
    return ham_data, wave_data


def energy_formula(weights, samples, ham_data):
    # energy_terms shape: (nsamples, terms)
    weights = jnp.atleast_1d(weights)
    samples = jnp.atleast_1d(samples)
    weight_tot, sample_mean, sample_err = sampling_exp.weighted_average(weights, samples)
    weight = weights.mean()
    energy = sample_mean
    energy_err = sample_err
    return weight, energy, energy_err