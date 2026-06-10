from jax import numpy as jnp
import opt_einsum as oe
from .. import slater_tools

# implementation in QMC sampling
def calc_overlap(trial, walker, wave_data):
    return slater_tools.r_overlap(wave_data["mo_coeff"], walker)

def calc_force_bias(trial, walker, ham_data, wave_data):
    chol = ham_data["chol"].reshape(trial.nchol, trial.norb, trial.norb)
    return slater_tools.r_force_bias(wave_data["mo_coeff"], walker, chol)

def calc_rot_force_bias(trial, walker, ham_data, wave_data):
    rot_chol = ham_data["rot_chol"]
    return slater_tools.r_rot_force_bias(wave_data["mo_coeff"], walker, rot_chol)

def calc_energy(trial, walker, ham_data, wave_data):
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

def calc_rot_energy(trial, walker, ham_data, wave_data):
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

def calc_intermediate(trial, ham_data: dict, wave_data: dict) -> dict:
    """Builds half rotated integrals for efficient force bias and energy calculations."""
    ham_data["h1"] = (
        ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
    )
    ham_data["h1"] = (
        ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
    )
    ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ (
        (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
    )
    ham_data["rot_chol"] = oe.contract(
        "pi,gij->gpj",
        wave_data["mo_coeff"].T.conj(),
        ham_data["chol"].reshape(-1, trial.norb, trial.norb), 
        backend="jax")
    return ham_data