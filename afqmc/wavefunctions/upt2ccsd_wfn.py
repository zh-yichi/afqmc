import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy as jsp
import opt_einsum as oe
from .. import slater_tools
from . import rpt2ccsd_wfn

from jax import jit
from functools import partial

energy_formula = rpt2ccsd_wfn.energy_formula

from afqmc import slater_tools, t2_tools


@partial(jit, static_argnums=0)
def calc_overlap(wave, walker, wave_data):
    return slater_tools.u_overlap(wave_data["mo_t"], walker)

@partial(jit, static_argnums=0)
def calc_energy(
        wave,
        walker: tuple,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
    terms = t2_tools.ut2h12(
        wave_data["mo_t"], 
        walker, 
        wave_data["t2"],
        ham_data["h1"],
        ham_data["chol"], 
        wave.mix_precision, 
        wave.nchol_chunk
        )
    return terms

@partial(jit, static_argnums=0)
def calc_intermediate(wave, ham_data: dict, wave_data: dict):
    wave_data["mo_t"] = slater_tools.thouless(wave_data["mo_coeff"], wave_data["t1"])
    return ham_data, wave_data

@partial(jit, static_argnums=0)
def calc_overlap_bar(wave, walker, wave_data):
    walker_up, walker_dn = walker
    walker_bar_up = wave_data['exp_t1a'] @ walker_up
    walker_bar_dn = wave_data['exp_t1b'] @ walker_dn
    walker_bar = (walker_bar_up, walker_bar_dn)
    return slater_tools.u_overlap(wave_data["mo_coeff"], walker_bar)

@partial(jit, static_argnums=0)
def calc_energy_bar(
        wave,
        walker: tuple,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
    mo = wave_data["mo_coeff"]
    t2 = wave_data["t2"]
    h1_bar = ham_data["h1_bar"]
    chol_bar = ham_data["chol_bar"]
    walker_up, walker_dn = walker
    walker_bar_up = wave_data['exp_t1a'] @ walker_up
    walker_bar_dn = wave_data['exp_t1b'] @ walker_dn
    walker_bar = (walker_bar_up, walker_bar_dn)
    terms = t2_tools.ut2h12_delta(
        mo, walker_bar, t2, h1_bar, chol_bar, 
        wave.mix_precision, wave.nchol_chunk
        )
    return terms

@partial(jit, static_argnums=0)
def calc_intermediate_bar(wave, ham_data: dict, wave_data: dict):
    nocc_a, nocc_b = wave.nelec
    norb_a, norb_b = wave.norb
    t1a, t1b = wave_data["t1"]
    t1a_full = jnp.zeros((norb_a, norb_a), dtype=jnp.float64)
    t1b_full = jnp.zeros((norb_b, norb_b), dtype=jnp.float64)
    t1a_full = t1a_full[:nocc_a, nocc_a:].set(t1a)
    t1b_full = t1b_full[:nocc_b, nocc_b:].set(t1b)
    wave_data['exp_t1a'] = jsp.linalg.expm(t1a_full)
    wave_data['exp_mt1a'] = jsp.linalg.expm(-t1a_full)
    wave_data['exp_t1b'] = jsp.linalg.expm(t1b_full)
    wave_data['exp_mt1b'] = jsp.linalg.expm(-t1b_full)
    chola = ham_data["chol"][0].reshape(-1, norb_a, norb_a)
    cholb = ham_data["chol"][1].reshape(-1, norb_b, norb_b)
    h1bar_a = wave_data['exp_t1a'] @ ham_data['h1'][0] @ wave_data['exp_mt1a']
    h1bar_b = wave_data['exp_t1b'] @ ham_data['h1'][1] @ wave_data['exp_mt1b']
    h1_bar = (h1bar_a, h1bar_b)
    ham_data["h1_bar"] = h1_bar
    chol_bar_a = oe.contract('pr,grs,sq->gpq', wave_data['exp_t1a'], chola, wave_data['exp_mt1a'], backend='jax')
    chol_bar_b = oe.contract('pr,grs,sq->gpq', wave_data['exp_t1b'], cholb, wave_data['exp_mt1b'], backend='jax')
    chol_bar = (chol_bar_a, chol_bar_b)
    ham_data["chol_bar"] = chol_bar
    return ham_data, wave_data
