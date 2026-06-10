import jax
import jax.numpy as jnp
from jax import lax, random, vmap, jit
import opt_einsum as oe
from . import slater_tools, t2_tools
from functools import partial

def calc_walkers_norm(walkers):

    def scan_walkers(carry, walker):
        norm = slater_tools.r_overlap(walker, walker)
        return carry, norm

    init_carry = 0.0
    _, norms = lax.scan(scan_walkers, init_carry, walkers)

    return norms

def replicate_walker(walker, nwalker):
    if isinstance(walker, jax.Array):
        walkers = jnp.array([walker] * nwalker, dtype=jnp.complex128)
    elif isinstance(walker, (tuple, list)):
        walkers_a = jnp.array([walker[0]] * nwalker, dtype=jnp.complex128)
        walkers_b = jnp.array([walker[1]] * nwalker, dtype=jnp.complex128)
        walkers = [walkers_a, walkers_b]
    return walkers

@partial(jit, static_argnames=("n_walkers"))
def get_rccsd_walkers(prop_data, wave_data, n_walkers):
    prop_data["key"], subkey = random.split(prop_data["key"])
    
    fieldy = random.normal(
        subkey,
        shape=(
            n_walkers,
            wave_data['tau'].shape[0],
        ),
    )
    # ytaus shape (nwalker, nocc, nvir)
    ytaus = oe.contract("wg,gia->wia", fieldy, wave_data['tau'], backend='jax')

    slaters = vmap(lambda y: slater_tools.rthouless(wave_data['mo_t'], y))(ytaus)

    return slaters, prop_data

@partial(jit, static_argnames=("n_walkers"))
def get_uccsd_walkers(prop_data, wave_data, n_walkers):
    prop_data["key"], subkey = random.split(prop_data["key"])
    
    fieldy = random.normal(
        subkey,
        shape=(
            n_walkers,
            wave_data['tau'][0].shape[0],
        ),
    )
    # ytaus shape (nwalker, nocc, nvir)
    ytaus_up = oe.contract("wg,gia->wia", fieldy, wave_data['tau'][0], backend='jax')
    ytaus_dn = oe.contract("wg,gia->wia", fieldy, wave_data['tau'][1], backend='jax')

    mo_t = (wave_data["mo_ta"], wave_data["mo_tb"])
    
    slaters_up, slaters_dn = vmap(
        lambda yu, yd: slater_tools.uthouless(mo_t, (yu, yd)))(ytaus_up, ytaus_dn)

    return [slaters_up, slaters_dn], prop_data

def get_ccsd_walkers(prop_data, wave_data, n_walkers, walker_type):
    
    if "tau" not in wave_data:
        wave_data["tau"] = t2_tools.decompose_t2(wave_data["t2"])

    if walker_type == "rhf":
        return get_rccsd_walkers(prop_data, wave_data, n_walkers)
    elif walker_type == "uhf":
        return get_uccsd_walkers(prop_data, wave_data, n_walkers)
    else:
        raise ValueError(f"unsupport CCSD initial walker_type: {walker_type}")