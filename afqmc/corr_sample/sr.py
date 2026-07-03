import jax
import jax.numpy as jnp
from jax import jit, random
from afqmc import sr

# @jit
# def stochastic_reconfiguration_core(walkers1, weights1, walkers2, weights2, zeta):
#     '''
#     Correlated SR: system 1 is the reference; system 2 is replicated/killed
#     with the SAME indices. System-2 survivors are reweighted by the expected
#     birth rate of system one r1 = w1 / w1_avg, w2 -> w2 / r1 to stay unbiased.
#     '''
#     nwalkers = weights1.shape[0]
#     average_weight1 = jnp.sum(jnp.abs(weights1)) / nwalkers
#     safe_w1 = jnp.where(weights1 == 0.0, 1.0, weights1)
#     birth_r1 = safe_w1 / average_weight1
#     weights2  = jnp.where(weights1 == 0.0, 0.0, weights2 / birth_r1)

#     indices, weights1 = sr.get_replicant(weights1, zeta)
    
#     walkers1 = jax.tree.map(lambda w: w[indices], walkers1)
#     walkers2 = jax.tree.map(lambda w: w[indices], walkers2)
#     weights2 = weights2[indices]

#     return walkers1, weights1, walkers2, weights2

# @jit
# def stochastic_reconfiguration(prop_data1, prop_data2):

#     walkers1, weights1 = prop_data1["walkers"], prop_data1["weights"]
#     walkers2, weights2 = prop_data2["walkers"], prop_data2["weights"]
    
#     prop_data1["key"], subkey = random.split(prop_data1["key"])
#     zeta = random.uniform(subkey)

#     walkers1, weights1, walkers2, weights2 \
#             = stochastic_reconfiguration_core(walkers1, weights1, walkers2, weights2, zeta)
        
#     prop_data1["walkers"], prop_data1["weights"] = walkers1, weights1
#     prop_data2["walkers"], prop_data2["weights"] = walkers2, weights2

#     return prop_data1, prop_data2

@jit
def _sr_directed(ref_walkers, ref_weights, fol_walkers, fol_weights, zeta):
    """One-directional correlated SR: `ref` drives birth/death, `fol` follows
    and is reweighted by fol / (ref_birth_rate) to stay unbiased."""
    nwalkers = ref_weights.shape[0]
    average_ref = jnp.sum(jnp.abs(ref_weights)) / nwalkers

    safe_ref = jnp.where(ref_weights == 0.0, 1.0, ref_weights)
    birth_r  = safe_ref / average_ref                       # <n_i> = w_ref / w̄_ref
    fol_weights = jnp.where(ref_weights == 0.0, 0.0, fol_weights / birth_r)

    indices, ref_weights = sr.get_replicant(ref_weights, zeta)
    ref_walkers = jax.tree.map(lambda w: w[indices], ref_walkers)
    fol_walkers = jax.tree.map(lambda w: w[indices], fol_walkers)
    fol_weights = fol_weights[indices]
    return ref_walkers, ref_weights, fol_walkers, fol_weights


@jit
def stochastic_reconfiguration_core(walkers1, weights1, walkers2, weights2, zeta, coin):
    """Correlated SR with a randomly chosen reference system.
    coin < 0.5  -> system 1 is the reference (system 2 follows)
    coin >= 0.5 -> system 2 is the reference (system 1 follows)
    Both branches are individually unbiased, so the mixture is unbiased."""

    def ref1(_):
        return _sr_directed(walkers1, weights1, walkers2, weights2, zeta)

    def ref2(_):
        # swap roles, then swap the outputs back to (1, 2) order
        w2, wt2, w1, wt1 = _sr_directed(walkers2, weights2, walkers1, weights1, zeta)
        return w1, wt1, w2, wt2

    return jax.lax.cond(coin < 0.5, ref1, ref2, operand=None)


def stochastic_reconfiguration(prop_data1, prop_data2):   # no @jit on the wrapper
    walkers1, weights1 = prop_data1["walkers"], prop_data1["weights"]
    walkers2, weights2 = prop_data2["walkers"], prop_data2["weights"]

    key, sub_zeta, sub_coin = random.split(prop_data1["key"], 3)
    zeta = random.uniform(sub_zeta)
    coin = random.uniform(sub_coin)

    walkers1, weights1, walkers2, weights2 = stochastic_reconfiguration_core(
        walkers1, weights1, walkers2, weights2, zeta, coin)

    prop_data1 = {**prop_data1, "walkers": walkers1, "weights": weights1, "key": key}
    prop_data2 = {**prop_data2, "walkers": walkers2, "weights": weights2}
    return prop_data1, prop_data2