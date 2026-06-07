import numpy as np
import opt_einsum as oe

import jax
import jax.numpy as jnp
from jax import jit, lax, random

from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD

from afqmc import slater_tools

from functools import partial
print = partial(print, flush=True)

def get_cc_amps(cc, save2disk=False, amp_file="amplitudes.npz"):

    if isinstance(cc, CCSD):
        t1 = jnp.array(cc.t1)
        t2 = jnp.array(cc.t2).transpose(0, 2, 1, 3)
        if save2disk:
            np.savez(
                amp_file, 
                t1=np.array(t1), 
                t2=np.array(t2),
                )

    elif isinstance(cc, UCCSD):
        t1a = jnp.array(cc.t1[0])
        t1b = jnp.array(cc.t1[1])
        t2aa, t2ab, t2bb = cc.t2
        t2aa = (t2aa - t2aa.transpose(0, 1, 3, 2)) / 2
        t2bb = (t2bb - t2bb.transpose(0, 1, 3, 2)) / 2
        t2aa = jnp.array(t2aa).transpose(0, 2, 1, 3)
        t2bb = jnp.array(t2bb).transpose(0, 2, 1, 3)
        t2ab = jnp.array(t2ab).transpose(0, 2, 1, 3)
        t1 = (t1a, t1b)
        t2 = (t2aa, t2ab, t2bb)
        if save2disk:
            np.savez(
                amp_file,
                t1a=np.array(t1a),
                t1b=np.array(t1b),
                t2aa=np.array(t2aa),
                t2ab=np.array(t2ab),
                t2bb=np.array(t2bb),
            )

    return t1, t2


def read_cc_amps(amp_file="amplitudes.npz"):
    data = np.load(amp_file)

    if "t1" in data:     # CCSD
        t1 = jnp.array(data["t1"])
        t2 = jnp.array(data["t2"])

    elif "t1a" in data:  # UCCSD
        t1 = (jnp.array(data["t1a"]), 
              jnp.array(data["t1b"]))
        t2 = (jnp.array(data["t2aa"]), 
              jnp.array(data["t2ab"]), 
              jnp.array(data["t2bb"]))

    return t1, t2

def get_ci_amps(cc, save2disk=False, amp_file="amplitudes.npz"):
    
    if isinstance(cc, CCSD):
        t1 = jnp.array(cc.t1)
        t2 = jnp.array(cc.t2).transpose(0, 2, 1, 3)
        ci1 = t1
        ci2 = t2 + oe.contract("ia,jb->iajb", t1, t1, backend="jax")
        if save2disk:
            np.savez(amp_file, 
                     ci1=np.array(ci1), 
                     ci2=np.array(ci2))
    
    elif isinstance(cc, UCCSD):
        t1a = jnp.array(cc.t1[0])
        t1b = jnp.array(cc.t1[1])
        t2aa, t2ab, t2bb = cc.t2
        t2aa = (t2aa - t2aa.transpose(0, 1, 3, 2)) / 2
        t2bb = (t2bb - t2bb.transpose(0, 1, 3, 2)) / 2
        t2aa = jnp.array(t2aa).transpose(0, 2, 1, 3)
        t2bb = jnp.array(t2bb).transpose(0, 2, 1, 3)
        t2ab = jnp.array(t2ab).transpose(0, 2, 1, 3)
        ci1a, ci1b = t1a, t1b
        ci2aa = t2aa + 2 * oe.contract("ia,jb->iajb", t1a, t1a, backend="jax")
        ci2ab = t2ab + oe.contract("ia,jb->iajb", t1a, t1b, backend="jax")
        ci2bb = t2bb + 2 * oe.contract("ia,jb->iajb", t1b, t1b, backend="jax")
        ci2aa = (ci2aa - ci2aa.transpose(0, 3, 2, 1)) / 2
        ci2bb = (ci2bb - ci2bb.transpose(0, 3, 2, 1)) / 2
        ci1 = (ci1a, ci1b)
        ci2 = (ci2aa, ci2ab, ci2bb)
        if save2disk:
            np.savez(
                amp_file,
                ci1a=np.array(ci1a),
                ci1b=np.array(ci1b),
                ci2aa=np.array(ci2aa),
                ci2ab=np.array(ci2ab),
                ci2bb=np.array(ci2bb),
            )

    return ci1, ci2

def read_ci_amps(amp_file="amplitudes.npz"):
    data = np.load(amp_file)

    if "ci1" in data:    # CCSD
        ci1 = data["ci1"]
        ci2 = data["ci2"]
    
    elif "ci1a" in data:  # UCCSD
        ci1 = [data["ci1a"], data["ci1b"]]
        ci2 = [data["ci2aa"], data["ci2ab"], data["ci2bb"]]

    return ci1, ci2

def decompose_rt2(t2, thresh=1e-8):

    nocc, nvir, _, _ = t2.shape
    npair = nocc * nvir

    assert t2.shape == (nocc, nvir, nocc, nvir)
    
    t2 = t2.reshape(npair, npair)
    e_val, e_vec = jnp.linalg.eigh(t2)

    # Keep only important modes
    mask = jnp.abs(e_val) > thresh
    e_val_trunc = e_val[mask]
    e_vec_trunc = e_vec[:, mask]

    tau = e_vec_trunc @ jnp.diag(jnp.sqrt(e_val_trunc + 0.0j))
    
    err = jnp.linalg.norm(t2 - tau @ tau.T)
    assert err < 10 * thresh

    tau = tau.T.reshape(-1, nocc, nvir)

    return tau

def decompose_ut2(t2, thresh=1e-8):

    t2aa, t2ab, t2bb = t2
    nocca, nvira, noccb, nvirb = t2ab.shape
    # Number of excitation pairs
    npaira = nocca * nvira
    npairb = noccb * nvirb

    assert t2aa.shape == (nocca, nvira, nocca, nvira)
    assert t2bb.shape == (noccb, nvirb, noccb, nvirb)

    # print('Decomposing Unrestricted T2 amplitudes')

    t2aa = t2aa.reshape(npaira, npaira)
    t2ab = t2ab.reshape(npaira, npairb)
    t2bb = t2bb.reshape(npairb, npairb)

    # Symmetric full t2 
    # [[ t2aa/2  t2ab   ]]
    # [[ t2ab^T  t2bb/2 ]]
    t2full = np.zeros((npaira + npairb, npaira + npairb))
    t2full[:npaira, :npaira] = 0.5 * t2aa
    t2full[npaira:, :npaira] = t2ab.T
    t2full[:npaira, npaira:] = t2ab
    t2full[npaira:, npaira:] = 0.5 * t2bb
    t2full = jnp.array(t2full)

    # t2 = LL^T
    e_val, e_vec = jnp.linalg.eigh(t2full)

    # Keep only important modes
    mask = jnp.abs(e_val) > thresh
    e_val_trunc = e_val[mask]
    e_vec_trunc = e_vec[:, mask]
    
    tau = e_vec_trunc @ jnp.diag(np.sqrt(e_val_trunc + 0.0j))
    err = jnp.linalg.norm(t2full - tau @ tau.T)
    assert err < 10 * thresh

    # alpha/beta operators for HS
    # Summation on the left to have a list of operators
    taua = tau.T[:,:npaira]
    taub = tau.T[:, npaira:]
    taua = taua.reshape(-1, nocca, nvira)
    taub = taub.reshape(-1, noccb, nvirb)

    return [taua, taub]

# @partial(jit, static_argnames=("n_walkers"))
def get_rstoccsd(mo_t1, tau, nslater, rand_key):
    rand_key, subkey = random.split(rand_key)
    
    fieldy = random.normal(
        subkey,
        shape=(nslater, tau.shape[0],),
        )
    # ytaus shape (nwalker, nocc, nvir)
    ytaus = oe.contract("wg,gia->wia", fieldy, tau, backend='jax')

    def scan_body(carry, ytau):
        # ytau_up, ytau_dn = ytau
        slater = slater_tools.rthouless(mo_t1, ytau)
        return carry, slater

    # scan iterates over leading axis (n_walkers) of (ytaus_up, ytaus_dn)
    _, slaters = lax.scan(scan_body, None, ytaus)

    return slaters, rand_key

def get_ustoccsd(mo_t1, tau, nslater, rand_key):
    rand_key, subkey = random.split(rand_key)
    
    field_y = random.normal(
        subkey,
        shape=(nslater, tau[0].shape[0],),
        )
    # ytau shape (nslater, nocc, nvir)
    ytau_a = oe.contract("wg,gia->wia", field_y, tau[0], backend='jax')
    ytau_b = oe.contract("wg,gia->wia", field_y, tau[1], backend='jax')
    ytau = (ytau_a, ytau_b)

    def scan_body(carry, ytau):
        # ytau_up, ytau_dn = ytau
        slater = slater_tools.uthouless(mo_t1, ytau)
        return carry, slater

    # scan iterates over leading axis (n_walkers) of (ytaus_up, ytaus_dn)
    _, slaters = lax.scan(scan_body, None, ytau)

    return slaters, rand_key

def get_stoccsd(mo_t1, tau, nslater, rand_key):
    if isinstance(mo_t1, jax.Array) and isinstance(tau, jax.Array):
        return get_rstoccsd(mo_t1, tau, nslater, rand_key)
    elif isinstance(mo_t1, (tuple, list)) and isinstance(tau, (tuple, list)):
        return get_ustoccsd(mo_t1, tau, nslater, rand_key)

# @partial(jit, static_argnames=("n_walkers"))
# def get_rccsd_walkers(prop_data, wave_data, n_walkers):
#     prop_data["key"], subkey = random.split(prop_data["key"])
    
#     fieldy = random.normal(
#         subkey,
#         shape=(
#             n_walkers,
#             wave_data['tau'].shape[0],
#         ),
#     )
#     # ytaus shape (nwalker, nocc, nvir)
#     ytaus = oe.contract("wg,gia->wia", fieldy, wave_data['tau'], backend='jax')

#     slaters = vmap(lambda y: rthouless(wave_data['mo_t'], y))(ytaus)

#     # mo_t = wave_data['mo_t']

#     # def scan_body(carry, ytau):
#     #     # ytau_up, ytau_dn = ytau
#     #     slater = rthouless(wave_data['mo_t'], ytau)
#     #     return carry, slater

#     # # scan iterates over leading axis (n_walkers) of (ytaus_up, ytaus_dn)
#     # _, slaters = lax.scan(scan_body, None, ytaus)

#     return slaters, prop_data

# @partial(jit, static_argnames=("n_walkers"))
# def get_uccsd_walkers(prop_data, wave_data, n_walkers):
#     prop_data["key"], subkey = random.split(prop_data["key"])
    
#     fieldy = random.normal(
#         subkey,
#         shape=(
#             n_walkers,
#             wave_data['tau'][0].shape[0],
#         ),
#     )
#     # ytaus shape (nwalker, nocc, nvir)
#     ytaus_up = oe.contract("wg,gia->wia", fieldy, wave_data['tau'][0], backend='jax')
#     ytaus_dn = oe.contract("wg,gia->wia", fieldy, wave_data['tau'][1], backend='jax')

#     mo_t = (wave_data["mo_ta"], wave_data["mo_tb"])
    
#     slaters_up, slaters_dn = vmap(
#         lambda yu, yd: uthouless(mo_t, (yu, yd)))(ytaus_up, ytaus_dn)

#     # mo_t = [wave_data['mo_ta'], wave_data['mo_tb']]

#     # def scan_body(carry, ytau):
#     #     ytau_up, ytau_dn = ytau
#     #     slater_up, slater_dn = uthouless(mo_t, [ytau_up, ytau_dn])
#     #     return carry, (slater_up, slater_dn)

#     # # scan iterates over leading axis (n_walkers) of (ytaus_up, ytaus_dn)
#     # _, (slaters_up, slaters_dn) = lax.scan(scan_body, None, (ytaus_up, ytaus_dn),)

#     return [slaters_up, slaters_dn], prop_data


# def get_ccsd_walkers(prop_data, wave_data, n_walkers, walker_type):
#     if walker_type == "rhf":
#         if "tau" not in wave_data:
#             wave_data["tau"] = decompose_rt2(wave_data["t2"])
#         return get_rccsd_walkers(prop_data, wave_data, n_walkers)
#     elif walker_type == "uhf":
#         if "tau" not in wave_data:
#             wave_data["tau"] = decompose_ut2([wave_data["t2aa"],
#                                               wave_data["t2ab"],
#                                               wave_data["t2bb"]])
#         return get_uccsd_walkers(prop_data, wave_data, n_walkers)
#     else:
#         raise ValueError(f"unsupport CCSD initial walker_type: {walker_type}")