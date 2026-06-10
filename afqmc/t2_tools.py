import jax
import jax.numpy as jnp
import opt_einsum as oe
import numpy as np

from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD

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

    npaira = nocca * nvira
    npairb = noccb * nvirb

    assert t2aa.shape == (nocca, nvira, nocca, nvira)
    assert t2bb.shape == (noccb, nvirb, noccb, nvirb)

    t2aa = t2aa.reshape(npaira, npaira)
    t2ab = t2ab.reshape(npaira, npairb)
    t2bb = t2bb.reshape(npairb, npairb)

    # Symmetric full t2 
    # [[ t2aa/2  t2ab   ]]
    # [[ t2ab^T  t2bb/2 ]]
    # t2full = np.zeros((npaira + npairb, npaira + npairb))
    # t2full[:npaira, :npaira] = 0.5 * t2aa
    # t2full[npaira:, :npaira] = t2ab.T
    # t2full[:npaira, npaira:] = t2ab
    # t2full[npaira:, npaira:] = 0.5 * t2bb
    # t2full = jnp.array(t2full)
    t2full = jnp.block([[0.5 * t2aa, t2ab],
                        [t2ab.T, 0.5 * t2bb]])
    # t2 = LL^T
    e_val, e_vec = jnp.linalg.eigh(t2full)

    # Keep only important modes
    mask = jnp.abs(e_val) > thresh
    e_val_trunc = e_val[mask]
    e_vec_trunc = e_vec[:, mask]
    
    tau = e_vec_trunc @ jnp.diag(jnp.sqrt(e_val_trunc + 0.0j))
    err = jnp.linalg.norm(t2full - tau @ tau.T)
    assert err < 10 * thresh

    # alpha/beta operators for HS
    # Summation on the left to have a list of operators
    taua = tau.T[:,:npaira]
    taub = tau.T[:, npaira:]
    taua = taua.reshape(-1, nocca, nvira)
    taub = taub.reshape(-1, noccb, nvirb)

    return (taua, taub)

def decompose_t2(t2, thresh=1e-8):
    if isinstance(t2, jax.array) and len(t2.shape) == 4:
        return decompose_rt2(t2, thresh)
    elif isinstance(t2, tuple) and len(t2) == 3:
        return decompose_ut2(t2, thresh)
    else:
        raise TypeError(f"T2 amplitude should either be a rank-4 tensor"
                        f"or a tuple of 3 (aa,ab,bb) rank-4 tensors.")


def get_cc_amps(cc, save2disk, amp_file):

    if isinstance(cc, UCCSD):
        t1a = np.array(cc.t1[0])
        t1b = np.array(cc.t1[1])
        t2aa, t2ab, t2bb = cc.t2
        t2aa = (t2aa - t2aa.transpose(0, 1, 3, 2)) / 2
        t2bb = (t2bb - t2bb.transpose(0, 1, 3, 2)) / 2
        t2aa = t2aa.transpose(0, 2, 1, 3)
        t2bb = t2bb.transpose(0, 2, 1, 3)
        t2ab = t2ab.transpose(0, 2, 1, 3)
        if save2disk:
            np.savez(
                amp_file,
                t1a=t1a,
                t1b=t1b,
                t2aa=t2aa,
                t2ab=t2ab,
                t2bb=t2bb,
            )
        t1 = (jnp.array(t1a), 
              jnp.array(t1b))
        t2 = (jnp.array(t2aa), 
              jnp.array(t2ab), 
              jnp.array(t2bb))

    elif isinstance(cc, CCSD):
        t1 = np.array(cc.t1)
        t2 = cc.t2
        t2 = t2.transpose(0, 2, 1, 3)
        if save2disk:
            np.savez(amp_file, t1=t1, t2=t2)
        t1 = jnp.array(t1) 
        t2 = jnp.array(t2)

    return t1, t2


def read_cc_amps(amp_file):
    data = np.load(amp_file)

    if "t1a" in data:  # UCCSD
        t1 = (jnp.array(data["t1a"]), 
              jnp.array(data["t1b"]))
        t2 = (jnp.array(data["t2aa"]), 
              jnp.array(data["t2ab"]), 
              jnp.array(data["t2bb"]))
    else:              # RCCSD
        t1 = jnp.array(data["t1"])
        t2 = jnp.array(data["t2"])

    return t1, t2

def rcc2ci(t1, t2):
    ci1 = t1
    ci2 = t2 + oe.contract("ia,jb->iajb", t1, t1, backend='jax')
    return ci1, ci2

def ucc2ci(t1, t2):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    ci1 = (t1a, t1b)
    
    ci2aa = t2aa + 2 * oe.contract("ia,jb->iajb", t1a, t1a, backend='jax')
    ci2ab = t2ab + oe.contract("ia,jb->iajb", t1a, t1b, backend='jax')
    ci2bb = t2bb + 2 * oe.contract("ia,jb->iajb", t1b, t1b, backend='jax')
    ci2aa = (ci2aa - ci2aa.transpose(0, 3, 2, 1)) / 2
    ci2bb = (ci2bb - ci2bb.transpose(0, 3, 2, 1)) / 2
    ci2 = (ci2aa, ci2ab, ci2bb)
    return ci1, ci2

def cc2ci(t1, t2):
    if isinstance(t2, jax.Array) and len(t2.shape) == 4:
        return rcc2ci(t1, t2)
    elif isinstance(t2, tuple) and len(t2) == 3:
        return ucc2ci(t1, t2)
    else:
        raise TypeError(f"restricted T1 T2 should be jax.array "
                        f"and tuple for unrestricted case")