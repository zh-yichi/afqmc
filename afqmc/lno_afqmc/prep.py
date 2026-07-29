import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
from jax import scipy as jsp
import opt_einsum as oe

import h5py, pickle, time
import numpy as np
from pyscf import lib
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD
from functools import partial

from afqmc import cholesky, prep, t2_tools
from afqmc.lno_afqmc import propagation, sampling, integral, cfs_sampling
from afqmc.lno_afqmc import wavefunctions_restricted as lno_wavefunctions
from afqmc.lno_afqmc import wavefunctions_unrestricted as ulno_wavefunctions

from afqmc import prep

init_hf_prop_data = prep.init_hf_prop_data

print = partial(print, flush=True)


def las_size(mf, frozen):
    mol = mf.mol
    nocc = np.count_nonzero(mf.mo_occ)
    actfrag = np.array([i for i in range(mol.nao) if i not in frozen])
    # frzocc = np.array([i for i in range(nocc) if i in frozen])
    actocc = np.array([i for i in range(nocc) if i in actfrag])
    actvir = np.array([i for i in range(nocc, mol.nao) if i in actfrag])
    # nfrzocc = len(frzocc)
    nactocc = len(actocc)
    nactvir = len(actvir)
    # nactorb = len(actfrag)
    return nactocc, nactvir


def kind(x):
    """don't support general spin-orbital"""
    if isinstance(x, np.ndarray) and x.ndim == 2:
        return "restricted"
    if (isinstance(x, (tuple, list))
            and len(x) == 2
            and all(isinstance(m, np.ndarray) and m.ndim == 2 for m in x)):
        return "unrestricted"
    return "other"


def auto_qmc_options(options={}):

    options["dt"] = options.get("dt", 0.005)
    options["n_walkers"] = options.get("n_walkers", 300)
    options["n_prop_steps"] = options.get("n_prop_steps", 50)
    options["eql_time"] = options.get("eql_time", 20)
    options["n_blocks"] = options.get("n_blocks", 500)
    options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))
    options["n_batch"] = options.get("n_batch", 1)
    options['max_memory'] = options.get("max_memory", 2000)
    options["nchol_chunk"] = options.get("nchol_chunk", 100)
    options['mix_precision'] = options.get("mix_precision", True)
    options["max_error"] = options.get("max_error", 0.0)
    options["n_exp_terms"] = options.get("n_exp_terms",6)
    options["trial"] = options.get("trial", "rhf")

    if "u" not in options["trial"]:
        options["walker_type"] = options.get("walker_type", "rhf")
    elif "u" in options["trial"]:
        options["walker_type"] = options.get("walker_type", "uhf")
    
    return options

def proj_cc_amplitude(prjlo, wave_data, options):

    if "t1" in wave_data:
        # ---- restricted ----
        assert isinstance(prjlo, (np.ndarray, jax.Array))

        t1 = jnp.array(wave_data["t1"])
        t2 = jnp.array(wave_data["t2"])
        if "pt2" not in options["trial"]:
            wave_data["t1"] = oe.contract('ia,ik->ka', t1, prjlo, backend='jax')
        else:  # "pt2" in options["trial"]
            nocc, nvir = t1.shape
            norb = nocc + nvir
            t1_full = np.zeros((norb, norb))
            t1_full[:nocc, nocc:] = t1
            wave_data['exp_t1']  = jsp.linalg.expm(jnp.array(t1_full))
            wave_data['exp_mt1'] = jsp.linalg.expm(jnp.array(-t1_full))
        # t2 projected in both cases
        wave_data["t2"] = oe.contract('iajb,ik->kajb', t2, prjlo, backend='jax')

    elif "t1a" in wave_data:
        # ---- unrestricted ----
        assert isinstance(prjlo, (tuple, list))

        t1a = jnp.array(wave_data["t1a"])
        t1b = jnp.array(wave_data["t1b"])
        t2aa = jnp.array(wave_data["t2aa"])
        t2ab = jnp.array(wave_data["t2ab"])
        t2bb = jnp.array(wave_data["t2bb"])
        if "pt2" not in options["trial"]:
            wave_data["t1a"] = oe.contract('ia,ik->ka', t1a, prjlo[0], backend='jax')
            wave_data["t1b"] = oe.contract('ia,ik->ka', t1b, prjlo[1], backend='jax')
        else:  # "pt2" in options["trial"]
            nocca, nvira = t1a.shape
            noccb, nvirb = t1b.shape
            norba = nocca + nvira
            norbb = noccb + nvirb
            t1a_full = np.zeros((norba, norba))
            t1a_full[:nocca, nocca:] = t1a
            t1b_full = np.zeros((norbb, norbb))
            t1b_full[:noccb, noccb:] = t1b
            wave_data['exp_t1a']  = jsp.linalg.expm(jnp.array(t1a_full))
            wave_data['exp_mt1a'] = jsp.linalg.expm(jnp.array(-t1a_full))
            wave_data['exp_t1b']  = jsp.linalg.expm(jnp.array(t1b_full))
            wave_data['exp_mt1b'] = jsp.linalg.expm(jnp.array(-t1b_full))
        # t2 projected in both cases
        wave_data["t2aa"] = oe.contract('iajb,ik->kajb', t2aa, prjlo[0], backend='jax')
        wave_data["t2ab"] = oe.contract('iajb,ik->kajb', t2ab, prjlo[0], backend='jax')
        wave_data["t2ba"] = oe.contract('jbia,ik->kajb', t2ab, prjlo[1], backend='jax')
        wave_data["t2bb"] = oe.contract('iajb,ik->kajb', t2bb, prjlo[1], backend='jax')

    else:
        raise KeyError(
            "wave_data has neither 't1' (restricted) nor 't1a' (unrestricted) "
            "amplitudes."
        )
    
    return wave_data

def get_wavefunction(options, nocc, norb, prjlo, amp_file):
    wave_data = {}
    if "cc" in options["trial"]:
        wave_data = prep.load_cc_amplitude(wave_data, amp_file=amp_file)
        wave_data = proj_cc_amplitude(prjlo, wave_data, options)

    if "u" not in options["trial"]:
        assert isinstance(prjlo, (np.ndarray, jax.Array))
        nelec_sp, norb = (int(nocc), int(nocc)), int(norb)
        wave_data["mo_coeff"] = jnp.eye(norb)[:, :nocc]
        wave_data['prjlo'] = jnp.array(prjlo)
        if options["trial"] == "rhf":
            trial = lno_wavefunctions.rhf(norb, nelec_sp,
                                          n_batch=options["n_batch"])
        elif options["trial"] == "ptccsd_ad":
            trial = lno_wavefunctions.ptccsd_ad(norb, nelec_sp,
                                                n_batch=options["n_batch"])
        elif options["trial"] == "ptccsd":
            trial = lno_wavefunctions.ptccsd(norb, nelec_sp,
                                             n_batch=options["n_batch"])
        elif "pt2ccsd" in options["trial"]:
            if options["trial"] == "pt2ccsd":
                trial = lno_wavefunctions.pt2ccsd(
                    norb, nelec_sp,
                    n_batch=options["n_batch"],
                    nchol_chunk=options["nchol_chunk"],
                    mix_precision=options["mix_precision"],
                )
            elif "ad" in options["trial"]:
                trial = lno_wavefunctions.pt2ccsd_ad(norb, nelec_sp,
                                                     n_batch=options["n_batch"])
            else:
                raise ValueError(f"Unrecognized restricted trial: {options['trial']!r}")
        else:
            raise ValueError(f"Unrecognized restricted trial: {options['trial']!r}")

    elif "u" in options["trial"]:
        assert isinstance(prjlo, (tuple, list))
        nelec_sp = (int(nocc[0]), int(nocc[1]))
        norb = (int(norb[0]), int(norb[1]))
        wave_data["mo_coeff"] = (jnp.eye(norb[0])[:, :nocc[0]],
                                 jnp.eye(norb[1])[:, :nocc[1]])
        wave_data['prjlo'] = (jnp.array(prjlo[0]), jnp.array(prjlo[1]))
        if options["trial"] == "uhf":
            trial = ulno_wavefunctions.uhf(norb, nelec_sp, n_batch=options["n_batch"])
        elif options["trial"] == "uptccsd_ad":
            trial = ulno_wavefunctions.uptccsd_ad(norb, nelec_sp, n_batch=options["n_batch"])
        elif options["trial"] == "uptccsd":
            trial = ulno_wavefunctions.uptccsd(norb, nelec_sp, n_batch=options["n_batch"])
        elif "upt2ccsd" in options["trial"]:
            if "ad" in options["trial"]:
                trial = ulno_wavefunctions.upt2ccsd_ad(norb, nelec_sp, n_batch=options["n_batch"])
            elif "alpha" in options["trial"]:
                trial = ulno_wavefunctions.upt2ccsd_alpha(
                    norb, nelec_sp,
                    n_batch=options["n_batch"],
                    nchol_chunk=options["nchol_chunk"],
                    mix_precision=options['mix_precision'],
                )
            elif "beta" in options["trial"]:
                trial = ulno_wavefunctions.upt2ccsd_beta(
                    norb, nelec_sp,
                    n_batch=options["n_batch"],
                    nchol_chunk=options["nchol_chunk"],
                    mix_precision=options['mix_precision'],
                )
            elif options["trial"] == "upt2ccsd":
                trial = ulno_wavefunctions.upt2ccsd(
                    norb, nelec_sp,
                    n_batch=options["n_batch"],
                    nchol_chunk=options["nchol_chunk"],
                    mix_precision=options["mix_precision"],
                )
            else:
                raise ValueError(f"Unrecognized unrestricted trial: {options['trial']!r}")
        else:
            raise ValueError(f"Unrecognized unrestricted trial: {options['trial']!r}")
    else:
        raise ValueError(f"Cannot classify trial as restricted or unrestricted: {options['trial']!r}")

    return wave_data, trial

def get_propagator(options):
    if options["walker_type"] == "rhf":
        prop = propagation.propagator_restricted(
            options["dt"], 
            options["n_walkers"], 
            options["n_exp_terms"],
            options["n_batch"]
        )
    elif options["walker_type"] == "uhf":
        prop = propagation.propagator_unrestricted(
                options["dt"],
                options["n_walkers"],
                n_batch=options["n_batch"],
            )
    return prop

def get_sampler(options, nchol):
    if  'pt' in options['trial']:
        if '2' in options['trial']:
            sampler = sampling.sampler_pt2(
                options["n_prop_steps"],
                options["n_blocks"],
                nchol,)
        else:
            sampler = sampling.sampler_pt(
                options["n_prop_steps"],
                options["n_blocks"],
                nchol,)
    else:
        sampler = sampling.sampler(
                options["n_prop_steps"],
                options["n_blocks"],
                nchol,)
    return sampler

def get_cfs_sampler(options, nchol):
    if  'pt2' in options['trial']:
        sampler = cfs_sampling.sampler_pt2(
            n_prop_steps=options["n_prop_steps"],
            n_blocks=options["n_blocks"],
            n_walkers=options["n_walkers"],
            n_chol=nchol)
    else:
        sampler = cfs_sampling.sampler(
                on_prop_steps=options["n_prop_steps"],
                n_blocks=options["n_blocks"],
                n_walkers=options["n_walkers"],
                n_chol=nchol)
    return sampler

def get_hamiltonian(h0, h1, chol, emf):
    ham_data = {}
    ham_data["h0"] = h0
    ham_data["E0"] = emf
    # h1 symmetrization is identical in both branches

    if isinstance(chol, (jax.Array, np.ndarray)):
        ham_data["h1"] = (jnp.array(h1 + h1.T) / 2.0,
                          jnp.array(h1 + h1.T) / 2.0)
        ham_data["chol"] = chol.reshape(chol.shape[0], -1)
    elif isinstance(chol, (list, tuple)):
        nchola = chol[0].shape[0]
        ncholb = chol[1].shape[0]
        assert nchola == ncholb, f"nchol mismatch: alpha={nchola}, beta={ncholb}"
        nchol = nchola
        ham_data["h1"] = (jnp.array(h1[0] + h1[0].T) / 2.0,
                          jnp.array(h1[1] + h1[1].T) / 2.0)
        ham_data["chol"] = (jnp.array(chol[0].reshape(nchol, -1)),
                            jnp.array(chol[1].reshape(nchol, -1)))
    else:
        raise TypeError(
            f"chol must be an ndarray/jax.Array or a list/tuple of two, "
            f"got {type(chol).__name__}."
        )
    return ham_data


def init_afqmc(
        options=None,
        option_file="options.bin",
        amp_file="amplitudes.npz",
        chol_file="FCIDUMP_chol"
        ):
    
    options = prep.get_qmc_options(options, option_file)

    nocc, norb, nchol, h0, h1, chol, emf, prjlo, = integral.load_integral(filename=chol_file)

    print("\nQMC System")
    print(f"Number of Occ. Orbitals:      {nocc}")
    print(f"Number of Act. Orbitals:      {norb}")
    print(f"Number of Chol. Vectors:      {nchol}")
    print(f"Mean-Field Energy:            {emf:.8f}")

    options["nchol_chunk"] = cholesky.chunk_chol(chol, options["nchol_chunk"], 
                                                 options["max_memory"]/options["n_walkers"])

    ham_data = get_hamiltonian(h0, h1, chol, emf)

    wave_data, trial = get_wavefunction(options, nocc, norb, prjlo, amp_file)

    prop = get_propagator(options)

    sampler = get_sampler(options, nchol)

    return ham_data, prop, trial, wave_data, sampler, options

def init_cfs_afqmc(options=None, option_file="options.bin",
                   amp_file1="amplitudes1.npz", chol_file1="FCIDUMP_chol1",
                   amp_file2="amplitudes2.npz", chol_file2="FCIDUMP_chol2"
                   ):
    
    options = prep.get_qmc_options(options, option_file)

    nocc1, norb1, nchol1, h01, h11, chol1, emf1, prjlo1 \
        = integral.load_integral(filename=chol_file1)
    nocc2, norb2, nchol2, h02, h12, chol2, emf2, prjlo2 \
        = integral.load_integral(filename=chol_file2)
    
    assert nchol1 == nchol2

    print(f"\n{'':<16}  {'QMC System 1':>12}  {'QMC System 2':>12}")
    print(f"{'nelectron:':<16}  {str(nocc1):>12}  {str(nocc2):>12}")
    print(f"{'norbital:':<16}  {str(norb1):>12}  {str(norb2):>12}")
    print(f"{'ncholesky:':<16}  {str(nchol1):>12}  {str(nchol2):>12}")
    print(f"{'E(mean-field):':<16}  {emf1:>12.8f}  {emf2:>12.8f}")

    options["nchol_chunk"] = cholesky.chunk_chol(chol1, options["nchol_chunk"], 
                                                 options["max_memory"]/options["n_walkers"])

    ham_data1 = get_hamiltonian(h01, h11, chol1, emf1)
    ham_data2 = get_hamiltonian(h02, h12, chol2, emf2)

    wave_data1, trial1 = get_wavefunction(options, nocc1, norb1, prjlo1, amp_file1)
    wave_data2, trial2 = get_wavefunction(options, nocc2, norb2, prjlo2, amp_file2)

    prop = get_propagator(options)
    sampler = get_cfs_sampler(options, nchol1)

    return (prop, sampler, options, 
            trial1, ham_data1, wave_data1, 
            trial2, ham_data2, wave_data2)
