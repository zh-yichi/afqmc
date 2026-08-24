import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

import numpy as np

from pyscf import lib, scf
from pyscf.data import elements
from pyscf.lno import lnoccsd
from pyscf.lno import ulnoccsd

from afqmc import cholesky
from afqmc.lno_afqmc import lno_afqmc, tools, integral

from matplotlib import pyplot as plt

@jax.jit
def spectrum_from_cderi(cderi):
    """Eigenvalues of L^T L, via SVD of L. Descending, numerically clean."""
    s = jnp.linalg.svd(jnp.asarray(cderi), compute_uv=False)
    return jnp.asarray(s) ** 2

@jax.jit
def spectrum_from_eri(cderi):
    """Eigenvalues via explicit eri = L^T L + eigh. Squares the condition number."""
    L = jnp.asarray(cderi)
    eri = jnp.einsum("gp,gq->pq", L, L)
    w = jnp.linalg.eigvalsh(eri)          # ascending, symmetric PSD
    return jnp.asarray(w)[::-1]


def rank_at_thresholds(evals, decades=range(1, 13)):
    """M(delta): number of eigenvalues with lambda/lambda_max > 10^-n."""
    lam = evals / evals[0]
    return {n: int(np.count_nonzero(lam > 10.0 ** (-n))) for n in decades}


def decay_rate(evals, floor=1e-12):
    """
    Fit log10(lambda_k / lambda_0) ~ -k / N_eff.
    N_eff is 'vectors needed per decade of accuracy' -- the empirical n*N_bas rule
    says N_eff should come out near the number of basis functions.
    """
    lam = evals / evals[0]
    mask = lam > floor
    k = np.arange(len(lam))[mask]
    if k.size < 10:
        return np.nan
    slope = np.polyfit(k, np.log10(lam[mask]), 1)[0]
    return -1.0 / slope if slope < 0 else np.inf


def report(name, cderi, evals):
    naux, npair = np.shape(cderi)
    print(f"\n=== {name} ===")
    print(f"  cderi shape      : (naux={naux}, npair={npair})")
    print(f"  lambda_max       : {evals[0]:.6e}")
    print(f"  lambda_min       : {evals[-1]:.6e}")
    print(f"  numerical rank   : {int(np.count_nonzero(evals > 1e-14 * evals[0]))}"
          f"   (hard cap = min(naux, npair) = {min(naux, npair)})")
    print(f"  vectors / decade : {decay_rate(evals):.1f}")
    print("  M(delta):  " + "  ".join(
        f"1e-{n}:{m}" for n, m in rank_at_thresholds(evals, range(2, 11)).items()))


def plot_decay(spectra, fname="cderi_spectrum.png"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, ev in spectra.items():
        lam = np.maximum(ev, 1e-300) / ev[0]
        k = np.arange(1, len(lam) + 1)
        ax[0].semilogy(k, lam, lw=1.6, label=name)
        ax[1].semilogy(k / len(lam), lam, lw=1.6, label=name)

    for a, xl in zip(ax, ["eigenvalue index $k$", "fraction of spectrum $k/n$"]):
        for n in (4, 6, 8):
            a.axhline(10.0 ** (-n), color="0.7", ls=":", lw=0.8)
            a.text(a.get_xlim()[1], 10.0 ** (-n), f" $10^{{-{n}}}$",
                   va="center", fontsize=8, color="0.4")
        a.set_xlabel(xl)
        a.set_ylabel(r"$\lambda_k / \lambda_{\max}$")
        a.set_ylim(1e-16, 2)
        a.legend(frameon=False, fontsize=9)

    ax[0].set_title("spectral decay (straight line = geometric)")
    ax[1].set_title("rescaled by spectrum length")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f"\nwrote {fname}")
    return fig

def lno_cholesky_analysis(mf, lno_coeff, lno_frozen, chol_cut, 
                          spectrum_plt="cderi_spectrum.png"):
    ncore, nocc, ncas, actfrag = integral.get_las_idx(mf, lno_frozen)
    print('*** Correlation Space Size ***')
    print(f'N Occupied Orbitals: {nocc}')
    print(f'N Active Orbitals:   {ncas}')

    if isinstance(mf, scf.rhf.RHF):        
        lno_act = lno_coeff[:,actfrag]
        print("Composing CDERIs from DF")

        naux = mf.with_df.get_naoaux()
        npair = ncas*(ncas+1)//2
        naux = mf.with_df.get_naoaux()
        cderi_las = np.zeros((naux, npair))
        p1 = 0
        for cderi in mf.with_df.loop():
            cderi = lib.unpack_tril(cderi, axis=-1)
            cderi = cholesky.cderi2mo_gpu(jnp.array(cderi), lno_act)
            p0, p1 = p1, p1 + cderi.shape[0]
            cderi_las[p0:p1] = np.array(cderi)

        print(f"Raw CDERI in LAS shape: {cderi_las.shape}")
        spectra = {}
        ev = spectrum_from_cderi(cderi_las)
        spectra[f"eri"] = ev
        report(f"eri", cderi_las, ev)
        plot_decay(spectra, fname=spectrum_plt)

        print(f"Preforming Chol | DF2CHOL cutoff: {chol_cut}")
        chol_full, nchol_keep = cholesky.df2chol_gpu(jnp.array(cderi_las), max_error=chol_cut)
        chol = chol_full[:nchol_keep]
        # chol = cholesky.unpack_symmetric(chol, ncas)
        print(f'LAS Cholesky shape: {chol.shape}')

    elif isinstance(mf, scf.uhf.UHF):
        print("Composing CDERIs from DF")
        clas_coeff, a2c, b2c = integral.common_las(mf, lno_coeff, ncas, ncore, torr=1e-5)

        nclas = clas_coeff.shape[1]
        npair = nclas*(nclas+1)//2
        naux = mf.with_df.get_naoaux()
        cderi_c = np.zeros((naux, npair))
        p1 = 0
        for cderi in mf.with_df.loop():
            cderi = lib.unpack_tril(cderi, axis=-1)
            cderi = cholesky.cderi2mo_gpu(jnp.array(cderi), clas_coeff)
            p0, p1 = p1, p1 + cderi.shape[0]
            cderi_c[p0:p1] = np.array(cderi)

        print(f"Raw CDERI in cLAS shape: {cderi_c.shape}")
        # print(f"Compress CDERI in cLAS by SVD with cutoff: {chol_cut}")

        cderi_c = cholesky.unpack_symmetric(jnp.array(cderi_c), nclas)
        cderi_a = cholesky.cderi2mo_gpu(cderi_c, a2c)
        cderi_b = cholesky.cderi2mo_gpu(cderi_c, b2c)
        cderi_c = cholesky.pack_symmetric(cderi_c)

        spectra = {}
        for name, L in [("a", cderi_a), ("b", cderi_b), ("c", cderi_c)]:
            ev = spectrum_from_cderi(L)
            spectra[f"eri_{name}"] = ev
            report(f"eri_{name}", L, ev)
        plot_decay(spectra, fname=spectrum_plt)

        # cderi_clas = cholesky.compress_cderi_gpu(cderi_clas, thresh=chol_cut)
        # cderi_clas = cholesky.unpack_symmetric(cderi_clas, nclas)
        print(f"Preforming Chol | DF2CHOL cutoff: {chol_cut}")
        chol_full, nchol_keep = cholesky.df2chol_gpu(jnp.array(cderi_c), max_error=chol_cut)
        chol_c = chol_full[:nchol_keep]
        # chol_c = cholesky.unpack_symmetric(jnp.array(chol_c), nclas)
        chol_a = cholesky.cderi2mo_gpu(chol_c, a2c)
        chol_b = cholesky.cderi2mo_gpu(chol_c, b2c)
        chol_a = cholesky.unpack_symmetric(chol_a, ncas[0])
        chol_b = cholesky.unpack_symmetric(chol_b, ncas[1])
        print(f'LAS Alpha Cholesky shape: {chol_a.shape}')
        print(f'LAS Beta  Cholesky shape: {chol_b.shape}')
        chol_las = (chol_a, chol_b)

    return chol_las, spectra

def run_lno2get_chol(
        mf,
        lo_coeff,
        frag_list,
        frag_name,
        lno_thresh = 1e-5,
        chol_cut = 1e-5,
        run_frag = None,
        nfrozen = None,
        spectrum_plt="cderi_spectrum.png",
        ):

        
    print("\n ******* LNO-CALCULATION ******* \n")
    print(f"LNO THRESHOLD = {lno_thresh}")

    if nfrozen is None:
        print("LNO freezes at least the chemcore orbitals for each element!")
        nfrozen = elements.chemcore(mf.mol)

    tools.check_span(mf, lo_coeff, nfrozen, thresh=1e-6)

    mlno = lno_afqmc.get_lnoccsd(mf, lo_coeff, frag_list, nfrozen, lno_thresh)

    lno_thresh = mlno.lno_thresh
    lno_type = ['1h','1h']
    eris = mlno.ao2mo()

    nfrag_tot = len(frag_list)
    if run_frag is None:
        run_frag = range(nfrag_tot)

    frag_list = [frag_list[i] for i in run_frag]
    frag_name = [frag_name[i] for i in run_frag]
    nfrag_run = len(frag_list)

    lno_pct_occ = [None, None]
    lno_norb = [[None,None]] * nfrag_tot

    # Loop over fragment
    for ifrag, frag_idx in enumerate(run_frag):
        loidx = frag_list[ifrag]
        print("\n")
        width = 80
        msg = f" LNO-FRAGMENT [{frag_name[ifrag]}] {frag_idx+1}/({nfrag_run},{nfrag_tot}) "
        print(msg.center(width, '='))
        # print(f"Fragment Name - {frag_name[ifrag]}")
        print(f"LNO THRESHOLD - {mlno.lno_thresh}")
        print(f"PySCF NumPy Threads - {lib.num_threads()}")

        orbloc, lno_param = lno_afqmc.get_lnoparam(mf, lo_coeff, lno_thresh, lno_pct_occ, lno_norb, loidx, ifrag)

        lno_coeff, lno_frozen, uocc_loc, _ \
                    = mlno.make_las(eris, orbloc, lno_type, lno_param)
        
        if isinstance(mlno._scf, scf.rhf.RHF):
            lno_frozen, maskact \
                = lnoccsd.get_maskact(lno_frozen, mlno.mo_occ.size)
        elif isinstance(mlno._scf, scf.uhf.UHF):
            lno_frozen, maskact \
                = ulnoccsd.get_maskact(lno_frozen, [mlno.mo_occ[0].size, mlno.mo_occ[1].size])
        else:
            raise TypeError(f'unsupported mean-field type: {type(mlno._scf)}')

        cderi_las, spectra = lno_cholesky_analysis(mf, lno_coeff, lno_frozen, chol_cut, spectrum_plt)

    return cderi_las, spectra