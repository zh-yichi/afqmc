from functools import partial
print = partial(print, flush=True)

print("\nLNO-AFQMC (frozen virtual) Started")

from afqmc import config
config.setup_jax()

import time
import numpy as np
from jax import numpy as jnp

from afqmc import sampling as sp
from afqmc.lno_afqmc import sampling as lsp
from afqmc.lno_afqmc import prep

init_time = time.time()

ham_data, prop, trial, wave_data, sampler, options = (prep.init_afqmc())

print(f"Trial:   {trial}")
print(f"Sampler: {sampler}")

wave_data["rdm1"] = trial.get_rdm1(wave_data)
ham_data = trial._build_measurement_intermediates(ham_data, wave_data)
ham_data = prop._build_propagation_intermediates(ham_data, trial, wave_data)
prop_data = prep.init_hf_prop_data(trial, wave_data, ham_data, options)

# ---------------------------------------------------------------- options --
# The virtuals are assumed to be ordered by decreasing occupation (natural
# orbitals), so dropping the last frozen_vir of them from the T2 terms is a
# small, quickly converging perturbation:
#
#   E_frag = <E_frozen>_(all blocks) + <E_full - E_frozen>_(first n_corr blocks)
#
# The correction is measured on the same walkers as the frozen energy, so the
# noise common to both branches cancels and it converges in far fewer blocks
# than either branch on its own.
norb_sp = trial.norb if isinstance(trial.norb, (tuple, list)) else (trial.norb,) * 2
nocc_sp = trial.nelec if isinstance(trial.nelec, (tuple, list)) else (trial.nelec,) * 2
nvir_sp = tuple(no - ne for no, ne in zip(norb_sp, nocc_sp))

# a third of the virtuals is frozen by default; alpha and beta spaces differ for
# an unrestricted trial, so the count is taken from their average.  For a
# restricted trial norb/nelec are scalars and nvir_sp is just (nvir, nvir).
frozen_rate = options.get("frozen_vir_rate",  3)
frozen_vir = int(options.get("frozen_vir", sum(nvir_sp) / len(nvir_sp) / frozen_rate))
n_corr_blocks = int(options.get("n_corr_blocks", min(200, sampler.n_blocks // 5)))
delta_error = float(options.get("delta_error", 0.25 * options["max_error"]))

# Both the restricted and the unrestricted lno pt2ccsd trials (and their
# sto_chol variants) accept frozen_vir; anything else does not.
if "pt2ccsd" not in options["trial"]:
    raise ValueError("frozen virtual sampling needs an lno pt2ccsd trial "
                     f"(calc_ept2_frag of {options['trial']!r} takes no frozen_vir)")
if frozen_vir >= min(nvir_sp):
    raise ValueError(f"frozen_vir = {frozen_vir} exceeds the number of virtuals {nvir_sp}")
if frozen_vir == 0:
    print("\nWarning: frozen_vir = 0, the frozen branch is the full one")
n_corr_blocks = min(n_corr_blocks, sampler.n_blocks)
if n_corr_blocks < 2:
    raise ValueError("n_corr_blocks must be >= 2 to measure the correction")

print("\nFrozen Virtual Sampling")
print(f"Virtual orbitals (a,b):       {nvir_sp}")
print(f"Frozen virtuals:              {frozen_vir}")
print(f"Kept virtuals (a,b):          {tuple(nv - frozen_vir for nv in nvir_sp)}")
print(f"Correlated (paired) blocks:   {n_corr_blocks}")
print(f"Target correction error:      {delta_error:.6f}")


def get_ept2orb(trial, prop_data, ham_data, wave_data, frozen_vir=None):

    eg_sp, t1_sp, t2frg_sp, e0frg_sp, e1frg_sp, e0_sp \
        = trial.calc_ept2_frag(prop_data['walkers'], ham_data, wave_data,
                               frozen_vir=frozen_vir)

    wt_sp = prop_data["weights"]
    wp_sp = wt_sp * t1_sp

    wt     = jnp.sum(wt_sp)
    eg     = jnp.sum(wt_sp * eg_sp) / wt

    wp     = jnp.sum(wp_sp)
    t2frg  = jnp.sum(wp_sp * t2frg_sp) / wp
    e0frg  = jnp.sum(wp_sp * e0frg_sp) / wp
    e1frg  = jnp.sum(wp_sp * e1frg_sp) / wp
    e0     = jnp.sum(wp_sp * e0_sp) / wp

    ept2_frg = jnp.real(e0frg + e1frg - t2frg * e0)

    return eg.real, ept2_frg


_, ept2orb_frz = get_ept2orb(trial, prop_data, ham_data, wave_data, frozen_vir)
_, ept2orb = get_ept2orb(trial, prop_data, ham_data, wave_data, None)

print("\nEquilibration")
print(f"Initial Orbital energy (frozen): {ept2orb_frz:.5f}")
print(f"Initial Orbital energy (full):   {ept2orb:.5f}")
print(f"Initial difference:              {ept2orb - ept2orb_frz:.5f}")
print(f"{'inv_T':>5s}  {'nodes':>5s}  {'weight':>10s}  {'energy':>10s}  {'error':>8s}  "
      f"{'runTime':>8s}")

print(f"{0.:5.2f}  {prop_data['n_killed_walkers']:5d}  {np.sum(prop_data['weights']):10.5f}  "
      f"{prop_data['e_estimate']:10.5f}  {0.:8.5f}  {time.time()-init_time:8.2f}")

block_time = prop.dt * options["n_prop_steps"]
neql_block = int(-(-options["eql_time"] // block_time))

sampler_eq = sp.sampler(
    n_prop_steps = options["n_prop_steps"],
    n_blocks = neql_block,
    n_chol = sampler.n_chol,
    )

for n in range(1, neql_block+1):
    prop_data, (wt, eg, err) \
        = sampler_eq.block_sample(prop_data, ham_data, prop, trial, wave_data)

    prop_data["n_killed_walkers"] = 0

    if (n+1) % (min(max(neql_block // 10, 1), 20)) == 0 and n > 0:
        nodes = prop_data["n_killed_walkers"]
        print(f"{(n+1)*block_time:5.2f}  {nodes:5d}  {wt:10.5f}  {eg:10.5f}  {err:8.5f}  "
              f"{time.time() - init_time:8.2f}")

# ------------------------------------------------------- sampling buffers --
# Branch 0 (frozen) is accumulated over every block, branch 1 (full) only over
# the paired blocks. wt/eg/wp are shared: they do not depend on frozen_vir.
wt_sp = np.zeros(sampler.n_blocks,dtype="float64")
eg_sp = np.zeros(sampler.n_blocks,dtype="float64")

wp_sp    = np.zeros(sampler.n_blocks,dtype="complex128")
t2frg_sp = np.zeros(sampler.n_blocks,dtype="complex128")
e0frg_sp = np.zeros(sampler.n_blocks,dtype="complex128")
e1frg_sp = np.zeros(sampler.n_blocks,dtype="complex128")
e0_sp    = np.zeros(sampler.n_blocks,dtype="complex128")

wp1_sp    = np.zeros(n_corr_blocks,dtype="complex128")
t2frg1_sp = np.zeros(n_corr_blocks,dtype="complex128")
e0frg1_sp = np.zeros(n_corr_blocks,dtype="complex128")
e1frg1_sp = np.zeros(n_corr_blocks,dtype="complex128")
e01_sp    = np.zeros(n_corr_blocks,dtype="complex128")

nodes = 0
delta, delta_err = 0.0, 0.0

# ------------------------------------------------- 1) correlated sampling --
# A trial that samples its Cholesky sum needs a fresh key each block, otherwise it
# reuses one fixed key forever and its sampling noise never averages down.  The
# sto_chol pair sampler also hands the *same* key to both branches, so they draw
# the same tail vectors and that noise cancels in E_full - E_frozen.
_sto_chol = "sto_chol" in options["trial"]
_pair_cls = lsp.sampler_pt2_frozen_vir_sto_chol if _sto_chol else lsp.sampler_pt2_frozen_vir
sampler_pair = _pair_cls(
    n_prop_steps = options["n_prop_steps"],
    n_blocks = n_corr_blocks,
    n_chol = sampler.n_chol,
    frozen_vir = frozen_vir,
    )

print(f"pair sampler: {sampler_pair}")

print("\nCorrelated Sampling Blocks (frozen + full)")
print(f"{'N':>4s}  {'nodes':>5s}  {'weight':>10s}  {'E(Guide)':>12s}  {'Error':>8s}  "
      f"{'wtp(frz)':>10s}  {'E_frozen':>10s}  {'Error':>8s}  "
      f"{'wtp(full)':>10s}  {'E_full':>10s}  {'Error':>8s}  "
      f"{'Delta':>9s}  {'Error':>8s}  {'Time':>8s}")

n_paired = 0

for n in range(n_corr_blocks):
    # 0-frozen 1-full
    prop_data, (wt, eg, wp, t2frg, e0frg, e1frg, e0,
                    eg1, wp1, t2frg1, e0frg1, e1frg1, e01) = \
        sampler_pair.block_sample(prop_data, ham_data, prop, trial, wave_data)

    wt_sp[n] = wt
    eg_sp[n] = eg

    wp_sp[n] = wp
    t2frg_sp[n] = t2frg
    e0frg_sp[n] = e0frg
    e1frg_sp[n] = e1frg
    e0_sp[n]    = e0

    wp1_sp[n]    = wp1
    t2frg1_sp[n] = t2frg1
    e0frg1_sp[n] = e0frg1
    e1frg1_sp[n] = e1frg1
    e01_sp[n]    = e01

    n_paired = n + 1
    nodes += prop_data["n_killed_walkers"]
    prop_data["n_killed_walkers"] = 0

    if (n+1) % (min(max(n_corr_blocks // 10, 1), 20)) == 0 and n > 0:
        weight, guide, guide_err = sp.blocking(wt_sp[:n+1], eg_sp[:n+1])
        weighp, efrz, efrz_err = lsp.ept2frg_blocking(
            wp_sp[:n+1], t2frg_sp[:n+1], e0frg_sp[:n+1], e1frg_sp[:n+1], e0_sp[:n+1],)
        # the full branch carries its own weightp, which must track the frozen
        # one block by block: t1 does not depend on frozen_vir
        weighp1, efull, efull_err = lsp.ept2frg_blocking(
            wp1_sp[:n+1], t2frg1_sp[:n+1], e0frg1_sp[:n+1], e1frg1_sp[:n+1], e01_sp[:n+1],)
        delta, delta_err = lsp.ept2frg_delta_blocking(
            wp_sp[:n+1],
            t2frg_sp[:n+1], e0frg_sp[:n+1], e1frg_sp[:n+1], e0_sp[:n+1],
            t2frg1_sp[:n+1], e0frg1_sp[:n+1], e1frg1_sp[:n+1], e01_sp[:n+1],)

        print(f"{n+1:4d}  {nodes:5d}  {wt:10.5f}  {guide:12.5f}  {guide_err:8.5f}  "
              f"{weighp.real:10.5f}  {efrz.real:10.5f}  {efrz_err:8.5f}  "
              f"{weighp1.real:10.5f}  {efull.real:10.5f}  {efull_err:8.5f}  "
              f"{delta:9.5f}  {delta_err:8.5f}  {time.time() - init_time:8.2f}")

        prop_data["e_estimate"] = 0.8 * prop_data["e_estimate"] + 0.2 * eg.real

        if delta_error > 0 and delta_err < delta_error and n > 20:
            print(f"Correction converged after {n+1} paired blocks")
            break

# correction over every paired block, whether or not the loop stopped on a
# printing block
if n_paired > 1:
    delta, delta_err = lsp.ept2frg_delta_blocking(
        wp_sp[:n_paired],
        t2frg_sp[:n_paired], e0frg_sp[:n_paired], e1frg_sp[:n_paired], e0_sp[:n_paired],
        t2frg1_sp[:n_paired], e0frg1_sp[:n_paired], e1frg1_sp[:n_paired], e01_sp[:n_paired],)

# ----------------------------------------------------- 2) frozen sampling --
# The correction is fixed from here on: only the (cheaper, lower variance)
# frozen branch is propagated for the remaining blocks.
_frz_cls = lsp.sampler_pt2_sto_chol if _sto_chol else lsp.sampler_pt2
sampler_frz = _frz_cls(
    n_prop_steps = options["n_prop_steps"],
    n_blocks = sampler.n_blocks,
    n_chol = sampler.n_chol,
    frozen_vir = frozen_vir,
    )

print("\nFrozen Sampling Blocks")
print(f"Correction from {n_paired} paired blocks: {delta:.5f} +/- {delta_err:.5f}")
print(f"Target Final Error ~ {options['max_error']:.6f}")
print(f"{'N':>4s}  {'nodes':>5s}  {'weight':>10s}  {'E(Guide)':>12s}  {'Error':>8s}  "
      f"{'weightp':>10s}  {'E_frozen':>10s}  {'Error':>8s}  {'E_frag':>8s}  {'Error':>8s}  "
      f"{'Time':>8s}")

for n in range(n_paired, sampler.n_blocks):
    prop_data, (wt, eg, wp, t2frg, e0frg, e1frg, e0) = \
        sampler_frz.block_sample(prop_data, ham_data, prop, trial, wave_data)

    wt_sp[n] = wt
    eg_sp[n] = eg

    wp_sp[n] = wp
    t2frg_sp[n] = t2frg
    e0frg_sp[n] = e0frg
    e1frg_sp[n] = e1frg
    e0_sp[n]    = e0

    nodes += prop_data["n_killed_walkers"]
    prop_data["n_killed_walkers"] = 0

    if (n+1) % (min(max(sampler.n_blocks // 10, 1), 20)) == 0 and n > 0:
        weight, guide, guide_err = sp.blocking(wt_sp[:n+1], eg_sp[:n+1])
        weighp, efrz, efrz_err = lsp.ept2frg_blocking(
            wp_sp[:n+1], t2frg_sp[:n+1], e0frg_sp[:n+1], e1frg_sp[:n+1], e0_sp[:n+1],)

        efrg = efrz.real + delta
        efrg_err = np.sqrt(efrz_err**2 + delta_err**2)

        print(f"{n+1:4d}  {nodes:5d}  {wt:10.5f}  {guide:12.5f}  {guide_err:8.5f}  "
              f"{weighp.real:10.5f}  {efrz.real:10.5f}  {efrz_err:8.5f}  "
              f"{efrg:8.5f}  {efrg_err:8.5f}  {time.time() - init_time:8.2f}")

        prop_data["e_estimate"] = 0.8 * prop_data["e_estimate"] + 0.2 * eg.real

        if efrg_err < 0.75 * options["max_error"] and n > 120:
            break

print("\nPost Propagation")
nsamples = np.count_nonzero(wt_sp)
print(f'Total number of samples {nsamples}')
print(f'Paired (correlated) samples {n_paired}')
wt_sp = wt_sp[:nsamples]
eg_sp = eg_sp[:nsamples]

wp_sp    = wp_sp[:nsamples]
t2frg_sp = t2frg_sp[:nsamples]
e0frg_sp = e0frg_sp[:nsamples]
e1frg_sp = e1frg_sp[:nsamples]
e0_sp    = e0_sp[:nsamples]

wp1_sp    = wp1_sp[:n_paired]
t2frg1_sp = t2frg1_sp[:n_paired]
e0frg1_sp = e0frg1_sp[:n_paired]
e1frg1_sp = e1frg1_sp[:n_paired]
e01_sp    = e01_sp[:n_paired]

# the two branches share the walkers, so t1 - and with it wp - must agree
print(f"max|wp(full) - wp(frozen)| = {np.max(np.abs(wp1_sp - wp_sp[:n_paired])):.3e}")

print("Remove Outliers")
ept2frg_sp = (e0frg_sp + e1frg_sp - t2frg_sp * e0_sp).real
mask = sp.filter_outliers(ept2frg_sp, zeta=30)
print(f"Removed {np.sum(~mask)} Outliers")
print(f"Outliers Energy {ept2frg_sp[~mask]}")

# the correction is only defined on the paired blocks that survived, and is
# filtered on its own scale: it is far smaller than either branch
delta_sp = ((e0frg1_sp + e1frg1_sp - t2frg1_sp * e01_sp).real
            - ept2frg_sp[:n_paired])
mask_d = sp.filter_outliers(delta_sp, zeta=30) & mask[:n_paired]
print(f"Removed {np.sum(~mask_d)} Outliers from the correction")

# paired blocks: both branches restricted to the same surviving samples
wpd_sp     = wp_sp[:n_paired][mask_d]
t2frg0d_sp = t2frg_sp[:n_paired][mask_d]
e0frg0d_sp = e0frg_sp[:n_paired][mask_d]
e1frg0d_sp = e1frg_sp[:n_paired][mask_d]
e00d_sp    = e0_sp[:n_paired][mask_d]

t2frg1d_sp = t2frg1_sp[mask_d]
e0frg1d_sp = e0frg1_sp[mask_d]
e1frg1d_sp = e1frg1_sp[mask_d]
e01d_sp    = e01_sp[mask_d]

wp_sp    = wp_sp[mask]
t2frg_sp = t2frg_sp[mask]
e0frg_sp = e0frg_sp[mask]
e1frg_sp = e1frg_sp[mask]
e0_sp    = e0_sp[mask]

print("\nBlocking: guiding energy")
weight, eguide, guide_err = sp.blocking(wt_sp, eg_sp,  final=True)

print("\nBlocking: frozen virtual fragment energy")
weighp, efrz, efrz_err = lsp.ept2frg_blocking(
    wp_sp, t2frg_sp, e0frg_sp, e1frg_sp, e0_sp, final=True)

print("\nBlocking: frozen virtual correction")
delta, delta_err = lsp.ept2frg_delta_blocking(
    wpd_sp,
    t2frg0d_sp, e0frg0d_sp, e1frg0d_sp, e00d_sp,
    t2frg1d_sp, e0frg1d_sp, e1frg1d_sp, e01d_sp, final=True)

# independent cross check: the full branch on the paired blocks alone
print("\nBlocking: full branch on the paired blocks (cross check)")
_, efull, efull_err = lsp.ept2frg_blocking(
    wpd_sp, t2frg1d_sp, e0frg1d_sp, e1frg1d_sp, e01d_sp, final=True)

efrag = efrz.real + delta
efrag_err = np.sqrt(efrz_err**2 + delta_err**2)

print(f"\nFinal AFQMC/HF Guiding Energy:        {eguide:.4f} +/- {guide_err:.4f}")
print(f"Frozen Virtual Orbital Energy:        {efrz.real:.5f} +/- {efrz_err:.5f}"
      f"   ({frozen_vir} frozen, {len(wp_sp)} samples)")
print(f"Frozen Virtual Correction:            {delta:.5f} +/- {delta_err:.5f}"
      f"   ({len(wpd_sp)} paired samples)")
print(f"Final AFQMC/pt2CCSD Orbital Energy:   {efrag:.5f} +/- {efrag_err:.5f}")
print(f"Cross check, full branch only:        {efull.real:.5f} +/- {efull_err:.5f}")
print(f"<t1> = weightp/weight = {jnp.real(weighp/weight):.5f}")
print(f"Total run time: {time.time() - init_time:.2f}")
print("\nAFQMC Sampling Finished\n")
