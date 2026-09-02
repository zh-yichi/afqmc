"""LNO-AFQMC / pt2CCSD with a semistochastic Cholesky sum.

The fragment version of ph_afqmc_pt2ccsd_sto_chol.py: same sampling, applied to
the fragment T2*h2 term of each LNO fragment.

WHAT IS SAMPLED
    Per fragment, only e2_2_2_1, e2_2_2_2 and e2_2_3 of _t2eorb_tc -- the
    accumulators that contract with T2.  e2_0, and hence e2_2_1 = e2_0 * gt2g,
    is always summed exactly.  The sampling proposal comes free from
    _calc_e0bar_frag, which is evaluated on the same walker anyway: its
    per-Cholesky fragment two-body energies are turned into pi_g.

    The estimator is unbiased; the sample count sets the variance only.

HOW TO SET THE BUDGET   (identical to the canonical example)
        'chol_cost_ratio': 0.2      # 20% of nchol per walker, split 3:1
                                    # head : samples
    or override either half
        'head_chol_ratio': 0.125    # head as a fraction of nchol
        'n_chol_samples' : 32       # tail draws per walker per block
    and
        'n_chol_head'    : 'full'   # disable sampling, reproduce 'upt2ccsd'
        'head_sample_ratio': 3.0    # head:samples split for chol_cost_ratio

    Note nchol here is the *fragment* nchol, which is much smaller than the
    full-system one, so the same ratio buys fewer vectors than it would in a
    canonical run.  Check the "Number of Chol. Vectors" line in fragment.out*.

FROZEN VIRTUALS
    Adding 'frozen_vir' (a count) or 'frozen_vir_rate' (a fraction) switches the
    driver to run_lno_afqmc_pt2ccsd_frozen_vir.py automatically -- no need to
    pass qmc_script.  That script reports

        E_frag = <E_frozen> + <E_full - E_frozen>

    and with a sto_chol trial it uses sampler_pt2_frozen_vir_sto_chol, which
    hands BOTH branches the same Cholesky-sampling key.  They therefore draw the
    same tail vectors and that noise largely cancels in the difference (measured
    ~1.5x smaller error on the correction than with independent keys).

    'frozen_vir': 0 means "do not freeze" and keeps the ordinary script.

FIRST RUN
    Set 'n_chol_head': 'full' once and check it reproduces a plain 'upt2ccsd'
    run before trusting the sampled fragment energies.
"""

import numpy as np
from pyscf import gto, scf, lo

####  test O2 monomers ####
a = 1.20577 # bond length in a cluster
d = 100 # distance between each cluster
unit = 'A' # unit of length
na = 2 # size of a cluster (monomer)
nc = 5 # set as integer multiple of monomers
spin = 2 # spin per monomer
elmt = 'O'
basis = 'sto6g'
atoms = ""
for n in range(nc*na):
    shift = ((n - n % na) // na) * (d-a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"
###########################

mol = gto.M(atom=atoms,
            basis=basis,
            verbose=4,
            unit=unit,
            symmetry=0,
            charge=0,
            spin=spin,
            max_memory=20000,
            )

mf = scf.UHF(mol).density_fit()
mf.kernel()

stable = False
while not stable:
    print(f'mean-field stability test')
    if not stable:
        mo_i, _, stable,_ = mf.stability(return_status=True)
        dm = mf.make_rdm1(mo_i,mf.mo_occ)
        mf.kernel(dm0=dm)
    elif stable:
        print(f'UHF Energy: {mf.e_tot}, stability {stable}')
        break

from afqmc.lno_afqmc import lno_afqmc, tools
lo_coeff, frag_list, frag_name = tools.iao_fragment(mf, frag_type='h2heavy', more_loc='pm')

options = {
           'n_prop_steps': 50,
           'n_blocks': 600,
           'n_walkers': 300,
           'max_memory': 2000,
           'mix_precision': False,
           'n_batch': 1,
           'seed': 17,
           'walker_type': 'uhf',
           'trial': 'upt2ccsd_sto_chol',
           # --- semistochastic Cholesky sum ------------------------------ #
           'chol_cost_ratio': 0.2,     # 20% of the fragment nchol per walker,
                                       # split 3:1 head : samples
           # 'head_chol_ratio': 0.125, # override the head half
           # 'n_chol_samples' : 32,    # override the tail half
           # 'n_chol_head'    : 'full',# no sampling; reproduces upt2ccsd
           # --- optional frozen virtuals --------------------------------- #
           # setting either of these switches to the frozen_vir driver script
           # 'frozen_vir_rate': 3,     # freeze 1/3 of the virtuals
           # 'frozen_vir'     : 10,    # or an explicit count
           }

lno_afqmc.run_afqmc(
    mf,
    lo_coeff, 
    frag_list,
    frag_name,
    lno_thresh = 1e-5,
    qmc_options = options, 
    chol_cut = 1e-5, 
    target_qmc_err = 1e-3, 
    run_frag = None, 
    nfrozen = None,
    run_mp = True,
    run_cc = True,
    run_qmc = True,
    plot_las = False,
    )
