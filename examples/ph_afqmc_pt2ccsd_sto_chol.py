"""ph-AFQMC / pt2CCSD with a semistochastic Cholesky sum.

Same system as ph_afqmc_pt2ccsd.py; only the trial and its options differ.

WHAT IS SAMPLED
    The energy needs, per walker per block, a sum over all nchol Cholesky
    vectors.  This trial splits that sum into

        sum_g  =  sum_{g in head}   +   sum_{g in tail}
                  (exact)               (importance sampled)

    and only for the terms that contract with T2 (e2_2_2_1, e2_2_2_2, e2_2_3).
    e2_0 -- and therefore e2_2_1 = e2_0 * gt2g -- is ALWAYS summed exactly: it
    is needed to build the sampling proposal anyway and costs only the cheap
    gl = green.chol contraction.  The T2 terms are the expensive ones
    ("gia,iajb->gjb" scales as nocc^2 nvir^2 per vector), which is what makes
    the trade worth it.

    The estimator is unbiased.  The sample count controls the variance, not the
    answer, so it is safe to tune for speed.

HOW TO SET THE BUDGET
    Easiest -- one knob, the fraction of the Cholesky vectors each walker
    touches per block:

        'chol_cost_ratio': 0.2      # 20% of nchol

    That budget is split head : samples = 3 : 1, which is close to the measured
    optimum (the variance at fixed cost is minimised where the head reaches the
    point at which a tail vector would be drawn about once).  With nchol = 688
    it gives n_head = 103 exact plus M = 34 sampled.

    Or set the two halves yourself; either one overrides its half of the split:

        'head_chol_ratio': 0.125    # head = 0.125 * nchol   (default)
        'n_chol_samples' : 32       # tail draws per walker  (default 128)

    Other knobs:

        'n_chol_head'      : 100     exact head size, or "full" to disable
                                     sampling entirely and reproduce the plain
                                     'upt2ccsd' result.  Beats both ratios.
        'head_sample_ratio': 3.0     head:samples split used by chol_cost_ratio
        'head_from_guide'  : False   rank the head per walker instead of taking
                                     the leading prefix.  Costs memory (the
                                     gather can no longer be shared across
                                     walkers) and measured *higher* variance,
                                     so leave it off unless you are testing it.

    Restricted analogue: use 'rpt2ccsd_sto_chol' with walker_type 'rhf'.

FIRST RUN
    Set 'n_chol_head': 'full' once.  It disables sampling and must reproduce a
    plain 'upt2ccsd' run to machine precision -- a cheap check that everything
    is wired up before you trust the sampled numbers.
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

from pyscf import gto, scf, cc
import os

#### test O2 monomers ####
m_list = [1] # number of monomers
d = 100 # distance between monomers
unit = 'A' # angstron 
for nc in m_list:
    atoms = ""
    for n in range(nc):
        shift = n*d
        atoms += f'O {0.0+shift} 0.0 0.0     \n'
        atoms += f'O {0.0+shift} 0.0 1.20577 \n'
    nfrozen = 2*nc
    spin = 2*nc
##########################

    mol = gto.M(atom=atoms, basis="sto6g", spin=spin, unit=unit, verbose=4)
    mol.build()

    mf = scf.UHF(mol)
    mf.kernel()

    # scf stability
    stable = False
    while not stable:
        print(f'mf stability test')
        if not stable:
            mo_i, _, stable,_ = mf.stability(return_status=True)
            dm = mf.make_rdm1(mo_i,mf.mo_occ)
            mf.kernel(dm0=dm)
        elif stable:
            print(f'mf energy: {mf.e_tot}, stability {stable}')
            break

    # CCSD 
    mycc = cc.CCSD(mf,frozen=nfrozen)
    mycc.kernel()

    options = {'n_blocks': 300,
               'n_walkers': 300,
               'nchol_chunk': 30,
               'max_memory': 3000,
               'seed': 17,
               'mix_precision': False,  # exact arithmetic, so any difference
                                        # from 'upt2ccsd' is the sampling
               'trial': 'upt2ccsd_sto_chol',
               # --- semistochastic Cholesky sum -------------------------- #
               'chol_cost_ratio': 0.2,  # each walker touches 20% of nchol,
                                        # split 3:1 head:samples
               # 'head_chol_ratio': 0.125,  # override the head half
               # 'n_chol_samples' : 32,     # override the tail half
               # 'n_chol_head'    : 'full', # no sampling; reproduces upt2ccsd
               }

    from afqmc import integral, launch_afqmc
    integral.prep_integral(mycc)
    launch_afqmc.ph_afqmc(options)
