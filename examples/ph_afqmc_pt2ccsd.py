import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

from pyscf import gto, scf, cc
import os

#### test O2 monomers ####
m_list = [3] # number of monomers
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

    from afqmc import prep, launch_afqmc
    prep.prep_afqmc(mycc, chol_cut=1e-5)

    # RHF Trial
    options = {'n_prop_steps': 50,
               'n_eql': 120, # 1 eql step = dt*n_prop_steps (here 120 = 30 au)
               'n_blocks': 1000, # tune this for how many samples you want
               'n_walkers': 300,
               'dt':0.005, # time every trot step
               'max_error': 0.0, # set to 0 to run the calculation till n_blocks
               'seed': 17,
               'walker_type': 'uhf',
               'trial': 'uccsd_pt2', # use unrestricted pt2ccsd trial
               'free_projection': False,
               'use_gpu': True}

    launch_afqmc.run_afqmc(options)
