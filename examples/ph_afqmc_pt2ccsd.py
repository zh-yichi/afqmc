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
               'trial': 'upt2ccsd',
               }

    from afqmc import integral, launch_afqmc
    integral.prep_integral(mycc)
    launch_afqmc.ph_afqmc(options)
