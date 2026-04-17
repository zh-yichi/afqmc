from pyscf import gto, scf, cc, fci
import numpy as np

basis = 'sto6g'
verbose = 4
memory = 120000
a = 1.596
d1 = 3
d2 = 100

atom1 = f'''
H 0 0 0
Li {a} 0 0
Li 0 0 {d1}
H {a} 0 {d1}
'''

atom2 = f'''
H 0 0 0
Li {a} 0 0
Li 0 0 {d2}
H {a} 0 {d2}
'''

mol1 = gto.Mole(
verbose=verbose,
atom=atom1,
basis=basis,
max_memory=memory
)
mol1.build()

mol2 = gto.Mole(
verbose=verbose,
atom=atom2,
basis=basis,
max_memory=memory
)
mol2.build()

# RHF
mf1 = scf.RHF(mol1).density_fit()
mf1 = mf1.newton()
mf1.kernel()

mf2 = scf.RHF(mol2).density_fit()
mf2 = mf2.newton()
mf2.kernel()

### coupled cluster ###
mc1 = cc.CCSD(mf1)
mc1.frozen = 0
mc1.kernel()
et1 = mc1.ccsd_t()

mc2 = cc.CCSD(mf2)
mc2.frozen = 0
mc2.kernel()
et2 = mc2.ccsd_t()

## fci
fci1 = fci.FCI(mf1)
fci_e1, fci_v1 = fci1.kernel()

fci2 = fci.FCI(mf2)
fci_e2, fci_v2 = fci2.kernel()

print(f"fci1 energy is {fci_e1}")
print(f"fci2 energy is {fci_e2}")

print(f"rhf energy difference is {mf1.e_tot-mf2.e_tot}")
print(f"ccsd energy difference is {mc1.e_tot-mc2.e_tot}")
print(f"ccsd(t) energy difference is {mc1.e_tot+et1-mc2.e_tot-et2}")
print(f"fci energy difference is {fci1.e_tot-fci2.e_tot}")

from mpi4py import MPI
MPI.Finalize() ### finalize MPI after ccsd
from ad_afqmc import pyscf_interface

mo_file1="mo1.npz"
chol_file1="chol1"
amp_file1="amp1.npz"
mo_file2="mo2.npz"
chol_file2="chol2"
amp_file2="amp2.npz"
pyscf_interface.prep_afqmc(mc1,mo_file=mo_file1,amp_file=amp_file1,chol_file=chol_file1)
pyscf_interface.prep_afqmc(mc2,mo_file=mo_file2,amp_file=amp_file2,chol_file=chol_file2)