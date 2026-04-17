from pyscf import gto, scf, cc

a = 2 # bond length in a cluster
d = 100 # distance between each cluster
unit = 'b' # unit of length
na = 2 # size of a cluster (monomer)
nc = 4 # set as integer multiple of monomers
spin = 0 # spin per monomer
frozen = 0 # frozen orbital per monomer
elmt = 'H'
basis = 'sto6g'
atoms = ""
for n in range(nc*na):
    shift = ((n - n % na) // na) * (d-a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms,basis=basis,spin=spin*nc,unit=unit,verbose=4)
mol.build()

mf = scf.RHF(mol)
mf.kernel()

nfrozen = nc*frozen

mycc = cc.CCSD(mf,frozen=nfrozen)
mycc.kernel()

options =  {'n_eql': 3,
            'n_prop_steps': 50,
            'n_ene_blocks': 1,
            'n_sr_blocks': 5,
            'n_blocks': 100,
            'dt': 0.005,
            'n_walkers': 100,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'stoccsd4',
            'nslater': 100, # number of slater determinants to sample the ccsd
            'use_gpu': False,
            }

from ad_afqmc.prop_unrestricted.mixed_wave import prep, launch_afqmc
import jax
jax.config.update("jax_enable_x64", True)
prep.prep_afqmc(mycc,chol_cut=1e-5)
launch_afqmc.run_afqmc(options)
