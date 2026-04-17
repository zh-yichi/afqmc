from pyscf import gto, scf, cc

a = 2 # bond length in a cluster
d = 10 # distance between each cluster
unit = 'b' # unit of length
na = 2  # size of a cluster (monomer)
nc = 1 # set as integer multiple of monomers
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

options = {'n_eql': 160, # 1 neq time = dt*n_prop_steps
           'n_prop_steps': 50,
            'n_ene_blocks': 1, # doesn't do anything
            'n_sr_blocks': 1, # doesn't do anything
            'n_blocks': 10, # just tune this for the number of samples
            'n_walkers': 100,
            'seed': 2,
            'walker_type': 'rhf',
            'trial': 'cisd',
            'dt':0.005,
            'free_projection':False,
            'ad_mode':None,
            'use_gpu': True,
            }

from ad_afqmc.prop_unrestricted import prep, launch_afqmc
prep.prep_afqmc(mycc,chol_cut=1e-5)
launch_afqmc.run_afqmc(options)
