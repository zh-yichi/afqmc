from pyscf import gto, scf

a = 1 
nH = 10
atoms1 = ""
for i in range(nH):
    atoms1 += f"H {i*a:.5f} 0.00000 0.00000 \n"

b = 1.5
atoms2 = ""
for i in range(nH):
    if i < nH-1:
        atoms2 += f"H {i*a:.5f} 0.00000 0.00000 \n"
    else:
        atoms2 += f"H {(i-1)*a+b:.5f} 0.00000 0.00000 \n"

basis = "sto6g"
mol1 = gto.M(atom=atoms1,basis=basis,verbose=4)
mol2 = gto.M(atom=atoms2,basis=basis,verbose=4)

mf1 = scf.RHF(mol1).density_fit()
mf1.kernel()

mf2 = scf.RHF(mol2).density_fit()
mf2.kernel()

e_mf1 = mf1.e_tot
e_mf2 = mf2.e_tot
print(f"mf energy difference: {e_mf2-e_mf1:.8f}")

options = {
    "dt": 0.005,
    "n_exp_terms": 6,
    "n_walkers": 30,
    "n_runs": 50,
    "rlx_steps": 0,
    "prop_steps": 10,
    "n_prop_steps": 10,
    "seed": 23,
    "walker_type": "rhf",
    "trial": "rhf",
    "use_gpu": False,
    "free_proj": False,
}

from ad_afqmc.corr_sample import lnocs
lnocs.run_cs_frags(mf1,mf2,None,options=options,nproc=5,lno_thresh=1e-4)
