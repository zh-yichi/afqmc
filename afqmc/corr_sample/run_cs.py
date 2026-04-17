from functools import partial
from jax import random
#from jax import numpy as jnp
# import argparse
import pickle
#from mpi4py import MPI
import numpy as np
from ad_afqmc.corr_sample import corr_sample
from ad_afqmc import mpi_jax, config
import time

print = partial(print, flush=True)

with open("files.bin", "rb") as f:
    files = pickle.load(f)

mo_file1=files["mo1"]
chol_file1=files["chol1"]
amp_file1=files["amp1"]
mo_file2=files["mo2"]
chol_file2=files["chol2"]
amp_file2=files["amp2"]

ham_data1, ham1, prop1, trial1, wave_data1, sampler, observable1, options, _ \
    = mpi_jax._prep_afqmc(mo_file=mo_file1,amp_file=amp_file1,chol_file=chol_file1)
ham_data2, ham2, prop2, trial2, wave_data2, _, observable2, options, _ \
    = mpi_jax._prep_afqmc(mo_file=mo_file2,amp_file=amp_file2,chol_file=chol_file2)

# options["dt"] = options.get("dt", 0.01)
# options["n_exp_terms"] = options.get("n_exp_terms",6)
# options["n_walkers"] = options.get("n_walkers", 50)
options["n_runs"] = options.get("n_runs", 100)
options["rlx_steps"] = options.get("rlx_steps", 0)
options["prop_steps"] = options.get("prop_steps", 10)
# options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))

nwalkers = options["n_walkers"]
seed = options["seed"]
rlx_steps = options["rlx_steps"]
prop_steps = options["prop_steps"]
n_runs = options["n_runs"]


if options["use_gpu"]:
    config.afqmc_config["use_gpu"] = True

config.setup_jax()
MPI = config.setup_comm()

init_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process rank
size = comm.Get_size()  # Total number of processes

ham_data1, ham1, prop1, trial1, wave_data1, sampler, observable1, options, _ \
    = mpi_jax._prep_afqmc(mo_file=mo_file1,amp_file=amp_file1,chol_file=chol_file1)
ham_data2, ham2, prop2, trial2, wave_data2, sampler, observable2, options, _ \
    = mpi_jax._prep_afqmc(mo_file=mo_file2,amp_file=amp_file2,chol_file=chol_file2)

prop_data1_init, ham_data1_init = \
    corr_sample.init_prop(ham_data1, ham1, prop1, trial1, wave_data1, seed, MPI)
prop_data2_init, ham_data2_init = \
    corr_sample.init_prop(ham_data2, ham2, prop2, trial2, wave_data2, seed, MPI)

### relaxation ###
comm.Barrier()
if rank == 0:
    if options["trial"] == "rhf":
        print(f'# relaxation from mean-field object using {nwalkers*size} walkers')
    if options["trial"] == "cisd":
        print(f'# relaxation from ccsd object using {nwalkers*size} walkers')
    print('# rlx_step \t system1_en \t system2_en \t en_diff')
    print(f'    {0}'
          f'\t \t {prop_data1_init["e_estimate"]:.6f}' 
          f'\t {prop_data2_init["e_estimate"]:.6f}'
          f'\t {prop_data1_init["e_estimate"]-prop_data2_init["e_estimate"]:.6f}')
comm.Barrier()

(prop_data1_rlx,prop_data2_rlx),(loc_en1,loc_weight1,loc_en2,loc_weight2) \
    = corr_sample.cs_steps_scan(rlx_steps,
                                prop_data1_init,ham_data1_init,prop1,trial1,wave_data1,
                                prop_data2_init,ham_data2_init,prop2,trial2,wave_data2, 
                                sampler)

comm.Barrier()
if rank == 0:
    glb_en1 = np.empty(size*loc_en1.size,dtype='float32')
    glb_en2 = np.empty(size*loc_en2.size,dtype='float32')
    glb_weight1 = np.empty(size*loc_weight2.size,dtype='float32')
    glb_weight2 = np.empty(size*loc_weight2.size,dtype='float32')
else:
    glb_en1 = None
    glb_en2 = None
    glb_weight1 = None
    glb_weight2 = None
comm.Barrier()

loc_en1 = np.asarray(loc_en1,dtype='float32')
loc_en2 = np.asarray(loc_en2,dtype='float32')
loc_weight1 = np.asarray(loc_weight1,dtype='float32')
loc_weight2 = np.asarray(loc_weight2,dtype='float32')

comm.Gather(loc_en1,glb_en1,root=0)
comm.Gather(loc_en2,glb_en2,root=0)
comm.Gather(loc_weight1,glb_weight1,root=0)
comm.Gather(loc_weight2,glb_weight2,root=0)


comm.Barrier()
if rank == 0:
    glb_en1 = glb_en1.reshape(size,rlx_steps).T
    glb_en2 = glb_en2.reshape(size,rlx_steps).T
    glb_weight1 = glb_weight1.reshape(size,rlx_steps).T
    glb_weight2 = glb_weight2.reshape(size,rlx_steps).T

    energy1 = np.zeros((rlx_steps))
    energy2 = np.zeros((rlx_steps))
    en_diff = np.zeros((rlx_steps))
    
    for step in range(rlx_steps):

        energy1[step] = sum(glb_en1[step,:])/sum(glb_weight1[step,:])
        energy2[step] = sum(glb_en2[step,:])/sum(glb_weight2[step,:])
        en_diff[step] = energy1[step] - energy2[step]

        print(f'    {step+1} \t \t {energy1[step]:.6f} \t {energy2[step]:.6f} \t {en_diff[step]:.6f}')
    
    now_time = time.time()
    print(f'# relaxation time: {now_time - init_time:.2f}')
comm.Barrier()

### post relaxation propagation ###
comm.Barrier()
if rank == 0:
    print()
    print(f'# post relaxation propagation independent runs')
    if options["corr_samp"]:
        print('# correlated sampling')
    else: print('# uncorrelated sampling')

    print(f'# tot_walkers: {nwalkers*size}')
    print(f'# propagation steps: {prop_steps}')
    print(f'number of independent runs: {n_runs}')
    print('# step' 
          '\t system1_en \t error' 
          '\t \t system2_en \t error'
          '\t \t energy_diff \t error')
    
comm.Barrier()

# if options["corr_samp"]:
seeds = random.randint(random.PRNGKey(options["seed"]),
                    shape=(n_runs,), minval=0, maxval=1000000*n_runs)

loc_en1,loc_weight1,loc_en2,loc_weight2 \
= corr_sample.scan_seeds(seeds,prop_steps,
                            prop_data1_rlx,ham_data1_init,prop1,trial1,wave_data1,
                            prop_data2_rlx,ham_data2_init,prop2,trial2,wave_data2, 
                            sampler,MPI)
# else:
#     seeds = random.randint(random.PRNGKey(options["seed"]),
#                         shape=(n_runs,2), minval=0, maxval=10000*n_runs)

#     loc_en1,loc_weight1,loc_en2,loc_weight2 \
#         = corr_sample.ucs_scan_seeds(seeds,prop_steps,
#                                      prop_data1_rlx,ham_data1_init,prop1,trial1,wave_data1,
#                                      prop_data2_rlx,ham_data2_init,prop2,trial2,wave_data2, 
#                                      MPI)

comm.Barrier()
if rank == 0:
    glb_en1 = np.empty(size*loc_en1.size,dtype='float32')
    glb_en2 = np.empty(size*loc_en2.size,dtype='float32')
    glb_weight1 = np.empty(size*loc_weight1.size,dtype='float32')
    glb_weight2 = np.empty(size*loc_weight2.size,dtype='float32')
else:
    glb_en1 = None
    glb_en2 = None
    glb_weight1 = None
    glb_weight2 = None
comm.Barrier()

loc_en1 = np.asarray(loc_en1,dtype='float32')
loc_en2 = np.asarray(loc_en2,dtype='float32')
loc_weight1 = np.asarray(loc_weight1,dtype='float32')
loc_weight2 = np.asarray(loc_weight2,dtype='float32')

comm.Gather(loc_en1,glb_en1,root=0)
comm.Gather(loc_en2,glb_en2,root=0)
comm.Gather(loc_weight1,glb_weight1,root=0)
comm.Gather(loc_weight2,glb_weight2,root=0)


comm.Barrier()
if rank == 0:
    glb_en1 = glb_en1.reshape(size,n_runs,prop_steps).T
    glb_en2 = glb_en2.reshape(size,n_runs,prop_steps).T
    glb_weight1 = glb_weight1.reshape(size,n_runs,prop_steps).T
    glb_weight2 = glb_weight2.reshape(size,n_runs,prop_steps).T

    energy1 = np.zeros((prop_steps,n_runs))
    energy2 = np.zeros((prop_steps,n_runs))
    en_diff = np.zeros((prop_steps,n_runs))

    for step in range(prop_steps):

        for run in range(n_runs):
            energy1[step,run] = sum(glb_en1[step,run,:])/sum(glb_weight1[step,run,:])
            energy2[step,run] = sum(glb_en2[step,run,:])/sum(glb_weight2[step,run,:])
            en_diff[step,run] = energy1[step,run] - energy2[step,run]

        en_mean1 = energy1[step,:].mean()
        en_mean2 = energy2[step,:].mean()
        en_diff_mean = en_diff[step,:].mean()
        en_err1 = energy1[step,:].std()/np.sqrt(n_runs)
        en_err2 = energy2[step,:].std()/np.sqrt(n_runs)
        en_diff_mean_err = en_diff[step,:].std()/np.sqrt(n_runs)
        print(f'  {step+1}'
              f'\t {en_mean1:.6f} \t {en_err1:.6f}' 
              f'\t {en_mean2:.6f} \t {en_err2:.6f}'
              f'\t {en_diff_mean:.6f} \t {en_diff_mean_err:.6f}')

    end_time = time.time()
    print(f'# total run time: {end_time - init_time:.2f}')
comm.Barrier()
