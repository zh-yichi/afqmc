from functools import partial
from jax import random
# import pickle
#from mpi4py import MPI
import numpy as np
from ad_afqmc.corr_sample import corr_sample
from ad_afqmc import mpi_jax, config
from ad_afqmc.corr_sample import lnocs
import time

print = partial(print, flush=True)

config.setup_jax()
MPI = config.setup_comm()

init_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process rank
size = comm.Get_size()  # Total number of processes

mo_file1='mo1.npz'
chol_file1='chol1'
amp_file1='amp1'
mo_file2='mo2.npz'
chol_file2='chol2'
amp_file2='amp2'


ham_data1, ham1, prop1, trial1, wave_data1, sampler, observable1, options, _ \
    = lnocs.prep_afqmc(mo_file=mo_file1,amp_file=amp_file1,chol_file=chol_file1)
ham_data2, ham2, prop2, trial2, wave_data2, _, observable2, options, _ \
    = lnocs.prep_afqmc(mo_file=mo_file2,amp_file=amp_file2,chol_file=chol_file2)

seed = options["seed"]
nwalkers = options["n_walkers"]
rlx_steps = options["rlx_steps"]
prop_steps = options["prop_steps"]
n_runs = options["n_runs"]

if options['use_gpu']:
    config.afqmc_config["use_gpu"] = True

config.setup_jax()
MPI = config.setup_comm()
init_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process rank
size = comm.Get_size()  # Total number of processes

prop_data1, ham_data1 = \
    corr_sample.init_prop(ham_data1, ham1, prop1, trial1, wave_data1, seed, MPI)
prop_data2, ham_data2 = \
    corr_sample.init_prop(ham_data2, ham2, prop2, trial2, wave_data2, seed, MPI)

### relaxation ###
comm.Barrier()
if rank == 0:

    print(f'# relaxation using {nwalkers*size} walkers')
    print('# rlx_step \t system1_en \t system2_en \t en_diff \t orb1_en \t orb2_en \t orb_en_diff')
    print(f'   {0:2d}'
          f'\t \t {prop_data1["e_estimate"]:.6f}' 
          f'\t {prop_data2["e_estimate"]:.6f}'
          f'\t {prop_data2["e_estimate"]-prop_data1["e_estimate"]:.6f}'
          f'\t {0:.6f}'
          f'\t {0:.6f}'
          f'\t {0:.6f}')
comm.Barrier()

(prop_data1,prop_data2),(loc_en1,loc_orb_en1,loc_wt1,loc_en2,loc_orb_en2,loc_wt2) \
    = corr_sample.lno_cs_steps_scan(rlx_steps,
                                    prop_data1,ham_data1,prop1,trial1,wave_data1,
                                    prop_data2,ham_data2,prop2,trial2,wave_data2,
                                    sampler)

comm.Barrier()
if rank == 0:
    glb_en1 = np.empty(size*loc_en1.size,dtype='float32')
    glb_en2 = np.empty(size*loc_en2.size,dtype='float32')
    glb_orb_en1 = np.empty(size*loc_orb_en1.size,dtype='float32')
    glb_orb_en2 = np.empty(size*loc_orb_en2.size,dtype='float32')
    glb_wt1 = np.empty(size*loc_wt2.size,dtype='float32')
    glb_wt2 = np.empty(size*loc_wt2.size,dtype='float32')
else:
    glb_en1 = None
    glb_en2 = None
    glb_orb_en1 = None
    glb_orb_en2 = None
    glb_wt1 = None
    glb_wt2 = None
comm.Barrier()

loc_en1 = np.asarray(loc_en1,dtype='float32')
loc_en2 = np.asarray(loc_en2,dtype='float32')
loc_orb_en1 = np.asarray(loc_orb_en1,dtype='float32')
loc_orb_en2 = np.asarray(loc_orb_en2,dtype='float32')
loc_wt1 = np.asarray(loc_wt1,dtype='float32')
loc_wt2 = np.asarray(loc_wt2,dtype='float32')

comm.Gather(loc_en1,glb_en1,root=0)
comm.Gather(loc_en2,glb_en2,root=0)
comm.Gather(loc_orb_en1,glb_orb_en1,root=0)
comm.Gather(loc_orb_en2,glb_orb_en2,root=0)
comm.Gather(loc_wt1,glb_wt1,root=0)
comm.Gather(loc_wt2,glb_wt2,root=0)


comm.Barrier()
if rank == 0:
    glb_en1 = glb_en1.reshape(size,rlx_steps).T
    glb_en2 = glb_en2.reshape(size,rlx_steps).T
    glb_orb_en1 = glb_orb_en1.reshape(size,rlx_steps).T
    glb_orb_en2 = glb_orb_en2.reshape(size,rlx_steps).T
    glb_wt1 = glb_wt1.reshape(size,rlx_steps).T
    glb_wt2 = glb_wt2.reshape(size,rlx_steps).T

    en1 = np.zeros((rlx_steps))
    en2 = np.zeros((rlx_steps))
    en_diff = np.zeros((rlx_steps))
    orb_en1 = np.zeros((rlx_steps))
    orb_en2 = np.zeros((rlx_steps))
    orb_en_diff = np.zeros((rlx_steps))
    
    for step in range(rlx_steps):

        en1[step] = sum(glb_en1[step,:])/sum(glb_wt1[step,:])
        en2[step] = sum(glb_en2[step,:])/sum(glb_wt2[step,:])
        en_diff[step] = en1[step] - en2[step]
        orb_en1[step] = sum(glb_orb_en1[step,:])/sum(glb_wt1[step,:])
        orb_en2[step] = sum(glb_orb_en2[step,:])/sum(glb_wt2[step,:])
        orb_en_diff[step] = orb_en1[step] - orb_en2[step]

        print(f'    {step+1} \t \t {en1[step]:.6f} \t {en2[step]:.6f} \t {en_diff[step]:.6f}'
              f'\t {orb_en1[step]:.6f} \t {orb_en2[step]:.6f} \t {orb_en_diff[step]:.6f}')
    
    now_time = time.time()
    print(f'# relaxation time: {now_time - init_time:.2f}')
comm.Barrier()

### post relaxation propagation ###
comm.Barrier()
if rank == 0:
    print('\n')
    print(f'# tot_walkers: {nwalkers*size}, propagation steps: {prop_steps}, number of independent runs: {n_runs}')
    print('# step'
          '\t   orb1_en \t error' 
          '\t \t orb2_en \t error'
          '\t \t orb_en_diff \t error')
comm.Barrier()

# if options["corr_samp"]:
seeds = random.randint(random.PRNGKey(seed),
                    shape=(n_runs,), minval=0, maxval=10000*n_runs)

_,loc_orb_en1,loc_wt1,_,loc_orb_en2,loc_wt2 \
    = corr_sample.lno_cs_seeds_scan(seeds,prop_steps,
                                    prop_data1,ham_data1,prop1,trial1,wave_data1,
                                    prop_data2,ham_data2,prop2,trial2,wave_data2,
                                    sampler,MPI)


comm.Barrier()
if rank == 0:
    # glb_en1 = np.empty(size*loc_en1.size,dtype='float32')
    # glb_en2 = np.empty(size*loc_en2.size,dtype='float32')
    glb_orb_en1 = np.empty(size*loc_orb_en1.size,dtype='float32')
    glb_orb_en2 = np.empty(size*loc_orb_en2.size,dtype='float32')
    glb_wt1 = np.empty(size*loc_wt1.size,dtype='float32')
    glb_wt2 = np.empty(size*loc_wt2.size,dtype='float32')
else:
    # glb_en1 = None
    # glb_en2 = None
    glb_orb_en1 = None
    glb_orb_en2 = None
    glb_wt1 = None
    glb_wt2 = None
comm.Barrier()

# loc_en1 = np.asarray(loc_en1,dtype='float32')
# loc_en2 = np.asarray(loc_en2,dtype='float32')
loc_orb_en1 = np.asarray(loc_orb_en1,dtype='float32')
loc_orb_en2 = np.asarray(loc_orb_en2,dtype='float32')
loc_wt1 = np.asarray(loc_wt1,dtype='float32')
loc_wt2 = np.asarray(loc_wt2,dtype='float32')

# comm.Gather(loc_en1,glb_en1,root=0)
# comm.Gather(loc_en2,glb_en2,root=0)
comm.Gather(loc_orb_en1,glb_orb_en1,root=0)
comm.Gather(loc_orb_en2,glb_orb_en2,root=0)
comm.Gather(loc_wt1,glb_wt1,root=0)
comm.Gather(loc_wt2,glb_wt2,root=0)


comm.Barrier()
if rank == 0:
    # glb_en1 = glb_en1.reshape(size,n_runs,prop_steps).T
    # glb_en2 = glb_en2.reshape(size,n_runs,prop_steps).T
    glb_orb_en1 = glb_orb_en1.reshape(size,n_runs,prop_steps).T
    glb_orb_en2 = glb_orb_en2.reshape(size,n_runs,prop_steps).T
    glb_wt1 = glb_wt1.reshape(size,n_runs,prop_steps).T
    glb_wt2 = glb_wt2.reshape(size,n_runs,prop_steps).T

    # en1 = np.zeros((prop_steps,n_runs))
    # en2 = np.zeros((prop_steps,n_runs))
    en_diff = np.zeros((prop_steps,n_runs))
    orb_en1 = np.zeros((prop_steps,n_runs))
    orb_en2 = np.zeros((prop_steps,n_runs))
    orb_en_diff = np.zeros((prop_steps,n_runs))

    for step in range(prop_steps):

        for run in range(n_runs):
            # en1[step,run] = sum(glb_en1[step,run,:])/sum(glb_wt1[step,run,:])
            # en2[step,run] = sum(glb_en2[step,run,:])/sum(glb_wt2[step,run,:])
            # en_diff[step,run] = en1[step,run] - en2[step,run]
            orb_en1[step,run] = sum(glb_orb_en1[step,run,:])/sum(glb_wt1[step,run,:])
            orb_en2[step,run] = sum(glb_orb_en2[step,run,:])/sum(glb_wt2[step,run,:])
            orb_en_diff[step,run] = orb_en2[step,run] - orb_en1[step,run]

        # en_mean1 = en1[step,:].mean()
        # en_mean2 = en2[step,:].mean()
        # en_diff_mean = en_diff[step,:].mean()
        # en_err1 = en1[step,:].std()/np.sqrt(n_runs)
        # en_err2 = en2[step,:].std()/np.sqrt(n_runs)
        # en_diff_mean_err = en_diff[step,:].std()/np.sqrt(n_runs)
        orb_en_mean1 = orb_en1[step,:].mean()
        orb_en_mean2 = orb_en2[step,:].mean()
        orb_en_diff_mean = orb_en_diff[step,:].mean()
        orb_en_err1 = orb_en1[step,:].std()/np.sqrt(n_runs)
        orb_en_err2 = orb_en2[step,:].std()/np.sqrt(n_runs)
        orb_en_diff_mean_err = orb_en_diff[step,:].std()/np.sqrt(n_runs)

        print(f'  {step+1}'
            #   f'\t {en_mean1:.6f} \t {en_err1:.6f}' 
            #   f'\t {en_mean2:.6f} \t {en_err2:.6f}'
            #   f'\t {en_diff_mean:.6f} \t {en_diff_mean_err:.6f}'
              f'\t {orb_en_mean1:.6f} \t {orb_en_err1:.6f}' 
              f'\t {orb_en_mean2:.6f} \t {orb_en_err2:.6f}'
              f'\t {orb_en_diff_mean:.6f} \t {orb_en_diff_mean_err:.6f}')

    print(f'# final result')
    print(f'# orbital energy 1: {orb_en_mean1:.6f} +\- {orb_en_err1:.6f}')
    print(f'# orbital energy 2: {orb_en_mean2:.6f} +\- {orb_en_err2:.6f}')
    print(f'# correlated d_energy: {orb_en_diff_mean:.6f} +\- {orb_en_diff_mean_err:.6f}')
    end_time = time.time()
    print(f'# total run time: {end_time - init_time:.2f}')
comm.Barrier()
