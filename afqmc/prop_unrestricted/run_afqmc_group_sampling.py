from functools import partial
from jax import random
#from mpi4py import MPI
import numpy as np
from jax import numpy as jnp
from ad_afqmc import config, stat_utils
from ad_afqmc.prop_unrestricted import prep, sampling
import time
import argparse

from ad_afqmc import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    if args.use_gpu:
        config.afqmc_config["use_gpu"] = True

config.setup_jax()
MPI = config.setup_comm()
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print = partial(print, flush=True)

ham_data, ham, prop, trial, wave_data, sampler, observable, options = (prep._prep_afqmc())

init_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

### initialize propagation
seed = options["seed"]
init_walkers = None
trial_rdm1 = trial.get_rdm1(wave_data)
if "rdm1" not in wave_data:
    wave_data["rdm1"] = trial_rdm1
ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)
h0 = ham_data['h0']
# mywalkers = trial.get_init_walkers(wave_data, prop.n_walkers, restricted=True)

# comm.Barrier()
# if rank == 0:
#     print(f'# my walkers ', mywalkers)
#     print(f'# one walker ', mywalkers[0])
#     print(f'# one walker shape', mywalkers[0].shape)
# comm.Barrier()

prop_data = prop.init_prop_data(trial, wave_data, ham_data, init_walkers)
# comm.Barrier()
# if rank == 0:
#     print(f'# initial walkers shape ', prop_data["walkers"].shape)
# comm.Barrier()

if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError(
        "Initial overlaps are zero. Pass walkers with non-zero overlap."
    )
prop_data["key"] = random.PRNGKey(seed + rank)

prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["group_weights"] = jnp.ones(prop.n_walkers//sampler.group_size, dtype=jnp.complex128)
prop_data["n_killed_groups"] = 0
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]

comm.Barrier()
if rank == 0:
    e_init = prop_data["e_estimate"]
    print('# \n')
    print(f'# trial {trial}')
    print(f'# prop {prop}')
    print(f'# Propagating with {options["n_walkers"]*size} walkers')
    # print(f'# initial walkers shape ', prop_data["walkers"].shape)
    print("# Equilibration sweeps:")
    print("#   Iter \t energy \t Walltime")
    print(f"  {0:5d} \t {e_init:.6f} \t {time.time() - init_time:.2f}")
comm.Barrier()

sampler_eq = sampling.sampler_group(
    options["group_size"], n_prop_steps=50, n_ene_blocks=5, n_sr_blocks=10, n_chol = sampler.n_chol)

for n in range(1,options["n_eql"]+1):
    prop_data, (blk_wt, blk_e) = sampler_eq.propagate_phaseless(
        ham, ham_data, prop, prop_data, trial, wave_data)

    blk_wt = np.array([blk_wt.real], dtype="float64")
    blk_e = np.array([blk_e.real], dtype="float64")

    gather_wt = None
    gather_e = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float64")
        gather_e = np.zeros(size, dtype="float64")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_e, gather_e, root=0)

    comm.Barrier()
    if rank == 0:
        assert gather_wt is not None
        blk_wt= np.sum(gather_wt)
        blk_e = np.sum(gather_wt * gather_e) / blk_wt
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_e, root=0)

    # prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = (
         0.9 * prop_data["e_estimate"] + 0.1 * blk_e
         )

    comm.Barrier()
    if rank == 0:
        print(
            f"  {n:5d} \t {blk_e:.6f} \t {time.time() - init_time:.2f} "
        )
    comm.Barrier()

comm.Barrier()
if rank == 0:
    print("#\n# Sampling sweeps:")
    print("#  Iter \t energy \t error \t \t Walltime")
comm.Barrier()

glb_blk_wt = None
glb_blk_e = None

comm.Barrier()
if rank == 0:
    glb_blk_wt = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_blk_e = np.zeros(size * sampler.n_blocks,dtype="float64")
comm.Barrier()
    
for n in range(sampler.n_blocks):
    prop_data, (blk_wt, blk_e) = sampler.propagate_phaseless(
            ham, ham_data, prop, prop_data, trial, wave_data)
    
    blk_wt = np.array([blk_wt], dtype="float64")
    blk_e = np.array([blk_e], dtype="float64")

    gather_wt = None
    gather_e = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float64")
        gather_e = np.zeros(size, dtype="float64")
    comm.Barrier()

    comm.Gather(blk_wt, gather_wt, root=0)
    comm.Gather(blk_e, gather_e, root=0)

    comm.Barrier()
    if rank == 0:
        glb_blk_wt[n * size : (n + 1) * size] = gather_wt
        glb_blk_e[n * size : (n + 1) * size] = gather_e

        assert gather_wt is not None

        blk_wt= np.sum(gather_wt)
        blk_e = np.sum(gather_wt * gather_e) / blk_wt
    comm.Barrier()

    comm.Bcast(blk_wt, root=0)
    comm.Bcast(blk_e, root=0)

    # prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * blk_e


    if n % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:
        comm.Barrier()
        if rank == 0:
            e, e_err = \
                stat_utils.blocking_analysis(
                    glb_blk_wt[: (n + 1) * size],
                    glb_blk_e[: (n + 1) * size],
                    neql=0,
                )
            if e_err is not None:
                e_err = f"{e_err:.6f}"
            else:
                e_err = f"  {e_err}  "
            e = f"{e:.6f}"
            print(f"  {n:4d} \t \t {e} \t {e_err} \t"
                  f"  {time.time() - init_time:.2f}")
        comm.Barrier()

comm.Barrier()
if rank == 0:
    assert glb_blk_wt is not None

    samples_clean, idx = stat_utils.reject_outliers(
        np.stack((
                glb_blk_wt,
                glb_blk_e
                )).T,
                1,
            )

    print(
        f"# Number of outliers in post: {glb_blk_wt.size - samples_clean.shape[0]} "
        )
    
    glb_blk_wt = samples_clean[:, 0]
    glb_blk_e = samples_clean[:, 1]

    e, e_err = stat_utils.blocking_analysis(glb_blk_wt,glb_blk_e,neql=0,printQ=True)

    if e_err is not None:
        e_err = f"{e_err:.6f}"
    else:
        e_err = f"  {e_err}  "
    e = f"{e:.6f}"

    print(f"Final Results:")
    print(f"AFQMC energy: {e} +/- {e_err}")
    print(f"total run time: {time.time() - init_time:.2f}")

comm.Barrier()
