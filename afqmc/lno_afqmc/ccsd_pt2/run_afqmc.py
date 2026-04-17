import time
import argparse
import numpy as np
from jax import numpy as jnp
from jax import random
from functools import partial
from ad_afqmc import config, stat_utils
from ad_afqmc.lno_afqmc import sampling
from ad_afqmc.lno_afqmc import lno_afqmc

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
init_time = time.time()

print = partial(print, flush=True)

ham_data, prop, trial, wave_data, sampler, options = (
    lno_afqmc._prep_afqmc())

if rank == 0:
    print(f"# norb: {trial.norb}")
    print(f"# nelec: {trial.nelec}")
    print("#")
    for op in options:
        if options[op] is not None:
            print(f"# {op}: {options[op]}")

### initialize propagation
seed = options["seed"]
init_walkers = None
trial_rdm1 = trial.get_rdm1(wave_data)
if "rdm1" not in wave_data:
    wave_data["rdm1"] = trial_rdm1
ham_data = trial._build_measurement_intermediates(ham_data, wave_data)
ham_data = prop._build_propagation_intermediates(ham_data, trial, wave_data)

prop_data = prop.init_prop_data(trial, wave_data, ham_data, init_walkers)
if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
    raise ValueError(
        "Initial overlaps are zero. Pass walkers with non-zero overlap."
    )
prop_data["key"] = random.PRNGKey(seed + rank)
prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
e_init = prop_data["e_estimate"]
h0 = ham_data['h0']

e0, t1olp, eorb, t2eorb, t2orb, e0bar \
    = trial._calc_eorb_pt2(prop_data['walkers'][0], ham_data, wave_data)
e0 = jnp.real(e0)
t1olp = jnp.real(t1olp)
eorb = jnp.real(eorb)
t2eorb = jnp.real(t2eorb)
t2orb = jnp.real(t2orb)
e0bar = jnp.real(e0bar)
eorb_pt = jnp.real(eorb/t1olp + t2eorb/t1olp - t2orb*e0bar/t1olp**2)

comm.Barrier()
if rank == 0:
    print('# \n')
    print(f'# Propagating with {options["n_walkers"]*size} walkers')
    print(f"# Initial orbital_pt energy: {eorb_pt:.6f}")
    print("# Orbital energis are not calculated in equilibration sweeps!")
    print("# Equilibration sweeps:")
    print("#   Iter \t Energy_hf  \t time")
    print(f"  {0:5d} \t {e0:.6f} \t {time.time() - init_time:.2f}")
comm.Barrier()

sampler_eq = sampling.sampler_eq(
    n_prop_steps=50, n_ene_blocks=5, n_sr_blocks=10, n_chol = sampler.n_chol)
for n in range(1,options["n_eql"]+1):
    prop_data, (wt, e0) = \
        sampler_eq.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)

    wt = np.array([wt], dtype="float64")
    e0 = np.array([e0], dtype="float64")

    gather_wt = None
    gather_e0 = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float64")
        gather_e0 = np.zeros(size, dtype="float64")
    comm.Barrier()

    comm.Gather(wt, gather_wt, root=0)
    comm.Gather(e0, gather_e0, root=0)

    comm.Barrier()
    if rank == 0:
        assert gather_wt is not None

        wt= np.sum(gather_wt)
        e0 = np.sum(gather_wt * gather_e0) / wt
    comm.Barrier()

    comm.Bcast(wt, root=0)
    comm.Bcast(e0, root=0)
    comm.Bcast(eorb, root=0)
    comm.Bcast(t2eorb, root=0)
    comm.Bcast(t2orb, root=0)
    comm.Bcast(e0bar, root=0)
    comm.Bcast(t1olp, root=0)

    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = \
          0.9 * prop_data["e_estimate"] + 0.1 * e0
    # eorb_pt = eorb + t2eorb - t2orb*e0bar
    eorb_pt = eorb/t1olp + t2eorb/t1olp - t2orb*e0bar/t1olp**2

    comm.Barrier()
    if rank == 0:
        print(f"  {n:5d} \t {e0:.6f} \t {time.time() - init_time:.2f} ")
    comm.Barrier()

comm.Barrier()
if rank == 0:
    print("#\n Sampling sweeps:")
    print("#  Iter \t Energy_hf \t error \t \t Ept_orb \t error \t \t time ")
comm.Barrier()

glb_wt = None
glb_e0 = None
glb_eorb = None
glb_t2eorb = None
glb_t2orb = None
glb_e0bar = None
glb_t1olp = None

comm.Barrier()
if rank == 0:
    glb_wt = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_e0 = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_eorb = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_t2eorb = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_t2orb = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_e0bar = np.zeros(size * sampler.n_blocks,dtype="float64")
    glb_t1olp = np.zeros(size * sampler.n_blocks,dtype="float64")
    ept_samples = np.zeros(sampler.n_blocks,dtype="float64")
comm.Barrier()

eorb_pt_err = np.array([options["max_error"] + 1e-3])
for n in range(sampler.n_blocks):
    prop_data, (wt, e0, eorb, t2eorb, t2orb, e0bar, t1olp) = \
        sampler.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)
    
    wt = np.array([wt], dtype="float64")
    e0 = np.array([e0], dtype="float64")
    eorb = np.array([eorb], dtype="float64")
    t2eorb = np.array([t2eorb], dtype="float64")
    t2orb = np.array([t2orb], dtype="float64")
    e0bar = np.array([e0bar], dtype="float64")
    t1olp = np.array([t1olp], dtype="float64")

    gather_wt = None
    gather_e0 = None
    gather_eorb = None
    gather_t2eorb = None
    gather_t2orb = None
    gather_e0bar = None
    gather_t1olp = None

    comm.Barrier()
    if rank == 0:
        gather_wt = np.zeros(size, dtype="float64")
        gather_e0 = np.zeros(size, dtype="float64")
        gather_eorb = np.zeros(size, dtype="float64")
        gather_t2eorb = np.zeros(size, dtype="float64")
        gather_t2orb = np.zeros(size, dtype="float64")
        gather_e0bar = np.zeros(size, dtype="float64")
        gather_t1olp = np.zeros(size, dtype="float64")
    comm.Barrier()

    comm.Gather(wt, gather_wt, root=0)
    comm.Gather(e0, gather_e0, root=0)
    comm.Gather(eorb, gather_eorb, root=0)
    comm.Gather(t2eorb, gather_t2eorb, root=0)
    comm.Gather(t2orb, gather_t2orb, root=0)
    comm.Gather(e0bar, gather_e0bar, root=0)
    comm.Gather(t1olp, gather_t1olp, root=0)

    comm.Barrier()
    if rank == 0:
        assert gather_wt is not None

        glb_wt[n*size: (n+1)*size] = gather_wt
        glb_e0[n*size: (n+1)*size] = gather_e0
        glb_eorb[n*size: (n+1)*size] = gather_eorb
        glb_t2eorb[n*size: (n+1)*size] = gather_t2eorb
        glb_t2orb[n*size : (n+1)*size] = gather_t2orb
        glb_e0bar[n*size : (n+1)*size] = gather_e0bar
        glb_t1olp[n*size : (n+1)*size] = gather_t1olp

        wt= np.sum(gather_wt)
        e0 = np.sum(gather_wt * gather_e0) / wt
        eorb = np.sum(gather_wt * gather_eorb) / wt
        t2eorb = np.sum(gather_wt * gather_t2eorb) / wt
        t2orb = np.sum(gather_wt * gather_t2orb) / wt
        e0bar = np.sum(gather_wt * gather_e0bar) / wt
        t1olp = np.sum(gather_wt * gather_t1olp) / wt
    comm.Barrier()

    comm.Bcast(wt, root=0)
    comm.Bcast(e0, root=0)
    comm.Bcast(eorb, root=0)
    comm.Bcast(t2eorb, root=0)
    comm.Bcast(t2orb, root=0)
    comm.Bcast(e0bar, root=0)
    comm.Bcast(t1olp, root=0)

    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_global(prop_data, comm)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e0
    
    comm.Barrier()
    if rank == 0:
        eorb_pt = eorb/t1olp + t2eorb/t1olp - t2orb*e0bar/t1olp**2
        ept_samples[n] = eorb_pt
    comm.Barrier()

    if n % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:
        comm.Barrier()
        if rank == 0:                        
            wt = np.sum(glb_wt[:(n+1)*size])
            e0 = np.sum(glb_wt[:(n+1)*size] * glb_e0[:(n+1)*size]) / wt
            eorb = np.sum(glb_wt[:(n+1)*size] * glb_eorb[:(n+1)*size]) / wt
            t2eorb = np.sum(glb_wt[:(n+1)*size] * glb_t2eorb[:(n+1)*size]) / wt
            t2orb = np.sum(glb_wt[:(n+1)*size] * glb_t2orb[:(n+1)*size]) / wt
            e0bar = np.sum(glb_wt[:(n+1)*size] * glb_e0bar[:(n+1)*size]) / wt
            t1olp = np.sum(glb_wt[:(n+1)*size] * glb_t1olp[:(n+1)*size]) / wt
            
            e0_err = np.sqrt(np.sum(glb_wt[:(n+1)*size] * 
                         (glb_e0[:(n+1)*size] - e0)**2) / wt / (n+1)*size)

            eorb_pt = eorb/t1olp + t2eorb/t1olp - t2orb*e0bar/t1olp**2
            # (p_eorb,p_t2eorb,p_t2orb,p_t2orb,p_t1olp)
            dE = np.array([1/t1olp,1/t1olp,-e0bar/t1olp**2,-t2orb/t1olp**2,
                           -eorb/t1olp**2-t2eorb/t1olp**2+2*t2orb*e0bar/t1olp**3])
            cov = np.cov([glb_eorb[:(n+1)*size],
                          glb_t2eorb[:(n+1)*size],
                          glb_t2orb[:(n+1)*size],
                          glb_e0bar[:(n+1)*size], 
                          glb_t1olp[:(n+1)*size]])
            eorb_pt_err[0] = np.sqrt(dE @ cov @ dE)/np.sqrt((n+1)*size)
            
            print(f"  {n:4d} \t \t {e0:.6f} \t {e0_err:.6f} \t"
                  f"  {eorb_pt:.6f} \t {eorb_pt_err[0]:.6f} \t"
                  f"  {time.time() - init_time:.2f}")
        comm.Barrier()
        
        comm.Bcast(eorb_pt_err, root=0)
        if eorb_pt_err[0] < options["max_error"] and n > 10:
            break

comm.Barrier()
if rank == 0:
    assert glb_wt is not None
    samples_clean, idx = stat_utils.reject_outliers(
        np.stack((
                glb_wt[:(n+1)*size],
                glb_e0[:(n+1)*size],
                glb_eorb[:(n+1)*size],
                glb_t2eorb[:(n+1)*size],
                glb_t2orb[:(n+1)*size],
                glb_e0bar[:(n+1)*size],
                glb_t1olp[:(n+1)*size],
                )).T,
                1,
            )
    nsamples = samples_clean.shape[0]
    print(
        f"# Number of outliers in post: {glb_wt[:(n+1)*size].size - nsamples} "
        )
    
    glb_wt = samples_clean[:, 0]
    glb_e0 = samples_clean[:, 1]
    glb_eorb = samples_clean[:, 2]
    glb_t2eorb = samples_clean[:, 3]
    glb_t2orb = samples_clean[:, 4]
    glb_e0bar = samples_clean[:, 5]
    glb_t1olp = samples_clean[:, 6]

    wt = np.sum(glb_wt)
    e0 = np.sum(glb_wt * glb_e0) / wt
    eorb = np.sum(glb_wt * glb_eorb) / wt
    t2eorb = np.sum(glb_wt * glb_t2eorb) / wt
    t2orb = np.sum(glb_wt * glb_t2orb) / wt
    e0bar = np.sum(glb_wt * glb_e0bar) / wt
    t1olp = np.sum(glb_wt * glb_t1olp) / wt

    e0_err = np.sqrt(np.sum(glb_wt * (glb_e0 - e0)**2) / wt / nsamples)

    eorb_pt = eorb/t1olp + t2eorb/t1olp - t2orb*e0bar/t1olp**2
    # (p_eorb,p_t2eorb,p_t2orb,p_t2orb,p_t1olp)
    dE = np.array([1/t1olp,1/t1olp,-e0bar/t1olp**2,-t2orb/t1olp**2,
                   -eorb/t1olp**2-t2eorb/t1olp**2+2*t2orb*e0bar/t1olp**3])
    cov = np.cov([glb_eorb,glb_t2eorb,glb_t2orb,glb_e0bar,glb_t1olp])
    eorb_pt_err = np.sqrt(dE @ cov @ dE)/np.sqrt(nsamples)

    ept_samples = ept_samples[:n+1]
    d = np.abs(ept_samples-np.median(ept_samples))
    d_med = np.median(d) + 1e-7
    mask = d/d_med < 10
    ept_clean = ept_samples[mask]
    print('# remove outliers in direct sampling: ', len(ept_samples)-len(ept_clean))

    eorb_pt_do = np.mean(ept_clean)
    eorb_pt_do_err = np.std(ept_clean)/np.sqrt(n)

    print(f"# Final Results")
    print(f"# AFQMC/HF Energy: {e0:.6f} +/- {e0_err:.6f}")
    print(f"# AFQMC/CCSD_PT2 Orbital Ept (covariance): {eorb_pt:.6f} +/- {eorb_pt_err:.6f}")
    print(f"# AFQMC/CCSD_PT2 Orbital Ept (direct observation): {eorb_pt_do:.6f} +/- {eorb_pt_do_err:.6f}")
    print(f"# total run time: {time.time() - init_time:.2f}")

comm.Barrier()
