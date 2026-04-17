import time
import argparse
import numpy as np
import jax
from jax import random
from jax import numpy as jnp
from functools import partial
from ad_afqmc import config
from ad_afqmc.lno_afqmc import sampling
from ad_afqmc.lno_afqmc import lno_afqmc

init_time = time.time()
print = partial(print, flush=True)
jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    if args.use_gpu:
        config.afqmc_config["use_gpu"] = True

config.setup_jax()

ham_data, prop, trial, wave_data, sampler, options = (lno_afqmc._prep_afqmc())

print(f"# norb: {trial.norb}")
print(f"# nelec: {trial.nelec}")

for op in options:
    if options[op] is not None:
        print(f"# {op}: {options[op]}")

### initialize propagation
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
prop_data["key"] = random.PRNGKey(options["seed"])
prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
prop_data["n_killed_walkers"] = 0
prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
e_init = prop_data["e_estimate"]
# h0 = ham_data['h0']

e0, t1olp, eorb, t2eorb, t2orb, e0bar \
    = trial._calc_eorb_pt2(prop_data['walkers'][0], ham_data, wave_data)

e0 = jnp.real(e0)
t1olp = jnp.real(t1olp)
eorb = jnp.real(eorb)
t2eorb = jnp.real(t2eorb)
t2orb = jnp.real(t2orb)
e0bar = jnp.real(e0bar)
eorb_pt = jnp.real(eorb/t1olp + t2eorb/t1olp - t2orb*e0bar/t1olp**2)


print(f'# Propagating with {options["n_walkers"]} walkers')
print(f"# Initial orbital_pt energy: {eorb_pt:.6f}")
print("# Orbital energis are not calculated in equilibration sweeps!")
print("# Equilibration sweeps with HF Trial: ")
print("# Atom_Time \t E(<HF|H|AF>)  \t wall_Time")
print(f"   {0.:.2f} \t {e0:.6f} \t {time.time() - init_time:.2f}")

sampler_eq = sampling.sampler_eq(
    n_prop_steps=50,
    n_ene_blocks=5,
    n_sr_blocks=10,
    n_chol = sampler.n_chol
    )

block_time = prop.dt * sampler_eq.n_prop_steps * sampler_eq.n_ene_blocks * sampler_eq.n_sr_blocks

for n in range(options["n_eql"]):
    prop_data, (wt, e0) = \
        sampler_eq.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)

    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e0

    print(f"  {(n+1)*block_time:.2f} \t {e0:.6f} \t {time.time() - init_time:.2f} ")

print("#\n Sampling sweeps:")
print("#  NBlock \t E(<HF|H|AF>) \t error \t \t E(<pt2|H|AF>) \t error \t \t time ")

wt_sp = np.zeros(sampler.n_blocks,dtype="float64")
e0_sp = np.zeros(sampler.n_blocks,dtype="float64")
eorb_sp = np.zeros(sampler.n_blocks,dtype="float64")
t2eorb_sp = np.zeros(sampler.n_blocks,dtype="float64")
t2orb_sp = np.zeros(sampler.n_blocks,dtype="float64")
e0bar_sp = np.zeros(sampler.n_blocks,dtype="float64")
t1olp_sp = np.zeros(sampler.n_blocks,dtype="float64")
ept_sp = np.zeros(sampler.n_blocks,dtype="float64")

# eorb_pt_err = options["max_error"] + 1e-3

for n in range(sampler.n_blocks):
    prop_data, (wt, e0, eorb, t2eorb, t2orb, e0bar, t1olp) = \
        sampler.propagate_phaseless(prop_data, ham_data, prop, trial, wave_data)
    
    wt_sp[n] = wt
    e0_sp[n] = e0
    eorb_sp[n] = eorb
    t2eorb_sp[n] = t2eorb
    t2orb_sp[n] = t2orb
    e0bar_sp[n] = e0bar
    t1olp_sp[n] = t1olp

    prop_data = prop.orthonormalize_walkers(prop_data)
    prop_data = prop.stochastic_reconfiguration_local(prop_data)
    prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * e0
    
    eorb_pt = eorb/t1olp + t2eorb/t1olp - t2orb*e0bar/t1olp**2
    ept_sp[n] = eorb_pt

    if (n+1) % (max(sampler.n_blocks // 10, 1)) == 0 and n > 0:                
        wt = np.sum(wt_sp[:n+1])
        e0 = np.sum(wt_sp[:n+1] * e0_sp[:n+1]) / wt
        eorb = np.sum(wt_sp[:n+1] * eorb_sp[:n+1]) / wt
        t2eorb = np.sum(wt_sp[:n+1] * t2eorb_sp[:n+1]) / wt
        t2orb = np.sum(wt_sp[:n+1] * t2orb_sp[:n+1]) / wt
        e0bar = np.sum(wt_sp[:n+1] * e0bar_sp[:n+1]) / wt
        t1olp = np.sum(wt_sp[:n+1] * t1olp_sp[:n+1]) / wt
        
        e0_err = np.sqrt(np.sum(wt_sp[:n+1] * (e0_sp[:n+1] - e0)**2) / wt / (n+1))

        eorb_pt = eorb/t1olp + t2eorb/t1olp - t2orb*e0bar/t1olp**2
        # (p_eorb,p_t2eorb,p_t2orb,p_t2orb,p_t1olp)
        dE = np.array([
            1/t1olp,
            1/t1olp,
            -e0bar/t1olp**2,
            -t2orb/t1olp**2,
            -eorb/t1olp**2 - t2eorb/t1olp**2 + 2*t2orb*e0bar/t1olp**3
            ])
        cov = np.cov([
            eorb_sp[:n+1],
            t2eorb_sp[:n+1],
            t2orb_sp[:n+1],
            e0bar_sp[:n+1], 
            t1olp_sp[:n+1]
            ])
        
        eorb_pt_err = np.sqrt(dE @ cov @ dE) / np.sqrt((n+1))
        
        print(f"  {n+1:4d} \t \t {e0:.6f} \t {e0_err:.6f} \t"
              f"  {eorb_pt:.6f} \t {eorb_pt_err:.6f} \t"
              f"  {time.time() - init_time:.2f}")

        if eorb_pt_err < options["max_error"] and n > 20:
            break

print(f"# Finish Sampling")
nsamples = n + 1
print(f'# total number of samples {nsamples}')

wt_sp = wt_sp[:nsamples]
e0_sp = e0_sp[:nsamples]
eorb_sp = eorb_sp[:nsamples]
t2eorb_sp = t2eorb_sp[:nsamples]
t2orb_sp = t2orb_sp[:nsamples]
e0bar_sp = e0bar_sp[:nsamples]
t1olp_sp = t1olp_sp[:nsamples]
ept_sp = ept_sp[:nsamples]

wt = np.sum(wt_sp)
e0 = np.sum(wt_sp * e0_sp) / wt
eorb = np.sum(wt_sp * eorb_sp) / wt
t2eorb = np.sum(wt_sp * t2eorb_sp) / wt
t2orb = np.sum(wt_sp * t2orb_sp) / wt
e0bar = np.sum(wt_sp * e0bar_sp) / wt
t1olp = np.sum(wt_sp * t1olp_sp) / wt

e0_err = np.sqrt(np.sum(wt_sp * (e0_sp - e0)**2) / wt / nsamples)
print(f"# AFQMC/HF Energy: {e0:.6f} +/- {e0_err:.6f}")

eorb_pt = eorb/t1olp + t2eorb/t1olp - t2orb*e0bar/t1olp**2
# (p_eorb,p_t2eorb,p_t2orb,p_t2orb,p_t1olp)
dE = np.array([1/t1olp,
               1/t1olp,
               -e0bar/t1olp**2,
               -t2orb/t1olp**2,
               -eorb/t1olp**2 - t2eorb/t1olp**2 + 2*t2orb*e0bar/t1olp**3
               ])
cov = np.cov([
    eorb_sp,
    t2eorb_sp,
    t2orb_sp,
    e0bar_sp,
    t1olp_sp,
    ])
# print(eorb_sp)
# print(e0bar_sp)
# print(cov[:4,:4])

eorb_pt_err = np.sqrt(dE @ cov @ dE)/np.sqrt(nsamples)
print(f"# AFQMC/pt2CCSD energy (Raw): {eorb_pt:.6f} +/- {eorb_pt_err:.6f}")

print(f'# Remove outliners by Mahalanobis distance ')
t1_var = np.sum(t1olp_sp - t1olp)**2
if np.sqrt(t1_var) < 1e-12:
    print(f'# <exp(T1)> is not varying ({t1_var:.2e}) during the sampling, it might cause numerical in the variance')
    print(f'# <exp(T1)> and <H_bar> removed from the covariant matrix. Recommend using ptCCSD trial')
    x = np.vstack([eorb_sp, t2eorb_sp, t2orb_sp]).T
    mu = np.array([eorb, t2eorb, t2orb])
    d2 = np.zeros(nsamples)
    for i in range(nsamples):
        d2[i] = (x[i]-mu).T @ np.linalg.inv(cov[:3,:3]) @ (x[i]-mu)
else:
    x = np.vstack([eorb_sp, t2eorb_sp, t2orb_sp, e0bar_sp, t1olp_sp]).T
    mu = np.array([eorb, t2eorb, t2orb, e0bar, t1olp])
    d2 = np.zeros(nsamples)
    for i in range(nsamples):
        d2[i] = (x[i]-mu).T @ np.linalg.inv(cov) @ (x[i]-mu)

mask = d2 < 20
print(f'# remove outliers {nsamples - np.sum(mask)}')

wt_sp = wt_sp[mask]
eorb_sp = eorb_sp[mask]
t2eorb_sp = t2eorb_sp[mask]
t2orb_sp = t2orb_sp[mask]
e0bar_sp = e0bar_sp[mask]
t1olp_sp = t1olp_sp[mask]

wt = np.sum(wt_sp)
eorb = np.sum(wt_sp * eorb_sp) / wt
t2eorb = np.sum(wt_sp * t2eorb_sp) / wt
t2orb = np.sum(wt_sp * t2orb_sp) / wt
e0bar = np.sum(wt_sp * e0bar_sp) / wt
t1olp = np.sum(wt_sp * t1olp_sp) / wt

eorb_pt = eorb/t1olp + t2eorb/t1olp - t2orb*e0bar/t1olp**2
# (p_eorb,p_t2eorb,p_t2orb,p_t2orb,p_t1olp)
dE = np.array([1/t1olp,
               1/t1olp,
               -e0bar/t1olp**2,
               -t2orb/t1olp**2,
               -eorb/t1olp**2 - t2eorb/t1olp**2 + 2*t2orb*e0bar/t1olp**3
               ])
cov = np.cov([
    eorb_sp,
    t2eorb_sp,
    t2orb_sp,
    e0bar_sp,
    t1olp_sp,
    ])

eorb_pt_err = np.sqrt(dE @ cov @ dE)/np.sqrt(nsamples)
print(f"# AFQMC/pt2CCSD energy (Mahalanobis): {eorb_pt:.6f} +/- {eorb_pt_err:.6f}")

#### direct observation all samples are treated independently ####
# ept_sp = ept_sp[:n+1]
d = np.abs(ept_sp - np.median(ept_sp))
d_med = np.median(d) + 1e-7
mask = d/d_med < 20
ept_clean = ept_sp[mask]
print('# remove outliers in direct sampling: ', len(ept_sp)-len(ept_clean))

eorb_pt = np.mean(ept_clean)
eorb_pt_err = np.std(ept_clean)/np.sqrt(n)

print(f"# AFQMC/CCSD_PT2 Orbital Ept (direct observation): {eorb_pt:.6f} +/- {eorb_pt_err:.6f}")
print(f"# total run time: {time.time() - init_time:.2f}")
