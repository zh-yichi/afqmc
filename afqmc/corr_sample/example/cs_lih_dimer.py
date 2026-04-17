from ad_afqmc.corr_sample import corr_sample

options = {
    "dt": 0.01,
    "n_exp_terms": 6,
    "n_walkers": 30,
    "n_runs": 50,
    "rlx_steps": 0,
    "prop_steps": 10,
    "seed": 23,
    "walker_type": "rhf",
    "trial": "cisd",
    "corr_samp": True,
    "use_gpu": False,
    "free_proj": False,
}

files = {
    "mo1":"mo1.npz",
    "chol1":"chol1",
    "amp1":"amp1.npz",
    "mo2":"mo2.npz",
    "chol2":"chol2",
    "amp2":"amp2.npz",
}

corr_sample.run_cs_afqmc(options,files,nproc=5)