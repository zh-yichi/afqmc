import os
import pickle
from ad_afqmc import config

def run_afqmc(options,
              option_file='options.bin',
              script=None,
              ):

    with open(option_file, 'wb') as f:
        pickle.dump(options, f)
    
    if options["use_gpu"]:
        config.afqmc_config["use_gpu"] = True
        config.setup_jax()
        print(f'# running AFQMC on GPU')
        gpu_flag = "--use_gpu"
    else:
        print(f'# running AFQMC on CPU')
        gpu_flag = ""

    if script is None:
        if "stoccsd2" in options["trial"]:
            script='run_stoccsd2_sampling.py'
        elif "stoccsd" in options["trial"]:
            script='run_stoccsd_sampling.py'

    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/{script}"
    print(f'# AFQMC script: {script}')

    os.system(
        # f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1;"
        f" python {script} {gpu_flag} |tee afqmc.out"
    )