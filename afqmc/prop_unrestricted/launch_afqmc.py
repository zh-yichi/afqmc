import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import pickle
from afqmc import config

def run_afqmc(options,
              option_file='options.bin',
              script=None,
              ):

    with open(option_file, 'wb') as f:
        pickle.dump(options, f)
    
    if options["use_gpu"]:
        config.afqmc_config["use_gpu"] = True
        config.setup_jax()
        print(f'running AFQMC on GPU')
        gpu_flag = "--use_gpu"
    else:
        print(f'running AFQMC on CPU')
        gpu_flag = ""

    if script is None:
        if  'pt' in options['trial']:
            if '2' in options['trial']:
                script='run_afqmc_ccsd_pt2_nompi.py'
            else:
                script='run_afqmc_ccsd_pt_nompi.py'
        else:
            script='run_afqmc_sampling_nompi.py'

    if options["free_projection"]:
        script = 'run_fp_afqmc_sampling_nompi.py'

    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/{script}"
    print(f'QMC script: {script}')

    os.system(
        # f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1;"
        f" python {script} {gpu_flag} |tee afqmc.out"
    )