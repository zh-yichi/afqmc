import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import pickle

def run_afqmc(options,
              option_file='options.bin',
              script=None,
              ):

    with open(option_file, 'wb') as f:
        pickle.dump(options, f)
    
    # if options["use_gpu"]:
        # config.afqmc_config["use_gpu"] = True
        # config.setup_jax()
    #     gpu_flag = "--use_gpu"
    # else:
    #     gpu_flag = ""

    if script is None:
        if options["free_projection"]:
            if  'pt2' in options['trial']:
                script='run_fpafqmc_pt2.py'
            else:
                script = 'run_fpafqmc.py'

        else:
            if  'pt' in options['trial']:
                if '2' in options['trial']:
                    script='run_afqmc_pt2ccsd.py'
                else:
                    script='run_afqmc_ptccsd.py'
            
            elif 'sto' in options['trial']:
                if '2' in options['trial']:
                    script='run_afqmc_stoccsd2.py'
                else:
                    script='run_afqmc_stoccsd.py'

            else:
                script='run_afqmc.py'

    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/scripts/{script}"
    print(f'QMC script: {script}')

    os.system(
        # f" python {script} {gpu_flag} |tee afqmc.out"
        f" python {script} |tee afqmc.out"
    )