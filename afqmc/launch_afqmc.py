import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import pickle

def ph_afqmc(
        options,
        script=None,
        option_file='options.bin',
        ):

    with open(option_file, 'wb') as f:
        pickle.dump(options, f)

    if script is None:
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

    os.system(f" python {script} |tee afqmc.out")

def fp_afqmc(
        options,
        script=None,
        option_file='options.bin',
        ):

    with open(option_file, 'wb') as f:
        pickle.dump(options, f)

    if script is None:
        if  'pt2' in options['trial']:
            script='run_fpafqmc_pt2.py'
        else:
            script = 'run_fpafqmc.py'

    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/scripts/{script}"
    print(f'QMC script: {script}')

    os.system(f" python {script} |tee afqmc.out")