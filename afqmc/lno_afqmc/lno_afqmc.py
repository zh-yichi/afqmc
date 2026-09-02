"""CPU/GPU pipelined version of ``lno_afqmc.py``.

The serial driver in ``lno_afqmc.run_afqmc`` does, for every fragment:

    [CPU]  make_las -> LNO-MP2 -> LNO-CCSD        (pyscf, numpy/BLAS)
    [GPU]  prep_lno_integral -> AFQMC subprocess  (jax on the GPU)

The two stages of *different* fragments are completely independent: the CPU
stage only reads ``mlno``/``eris``/``mf`` (``make_las`` does not mutate the
LNO object, and every MP2/CCSD solve builds its own ``mcc``), and the GPU
stage of fragment ``i`` only needs ``(lno_coeff, lno_frozen, uocc_loc, t1,
t2)`` of that same fragment.  The AFQMC run itself is a *subprocess*
(``os.system``), so while it occupies the GPU the parent process is parked in
``waitpid`` with the GIL released -- a background thread can push the next
fragment's CCSD through at full BLAS speed.

This module therefore keeps a one-worker ``ThreadPoolExecutor`` running the
CPU stage ``prefetch`` fragments ahead of the AFQMC loop:

    frag 0 CPU | frag 1 CPU | frag 2 CPU | ...
                 frag 0 GPU | frag 1 GPU | ...

so the wall time per fragment becomes ``max(t_cpu, t_gpu)`` instead of
``t_cpu + t_gpu`` (the first fragment's CPU stage cannot be hidden).

Notes / caveats
---------------
* A *thread* (not a process) is used on purpose: the CPU stage needs ``mlno``
  and the (potentially huge) ``eris`` object, which a forked worker would
  either have to re-create or pickle.
* Memory: ``prefetch=1`` keeps the amplitudes of at most two fragments alive
  at the same time.  Lower it to 0 (or ``pipeline=False``) to fall back to the
  exactly serial behaviour of ``lno_afqmc.run_afqmc``.
* pyscf logging from the worker is captured into a per-fragment buffer and
  replayed by the main thread, so the printed log stays ordered by fragment.
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from jax import random

from pyscf import lib, scf
from pyscf.data import elements
from pyscf.lno import lnoccsd
from pyscf.lno import ulnoccsd
from collections.abc import Iterable

from afqmc.lno_afqmc import tools, integral
from afqmc.lno_afqmc import mod_lnoccsd

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
import io, time, gc, pickle

print = partial(print, flush=True)

def get_lnoccsd(mf, lo_coeff, frag_list, nfrozen, thresh, verbose=3):
    if isinstance(mf, scf.rhf.RHF):
        mlno = lnoccsd.LNOCCSD(mf, lo_coeff, frag_list, frozen=nfrozen).set(verbose=verbose)
    elif isinstance(mf, scf.uhf.UHF):
        mlno = ulnoccsd.ULNOCCSD(mf, lo_coeff, frag_list, frozen=nfrozen).set(verbose=verbose)
    else:
        raise NotImplementedError('LNO Only Support Restricted and Unrestricted MF!')

    if isinstance(thresh, float):
        mlno.lno_thresh = [thresh*10, thresh]
    elif isinstance(thresh, (list, tuple)):
        assert len(thresh) == 2
        mlno.lno_thresh = [thresh[0], thresh[1]]

    return mlno

def get_lnoparam(mf, lo_coeff, lno_thresh, lno_pct_occ, lno_norb, loidx, ifrag):
    if isinstance(mf, scf.rhf.RHF):
        orbloc = lo_coeff[:,loidx]
        lno_param = [{
            'thresh': lno_thresh[i],
            'pct_occ': lno_pct_occ[i],
            'norb': lno_norb[ifrag][i]
            } for i in [0,1]]
    elif isinstance(mf, scf.uhf.UHF):
        orbloc = [lo_coeff[0][:,loidx[0]], lo_coeff[1][:,loidx[1]]]
        lno_param = [
            [
                {
                    'thresh': (
                        lno_thresh[i][s] if isinstance(lno_thresh[i], Iterable)
                        else lno_thresh[i]
                    ),
                    'pct_occ': (
                        lno_pct_occ[i][s] if isinstance(lno_pct_occ[i], Iterable)
                        else lno_pct_occ[i]
                    ),
                    'norb': (
                        lno_norb[ifrag][i][s] if isinstance(lno_norb[ifrag][i], Iterable)
                        else lno_norb[ifrag][i]
                    ),
                } for i in [0, 1]
            ] for s in range(2)
        ]
    else:
        raise NotImplementedError('LNO Only Support Restricted and Unrestricted MF!')

    return orbloc, lno_param

def lnoccsd_kernel(mlno, lno_coeff, lno_frozen, uocc_loc, maskact, verbose=3):
    mf = mlno._scf
    if isinstance(mf, scf.rhf.RHF):
        mcc = lnoccsd.CCSD(mf, mo_coeff=lno_coeff, frozen=lno_frozen).set(verbose=verbose)
        mcc.conv_tol = 1e-6
        mcc.conv_tol_normt = 3e-5
        ecc_frag, t1, t2 = \
            mod_lnoccsd.rlnoccsd_solver(mcc, lno_coeff, uocc_loc, mlno.mo_occ, maskact)
    elif isinstance(mf, scf.uhf.UHF):
        mcc = ulnoccsd.UCCSD(mf, mo_coeff=lno_coeff, frozen=lno_frozen).set(verbose=verbose)
        mcc.conv_tol = 1e-6
        mcc.conv_tol_normt = 3e-5
        ecc_frag, t1, t2 = \
            mod_lnoccsd.ulnoccsd_solver(mcc, lno_coeff, uocc_loc, mlno.mo_occ, maskact)
    else:
        raise NotImplementedError('LNO Only Support Restricted and Unrestricted Orbitals!')
    return ecc_frag, t1, t2

def lnomp2_kernel(mlno, lno_coeff, lno_frozen, uocc_loc, maskact, verbose=3):
    # give canonicalized orbitals to MP2
    mf = mlno._scf
    if isinstance(mf, scf.rhf.RHF):
        mcc = lnoccsd.CCSD(mf, mo_coeff=lno_coeff, frozen=lno_frozen).set(verbose=verbose)
        emp_frag = mod_lnoccsd.rlnomp2_solver(mcc, lno_coeff, uocc_loc, mlno.mo_occ, maskact)
    elif isinstance(mf, scf.uhf.UHF):
        mcc = ulnoccsd.UCCSD(mf, mo_coeff=lno_coeff, frozen=lno_frozen).set(verbose=verbose)
        emp_frag = mod_lnoccsd.ulnomp2_solver(mcc, lno_coeff, uocc_loc, mlno.mo_occ, maskact)
    else:
        raise NotImplementedError('LNO Only Support Restricted and Unrestricted Orbitals!')
    return emp_frag

def run_lnoafqmc(options, option_file='options.bin', script=None):

    with open(option_file, 'wb') as f:
        pickle.dump(options, f)

    if 'pt2' in options['trial']:
        if script is None:
            # Frozen-virtual correlated sampling is requested by putting
            # `frozen_vir` (an explicit count) or `frozen_vir_rate` (a fraction the
            # script turns into a count) in the options.  `frozen_vir = 0` means
            # "do not freeze" and falls back to the ordinary script, since a
            # zero-width frozen branch is just the full one.  An explicit
            # `qmc_script` always wins over this detection.
            want_frozen_vir = bool(options.get('frozen_vir', 0)) \
                or 'frozen_vir_rate' in options
            if want_frozen_vir:
                script='script/run_lno_afqmc_pt2ccsd_frozen_vir.py'
            else:
                script='script/run_lno_afqmc_pt2ccsd.py'
        else:
            script=f'script/{script}'
    else:
        raise ValueError(
            f"no LNO-AFQMC script for trial {options['trial']!r}; "
            "expected a 'pt2' trial")

    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/{script}"
    print(f'AFQMC script: {script}')

    os.system(f" python {script} |tee afqmc.out")

def lnoafqmc_kernel(mlno, lno_coeff, uocc_loc, lno_frozen, t1, t2,
                    chol_cut, frag_idx, seeds, qmc_options, qmc_script=None):
    '''GPU stage of one fragment.

    Returns the fragment energy plus a (t_integral, t_sampling) timing pair so
    the pipeline efficiency can be reported.
    '''

    mf = mlno._scf

    time0 = time.perf_counter()
    integral.prep_lno_integral(mf, lno_coeff, lno_frozen, uocc_loc, t1, t2, chol_cut)
    t_int = time.perf_counter() - time0

    qmc_options["seed"] = seeds[frag_idx]

    time0 = time.perf_counter()
    run_lnoafqmc(qmc_options, script=qmc_script)
    t_qmc = time.perf_counter() - time0

    outfile = f'fragment.out{frag_idx+1}'
    os.system(f'mv afqmc.out {outfile}')
    eorb_afqmc = eorb_afqmc_err = 0.0
    with open(outfile, "r") as f:
        for line in f:
            if "Final AFQMC/pt2CCSD Orbital Energy" in line:
                eorb_afqmc = float(line.split()[-3])
                eorb_afqmc_err = float(line.split()[-1])

    return eorb_afqmc, eorb_afqmc_err, t_int, t_qmc


# --------------------------------------------------------------------------- #
#                      CPU stage (runs in a worker thread)                     #
# --------------------------------------------------------------------------- #

@contextmanager
def _capture_pyscf_log(obj):
    '''Redirect the pyscf logger of ``obj`` into a string buffer.

    ``logger.new_logger(obj)`` writes to ``obj.stdout``, which is looked up at
    call time -- swapping it lets the worker thread collect its log without
    touching the global ``sys.stdout`` shared with the main thread.
    '''
    buf = io.StringIO()
    old = getattr(obj, 'stdout', None)
    obj.stdout = buf
    try:
        yield buf
    finally:
        obj.stdout = old


def cpu_stage(mlno, mf, lo_coeff, loidx, lno_thresh, lno_pct_occ, lno_norb,
              lno_type, eris, ifrag, frag_idx, run_mp, run_cc, plot_las):
    '''Everything that runs on the CPU for one fragment.

    Pure with respect to shared state: ``make_las`` only reads ``mlno``/``eris``
    and each solver builds its own ``mcc`` object, so several invocations for
    different fragments could even run concurrently.  Output is buffered and
    returned instead of printed, so the main thread can replay it in order.
    '''
    log = io.StringIO()
    _log = partial(print, file=log)

    t_start = time.perf_counter()

    orbloc, lno_param = get_lnoparam(
        mf, lo_coeff, lno_thresh, lno_pct_occ, lno_norb, loidx, ifrag)

    with _capture_pyscf_log(mlno) as pyscf_log:
        lno_coeff, can_coeff, frozen_idx, lno_loc, can_loc, frag_msg \
             = tools.make_las(mlno, eris, orbloc, lno_type, lno_param)
        # lno_coeff, frozen_idx, lno_loc, frag_msg = \
        #     mlno.make_las(eris, orbloc, lno_type, lno_param)
    if pyscf_log.getvalue():
        log.write(pyscf_log.getvalue())
    _log(f'LNO-LAS: {frag_msg}')

    if isinstance(mf, scf.rhf.RHF):
        frozen_idx, maskact = lnoccsd.get_maskact(frozen_idx, mlno.mo_occ.size)
    elif isinstance(mf, scf.uhf.UHF):
        frozen_idx, maskact = ulnoccsd.get_maskact(
            frozen_idx, [mlno.mo_occ[0].size, mlno.mo_occ[1].size])
    else:
        raise TypeError(f'unsupported mean-field type: {type(mf)}')

    lno_split, nfrzocc, nactocc, nactvir, nfrzvir = tools.split_lno(mlno, lno_coeff, frozen_idx)
    can_split, _, _, _, _ = tools.split_lno(mlno, can_coeff, frozen_idx)

    if plot_las:
        tools.plot_density(mf, orbloc, lno_split, frag_idx+1)

    t_las = time.perf_counter() - t_start

    time0 = time.perf_counter()
    if run_mp:
        # fragment mp2 only support canonical orbital currently
        efrag_mp = lnomp2_kernel(mlno, can_coeff, frozen_idx, can_loc, maskact, verbose=0)
    else:
        efrag_mp = 0.0
    t_mp = time.perf_counter() - time0

    time0 = time.perf_counter()
    if run_cc:
        # fragment cc converges faster in canonical orbitals currently
        efrag_cc, t1, t2 = \
            lnoccsd_kernel(mlno, can_coeff, frozen_idx, can_loc, maskact, verbose=0)
        # rot amplitudes from can 2 lno
        t1, t2 = tools.can2lno_amplitude(mf, t1, t2, can_split, lno_split)
    else:
        efrag_cc, t1, t2 = 0.0, None, None
    t_cc = time.perf_counter() - time0

    return {
        'ifrag': ifrag,
        'frag_idx': frag_idx,
        'lno_coeff': lno_coeff,
        'lno_frozen': frozen_idx,
        'uocc_loc': lno_loc,
        't1': t1,
        't2': t2,
        'nactocc': nactocc,
        'nactvir': nactvir,
        'efrag_mp': efrag_mp,
        'efrag_cc': efrag_cc,
        't_las': t_las,
        't_mp': t_mp,
        't_cc': t_cc,
        't_cpu': time.perf_counter() - t_start,
        'log': log.getvalue(),
    }


class _CPUPipeline:
    '''Runs ``cpu_stage`` ``depth`` fragments ahead of the consumer.

    ``depth == 0`` executes inline in the calling thread (serial reference).
    A single worker is used so the CPU stages keep their original order and
    only ``depth + 1`` fragments' amplitudes are alive at once.
    '''

    def __init__(self, fn, nfrag, depth=1):
        self.fn = fn
        self.nfrag = nfrag
        self.depth = int(depth)
        self.futures = {}
        self.executor = (ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix='lno-cpu')
                         if self.depth > 0 else None)

    def schedule(self, i):
        if self.executor is None or i >= self.nfrag or i in self.futures:
            return
        self.futures[i] = self.executor.submit(self.fn, i)

    def fill(self, i):
        '''Keep fragments [i, i+depth] queued.'''
        for k in range(i, min(i + self.depth + 1, self.nfrag)):
            self.schedule(k)

    def get(self, i):
        if self.executor is None:
            return self.fn(i)
        self.schedule(i)
        return self.futures.pop(i).result()

    def shutdown(self, cancel=False):
        if self.executor is not None:
            try:
                self.executor.shutdown(wait=not cancel, cancel_futures=cancel)
            except TypeError:   # python < 3.9
                self.executor.shutdown(wait=not cancel)


# --------------------------------------------------------------------------- #
#                                 driver                                       #
# --------------------------------------------------------------------------- #

def run_afqmc(mf,
              lo_coeff,
              frag_list,
              frag_name,
              lno_thresh = 1e-6,
              qmc_options = {},
              chol_cut = 1e-5,
              target_qmc_err = 1e-3,
              run_frag = None,
              nfrozen = None,
              run_mp = True,
              run_cc = True,
              run_qmc = True,
              qmc_script=None, 
              plot_las = False,
              pipeline = True,
              prefetch = 1,
              ):

    print("\n ******* LNO-CALCULATION ******* \n")
    print(f"LNO THRESHOLD = {lno_thresh}")

    if run_qmc and not run_cc:
        raise ValueError("run_qmc=True requires run_cc=True: the pt2CCSD trial "
                         "needs the fragment t1/t2 amplitudes.")

    if nfrozen is None:
        print("LNO freezes at least the chemcore orbitals for each element!")
        nfrozen = elements.chemcore(mf.mol)

    tools.check_span(mf, lo_coeff, nfrozen, thresh=1e-6)

    mlno = get_lnoccsd(mf, lo_coeff, frag_list, nfrozen, lno_thresh)

    lno_thresh = mlno.lno_thresh
    lno_type = ['1h','1h']
    eris = mlno.ao2mo()

    nfrag_tot = len(frag_list)
    if run_frag is None:
        run_frag = range(nfrag_tot)
    else:
        # run_frag is indexed positionally below (run_frag[i]) and each entry
        # picks one fragment, so a repeated index would run that fragment
        # several times and double count its energy in the total.
        run_frag = list(run_frag)
        dup = sorted({i for i in run_frag if run_frag.count(i) > 1})
        if dup:
            raise ValueError(f"run_frag contains duplicate fragment indices: "
                             f"{dup} (run_frag = {run_frag})")

        # each entry is a 0-based fragment index, also used to index the seeds
        # array and to name the fragment output files, so negatives are out too
        bad = sorted({i for i in run_frag if not 0 <= i < nfrag_tot})
        if bad:
            raise ValueError(f"run_frag contains out-of-range fragment "
                             f"indices: {bad} (valid range 0 ... "
                             f"{nfrag_tot-1} for {nfrag_tot} fragments)")

    print(f"Run Fragment {run_frag}")
    frag_list = [frag_list[i] for i in run_frag]
    frag_name = [frag_name[i] for i in run_frag]
    nfrag_run = len(frag_list)

    lno_pct_occ = [None, None]
    lno_norb = [[None,None]] * nfrag_tot

    seeds = random.randint(random.PRNGKey(qmc_options["seed"]),
                           shape=(nfrag_tot,),
                           minval=0,
                           maxval=100*nfrag_tot
                           )

    qmc_options["max_error"] = target_qmc_err / np.sqrt(nfrag_tot)

    lno_size = [None]*nfrag_run
    lno_emp = np.zeros(nfrag_run, dtype='float64')
    lno_ecc  = np.zeros(nfrag_run, dtype='float64')
    lno_eqmc = np.zeros(nfrag_run, dtype='float64')
    lno_eqmc_err  = np.zeros(nfrag_run, dtype='float64')
    lno_cc_time = np.zeros(nfrag_run, dtype='float64')
    lno_qmc_time = np.zeros(nfrag_run, dtype='float64')
    lno_wait_time = np.zeros(nfrag_run, dtype='float64')

    depth = int(prefetch) if pipeline else 0
    depth = max(0, min(depth, max(nfrag_run - 1, 0)))

    print(f"\nCPU/GPU pipeline: {'ON' if depth > 0 else 'OFF'} (prefetch depth {depth})")
    if depth > 0:
        print("  the LNO-MP2/CCSD of the next fragment(s) run on the CPU while "
              "the current fragment's AFQMC occupies the GPU")

    def _cpu_task(i):
        return cpu_stage(mlno, mf, lo_coeff, frag_list[i],
                         lno_thresh, lno_pct_occ, lno_norb, lno_type, eris,
                         i, run_frag[i], run_mp, run_cc, plot_las)

    pipe = _CPUPipeline(_cpu_task, nfrag_run, depth)

    loop_time0 = time.perf_counter()

    try:
        # prime the pipeline: fragment 0 (+ prefetch) starts before any GPU work
        pipe.fill(0)

        # Loop over fragment
        for ifrag, frag_idx in enumerate(run_frag):
            print("\n")
            width = 80
            msg = f" LNO-FRAGMENT [{frag_name[ifrag]}] {ifrag+1}/({nfrag_run},{nfrag_tot}) "
            print(msg.center(width, '='))
            print(f"Fragment Num.  {ifrag+1}")
            print(f"Fragment Idx.  {frag_idx+1}")
            print(f"Fragment Name  {frag_name[ifrag]}")
            print(f"LNO THRESHOLD  {mlno.lno_thresh}")
            print(f"PySCF Threads  {lib.num_threads()}")

            # ---------------- CPU stage (possibly already finished) ----------
            time0 = time.perf_counter()
            data = pipe.get(ifrag)
            wait_time = time.perf_counter() - time0

            # refill immediately so the next CPU stage overlaps this AFQMC run
            pipe.fill(ifrag + 1)

            if data['log']:
                print(data['log'], end='')

            efrag_mp = data['efrag_mp']
            efrag_cc = data['efrag_cc']
            nactocc, nactvir = data['nactocc'], data['nactvir']
            frag_cc_time = data['t_cpu']

            print(f'LNO-MP2 Fragment Energy:  {efrag_mp:.8f}')
            print(f'LNO-CCSD Fragment Energy: {efrag_cc:.8f}')
            print(f"LNO-CPU time (s):         {frag_cc_time:.2f} "
                  f"(LAS {data['t_las']:.2f} | MP2 {data['t_mp']:.2f} | CCSD {data['t_cc']:.2f})")
            print(f"LNO-CPU wait time (s):    {wait_time:.2f} "
                  f"({100.0*(1.0 - wait_time/frag_cc_time) if frag_cc_time > 0 else 0.0:.1f}% hidden)")

            outfile = f'fragment.out{frag_idx+1}'

            # ---------------- GPU stage --------------------------------------
            time0 = time.perf_counter()
            if run_qmc:
                efrag_qmc, efrag_qmc_err, t_int, t_smp \
                    = lnoafqmc_kernel(
                        mlno, data['lno_coeff'], 
                        data['uocc_loc'], data['lno_frozen'],
                        data['t1'], data['t2'],
                        chol_cut, frag_idx, seeds, 
                        qmc_options, qmc_script)
                print(f"LNO-Integral time (s):    {t_int:.2f}")
                print(f"LNO-AFQMC time (s):       {t_smp:.2f}")
            else:
                efrag_qmc, efrag_qmc_err = 0.0, 0.0
            frag_qmc_time = time.perf_counter() - time0

            norb = np.array(nactocc)+np.array(nactvir)
            lno_size[ifrag] = norb
            lno_emp[ifrag] = efrag_mp
            lno_ecc[ifrag] = efrag_cc
            lno_cc_time[ifrag] = frag_cc_time
            lno_eqmc[ifrag] = efrag_qmc
            lno_eqmc_err[ifrag] = efrag_qmc_err
            lno_qmc_time[ifrag] = frag_qmc_time
            lno_wait_time[ifrag] = wait_time

            header = f' Fragment{run_frag[ifrag]+1} Results '
            width = 80
            with open(outfile, 'a') as f:
                f.write('\n')
                f.write(f'{header:=^{width}}\n')
                f.write("\t LNO Fragment " + frag_name[ifrag] + "\n")
                f.write('-' * width + '\n')
                f.write(f'\t LNO-Active Space electrons: {np.array(nactocc)} | orbitals: {norb} \n')
                f.write(f'\t LNO-MP2 Fragment Energy:    {efrag_mp:.8f} \n')
                f.write(f'\t LNO-CCSD Fragment Energy:   {efrag_cc:.8f} \n')
                f.write(f'\t LNO-AFQMC Fragment Energy:  {efrag_qmc:.5f} +/- {efrag_qmc_err:.5f} \n')
                f.write(f'\t LNO-CCSD Fragment Time:     {frag_cc_time:.2f} \n')
                f.write(f'\t LNO-CCSD Fragment Wait:     {wait_time:.2f} \n')
                f.write(f'\t LNO-AFQMC Fragment Time:    {frag_qmc_time:.2f} \n')
                f.write('=' * width + '\n')

            # drop the fragment payload before the next iteration
            data.clear()
            del data
            jax.clear_caches()
            gc.collect()

    except BaseException:
        pipe.shutdown(cancel=True)
        raise
    else:
        pipe.shutdown()

    loop_time = time.perf_counter() - loop_time0

    lno_size = np.array(lno_size, dtype=np.int32)
    lno_max = lno_size.max()
    # convert to list of string for print
    lno_size = list(map(lambda row: f"{row}", lno_size))
    e_mp = np.sum(lno_emp)
    e_cc = np.sum(lno_ecc)
    e_qmc = np.sum(lno_eqmc)
    e_qmc_err = np.sqrt(np.sum(lno_eqmc_err**2))
    tot_ccsd_time = np.sum(lno_cc_time)
    tot_qmc_time = np.sum(lno_qmc_time)
    tot_wait_time = np.sum(lno_wait_time)
    serial_time = tot_ccsd_time + tot_qmc_time

    with open(f'lno_result.out', 'w') as f:
        width = 120
        f.write('=' * width + '\n')
        f.write(f'{"LNO-AFQMC Results":^{width}}\n')
        f.write('=' * width + '\n')

        f.write(f'{"Num":>4s}  {"Fragment":>16s}  {"LAS SIZE":>10s}  '
                f'{"E(MP2)":>10s}  {"E(CCSD)":>10s}  '
                f'{"E(AFQMC)":>10s}  {"Error":>8s}  '
                f'{"t(CCSD)":>8s}  {"t(wait)":>8s}  {"t(AFQMC)":>8s}\n')
        f.write('-' * width + '\n')

        for n, i in enumerate(run_frag):
            f.write(f"{i+1:4d}  {frag_name[n]:>16s}  {lno_size[n]:10s}  "
                    f"{lno_emp[n]:10.8f}  {lno_ecc[n]:10.8f}  "
                    f"{lno_eqmc[n]:10.5f}  {lno_eqmc_err[n]:8.5f}  "
                    f"{lno_cc_time[n]:8.2f}  {lno_wait_time[n]:8.2f}  {lno_qmc_time[n]:8.2f}\n")

        f.write('-' * width + '\n')

        f.write(f'{"Summarize Fragments":^{width}}\n')
        f.write('-' * width + '\n')

        lno_thresh_str = "[" + ", ".join(f"{x:.2e}" for x in lno_thresh) + "]"
        f.write(f'{"LNO-Thresh":<20} {"Max LAS":>8} '
                f'{"E[MP2]":>12} {"E[CCSD]":>12} '
                f'{"E[AFQMC]":>10} {"Err[AFQMC]":>10} '
                f'{"CCSD-Time":>10} {"AFQMC-Time":>10}\n')

        f.write(f'{lno_thresh_str:<20} {lno_max:>8} '
                f'{e_mp:>12.8f} {e_cc:>12.8f} '
                f'{e_qmc:>10.5f} {e_qmc_err:>10.5f} '
                f'{tot_ccsd_time:>10.2f} {tot_qmc_time:>10.2f}\n')

        f.write('-' * width + '\n')
        f.write(f'{"Pipeline (prefetch depth " + str(depth) + ")":^{width}}\n')
        f.write('-' * width + '\n')
        f.write(f'{"Loop wall time":<28} {loop_time:>10.2f} s\n')
        f.write(f'{"Serial equivalent (CPU+GPU)":<28} {serial_time:>10.2f} s\n')
        f.write(f'{"CPU time hidden behind GPU":<28} '
                f'{tot_ccsd_time - tot_wait_time:>10.2f} s '
                f'({100.0*(tot_ccsd_time - tot_wait_time)/tot_ccsd_time if tot_ccsd_time > 0 else 0.0:.1f}%)\n')
        f.write(f'{"Speedup vs serial":<28} '
                f'{serial_time/loop_time if loop_time > 0 else 0.0:>10.2f} x\n')

        f.write('=' * width + '\n\n')

    print("\n" + "=" * 80)
    print(f"Loop wall time:              {loop_time:.2f} s")
    print(f"Serial equivalent (CPU+GPU): {serial_time:.2f} s")
    print(f"CPU time hidden behind GPU:  {tot_ccsd_time - tot_wait_time:.2f} s "
          f"of {tot_ccsd_time:.2f} s")
    print("=" * 80)

    return e_mp, e_cc, e_qmc, e_qmc_err, lno_max
