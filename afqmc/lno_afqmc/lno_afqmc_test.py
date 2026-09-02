"""Deprecated alias for `lno_afqmc`.

The CPU/GPU pipelined driver that used to live here IS `lno_afqmc` now: the
fragment local active space, MP2 and CCSD run on the CPU while the GPU keeps
sampling the previous fragment.  This module is kept so existing scripts that do

    from afqmc.lno_afqmc import lno_afqmc_test

keep working; it re-exports the same objects, so `lno_afqmc_test.run_afqmc is
lno_afqmc.run_afqmc`.  New code should import `lno_afqmc` directly.
"""

from afqmc.lno_afqmc.lno_afqmc import (  # noqa: F401
    get_lnoccsd,
    get_lnoparam,
    lnoccsd_kernel,
    lnomp2_kernel,
    run_lnoafqmc,
    lnoafqmc_kernel,
    cpu_stage,
    run_afqmc,
    _capture_pyscf_log,
    _CPUPipeline,
)
