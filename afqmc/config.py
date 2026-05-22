# import os
# import platform
# import socket

def _gpu_available():
    """Detect a usable NVIDIA GPU without initializing JAX."""
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return False


def setup_jax():
    import os
    import socket
    import platform

    # --- Decide backend BEFORE importing jax, so XLA_FLAGS take effect ---
    # Allow an explicit override in afqmc_config["use_gpu"]:
    #   True  -> force GPU (fail if unavailable)
    #   False -> force CPU
    #   None / "auto" / missing -> auto-detect
    # requested = config.afqmc_config.get("use_gpu", None)
    # if requested is True:
    #     use_gpu = True
    # elif requested is False:
    #     use_gpu = False
    # else:
    #     use_gpu = _gpu_available()
    # config.afqmc_config["use_gpu"] = use_gpu
    use_gpu = _gpu_available()
    
    if use_gpu:
        os.environ["XLA_FLAGS"] = (
            "--xla_gpu_enable_latency_hiding_scheduler=true "
            "--xla_gpu_enable_highest_priority_async_stream=true "
            "--xla_gpu_triton_gemm_any=true "
            "--xla_gpu_autotune_level=4 "
        )
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        # Intentionally NOT setting XLA_PYTHON_CLIENT_ALLOCATOR=platform
        # (the BFC default is much faster). Uncomment the next line only
        # if you need to share the GPU with another process.
        # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
    else:
        # CPU: only real XLA flags here.
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

    # --- Now it's safe to import and configure JAX ---
    from jax import config
    config.update("jax_enable_x64", True)
    config.update("jax_platform_name", "gpu" if use_gpu else "cpu")

    # Optional: persistent JIT cache (uncomment to enable)
    # config.update("jax_compilation_cache_dir", "./.jax_cache")
    # config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    # config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    # --- Verify and report ---
    import jax
    devices = jax.devices()
    backend_ok = (use_gpu and devices[0].platform == "gpu") or \
                 (not use_gpu and devices[0].platform == "cpu")

    uname_info = platform.uname()
    print(f"Hostname:     {socket.gethostname()}")
    print(f"System:       {uname_info.system}")
    print(f"Node:         {uname_info.node}")
    print(f"Release:      {uname_info.release}")
    print(f"Machine:      {uname_info.machine}")
    print(f"Processor:    {uname_info.processor or platform.processor()}")
    print(f"JAX backend:  {'GPU' if use_gpu else 'CPU'}")
    print(f"JAX devices:  {devices}")
    print(f"Device kind:  {devices[0].device_kind}")
    print(f"Platform:     {devices[0].platform}")
    # Memory stats (works on CUDA backends)
    # try:
    #     stats = devices[0].memory_stats()
    #     gb = 1024**3
    #     print(f"Memory:       {stats['bytes_in_use']/gb:.2f} GB in use | "
    #           f"{stats['bytes_limit']/gb:.2f} GB limit")
    # except Exception as e:
    #     print(f"memory_stats unavailable: {e}")

    if use_gpu and not backend_ok:
        print("WARNING: GPU was requested/detected but JAX did not initialize "
              "a GPU device. Check that jaxlib was installed with CUDA support "
              "(e.g. `pip install -U \"jax[cuda12]\"`).")


# def setup_jax():
#     from jax import config

#     config.update("jax_enable_x64", True)
#     if afqmc_config["use_gpu"] == True:
#         config.update("jax_platform_name", "gpu")
#         os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#         os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
#         # TODO: add gpu performance xla flags
#         hostname = socket.gethostname()
#         system_type = platform.system()
#         machine_type = platform.machine()
#         processor = platform.processor()
#         print(f"Hostname: {hostname}")
#         print(f"System Type: {system_type}")
#         print(f"Machine Type: {machine_type}")
#         print(f"Processor: {processor}")
#         uname_info = platform.uname()
#         print("Using GPU.")
#         print(f"System: {uname_info.system}")
#         print(f"Node Name: {uname_info.node}")
#         print(f"Release: {uname_info.release}")
#         print(f"Version: {uname_info.version}")
#         print(f"Machine: {uname_info.machine}")
#         print(f"Processor: {uname_info.processor}")
#     else:
#         afqmc_config["use_gpu"] = False
#         config.update("jax_platform_name", "cpu")
#         os.environ["XLA_FLAGS"] = (
#             "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
#         )


# def setup_comm():
#     if "use_mpi" not in afqmc_config:
#         if afqmc_config["use_gpu"] == True:
#             afqmc_config["use_mpi"] = False
#         else:
#             afqmc_config["use_mpi"] = True
#     if afqmc_config["use_mpi"] == True:
#         from mpi4py import MPI
#     else:
#         MPI = not_MPI()
#     rank = MPI.COMM_WORLD.Get_rank()
#     if rank == 0 and afqmc_config["use_gpu"] == False:
#         hostname = socket.gethostname()
#         system_type = platform.system()
#         machine_type = platform.machine()
#         processor = platform.processor()
#         print(f"# Hostname: {hostname}")
#         print(f"# System Type: {system_type}")
#         print(f"# Machine Type: {machine_type}")
#         print(f"# Processor: {processor}")
#     return MPI
