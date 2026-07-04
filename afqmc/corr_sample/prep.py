from afqmc import prep, cholesky
from afqmc.corr_sample import sampling as csp

print_start = prep.print_start
init_hf_prop_data = prep.init_hf_prop_data

def get_sampler(options, nchol):

    if  'pt2ccsd' in options['trial']:
        sampler = csp.sampler_pt2(
            n_prop_steps=options["n_prop_steps"],
            n_blocks=options["n_blocks"],
            n_walkers=options["n_walkers"],
            n_chol=nchol)           
    else:
        sampler = csp.sampler(
            n_prop_steps=options["n_prop_steps"],
            n_blocks=options["n_blocks"],
            n_walkers=options["n_walkers"],
            n_chol=nchol)  
    
    return sampler

def init_afqmc(options=None, option_file="options.bin",
               amp_file1="amplitudes1.npz", chol_file1="FCIDUMP_chol1",
               amp_file2="amplitudes2.npz", chol_file2="FCIDUMP_chol2",
               ):
    
    options = prep.get_qmc_options(options, option_file)

    print("\nLoad system")

    h01, h11, chol1, ms1, nelec_sp1, norb1, spin_type1 = prep.load_chol(chol_file1)
    h02, h12, chol2, ms2, nelec_sp2, norb2, spin_type2 = prep.load_chol(chol_file2)

    _, ham_data1, nchol1 = prep.get_hamiltonian(h01, h11, chol1, norb1, spin_type1)
    _, ham_data2, nchol2 = prep.get_hamiltonian(h02, h12, chol2, norb2, spin_type2)
    
    assert nchol1 == nchol2

    nchol_chunk1 = cholesky.chunk_chol(
        chol1, options["nchol_chunk"], options["max_memory"]/options["n_walkers"])
    
    nchol_chunk2 = cholesky.chunk_chol(
        chol2, options["nchol_chunk"], options["max_memory"]/options["n_walkers"])
    
    trial1, wave_data1 = prep.get_wavefunction(
        spin_type1, norb1, nelec_sp1, nchol_chunk1, options, amp_file1)
    
    trial2, wave_data2 = prep.get_wavefunction(
        spin_type2, norb2, nelec_sp2, nchol_chunk2, options, amp_file2)


    prop = prep.get_propagator(options)
    sampler = get_sampler(options, nchol1)

    print(f"\n{'':<22}  {'QMC System 1':>14}  {'QMC System 2':>14}")
    print(f"{'Number of electrons:':<22}  {str(nelec_sp1):>14}  {str(nelec_sp2):>14}")
    print(f"{'Spin Multiplicity:':<22}  {str(ms1):>14}  {str(ms2):>14}")
    print(f"{'Number of orbitals:':<22}  {str(norb1):>14}  {str(norb2):>14}")
    print(f"{'Number of Chol:':<22}  {str(nchol1):>14}  {str(nchol2):>14}")

    return (prop, sampler, options, 
            trial1, ham_data1, wave_data1, 
            trial2, ham_data2, wave_data2)