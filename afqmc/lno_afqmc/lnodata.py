import numpy as np
import re

# def lno_data(nfrag, lno_thresh, outfile='lno_result.out'):
#     lno_thresh = [lno_thresh*10, lno_thresh]

#     frg_list = range(nfrag)
#     nelec_list = np.zeros(nfrag,dtype='int32')
#     norb_list = np.zeros(nfrag,dtype='int32')
#     # eorb_pt2raw = np.zeros(nfrag,dtype='float64')
#     # eorb_pt2raw_err = np.zeros(nfrag,dtype='float64')
#     # eorb_pt2mah = np.zeros(nfrag,dtype='float64')
#     # eorb_pt2mah_err = np.zeros(nfrag,dtype='float64')
#     eorb = np.zeros(nfrag,dtype='float64')
#     eorb_err = np.zeros(nfrag,dtype='float64')
#     run_time = np.zeros(nfrag,dtype='float64')
#     for n, i in enumerate(frg_list):
#         with open(f"lnoafqmc.out{i+1}", "r") as rf:
#             for line in rf:
#                 if line.startswith('# nelec:'):
#                     nums = re.findall(r'\d+', line)
#                     nelec_list[n] = int(nums[0]) + int(nums[1])
#                 if line.startswith('# norb:'):
#                     # nums = re.findall(r'\d+', line)
#                     norb_list[n] = int(line.split()[-1]) # int(nums[0]), int(nums[1])])
#                 if "energy (covariance)" in line:
#                     eorb[n] = float(line.split()[-3])
#                     eorb_err[n] = float(line.split()[-1])
#                 # if "energy (Mahalanobis)" in line:
#                 #     eorb_pt2mah[n] = float(line.split()[-3])
#                 #     eorb_pt2mah_err[n] = float(line.split()[-1])
#                 # if "Ept (direct observation)" in line:
#                 #     eorb_pt2dir[n] = float(line.split()[-3])
#                 #     eorb_pt2dir_err[n] = float(line.split()[-1])
#                 if "total run time" in line:
#                     run_time[n] = float(line.split()[-1])

#     nelec = np.array(nelec_list, dtype=np.int32)
#     norb  = np.array(norb_list,  dtype=np.int32)

#     # eafqmc_pt2raw = np.sum(eorb_pt2raw)
#     # eafqmc_pt2raw_err = np.sqrt(sum(eorb_pt2raw_err**2))
#     # eafqmc_pt2mah = np.sum(eorb_pt2mah)
#     # eafqmc_pt2mah_err = np.sqrt(sum(eorb_pt2mah_err**2))
#     eafqmc = np.sum(eorb)
#     eafqmc_err = np.sqrt(sum(eorb_err**2))
#     tot_time = np.sum(run_time)

#     with open(outfile, 'w') as out_file:
#         print(f"{'frag':>4s}  {'eorb_afqmc':>10s}  {'error':>8s}  "
#               f"{'nelec'}  {'norb'}  {'time'}", file=out_file)
#         for n, i in enumerate(frg_list):
#             print(f'{i+1:3d} \t'#  {eorb_mp2[n]:.8f}  {eorb_ccsd[n]:.8f}'
#                 #   f'{eorb_pt2raw[n]:.6f} +/- {eorb_pt2raw_err[n]:.6f} \t \t'
#                 #   f'{eorb_pt2mah[n]:.6f} +/- {eorb_pt2mah_err[n]:.6f} \t \t'
#                   f'{eorb_pt2dir[n]:.6f} +/- {eorb_pt2dir_err[n]:.6f} \t \t'
#                   f'{nelec[n]} \t {norb[n]} \t {run_time[n]:.2f}', file=out_file)
#         print(f'# LNO Thresh: ({lno_thresh[0]:.2e},{lno_thresh[1]:.2e})',file=out_file)
#         print(f'# LNO Average Number of Electrons: ({nelec[:,0].mean():.1f},{nelec[:,1].mean():.1f})',file=out_file)
#         print(f'# LNO Average Number of Basis: ({norb[:,0].mean():.1f},{norb[:,1].mean():.1f})',file=out_file)
#         # print(f'# LNO-AFQMC/CCSD_PT(raw) Energy: {eafqmc_pt2raw:.6f} +/- {eafqmc_pt2raw_err:.6f}',file=out_file)
#         # print(f'# LNO-AFQMC/CCSD_PT(mah) Energy: {eafqmc_pt2mah:.6f} +/- {eafqmc_pt2mah_err:.6f}',file=out_file)
#         print(f'# LNO-AFQMC/CCSD_PT(dir) Energy: {eafqmc_pt2dir:.6f} +/- {eafqmc_pt2dir_err:.6f}',file=out_file)
#         print(f"# total run time: {tot_time:.2f}",file=out_file)


def ulno_data(lno_thresh, nfrag=None, frg_list=None, outfile='lno_result.out'):
    lno_thresh = [lno_thresh*10, lno_thresh]

    if nfrag is not None:
        frg_list = range(nfrag)
    # elif frag_list is not None:
    #     frg_list = frag_list
    nelec_list = []
    norb_list = []
    eorb_pt2 = np.zeros(nfrag,dtype='float64')
    eorb_pt2_err = np.zeros(nfrag,dtype='float64')
    run_time = np.zeros(nfrag,dtype='float64')
    for n, i in enumerate(frg_list):
        with open(f"lnoafqmc.out{i+1}", "r") as rf:
            for line in rf:
                if line.startswith('nelec:'):
                    nums = re.findall(r'\d+', line)
                    nelec_list.append([int(nums[0]), int(nums[1])])
                if line.startswith('norb:'):
                    nums = re.findall(r'\d+', line)
                    norb_list.append([int(nums[0]), int(nums[1])])
                if "Clean AFQMC/pt2CCSD Orbital Energy (blocking)" in line:
                    eorb_pt2[n] = float(line.split()[-3])
                    eorb_pt2_err[n] = float(line.split()[-1])
                if "total run time" in line:
                    run_time[n] = float(line.split()[-1])

    nelec = np.array(nelec_list, dtype=np.int32)
    norb  = np.array(norb_list,  dtype=np.int32)

    eafqmc_pt2 = np.sum(eorb_pt2)
    eafqmc_pt2_err = np.sqrt(sum(eorb_pt2_err**2))
    tot_time = np.sum(run_time)

    with open(outfile, 'w') as out_file:
        print(f"{'frag':>4s}  "
              f"{'eorb_afqmc':>10s} {'error':>8s}  "
              f"{'nelec'}  {'norb'}  {'time'}", 
              file=out_file)
        for n, i in enumerate(frg_list):
            print(f"{i+1:4d} "
                  f"{eorb_pt2[n]:10.6f}  {eorb_pt2_err[n]:8.6f}"
                  f"{nelec[n]}  {norb[n]}  {run_time[n]:.2f}",
                  file=out_file)
        print(f'# LNO Thresh: ({lno_thresh[0]:.2e},{lno_thresh[1]:.2e})',file=out_file)
        print(f'# LNO Average Number of Electrons: ({nelec[:,0].mean():.1f},{nelec[:,1].mean():.1f})',file=out_file)
        print(f'# LNO Average Number of Basis: ({norb[:,0].mean():.1f},{norb[:,1].mean():.1f})',file=out_file)
        print(f'# LNO-AFQMC/pt2CCSD Energy: {eafqmc_pt2:.6f} +/- {eafqmc_pt2_err:.6f}',file=out_file)
        print(f"# total run time: {tot_time:.2f}",file=out_file)