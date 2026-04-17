import numpy as np

#### results analysis ####
def frg2result(lno_thresh,nfrag,e_mf,e_mp2):
    with open('results.out', 'w') as out_file:
        print('# frag  mp2_orb_corr  ccsd_orb_corr' \
              '  afqmc_orb_corr    err    norb  nelec  time',file=out_file)
        for ifrag in range(nfrag):
            with open(f"frg_{ifrag+1}.out","r") as read_file:
                for line in read_file:
                    if "lno-mp2 orb_corr" in line:
                        mp2_eorb = line.split()[-1]
                    if "lno-ccsd orb_corr" in line:
                        ccsd_eorb = line.split()[-1]
                    if "lno-afqmc/cisd orbital energy" in line:
                        afqmc_eorb = line.split()[-3]
                        afqmc_eorb_err = line.split()[-1]
                    if "number of active electrons" in line:
                        nelec = line.split()[-1]
                    if "number of active orbitals" in line:
                        norb = line.split()[-1]
                    if "total run time" in line:
                        tot_time = line.split()[-1]
                print(f'{ifrag+1:3d}  '
                      f'{mp2_eorb}  {ccsd_eorb}  '
                      f'{afqmc_eorb}  {afqmc_eorb}  '
                      f'{norb}  {nelec}  {tot_time}', file=out_file)

    data = []
    with open('results.out', 'r') as read_file:
        for line in read_file:
            if not line.startswith("#"):
                data = np.hstack((data,line.split()))
                data[data == 'None'] = '0.000000' 

    data = np.array(data.reshape(nfrag,8))
    mp2_eorb = np.array(data[:,1],dtype='float32')
    ccsd_eorb = np.array(data[:,2],dtype='float32')
    afqmc_eorb = np.array(data[:,3],dtype='float32')
    afqmc_eorb_err = np.array(data[:,4],dtype='float32')
    norb = np.array(data[:,5],dtype='int32')
    nelec = np.array(data[:,6],dtype='int32')
    tot_time = np.array(data[:,7],dtype='float32')

    mp2_corr = sum(mp2_eorb)
    mp2_cr = e_mp2 - mp2_corr
    ccsd_corr = sum(ccsd_eorb)
    afqmc_corr = sum(afqmc_eorb)
    afqmc_corr_err = np.sqrt(sum(afqmc_eorb_err**2))
    nelec_avg = np.mean(nelec)
    norb_avg = np.mean(norb)
    nelec_max = max(nelec)
    norb_max = max(norb)
    tot_time = sum(tot_time)

    mp2_corr = f'{mp2_corr:.8f}'
    ccsd_corr = f'{ccsd_corr:.8f}'
    afqmc_corr = f'{afqmc_corr:.6f}'
    afqmc_corr_err = f'{afqmc_corr_err:.6f}'

    with open('results.out', 'a') as out_file:
        out_file.write(f'# final results \n')
        out_file.write(f'# mean-field energy: {e_mf:.8f}\n')
        out_file.write(f'# lno-thresh {lno_thresh}\n')
        out_file.write(f'# lno-mp2_corr: {mp2_corr}\n')
        out_file.write(f'# lno-ccsd_corr: {ccsd_corr}\n')
        out_file.write(f'# lno-afqmc_corr: {afqmc_corr} +/- {afqmc_corr_err}\n')
        out_file.write(f'# mp2_correction: {mp2_cr:.8f}\n')
        out_file.write(f'# number of electrons: average {nelec_avg:.2f} maxium {nelec_max}\n')
        out_file.write(f'# number of orbitals: average {norb_avg:.2f} maxium {norb_max}\n')
        out_file.write(f'# total run time: {tot_time:.2f}\n')
    
    return None

def frg2result_dbg(lno_thresh,nfrag,e_mf,e_mp2):
    with open('results.out', 'w') as out_file:
        print('# frag  mp2_orb_corr  ccsd_orb_corr' \
              '  olp_ratio  err  qmc_hf_orb_cr  err'
              '  qmc_cc_orb_cr0  err  qmc_cc_orb_cr1  err' \
              '  qmc_cc_orb_cr2  err  qmc_cc_orb_cr  err'
              '  qmc_orb_en  err  nelec  norb  time',file=out_file)
        for ifrag in range(nfrag):
            # ccsd_orb_en = None
            with open(f"frg_{ifrag+1}.out","r") as read_file:
                for line in read_file:
                    if "lno-mp2 orb_corr:" in line:
                        mp2_orb_en = line.split()[2]
                    if "lno-ccsd orb_corr:" in line:
                        ccsd_orb_en = line.split()[2]
                    if "lno-afqmc/cc olp_r:" in line:
                        olp_r = line.split()[2]
                        olp_r_err = line.split()[4]
                    if "lno-afqmc/cc hf_orb_cr:" in line:
                        hf_orb_cr = line.split()[2]
                        hf_orb_cr_err = line.split()[4]
                    if "lno-afqmc/cc cc_orb_cr0:" in line:
                        cc_orb_cr0 = line.split()[2]
                        cc_orb_cr0_err = line.split()[4]
                    if "lno-afqmc/cc cc_orb_cr1:" in line:
                        cc_orb_cr1 = line.split()[2]
                        cc_orb_cr1_err = line.split()[4]
                    if "lno-afqmc/cc cc_orb_cr2:" in line:
                        cc_orb_cr2 = line.split()[2]
                        cc_orb_cr2_err = line.split()[4]
                    if "lno-afqmc/cc cc_orb_cr:" in line:
                        cc_orb_cr = line.split()[2]
                        cc_orb_cr_err = line.split()[4]
                    if "lno-afqmc/cc orb_en:" in line:
                        qmc_orb_en = line.split()[2]
                        qmc_orb_en_err = line.split()[4]
                    if "number of active electrons:" in line:
                        nelec = line.split()[4]
                    if "number of active orbitals:" in line:
                        norb = line.split()[4]
                    if "total run time" in line:
                        tot_time = line.split()[3]
                if ccsd_orb_en is None:
                    ccsd_orb_en = '  None  '
                print(f'{ifrag+1:3d}  '
                      f'{mp2_orb_en}  {ccsd_orb_en}  '
                      f'{olp_r}  {olp_r_err}  {hf_orb_cr}  {hf_orb_cr_err}  '
                      f'{cc_orb_cr0}  {cc_orb_cr0_err}  {cc_orb_cr1}  {cc_orb_cr1_err}  '
                      f'{cc_orb_cr2}  {cc_orb_cr2_err}  {cc_orb_cr}  {cc_orb_cr_err}  '
                      f'{qmc_orb_en}  {qmc_orb_en_err}  {nelec}  {norb}  {tot_time}  ',
                      file=out_file)

    data = []
    with open('results.out', 'r') as read_file:
        for line in read_file:
            if not line.startswith("#"):
                data = np.hstack((data,line.split()))
                data[data == 'None'] = '0.000000' 

    data = np.array(data.reshape(nfrag,20))
    mp2_orb_en = np.array(data[:,1],dtype='float32')
    ccsd_orb_en = np.array(data[:,2],dtype='float32')
    olp_r = np.array(data[:,3],dtype='float32')
    olp_r_err = np.array(data[:,4],dtype='float32')
    hf_orb_cr = np.array(data[:,5],dtype='float32')
    hf_orb_cr_err = np.array(data[:,6],dtype='float32')
    cc_orb_cr0 = np.array(data[:,7],dtype='float32')
    cc_orb_cr0_err = np.array(data[:,8],dtype='float32')
    cc_orb_cr1 = np.array(data[:,9],dtype='float32')
    cc_orb_cr1_err = np.array(data[:,10],dtype='float32')
    cc_orb_cr2 = np.array(data[:,11],dtype='float32')
    cc_orb_cr2_err = np.array(data[:,12],dtype='float32')
    cc_orb_cr = np.array(data[:,13],dtype='float32')
    cc_orb_cr_err = np.array(data[:,14],dtype='float32')
    qmc_orb_en = np.array(data[:,15],dtype='float32')
    qmc_orb_en_err = np.array(data[:,16],dtype='float32')
    nelec = np.array(data[:,17],dtype='int32')
    norb = np.array(data[:,18],dtype='int32')
    tot_time = np.array(data[:,19],dtype='float32')

    mp2_corr = sum(mp2_orb_en)
    mp2_cr = e_mp2 - mp2_corr
    ccsd_corr = sum(ccsd_orb_en)
    olp_r = np.mean(olp_r)
    olp_r_err = np.sqrt(sum(olp_r_err**2))/len(olp_r_err)
    qmc_hf_cr = sum(hf_orb_cr)
    qmc_hf_cr_err = np.sqrt(sum(hf_orb_cr_err**2))
    qmc_cc_cr0 = sum(cc_orb_cr0)
    qmc_cc_cr0_err = np.sqrt(sum(cc_orb_cr0_err**2))
    qmc_cc_cr1 = sum(cc_orb_cr1)
    qmc_cc_cr1_err = np.sqrt(sum(cc_orb_cr1_err**2))
    qmc_cc_cr2 = sum(cc_orb_cr2)
    qmc_cc_cr2_err = np.sqrt(sum(cc_orb_cr2_err**2))
    qmc_cc_cr = sum(cc_orb_cr)
    qmc_cc_cr_err = np.sqrt(sum(cc_orb_cr_err**2))
    qmc_corr = sum(qmc_orb_en)
    qmc_corr_err = np.sqrt(sum(qmc_orb_en_err**2))
    nelec_avg = np.mean(nelec)
    norb_avg = np.mean(norb)
    nelec_max = max(nelec)
    norb_max = max(norb)
    tot_time = sum(tot_time)

    mp2_corr = f'{mp2_corr:.8f}'
    ccsd_corr = f'{ccsd_corr:.8f}'
    olp_r = f'{olp_r:.6f}'
    olp_r_err = f'{olp_r_err:.6f}'
    qmc_hf_cr = f'{qmc_hf_cr:.6f}'
    qmc_hf_cr_err = f'{qmc_hf_cr_err:.6f}'
    qmc_cc_cr0 = f'{qmc_cc_cr0:.6f}'
    qmc_cc_cr0_err = f'{qmc_cc_cr0_err:.6f}'
    qmc_cc_cr1 = f'{qmc_cc_cr1:.6f}'
    qmc_cc_cr1_err = f'{qmc_cc_cr1_err:.6f}'
    qmc_cc_cr2 = f'{qmc_cc_cr2:.6f}'
    qmc_cc_cr2_err = f'{qmc_cc_cr2_err:.6f}'
    qmc_cc_cr = f'{qmc_cc_cr:.6f}'
    qmc_cc_cr_err = f'{qmc_cc_cr_err:.6f}'
    qmc_corr = f'{qmc_corr:.6f}'
    qmc_corr_err = f'{qmc_corr_err:.6f}'

    with open('results.out', 'a') as out_file:
        out_file.write(f'# final results \n')
        out_file.write(f'# mean-field energy: {e_mf:.8f}\n')
        out_file.write(f'# lno-thresh {lno_thresh}\n')
        out_file.write(f'# lno-mp2_corr: {mp2_corr}\n')
        out_file.write(f'# lno-ccsd_corr: {ccsd_corr}\n')
        out_file.write(f'# lno-afqmc/cc olp_r: {olp_r} +/- {olp_r_err}\n')
        out_file.write(f'# lno-afqmc/cc hf_cr: {qmc_hf_cr} +/- {qmc_hf_cr_err}\n')
        out_file.write(f'# lno-afqmc/cc cc_cr0: {qmc_cc_cr0} +/- {qmc_cc_cr0_err}\n')
        out_file.write(f'# lno-afqmc/cc cc_cr1: {qmc_cc_cr1} +/- {qmc_cc_cr1_err}\n')
        out_file.write(f'# lno-afqmc/cc cc_cr2: {qmc_cc_cr2} +/- {qmc_cc_cr2_err}\n')
        out_file.write(f'# lno-afqmc/cc cc_cr: {qmc_cc_cr} +/- {qmc_cc_cr_err}\n')
        out_file.write(f'# lno-afqmc/cc corr: {qmc_corr} +/- {qmc_corr_err}\n')
        out_file.write(f'# mp2_correction: {mp2_cr:.8f}\n')
        out_file.write(f'# number of electrons: average {nelec_avg:.2f} maxium {nelec_max}\n')
        out_file.write(f'# number of orbitals: average {norb_avg:.2f} maxium {norb_max}\n')
        out_file.write(f'# total run time: {tot_time:.2f}\n')
    
    return None

def sum_results(n_results):
    with open('sum_results.out', 'w') as out_file:
        print("# lno-thresh(occ,vir) \t"
              "  mp2_corr \t ccsd_corr \t"
              "  afqmc_corr \t   err   \t"
              "  mp2_cr \t avg(nelec,norb) \t"
              "  max(nelec,norb) \t"
              "  run_time",file=out_file)
        for i in range(n_results):
            with open(f"results.out{i+1}","r") as read_file:
                for line in read_file:
                    if "lno-thresh" in line:
                        thresh_occ = line.split()[-2]
                        thresh_vir = line.split()[-1]
                        thresh_occ = float(thresh_occ.strip('()[],'))
                        thresh_vir = float(thresh_vir.strip('()[],'))
                    if "lno-mp2_corr:" in line:
                        mp2_corr = line.split()[-1]
                    if "lno-ccsd_corr:" in line:
                        ccsd_corr = line.split()[-1]
                    if "lno-afqmc_corr:" in line:
                        afqmc_corr = line.split()[-3]
                        afqmc_corr_err = line.split()[-1]
                    if "mp2_correction:" in line:
                        mp2_cr = line.split()[-1]
                    if "electrons:" in line:
                        nelec_avg = line.split()[-3]
                        nelec_max = line.split()[-1]
                    if "orbitals:" in line:
                        norb_avg = line.split()[-3]
                        norb_max = line.split()[-1]
                    if "time:" in line:
                        run_time = line.split()[-1]
            print(f" ({thresh_occ:.2e},{thresh_vir:.2e}) \t"
                  f" {mp2_corr} \t {ccsd_corr} \t"
                  f" {afqmc_corr} \t {afqmc_corr_err} \t"
                  f" {mp2_cr} \t ({nelec_avg},{norb_avg}) \t"
                  f" \t ({nelec_max},{norb_max}) \t {run_time}",file=out_file)
    return None

def sum_results_dbg(n_results):
    with open('sum_results.out', 'w') as out_file:
        print("# thresh(occ,vir) "
              "  mp2_corr  ccsd_corr  olp_ratio  err"
              "  qmc_hf_cr  err  qmc_cc_cr0   err"
              "  qmc_cc_cr1   err  qmc_cc_cr2   err"
              "  qmc_cc_cr   err  qmc_corr   err  mp2_correction"
              "  nelec_avg   nelec_max  norb_avg   norb_max  "
              "  run_time",file=out_file)
        for i in range(n_results):
            with open(f"results.out{i+1}","r") as read_file:
                for line in read_file:
                    if "lno-thresh" in line:
                        thresh_occ = line.split()[-2]
                        thresh_vir = line.split()[-1]
                        thresh_occ = float(thresh_occ.strip('()[],'))
                        thresh_vir = float(thresh_vir.strip('()[],'))
                    if "lno-mp2_corr:" in line:
                        mp2_corr = line.split()[-1]
                    if "lno-ccsd_corr:" in line:
                        ccsd_corr = line.split()[-1]
                    if "lno-afqmc/cc olp_r:" in line:
                        olp_r = line.split()[-3]
                        olp_r_err = line.split()[-1]
                    if "lno-afqmc/cc hf_cr:" in line:
                        qmc_hf_cr = line.split()[-3]
                        qmc_hf_cr_err = line.split()[-1]
                    if "lno-afqmc/cc cc_cr0:" in line:
                        qmc_cc_cr0 = line.split()[-3]
                        qmc_cc_cr0_err = line.split()[-1]
                    if "lno-afqmc/cc cc_cr1:" in line:
                        qmc_cc_cr1 = line.split()[-3]
                        qmc_cc_cr1_err = line.split()[-1]
                    if "lno-afqmc/cc cc_cr2:" in line:
                        qmc_cc_cr2 = line.split()[-3]
                        qmc_cc_cr2_err = line.split()[-1]
                    if "lno-afqmc/cc cc_cr:" in line:
                        qmc_cc_cr = line.split()[-3]
                        qmc_cc_cr_err = line.split()[-1]
                    if "lno-afqmc/cc corr:" in line:
                        qmc_corr = line.split()[-3]
                        qmc_corr_err = line.split()[-1]
                    if "mp2_correction:" in line:
                        mp2_cr = line.split()[-1]
                    if "electrons:" in line:
                        nelec_avg = line.split()[-3]
                        nelec_max = line.split()[-1]
                    if "orbitals:" in line:
                        norb_avg = line.split()[-3]
                        norb_max = line.split()[-1]
                    if "time:" in line:
                        run_time = line.split()[-1]
            print(f"  ({thresh_occ:.2e},{thresh_vir:.2e})"
                  f"  {mp2_corr}  {ccsd_corr}  {olp_r}  {olp_r_err}"
                  f"  {qmc_hf_cr}  {qmc_hf_cr_err}"
                  f"  {qmc_cc_cr0}  {qmc_cc_cr0_err}"
                  f"  {qmc_cc_cr1}  {qmc_cc_cr1_err}"
                  f"  {qmc_cc_cr2}  {qmc_cc_cr2_err}"
                  f"  {qmc_cc_cr}  {qmc_cc_cr_err}"
                  f"  {qmc_corr} \t {qmc_corr_err} \t"
                  f"  {mp2_cr}  {nelec_avg} \t {nelec_max} \t"
                  f"  {norb_avg}  \t {norb_max} \t {run_time}",
                  file=out_file)
    return None

def lno_data(data):
      new_data = []
      lines = data.splitlines()
      for line in lines:
            columns = line.split()
            if len(columns)>1:
                  if not line.startswith("#"): 
                        new_data.append(columns)

      new_data = np.array(new_data)

      lno_thresh = []
      for i in range(new_data.shape[0]):
            thresh_vir = new_data[:,0][i].split(sep=',')[1]
            thresh_vir = float(thresh_vir.strip('(),'))
            lno_thresh.append(thresh_vir)

      lno_data = np.array(new_data[:,1:],dtype="float32")

      lno_thresh = np.array(lno_thresh,dtype="float32")
      lno_mp2_corr = lno_data[:,0]
      lno_cc_corr = lno_data[:,1]
    #   lno_qmc_hf_corr = lno_data[:,2]
    #   lno_qmc_hf_err = lno_data[:,3]
      lno_qmc_cc_corr = lno_data[:,4]
      lno_qmc_cc_err = lno_data[:,5]
      mp2_cr = lno_data[:,6]

      return lno_thresh,lno_mp2_corr,lno_cc_corr,lno_qmc_cc_corr,lno_qmc_cc_err,mp2_cr

def lno_data_dbg(data):
      new_data = []
      lines = data.splitlines()
      for line in lines:
            columns = line.split()
            if len(columns)>1:
                  if not line.startswith("#"): 
                        new_data.append(columns)

      new_data = np.array(new_data)

      lno_thresh = []
      for i in range(new_data.shape[0]):
            thresh_vir = new_data[:,0][i].split(sep=',')[1]
            thresh_vir = float(thresh_vir.strip('(),'))
            lno_thresh.append(thresh_vir)

      lno_data = np.array(new_data[:,1:],dtype="float32")

      lno_thresh = np.array(lno_thresh,dtype="float32")
      mp2_corr = lno_data[:,0]
      ccsd_corr = lno_data[:,1]
      olp_r = lno_data[:,2]
      olp_r_err = lno_data[:,3]
      qmc_hf_cr = lno_data[:,4]
      qmc_hf_cr_err = lno_data[:,5]
      qmc_cc_cr0 = lno_data[:,6]
      qmc_cc_cr0_err = lno_data[:,7]
      qmc_cc_cr1 = lno_data[:,8]
      qmc_cc_cr1_err = lno_data[:,9]
      qmc_cc_cr2 = lno_data[:,10]
      qmc_cc_cr2_err = lno_data[:,11]
      qmc_cc_cr = lno_data[:,12]
      qmc_cc_cr_err = lno_data[:,13]
      qmc_corr = lno_data[:,14]
      qmc_corr_err = lno_data[:,15]
      mp2_cr = lno_data[:,16]
      nelec_avg = lno_data[:,17]
      nelec_max = lno_data[:,18]
      norb_avg = lno_data[:,19]
      norb_max = lno_data[:,20]
      time = lno_data[:,21]

      return (lno_thresh,mp2_corr,ccsd_corr,olp_r,olp_r_err,
              qmc_hf_cr,qmc_hf_cr_err,qmc_cc_cr0,qmc_cc_cr0_err,
              qmc_cc_cr1,qmc_cc_cr1_err,qmc_cc_cr2,qmc_cc_cr2_err,
              qmc_cc_cr,qmc_cc_cr_err,qmc_corr,qmc_corr_err,mp2_cr,
              nelec_avg,nelec_max,norb_avg,norb_max,time)

def each_frg_results(nresults):
    for i in range(nresults):
        with open(f"results.out{i+1}","r") as read_file:
            for line in read_file:
                if "lno-thresh" in line:
                    thresh_occ = line.split()[-2]
                    thresh_vir = line.split()[-1]
                    thresh_occ = float(thresh_occ.strip('()[],'))
                    thresh_vir = float(thresh_vir.strip('()[],'))
        with open(f"results.out{i+1}","r") as read_file:
            for line in read_file:
                if not line.lstrip().startswith("#"):
                    # print(line.split())
                    (ifrag,mp2_corr,ccsd_corr,olp_r,olp_r_err,
                    qmc_hf_cr,qmc_hf_cr_err,qmc_cc_cr0,qmc_cc_cr0_err,
                    qmc_cc_cr1,qmc_cc_cr1_err,qmc_cc_cr2,qmc_cc_cr2_err,
                    qmc_cc_cr,qmc_cc_cr_err,qmc_orb_en,qmc_orb_en_err,
                    nelec,norb,time) = line.split()
                    if i == 0:
                        with open(f'frg{ifrag}_results.out', 'w') as out_file:
                            print("# thresh(occ,vir) "
                                "  mp2_corr  ccsd_corr  olp_ratio  err"
                                "  qmc_hf_cr  err  qmc_cc_cr0   err"
                                "  qmc_cc_cr1   err  qmc_cc_cr2   err"
                                "  qmc_cc_cr   err  qmc_orb_en   err"
                                "  nelec   norb  time", file=out_file)
                            print(f"  ({thresh_occ:.2e},{thresh_vir:.2e})"
                                f"  {mp2_corr}  {ccsd_corr}  {olp_r}  {olp_r_err}"
                                f"  {qmc_hf_cr}  {qmc_hf_cr_err}"
                                f"  {qmc_cc_cr0}  {qmc_cc_cr0_err}"
                                f"  {qmc_cc_cr1}  {qmc_cc_cr1_err}"
                                f"  {qmc_cc_cr2}  {qmc_cc_cr2_err}"
                                f"  {qmc_cc_cr}  {qmc_cc_cr_err}"
                                f"  {qmc_orb_en}  {qmc_orb_en_err}"
                                f"  {nelec}  {norb}  {time}", file=out_file)
                    else: 
                        with open(f'frg{ifrag}_results.out', 'a') as out_file:
                                print(f"  ({thresh_occ:.2e},{thresh_vir:.2e})"
                                f"  {mp2_corr}  {ccsd_corr}  {olp_r}  {olp_r_err}"
                                f"  {qmc_hf_cr}  {qmc_hf_cr_err}"
                                f"  {qmc_cc_cr0}  {qmc_cc_cr0_err}"
                                f"  {qmc_cc_cr1}  {qmc_cc_cr1_err}"
                                f"  {qmc_cc_cr2}  {qmc_cc_cr2_err}"
                                f"  {qmc_cc_cr}  {qmc_cc_cr_err}"
                                f"  {qmc_orb_en}  {qmc_orb_en_err}"
                                f"  {nelec}  {norb}  {time}", file=out_file)
    return None