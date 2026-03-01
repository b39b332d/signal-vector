

import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from utils.signal_utils import *
from data_parser import data_align 
from inf_tests import test
from scipy.stats import pearsonr



def get_error1(signals,sig_ref,fs,w_len,stride):
    dft  = sp.DFT(20000,(0.5,3.0),fs)
    win = np.hanning(w_len)
    ms=[]
    mrefs=[]
    snrs = []
    dtws=[]
    accs=[]
    for i in range(0,len(sig_ref)-w_len+1,stride):
        fft_ref = np.abs(dft.rfft(win*sig_ref[i:i+w_len]))
        m_ref = dft.toFreq(np.argmax(fft_ref))
        mrefs.append(m_ref)
        delta_n = dft.toIdx(m_ref-0.2)
        delta_p = dft.toIdx(m_ref+0.2)
        delta_n = delta_n if delta_n>0 else delta_n
        delta_p = delta_p if delta_p<len(fft_ref) else len(fft_ref)-1
        ms.append([])
        snrs.append([])
        dtws.append([])
        accs.append([])
        for s in signals:
            fft = np.abs(dft.rfft(win*s[i:i+w_len]))
            br = dft.toFreq(np.argmax(fft))
            ms[-1].append(br)
            accs[-1].append(1 if np.abs(br-m_ref)<5/60 else 0)
            pow_s = np.sum(fft[delta_n:delta_p])
            pow_n = np.sum(fft)-pow_s
            snrs[-1].append( pow_s/pow_n)
    mae = np.mean(np.abs(np.array(ms).T-mrefs),axis=1)
    rmse = np.sqrt(np.mean((np.array(ms).T-mrefs)**2,axis=1))
    snrs = np.mean(snrs,axis=0)
    acc = np.mean(accs,axis=0)
    coors = np.array([pearsonr(sig_ref,s)[0] for s in signals])
    return {"mae":mae,"rmse":rmse,"snrs":snrs,"coors":coors,
            "PCC":[pearsonr(p,mrefs)[0] for p in np.array(ms).T],"acc":acc}
def showSpec(ds_path):
    fs = 30
    window_length = fs*20
    
    data_align_all = data_align.GetDataWrapper(ds_path,fs,window_length,use_cache=False)

    signal_interped  = data_align_all.get_signal_interp()
    inf_sig = []
    if not os.path.isfile(data_align_all.fit_output):
        inf_sig = test.getFit(ds_path)
    else:
        inf_sig = np.load(data_align_all.fit_output)
    # inf = test.InfTest()
    # inf_sig = inf.inf(signal_interped['rois'])
    
    aligned_signal = data_align_all.get_signal_aligned(**signal_interped,reference_rppg = inf_sig)
    f=sp.Filter(6,[0.7,2.5],fs=data_align_all.fs)
    err = get_error1 ([f.filtfilt(aligned_signal['reference_rppg']),f.filtfilt(aligned_signal["rppgs"]["POS"]),f.filtfilt(aligned_signal["rppgs"]["CHROM"])],
                      aligned_signal["reference"],fs,window_length,1)

    print(f"{data_align_all.dataset_name}:{data_align_all.fname}",
          *err["mae"],*err["rmse"],*err["snrs"],*err["coors"],*err["PCC"],*err["acc"],sep=",")

    # fig = data_align_all.plt_signals(draw_signal=False,**aligned_signal)
    # # data_align_all.save_figure(fig,False)
    # plt.show()


if __name__ == "__main__":
    block = True
    if len(sys.argv) !=1:
        showSpec(sys.argv[1])
    else:
        # showSpec(r"/tank/数据集/MMPD/subject22/p22_7.mat")
        # showSpec(r"/tank/数据集/ECG-Fitness/视频版/13/03/c920-1.avi")        
        # showSpec(r"/tank/数据集/UBFC-Phys_dataset/s1/vid_s1_T1.avi")
        # showSpec(r"/tank/数据集/UBFC-Phys_dataset/s33/vid_s33_T1.avi")
        showSpec(r"/tank/数据集/ECG-Fitness/视频版/00/02/c920-2.avi")
        # showSpec(r"/tank/数据集/UBFC/UBFC2/subject17/vid.avi")
        # showSpec("/tank/数据集/UBFC/UBFC2/subject27/vid.avi")  
        # showSpec(r"/tank/数据集/PURE/08-01/08-01.json")
        # showSpec(r"/tank/数据集/LGI_PPGI/id1/alex/alex_gym/cv_camera_sensor_stream_handler.avi")
        # showSpec(r"/tank/数据集/VitalVideo/vv250/a2b5cd11cec944229265274feec54cb8_1.mp4")
        # showSpec(r"/tank/数据集/iBVP/p21_d/p21_d_bvp.csv")
        # showSpec(r"/tank/数据集/MMPD/subject21/p21_12.mat")