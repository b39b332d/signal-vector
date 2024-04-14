
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import sys,os
sys.path.insert(0,'.')
import utils.opencv as ocv
import utils.sp as sp
import utils.fusion as fusion
from utils.signal_utils import *
import numpy as np
from sklearn.preprocessing import normalize
from scipy.optimize import minimize
from scipy.optimize import Bounds
import cv2

def getLabel(ds_path):
    fs=30
    filter_sensor = sp.Filter(6,[0.5,4],fs=fs)
    stride = fs
    window_length = fs*10
    sig_dft = sp.DFT(2000,(0.1,3),fs)
    ecg_max_idxs=[]
    def draw_spec(sig,name):
        spec = sig_dft.get_spectrogram(sig,window_length,1,lambda sig:sig*np.hanning(len(sig)))
        img = ocv.draw_spectrum(np.array(spec),sig_dft.freq_resolution,min_idx=sig_dft.roi_idx[0],ref_idxs=[ecg_max_idxs] if len(ecg_max_idxs) !=0 else None)
        cv2.imshow(name,img)
    

    fpaths = ds_path.split(os.sep)
    file_path = f"/tank/数据集/ECG-Fitness/RGB信号版/ECG_FITNESS Signal/{fpaths[-3]}/{fpaths[-2]}/"
    rois_data = f"./data/raw/{fpaths[-3]}_{fpaths[-2]}out.npy"
    gth_fname = f"./data/gth/{fpaths[-3]}_{fpaths[-2]}_"
    label_fname = f"./data/label/{fpaths[-3]}_{fpaths[-2]}_"
    train_fname = f"./data/train/{fpaths[-3]}_{fpaths[-2]}_"
    if not (os.path.isfile(rois_data) and\
        os.path.isfile(file_path+"c920-1-B.txt") and \
        os.path.isfile(file_path+"c920-1-G.txt") and \
        os.path.isfile(file_path+"c920-1-R.txt") and \
        os.path.isfile(file_path+"viatom-raw.csv") ):
        exit(0)
    
    ecg_max_idxs=[]
    ts = np.loadtxt(file_path+"c920.csv",delimiter=",")[:,0]
    ts_ofs = np.loadtxt(file_path+"c920.csv",delimiter=",")[:,1]
    ecg_ts,ecg_raw = np.loadtxt(file_path+"viatom-raw.csv",delimiter=",",skiprows=1)[int(ts_ofs[0]):int(ts_ofs[-1]),:2].T



    ts_dst = np.arange(ts[0],ts[-1],1000/fs) #ms
    from biosppy.signals import ecg
    _,_,_,_,_,_,hr = ecg.ecg(signal=ecg_raw, sampling_rate=125,show=False)
    hr_est = np.mean(hr/60)
    ecg_filter = sp.Filter(6,[hr_est-0.5,hr_est+0.5],fs=125).filtfilt(ecg_raw)
    ecg_filter = np.interp(ts_dst,np.linspace(ecg_ts[0],ecg_ts[-1],len(ecg_filter)),ecg_filter)
    ecg_spectrogram = sig_dft.get_spectrogram(ecg_filter, window_length, 1,lambda sig:sig*np.hanning(len(sig)))
    ecg_max_idxs = np.argmax(ecg_spectrogram,1)


    raw_r = np.interp(ts_dst,ts,np.loadtxt(file_path+"c920-1-R.txt"))
    raw_g = np.interp(ts_dst,ts,np.loadtxt(file_path+"c920-1-G.txt"))
    raw_b = np.interp(ts_dst,ts,np.loadtxt(file_path+"c920-1-B.txt"))
    motion_x = norm(filter_sensor.filtfilt(np.interp(ts_dst,ts,np.mean(np.loadtxt(file_path+"c920-1-X_Motion.txt"),axis=0))))
    motion_y = norm(filter_sensor.filtfilt(np.interp(ts_dst,ts,np.mean(np.loadtxt(file_path+"c920-1-Y_Motion.txt"),axis=0))))
    motion_z = norm(filter_sensor.filtfilt(np.interp(ts_dst,ts,np.loadtxt(file_path+"c920-1-Z_Motion.txt"))))
    S_raw=np.vstack([raw_r,raw_g,raw_b])
    S=np.vstack([filter_sensor.filtfilt(raw_r),
                filter_sensor.filtfilt(raw_g),
                filter_sensor.filtfilt(raw_b)])
    D=np.vstack([motion_z])

    load_data = np.load(rois_data)#[:,:,0:3]
    rois_sigs = np.reshape(load_data.T,(-1,load_data.shape[0]))
    roi_sig_interp = []
    for rois_sig in rois_sigs:
        roi_sig_interp.append(filter_sensor.filtfilt(np.interp(ts_dst,ts[:load_data.shape[0]],rois_sig)))
    roi_sig_interp = np.array(roi_sig_interp)

    
    rppg_raw_signal = getDIS(S,fs,D)
    rppg_ref_sig = rppg_raw_signal
    rppg_ref_sig =   near_filt(rppg_ref_sig,fs,sig_dft.toFreq(ecg_max_idxs),window_length)   
    
    make_align = align_sigs(rppg_ref_sig,ecg_filter,fs)
    ecg_ref = make_align.get_ref()
    rppg_ref = make_align(rppg_ref_sig)
    roi_sig_interp = make_align(roi_sig_interp)

    ref_sig = ecg_ref
    time_len=256
    stride = int(fs/2)
    rg = np.arange(0,len(ref_sig)-time_len,stride)
    rg[-1] = len(ref_sig)-time_len

    sig_n = np.zeros(ref_sig.shape)
    sig_r = np.zeros(ref_sig.shape)
    aa=0
    for i in rg:
        S_win = roi_sig_interp[:,i:i+time_len].T
        S_win -= np.min(S_win,axis=0)
        S_win = S_win/np.max(S_win,axis=0)
        D_win = ref_sig[i:i+time_len].copy()
        D_win -= np.min(D_win)
        D_win = D_win/np.max(D_win)
        
        ls_out = np.linalg.lstsq(S_win, D_win, rcond=None)[0]
        gi = np.reshape(ls_out,(1,-1))
        ri = S_win.T
        np.save(f"{train_fname}{aa}.npy",ri)
        np.save(f"{label_fname}{aa}.npy",gi)
        np.save(f"{gth_fname}{aa}.npy",np.reshape(D_win,(1,-1)))
        aa+=1

        out = S_win@ls_out
        # out = s1+s2/np.std(s2)*np.std(s1)
        # out = np.sqrt(s1*s1+s2*s2)
        sig_n[i:i+time_len] += out
        sig_r[i:i+time_len] += D_win
    print(f"{fpaths[-3]}_{fpaths[-2]}: {np.corrcoef(sig_n,sig_r)[0,1]}")
    # rppg_raw_signal = getDIS(S,fs,D)
    # draw_spec(rppg_raw_signal,"DIS")
    # draw_spec(ref_sig,"ref")
    # draw_spec(sig_n,"sig")
    # cv2.waitKey(0)
    

if __name__ == "__main__":
    if len(sys.argv) != 1:
        plt.ion()
        param = sys.argv[1]
        getLabel(param)
    else:
        #getRawSignal(r"E:\数据集压缩包备份\UBFC\UBFC2\subject9\vid.avi")
        getLabel(r"/tank/数据集/ECG-Fitness/视频版/07/01/c920-1.avi")