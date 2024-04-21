

import sys,os
sys.path.insert(0,'.')
import utils.sp as sp
import numpy as np
from utils.signal_utils import *
from biosppy.signals import ecg
import matplotlib.pyplot as plt
import json


class GetData:
    fname : str
    rois_data : str
    rgb_data : str
    label_gth_prefix : str
    label_fit_prefix : str
    train_prefix : str
    dataset_name : str
    def __init__(self,file,fs,window_length):
        self.file = file
        if fs is not None and window_length is not None:
            self.fs = fs
            self.sig_dft = sp.DFT(2000,(0.1,4),fs)
            self.window_length = window_length
            self.filter_sensor = sp.Filter(6,[0.5,4],fs=fs)
    
    def get_signal_interp(self):
        ref_signals = self.parse_ref_signal()
        load_data = np.load(self.rois_data)[:,:,:3]
        rois_sigs = np.reshape(load_data.T,(-1,load_data.shape[0]))
        roi_sig_interp = []
        for rois_sig in rois_sigs:
            roi_sig_interp.append(self.filter_sensor.filtfilt(np.interp(self.ts_dst,self.ts[:len(rois_sig)],rois_sig)))
        roi_sig_interp = np.array(roi_sig_interp)
        ref_signals['rois']=roi_sig_interp
        return ref_signals

    def read_rppg(self):
        raw_rgb = np.load(self.rgb_data)
        raw_rgb_interp=[]
        for raw_s in raw_rgb.T:
            raw_rgb_interp.append(self.filter_sensor.filtfilt(np.interp(self.ts_dst,self.ts[:len(raw_s)],raw_s)))
        raw_rgb_interp = np.array(raw_rgb_interp)
        rppg_raw_signal = getDIS(raw_rgb_interp[:3],self.fs,raw_rgb_interp[None,3])
        return rppg_raw_signal

    def get_signal_aligned(self,reference,reference_rppg,max_idx=None,**others):
        make_align = align_sigs(reference_rppg,reference,self.fs)
        ecg_ref = make_align.get_ref()
        aligned_signal = {"reference":ecg_ref}
        for keys in others:
            aligned_signal[keys] = make_align(others[keys])
        ecg_spectrogram = self.sig_dft.get_spectrogram(ecg_ref, self.window_length, 1,lambda sig:sig*np.hanning(len(sig)))
        #ecg_max_idxs_aligend = np.argmax(ecg_spectrogram,1)
        aligned_signal['max_idx'] = np.interp(np.linspace(0,1,len(ecg_spectrogram)),np.linspace(0,1,len(max_idx)),max_idx)
        return aligned_signal
    
    def save_slices(self,reference,rois,max_idx=None,**others):
        time_len=256
        stride = int(self.fs/2)
        rg = np.arange(0,len(reference)-time_len,stride)
        rg[-1] = len(reference)-time_len

        sig_n = np.zeros_like(reference)
        aa=0
        ofs_array=np.zeros_like(reference)
        for i in rg:
            S_win = rois[:,i:i+time_len].T
            S_win = norm_std(S_win)
            D_win = reference[i:i+time_len]
            D_win = norm_std(D_win)

            ls_out = np.linalg.lstsq(S_win, D_win, rcond=None)[0]
            gi = np.reshape(ls_out,(1,-1))
            ri = S_win.T
            out = S_win@ls_out
            np.save(f"{self.train_prefix}{aa}.npy",ri)
            np.save(f"{self.label_fit_prefix}{aa}.npy",gi)
            np.save(f"{self.label_gth_prefix}{aa}.npy",np.reshape(D_win,(1,-1)))
            aa+=1
            sig_n[i:i+time_len] += out
            ofs_array[i:i+time_len] += 1
        return sig_n/ofs_array

    def plt_signals(self,fit_signal,reference,max_idx,draw_signal=False,draw_keys=["rppg"],**others):
        import matplotlib.lines as mlines
        import matplotlib.colors as colors
        import matplotlib.patches as mpatches
        import utils.opencv as ocv
        rows = 2+len(draw_keys)
        if draw_signal:
            rows +=1
        fig, ax = plt.subplots(nrows=rows, ncols=1, sharex=True, sharey=False,gridspec_kw = {'wspace':0.01, 'hspace':0.01, 'top':0.95, 'bottom':0.05,'right':0.95,'left':0.05})
        fig.canvas.manager.set_window_title(f"{self.fname}: {np.corrcoef(fit_signal,reference)[0,1]}")
        self.draw_ii = 0
        gth = mlines.Line2D([], [], color=colors.to_rgb('red'),label="GTH")
        def draw_spec(sig,name):
            spec = self.sig_dft.get_spectrogram(sig,self.window_length,1,lambda sig:sig*np.hanning(len(sig)))
            img, ext = ocv.draw_spectrum(np.array(spec),self.sig_dft.freq_resolution,min_idx=self.sig_dft.roi_idx[0],
                                    ref_idxs=[max_idx],axis_step_bpm = None)
            ax[self.draw_ii].grid(False)
            ext[1] = len(sig)
            ax[self.draw_ii].imshow(img[:,:,::-1],aspect='auto',interpolation = 'bicubic',extent=ext)
            ax[self.draw_ii].legend(handles=[mpatches.Patch(ec=(0.267, 0, 0.325),label=name,color=(0.984, 0.906, 0.141)),gth])
            self.draw_ii +=1

        ax[0].set_ylabel('Bpm', loc='top')
        draw_spec(reference,"REF")
        draw_spec(fit_signal,"FIT")
        for k in draw_keys:
            draw_spec(others[k],"DIS")
        if draw_signal:
            ax[-1].plot(norm_std(reference),label="REF")
            ax[-1].plot(norm_std(fit_signal),label="FIT")
            for k in draw_keys:
                ax[-1].plot(norm_std(others[k]),label="DIS")
            ax[-1].legend()
        ax[0].set_xticks(np.arange(0,len(reference),5*30),np.char.mod("%ds",np.arange(0,len(reference)/30,5)))
        fig.set_size_inches(13,7.5)
        return fig
    def save_figure(self,fig,block=False):
        fig.savefig(f"./fig/{self.dataset_name}/{self.fname}.png")
        if block:
            plt.show()





class GetDataEcgFitness(GetData):
    def __init__(self,file,fs=None,window_length=None):
        super().__init__(file,fs,window_length)
        fpaths = file.split(os.sep)
        self.fname = f"{fpaths[-3]}_{fpaths[-2]}"
        self.dataset_path = f"/tank/数据集/ECG-Fitness/RGB信号版/ECG_FITNESS Signal/{fpaths[-3]}/{fpaths[-2]}/"
        self.rois_data = f"./data/ecgfitness/raw/{self.fname}out.npy"
        self.rgb_data = f"./data/ecgfitness/raw/{self.fname}rgb.npy"
        self.label_gth_prefix = f"./data/ecgfitness/label_gth/{self.fname}_"
        self.label_fit_prefix = f"./data/ecgfitness/label_fit/{self.fname}_"
        self.train_prefix = f"./data/ecgfitness/train/{self.fname}_"
        self.dataset_name = "ECG-Fitness"

    def parse_ref_signal(self):
        if not (os.path.isfile(self.rois_data) and\
                os.path.isfile(self.rgb_data) and\
                os.path.isfile(self.dataset_path+"c920.csv") and\
            os.path.isfile(self.dataset_path+"viatom-raw.csv") ):
            exit(0)
        
        self.ts = np.loadtxt(self.dataset_path+"c920.csv",delimiter=",")[:,0]
        ts_ofs = np.loadtxt(self.dataset_path+"c920.csv",delimiter=",")[:,1]
        ecg_ts,ecg_raw = np.loadtxt(self.dataset_path+"viatom-raw.csv",delimiter=",",skiprows=1)[int(ts_ofs[0]):int(ts_ofs[-1]),:2].T

        self.ts_dst = np.arange(self.ts[0],self.ts[-1],1000/self.fs) #ms
        _,_,rpeaks,_,_,_,hr = ecg.ecg(signal=ecg_raw, sampling_rate=125,show=False)
        ecg_ppg = np.zeros_like(ecg_raw)
        ecg_ppg[rpeaks] = -1
        hr_est = np.mean(hr/60)
        ecg_filter = sp.Filter(6,[hr_est*0.5,hr_est*1.5],fs=125).filtfilt(ecg_ppg)
        ecg_filter = np.interp(self.ts_dst,np.linspace(ecg_ts[0],ecg_ts[-1],len(ecg_filter)),ecg_filter)
        ecg_spectrogram = self.sig_dft.get_spectrogram(ecg_filter, self.window_length, 1,lambda sig:sig*np.hanning(len(sig)))
        ecg_max_idxs = np.argmax(ecg_spectrogram,1)

        rppg_raw_signal = self.read_rppg()
        rppg_ref_sig =   near_filt(rppg_raw_signal,self.fs,self.sig_dft.toFreq(ecg_max_idxs),self.window_length)   

        return dict(reference=ecg_filter,reference_rppg=rppg_ref_sig,rppg=rppg_raw_signal,max_idx=ecg_max_idxs)

class GetDataPure(GetData):
    def __init__(self,file,fs=None,window_length=None):
        super().__init__(file,fs,window_length)
        self.fname = file.split(os.sep)[-1].split('.')[0]

        self.rois_data = f"./data/pure/raw/{self.fname}out.npy"
        self.rgb_data = f"./data/pure/raw/{self.fname}bgr.npy"
        self.label_gth_prefix = f"./data/pure/label_gth/{self.fname}_"
        self.label_fit_prefix = f"./data/pure/label_fit/{self.fname}_"
        self.train_prefix = f"./data/pure/train/{self.fname}_"
        self.dataset_name = "PURE"

    def parse_ref_signal(self):    
        if not (os.path.isfile(self.rgb_data)):
            exit(0)
        ecg_max_idxs=[]
        all_Info = json.load(open(self.file))
        self.ts = [ n['Timestamp']/1e6 for n in all_Info['/Image']]
        ts_signal = [ n['Timestamp']/1e6 for n in all_Info['/FullPackage']]
        ppg_raw = [ n['Value']['waveform'] for n in all_Info['/FullPackage']]
        bpm_raw = [ n['Value']['pulseRate'] for n in all_Info['/FullPackage']]

        hr_est = np.mean(bpm_raw)/60
        self.ts_dst = np.arange(self.ts[0],self.ts[-1],1000/self.fs) #ms
        ppg_filter = np.interp(self.ts_dst,ts_signal,ppg_raw)
        ppg_filter = sp.Filter(6,[hr_est*0.5,hr_est*1.5],fs=self.fs).filtfilt(ppg_filter)
        ecg_spectrogram = self.sig_dft.get_spectrogram(ppg_filter, self.window_length, 1,lambda sig:sig*np.hanning(len(sig)))
        ecg_max_idxs = np.argmax(ecg_spectrogram,1)
        
        
        rppg_raw_signal = self.read_rppg()
        rppg_ref_sig =   near_filt(rppg_raw_signal,self.fs,self.sig_dft.toFreq(ecg_max_idxs),self.window_length)   

        max_idx_ref = np.interp(np.linspace(0,1,len(ecg_max_idxs)),np.linspace(0,1,len(bpm_raw)),bpm_raw)

        return dict(reference=ppg_filter,reference_rppg=rppg_ref_sig,rppg=rppg_raw_signal,max_idx=max_idx_ref)
    

def GetDataWrapper(ds_path,fs=None,window_length=None) -> GetData:
    if "ECG-Fitness" in ds_path.split(os.sep):
        return GetDataEcgFitness(ds_path,fs,window_length)
    elif "PURE" in ds_path.split(os.sep):
        return GetDataPure(ds_path,fs,window_length)