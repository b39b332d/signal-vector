import torch
import numpy as np

import sys,os
sys.path.append(os.getcwd())
import torch.nn as nn   
import os,sys
import utils.sp as sp
import utils.opencv as ocv
import cv2
from utils.signal_utils import *
from biosppy.signals import ecg
import matplotlib.pyplot as plt
import model.model as module_arch
from data_loader import data_align 

draw_ii=0
def getSpectrum(ds_path):
    fs = 30
    window_length = fs*10
    
    data_align_all = data_align.GetDataWrapper(ds_path,fs,window_length)

    signal_interped  = data_align_all.get_signal_interp()
    model = module_arch.load_model('model_scripted2.pt')
    
    time_len=256
    stride = 1
    sig_len = len(signal_interped['reference'])
    sig_o = torch.zeros(sig_len).cuda()
    roi_sig_interp = signal_interped['rois']
    with torch.no_grad():
    
        for i in np.arange(0,sig_len-time_len,stride):
            S_win = roi_sig_interp[None,:,i:i+time_len]
            out = model(torch.from_numpy(norm_std(S_win)).float().cuda())
            sig_o[i:i+time_len] += out[0,0]
            del out
        
        sig_o = sig_o.cpu().detach().numpy()
    sig_o /= np.hstack([np.arange(1, time_len + 1), np.ones(len(sig_o) - 2 * time_len) * time_len,
                      np.arange(1, time_len + 1)[::-1]])
    
    aligned_signal = data_align_all.get_signal_aligned(**signal_interped,fit_signal = sig_o)
    
    fig = data_align_all.plt_signals(draw_signal=True,**aligned_signal)

    
    sig_o_align = aligned_signal['fit_signal']
    ref_sig = aligned_signal["reference"]
    mae,mse,rmse = get_error(sig_o_align,ref_sig,fs,window_length,fs)
    corr = np.corrcoef(sig_o_align,ref_sig)[0,1]
    print(f"{data_align_all.fname},{mae},{mse},{rmse},{corr}")

    data_align_all.save_figure(fig,True)


if __name__ == "__main__":
    block = True
    if len(sys.argv) != 1:
        param = sys.argv[1]
        getSpectrum(param)
    else:
        # getSpectrum(r"/tank/数据集/ECG-Fitness/视频版/15/01/c920-1.avi")        
        getSpectrum(r"/tank/数据集/UBFC-Phys_dataset/s1/vid_s1_T1.avi")

        # getSpectrum(r"/tank/数据集/PURE/05-02.json")
