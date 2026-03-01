import torch
device = torch.device('cuda:1')
torch.set_default_device(device)
import numpy as np
import sys,os
sys.path.append(os.getcwd())
import os,sys
from utils.signal_utils import *
import model.model as module_arch
from data_loader.data_loaders import preprocess

class InfTest:
    def __init__(self,stride=10) -> None:
        self.model = module_arch.load_model('/home/hexingyan/signal_vector/model_save/1t_27_160746_87_0.4370833933353424')
        self.time_len=256
        self.stride = stride
    def inf(self,roi_sig_interp):
        # roi_sig_interp = roi_sig_interp[:int(-roi_sig_interp.shape[0]/4)]

        sig_len = roi_sig_interp.shape[-1]
        sig_o = torch.zeros(sig_len).to(device)
        sig_ofs = np.zeros(sig_len)
        rg = np.arange(0,sig_len-self.time_len+1,self.stride)
        if rg[-1] != sig_len-self.time_len:
            rg = np.append(rg,sig_len-self.time_len)
        with torch.no_grad():
            roi_sig_interp = torch.from_numpy(roi_sig_interp[None,:,:]).float().to(device)
            for i in rg:
                S_win = roi_sig_interp[:,:,i:i+self.time_len]
                out = self.model(preprocess(S_win[0],torch)[None,:])
                sig_o[i:i+self.time_len] += out[0][0]
                sig_ofs[i:i+self.time_len] += 1
                del out
            sig_o = sig_o.cpu().detach().numpy()/sig_ofs
        return sig_o

def getFit(ds_path):
    from data_parser import data_align 
    from inf_tests import test
    fs = 30
    window_length = fs*10
    
    data_align_all = data_align.GetDataWrapper(ds_path,fs,window_length)

    signal_interped  = data_align_all.get_signal_interp()
    inf = test.InfTest()
    sig_o = inf.inf(signal_interped['rois'])
    np.save(data_align_all.fit_output,sig_o)
    return sig_o


if __name__ == "__main__":
    block = True
    if len(sys.argv) != 1:
        getFit(sys.argv[1])
    else:
        # getSpectrum(r"/tank/数据集/ECG-Fitness/视频版/15/01/c920-1.avi")        
        # getSpectrum(r"/tank/数据集/UBFC-Phys_dataset/s1/vid_s1_T1.avi")
        # getSpectrum(r"/tank/数据集/UBFC-Phys_dataset/s8/vid_s8_T2.avi")
        # getSpectrum(r"/tank/数据集/ECG-Fitness/视频版/08/06/c920-1.avi")
        # getSpectrum(r"/tank/数据集/LGI_PPGI/id3/cpi/cpi_talk/cv_camera_sensor_stream_handler.avi")
        getFit("/tank/数据集/ECG-Fitness/视频版/08/06/c920-1.avi")
