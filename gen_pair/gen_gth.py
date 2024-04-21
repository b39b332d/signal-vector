import matplotlib.pyplot as plt
import sys,os
sys.path.insert(0,'.')
import numpy as np
from data_loader import data_align 

def getLabel(ds_path):
    fs=30
    window_length = fs*10
    data_align_handler = data_align.GetDataWrapper(ds_path,fs,window_length)

    aligned_signal = data_align_handler.get_signal_aligned(**data_align_handler.get_signal_interp())
    fit_signal = data_align_handler.save_slices(**aligned_signal)
    print(f"{data_align_handler.fname}: {np.corrcoef(aligned_signal['reference'],fit_signal)[0,1]}")
    data_align_handler.plt_signals(fit_signal,draw_signal=True,**aligned_signal)
    plt.show()
    

if __name__ == "__main__":
    if len(sys.argv) != 1:
        plt.ion()
        param = sys.argv[1]
        getLabel(param)
    else:
        getLabel(r"/tank/数据集/UBFC-Phys_dataset/s1/vid_s1_T1.avi")
        # getLabel(r"/tank/数据集/PURE/01-01.json")