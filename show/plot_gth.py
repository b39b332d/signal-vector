import matplotlib.pyplot as plt
import sys,os
sys.path.insert(0,'.')
import numpy as np
from data_parser import data_align 

def getLabel(ds_path):
    fs=30
    window_length = fs*10
    data_align_handler = data_align.GetDataWrapper(ds_path,fs,window_length)

    raw_signal = data_align_handler.parse_ref_signal(ref_only=True)
    plt.plot(raw_signal)
    # data_align_handler.plt_signals(draw_signal=True,**aligned_signal)
    plt.title(ds_path)
    plt.show()
    

if __name__ == "__main__":
    if len(sys.argv) != 1:
        #plt.ion()
        param = sys.argv[1]
        getLabel(param)
    else:
        getLabel(r"/tank/数据集/PURE/01-03.json")