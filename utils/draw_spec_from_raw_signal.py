import matplotlib.pyplot as plt
import sys,os
sys.path.insert(0,r'C:\Users\b39b3\Documents\SRC\Python\RobustRPPGExtractor')
import utils.opencv as ocv
import utils.sp as sp
import numpy as np
import cv2

all_input=np.loadtxt(r"C:/Users/b39b3/source/repos/respiheartembed/out/build/x86-debug/msvc/out")
ref_raw=None
sig=None
if len(all_input[0]) != 1:
    sig = all_input[:,0]
    ref_raw=[]
    for i in range(1, len(all_input[0])):
        if all_input[:,i][0] > 50:
            ref_raw.append(all_input[:,i]/60)
        else:
            ref_raw.append(all_input[:,i])
else:
    sig = all_input

# filt_hr = sp.Filter(6,(0.9, float('nan')),50)
# sig = filt_hr.filt(sig)

fft_hr = sp.DFT(2048,(0.5,3),50)
spec = fft_hr.get_spectrogram(sig,2048,1)
ref=None
if ref_raw is not None:
    ref = []
    for ref_i in ref_raw:
        ref.append(np.interp(np.linspace(0,1,len(spec)),np.linspace(0,1,len(ref_i)),ref_i))

imag = ocv.draw_spectrum(spec,50/2048,min_freq=0.5,ref_freqs=ref)
cv2.imshow("",imag)
cv2.waitKey(0)