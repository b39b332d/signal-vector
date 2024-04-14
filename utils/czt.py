import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cmath
from scipy.fft import fft, ifft

def next_fast_len(len):
    if len < 16:
        return 16
    elif len < 32:
        return 32
    elif len < 64:
        return 64
    elif len < 128:
        return 128
    elif len < 256:
        return 256
    elif len < 512:
        return 512
    elif len < 1024:
        return 1024
    elif len < 2048:
        return 2048
    elif len < 4096:
        return 4096
    else:
        return -1
filename = "../realsense/out/1660036902_rppg_green.npy"
rppg_signal = np.load(filename)

f1=0
f2=2
fs=50
n=20
m=10
x=[1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1,6]
czt = signal.ZoomFFT(n,[f1,f2],m,fs=50)
o_dst = czt(x)
plt.subplot(211)
plt.plot(np.abs(o_dst))

f2 /=(fs)
f2*=(2*np.pi)
f1 /=(fs)
f1*=(2*np.pi)
scale = f2 - f1
k = np.arange(max(m, n), dtype=np.int32)
wk2 = np.exp(-(1j * scale * k ** 2) /2/ m)
ak = np.exp(-1j * f1 * k[:n])
_Awk2 = ak * wk2[:n]
nfft = next_fast_len(n + m - 1)
_Fwk2 =1/np.hstack((wk2[n-1:0:-1], wk2[:m]))
_Fwk2 = fft(_Fwk2, nfft)
wk2 = wk2[:m]
_yidx = slice(n-1, n+m-1)


y = ifft(_Fwk2 * fft(x*_Awk2, nfft))
y = y[..., _yidx] * wk2
o_est = y
print(o_dst)
plt.subplot(212)
plt.plot(np.abs(o_est))
plt.show()