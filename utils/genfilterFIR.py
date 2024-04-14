import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
sr = 3422
filt_2 = np.log2((signal.firwin(64,cutoff=10,fs=sr,pass_zero='lowpass')*2**16)).astype(int)
filter_t = (2**filt_2)/2**15
filter_t = np.ones(135)/135
w, h = signal.freqz(b=filter_t, a=1,fs = sr)
# x = w * sr * 1.0 / (2 * np.pi)
x = w
y = abs(h)
plt.figure(figsize=(10,5))
plt.semilogx(x, y)
plt.ylabel('Amplitude')
plt.xlabel('Frequency [Hz]')
plt.title('Frequency response')
plt.grid(which='both', linestyle='-', color='grey')
plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], ["20", "50", "100", "200", "500", "1K", "2K", "5K", "10K", "20K"])
plt.show()

y = 20 * np.log10(abs(h))
plt.figure(figsize=(10,5))
plt.semilogx(x, y)
plt.ylabel('Amplitude [dB]')
plt.xlabel('Frequency [Hz]')
plt.title('Frequency response')
plt.grid(which='both', linestyle='-', color='grey')
plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], ["20", "50", "100", "200", "500", "1K", "2K", "5K", "10K", "20K"])
plt.show()