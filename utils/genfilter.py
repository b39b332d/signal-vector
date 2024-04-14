import numpy as np
from scipy import signal
import struct
import matplotlib.pyplot as plt

def float_to_hex(f):
 return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def double_to_hex(f):
 return hex(struct.unpack('<Q', struct.pack('<d', f))[0])

def output1d(c):
 for i in c:
  print(float_to_hex(i),end=',')
def outputf(c):
 for i in c:
  for ii in i:
   print(float_to_hex(ii),end=',')
  print("")
def outputd(c):
 for i in c:
  for ii in i:
   print(double_to_hex(ii),end=',')
  print("")
def outputlong(c):
 for i in c:
  for ii in i:
   print(int(ii*2**16),end='L,')
  print("")
o=[]
fs=3431
lcut_hr,hcut_hr = np.array((0.3,15))
sos = signal.butter(2,45*2/fs,'lowpass',output='sos')
outputf(np.delete(sos.astype(np.float32),3,1))
outputd(sos)
outputlong(np.delete(sos,3,1))
print(sos)
print(np.delete(sos.astype(np.float32),3,1))
print(signal.sosfilt(sos.astype(np.float32),(np.ones(75000)*2345).astype(np.float32))[10:30])
### filter test
# sig = np.arange(1000).astype(np.float32)
# out = signal.sosfilt(sos.astype(np.float32),sig)
# np.savetxt(r"C:\Users\b39b3\source\repos\respiheartembed\msvc\data\out_bandpass_test_0_to_1000.txt",out)
# print(out)


exit(0)
sos=signal.butter(6,[0.9*2/150,2.5*2/150],"bandpass",output="sos")
sos2=signal.butter(6,[0.9*2/150,2*2/150],"bandpass",output="sos")
#sos= signal.iirfilter(5,(0.5*2/100,3.1*2/100),btype="bandpass",output="sos",ftype="ellip",rp=0.1,rs=30)
w, h =signal.sosfreqz(sos, worN=3000)
plt.subplot(2, 1, 1)
db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
plt.plot(w/np.pi*50*60, np.abs(h))
#plt.ylim(-75, 5)
plt.grid(True)
#plt.yticks([0, -20, -40, -60])
plt.ylabel('Gain [dB]')
plt.title('Frequency Response')
plt.subplot(2, 1, 2)
plt.plot(np.angle(h))
np.save("phase.npy",np.angle(h))
plt.grid(True)
plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
           [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
plt.ylabel('Phase [rad]')
plt.xlabel('Normalized frequency (1.0 = Nyquist)')
plt.show()

exit(0)
