import numpy as np
import matplotlib.pyplot as plt

ts= np.load("E:/rec/1658147634/ppg_ts.npy")
phase = np.cos(2*np.pi*0.01*ts)
sig = np.sin(2*np.pi*1.5*ts+0.1*phase)
plt.plot(sig)
plt.show()
np.save("D:/sig.npy",sig)
