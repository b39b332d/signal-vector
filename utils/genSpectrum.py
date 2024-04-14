import numpy as np
import cv2

a=np.load("br_out.npy")
b = np.reshape(a, (-1, 200))
np.save("br_out.npy", b)

m=np.max(b,1)
m[m==0]=1
o=b.T/m
m=o*255
o= np.uint8(m)
pic=cv2.applyColorMap(o,cv2.COLORMAP_HOT)
cv2.imshow("",pic)
cv2.waitKey(0)