import numpy as np
import os
import matplotlib.pyplot as plt

label_files = os.listdir("./data/gth/")
train_files = os.listdir("./data/train/")

label_files_head = [ f.split(".")[0] for f in label_files ]
train_files_head = [ f.split(".")[0] for f in train_files ]

datas = [x for x in label_files_head if x in train_files_head]

vecs = [ np.load("./data/label/"+x+".npy") for x in datas ]
vecs = np.array(vecs)