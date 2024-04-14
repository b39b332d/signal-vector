import pandas as pd
import numpy as np
import os,sys

data_path = "/home/a406/Source/signal_vector/data/"
gth_files = os.listdir(data_path+"gth/")
raw_files = os.listdir(data_path+"raw/")

gth_files_head = [ f[:5] for f in gth_files ]
raw_files_head = [ f[:5] for f in raw_files ]

all_heads = [x for x in gth_files_head if x in raw_files_head]

missing_head_in_raw = [x for x in gth_files_head if x not in raw_files_head]
missing_head_in_gth = [x for x in raw_files_head if x not in gth_files_head]
print("missing_head_in_raw")
print(missing_head_in_raw)
print("missing_head_in_gth")
print(missing_head_in_gth)

time_len=256
stride = 50
labels = []
for f in all_heads:
    r=np.load(data_path+"raw/"+f+"out.npy")
    g=np.load(data_path+"gth/"+f+"gth.npy")
    len_sig = len(r)
    rg = np.arange(0,len_sig-time_len,stride)
    rg[-1]  = len_sig-time_len
    a=0
    for i in rg:
        a+=1
        ri = r[i:i+time_len]
        gi = g[i:i+time_len]
        

        ri = np.reshape(ri,(-1,52)).T
        gi = np.reshape(gi,(1,-1))
        labels.append(f"{f}_{a}.npy")
        np.save(f"{data_path}train/{f}_{a}.npy",ri)
        np.save(f"{data_path}label/{f}_{a}.npy",gi)

df = pd.DataFrame({'files':labels})
df.to_csv(data_path+"train.csv")