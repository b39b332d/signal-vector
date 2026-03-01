import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

all_val = pd.read_csv("./outoutout.csv",header=None,sep=',|:',engine='python')
labels = pd.unique(all_val[0])
ds_fit_val=[]
ds_dis_val=[]
for ds_name in labels:
    ds_fit_val.append(all_val.loc[all_val[0]==ds_name][2])
    ds_dis_val.append(all_val.loc[all_val[0]==ds_name][5])

def plot(data,ax,title):
    m1 = [ m.mean() for m in data]
    st1 = [ m.std() for m in data]

    bp = ax.boxplot(data,labels=labels, showmeans=True)

    for i, line in enumerate(bp['medians']):
        x, y = line.get_xydata()[1]
        text = ' μ={:.2f}\n σ={:.2f}'.format(m1[i], st1[i])
        ax.annotate(text, xy=(x, y))
    ax.set_ylabel("Bpm")
    #ax.set_yscale('log')
    ax.set_title(title,loc='right')


fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
plot(ds_fit_val,ax[0],"FIT")
plot(ds_dis_val,ax[1],"POS")
ax[0].set_title("train on ECG-Fitness: MAE")
plt.show()
