import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import matplotlib.font_manager as fm

# 字体设置
font_S = fm.FontProperties(family='Times New Roman',math_fontfamily='dejavuserif', size=14)

# 自定义颜色和透明度
colors = ['#6E8FB2', '#EAB67A', '#E5A79A', '#BD9AAE', '#ABC8E6',
          '#D0D08A', '#F5DDB7', '#F5C9BC', '#F3B4BD', '#CEA1B5']
alpha = 0.6
vs_label = ['01-01', '02-01','03-01','04-01','05-01','06-01', '07-01', '08-01', '09-01', '10-01','10-06']

data=pd.read_csv("./vout",sep=" ",header=None)
data = data.sort_values(0)
xs = np.arange(len(data[data[1]=="moti"][3]))

plt.figure(figsize=(8, 5))
plt.scatter(xs,np.arccos(data[data[1]=="moti"][3])/np.pi*180,c=colors[8],label=r'$\theta_{noise,raw}$',s=22)
plt.scatter(xs,np.arccos(data[data[1]=="rppg"][3])/np.pi*180,c=colors[4],label=r'$\theta_{bvp,raw}$',s=22)
plt.xticks(xs)
plt.tick_params(axis='x', labelsize=5)  # 只改变x轴刻度标签大小
# plt.grid(True,axis='x',linewidth=1)
plt.ylabel("Angle(°)", fontproperties=font_S)
plt.xlabel("Video", fontproperties=font_S)

lbs = [x.split("/")[-2] for x in data[data[1]=="moti"][0]]
v_l = [ l if l in vs_label else ""  for l in lbs]
for i, label in enumerate(v_l):
    if label == '':
        plt.gca().xaxis.get_major_ticks()[i].tick1line.set_alpha(0.3)
        plt.gca().xaxis.get_major_ticks()[i].tick2line.set_alpha(0.3)
    else:
        plt.gca().xaxis.get_major_ticks()[i].tick1line.set_alpha(1.0)
        plt.gca().xaxis.get_major_ticks()[i].tick1line.set_linewidth(1.0)
        plt.gca().xaxis.get_major_ticks()[i].tick2line.set_alpha(1.0)
        plt.gca().xaxis.get_major_ticks()[i].tick2line.set_linewidth(1.0)

plt.xticks(xs)
font_s = fm.FontProperties(family='Times New Roman', size=10)
plt.gca().set_xticklabels( v_l,fontproperties=font_s)

plt.legend(prop=font_S)
# plt.ylim(0.99,1)
plt.show()