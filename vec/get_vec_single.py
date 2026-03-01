import matplotlib.pyplot as plt
import sys,os
sys.path.insert(0,'.')
import utils.opencv as ocv
import utils.sp as sp
import utils.fusion as fusion
import numpy as np
import cv2
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def norm(sig):
    return (sig-np.mean(sig))/np.std(sig)
def getDIS(sig,fs,noise=[]):
    e = np.append([0.3, 0.8, 0.5],np.zeros(len(noise)))
    #去直流归一化
    S=np.vstack([sig,noise]) if len(noise)!=0 else sig
    sig_len = len(sig[0])
    out = np.zeros(sig_len)

    w_size=int(fs*7)
    for i in range(0,sig_len-w_size):
        S_win = S[:,i:i+w_size]
        out[i:i+w_size]  = e @ np.linalg.pinv(S_win @ S_win.T) @ S_win
    # Output
    return out
def ls_line(points):
    """
    最小二乘拟合空间直线
    :param points: 三维点集合
     直线方程：
     x = a * z + b
     y = c * z + d
    :return: 直线参数a,b,c,d
    """
    points=points.T
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    n = points.shape[0]
    a = (n * sum(x * z) - sum(x) * sum(z)) / (n * sum(z * z) - sum(z) * sum(z))
    b = (sum(x) - a * sum(z)) / n
    c = (n * sum(y * z) - sum(y) * sum(z)) / (n * sum(z * z) - sum(z) * sum(z))
    d = (sum(y) - c * sum(z)) / n

    dv = np.array([a,c,1])
    dv/=np.linalg.norm(dv)

    return dv, np.array([b,d,0])


# file_path=sys.argv[1]
def draw_vec(file_path):

    fs=50
    filter_sensor = sp.Filter(6,[0.5,3],fs=fs)
    stride = fs
    window_length = fs*10
    sig_dft = sp.DFT(2000,(0.1,3),fs)
    ecg_max_idxs=[]
    def draw_spec(sig,name):
        spec = sig_dft.get_spectrogram(sig,window_length,1,lambda sig:sig*np.hanning(len(sig)))
        img = ocv.draw_spectrum(np.array(spec),sig_dft.freq_resolution,min_idx=sig_dft.roi_idx[0],ref_idxs=[ecg_max_idxs] if len(ecg_max_idxs) !=0 else None)
        cv2.imshow(name,img)
    f_name = os.path.split(file_path)[-1].split('.')[0]
    import json
    rgb_path = f"data/PURE/raw/{f_name}rgb.npy"
    raw_signal = np.load(rgb_path)
    all_Info = json.load(open(file_path))
    ppg_raw = [ n['Value']['waveform'] for n in all_Info['/FullPackage']]
    bpm_raw = [ n['Value']['pulseRate'] for n in all_Info['/FullPackage']]
    ts = np.array([ n['Timestamp']/1e6 for n in all_Info['/Image']])
    ts_signal = [ n['Timestamp']/1e6 for n in all_Info['/FullPackage']]

    hr_est = np.mean(bpm_raw)/60
    ts_dst = np.arange(ts[0],ts[-1],1000/fs) #ms
    ppg_filter = np.interp(ts_dst,ts_signal,ppg_raw)
    ppg_filter = sp.Filter(6,[hr_est*0.5,hr_est*1.5],fs=fs).filtfilt(ppg_filter)

    ecg_spectrogram = sig_dft.get_spectrogram(ppg_filter, window_length, 1,lambda sig:sig*np.hanning(len(sig)))
    ecg_max_idxs = np.argmax(ecg_spectrogram,1)
    # 
    raw_r = np.interp(ts_dst,ts,raw_signal[:,2])
    raw_g = np.interp(ts_dst,ts,raw_signal[:,1])
    raw_b = np.interp(ts_dst,ts,raw_signal[:,0])
    motion = np.interp(ts_dst,ts,raw_signal[:,3])

    filter_hp = sp.Filter(6,[0.5,np.nan],fs=fs)

    signal_len = len(raw_r)
    # S_raw=np.vstack([filter_hp.filtfilt(raw_r),filter_hp.filtfilt(raw_g),filter_hp.filtfilt(raw_b)])
    S_raw=np.vstack([raw_r,raw_g,raw_b])
    S=np.vstack([filter_sensor.filtfilt(raw_r),
                filter_sensor.filtfilt(raw_g),
                filter_sensor.filtfilt(raw_b)])
    D=np.vstack([motion])

    ref_sig = getDIS(S,fs)
    def near_filt(sig,fs,freqs,wlen):
        out = np.zeros_like(sig)
        for i in range(0,len(sig)-wlen):
            sig_win = sig[i:i+wlen]
            hr_est = freqs[i]
            o = sp.Filter(6,[hr_est-0.1,hr_est+0.1],fs=fs).filtfilt(sig_win)
            out[i:i+wlen] = o
        return out
    is_draw = False     
    if is_draw:    
    # ref_sig =   near_filt(ref_sig,fs,sig_dft.toFreq(ecg_max_idxs),window_length)     
        draw_spec(ref_sig,"asdsad")


    hr_filter = sp.Filter(6,[hr_est-0.5,hr_est+0.5],fs=fs)
    if is_draw:
        w, h = hr_filter.get_freqz(signal_len)
        db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
        plt.plot(w/np.pi*fs/2, db)
        plt.grid(True)
        plt.ylim(-0.2,0.1)
        plt.ylabel('Gain [dB]')
        plt.title('Frequency Response')
        plt.xlim(0.5,3)
        plt.subplot(2, 1, 2)
        plt.plot(w/np.pi*fs/2, np.angle(h))
        plt.grid(True)
        plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
                [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.ylabel('Phase [rad]')
        plt.xlabel('Normalized frequency (1.0 = Nyquist)')
        plt.xlim(0.5,3)
        plt.show()


    H=np.vstack([hr_filter.filtfilt(raw_r),
                hr_filter.filtfilt(raw_g),
                hr_filter.filtfilt(raw_b)])

    M = S_raw-H

    if is_draw:
        plt.plot(np.abs(np.fft.fft(M[1]))[int(0.1/fs*signal_len):int(5/fs*signal_len)])
        plt.plot(np.abs(np.fft.fft(S_raw[1]))[int(0.1/fs*signal_len):int(5/fs*signal_len)])
        plt.show()

    window_length=signal_len#fs*30
    i=0
    S_win = M[:,i:i+window_length]
    D_win = D[:1,i:i+window_length]
    # S_win = (S_win.T-np.mean(S_win,axis=1)).T
    v_motion,k = ls_line(S_win)

    S_win = S_raw[:,i:i+window_length]
    D_win = D[:1,i:i+window_length]
    v_raw,k = ls_line(S_win)

    S_win = H[:,i:i+window_length]
    D_win = D[:1,i:i+window_length]
    v_rppg,k = ls_line(S_win)

    print("moti ",np.arccos(np.dot(v_raw,v_motion))/np.pi*180)
    print("rppg ",np.arccos(np.dot(v_raw,v_rppg))/np.pi*180)
    print("morp ",np.arccos(np.dot(v_motion,v_rppg))/np.pi*180)

    def plot3d(ax,S_win,v,title,legend=None):
        half_range = np.max(np.max(S_win,axis=1)-np.min(S_win,axis=1))/2
        centers = (np.max(S_win,axis=1)-np.min(S_win,axis=1))/2+np.min(S_win,axis=1)
        ax.scatter(S_win[0], S_win[1], S_win[2],s=1,label=title[0])
        q0=np.array((centers[0]-half_range,centers[1]-half_range,centers[2]-half_range))
        q1=np.array((centers[0]+half_range,centers[1]+half_range,centers[2]+half_range))
        ax.set_xlim3d(q0[0],q1[0])
        ax.set_ylim3d(q0[1],q1[1])
        ax.set_zlim3d(q0[2],q1[2])
        ax.view_init(elev=30, azim=-60, roll=0)
        
        cam_vec = 1/np.array((1,np.tan(-60),1/np.cos(-60)*np.tan(30)))
        ax.set_xlabel("R")
        ax.set_ylabel("G")
        ax.set_zlabel("B")

        # ax.quiver(*(k+v*(v@(q0-k))), *v,
        #         length = v@(q1-q0)*1.0,color='g',arrow_length_ratio=0.1,label=title[1])
        # #ax.quiver(0, 0, 0, *k, color='g')
        # vec_view = q1-q0
        # print(k+v*(v@(q1-k))- v*(v@(vec_view))*0.5,(k+v*(v@(q1-k))- v*(v@(vec_view))*0.5 + np.cross(cam_vec/np.linalg.norm(cam_vec),v)))
        # ax.text3D(*(k+v*(v@(q1-k))- v*(v@(vec_view))*0.5 + 0.1*np.cross(cam_vec/np.linalg.norm(cam_vec),v*(v@(vec_view)))),"({:.3f},{:.3f},{:.3f})".format(*v),v)        


        # ax.quiver(*(k+v*(v@(q0-k))), *cam_vec,
        #         length = v@(q1-q0)*1.0,color='g',arrow_length_ratio=0.1,label='Fit Vector')
        ax.autoscale(False)
        ax.set_title(title[2])
        # if legend is not None:
        ax.legend(loc='upper left')

    if True:
        fig = plt.figure(figsize=(5, 4))
        plt.tight_layout()
        S_win = S_raw[:,i:i+window_length]*255
        v,k = ls_line(S_win)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        print(v)
        plot3d(ax,S_win,v,[r'${RGB}_{raw}$',r'${v}_{raw}$',r'RAW'])

        # ax = fig.add_subplot(1, 3, 2, projection='3d')
        # S_win = H[:,i:i+window_length]
        # v,k = ls_line(S_win)
        # print(v)
        # plot3d(ax,S_win,v,[r'${RGB}_{bvp}$',r'${v}_{bvp}$',r'BVP'])

        # ax = fig.add_subplot(1, 3, 3, projection='3d')
        # S_win = M[:,i:i+window_length]
        # v,k = ls_line(S_win)
        # print(v)
        # plot3d(ax,S_win,v,[r'${RGB}_{noise}$',r'${v}_{noise}$',r'Noise'])

        plt.show()
    # #%%


if __name__ == "__main__":
    # file_path=sys.argv[1]
    file_path="/tank/数据集/PURE/01-01/01-01.json"
    draw_vec(file_path)