import cv2
import numpy as np
from matplotlib import colors
__draw_spectrum_all_colors = ['red','lime', 'tab:blue', 'violet', 'mdeiumturquoise', 'mediumpurple']
from scipy.interpolate import interp1d
import warnings

# from matplotlib import *
# from pylab import *

# def plt_specgram(ax, x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
#              window=mlab.window_hanning, noverlap=128,
#              cmap=None, xextent=None, pad_to=None, sides='default',
#              scale_by_freq=None, minfreq = None, maxfreq = None, **kwargs):


#     #####################################
#     # modified  axes.specgram() to limit
#     # the frequencies plotted
#     #####################################

#     # this will fail if there isn't a current axis in the global scope
#     Pxx, freqs, bins = mlab.specgram(x, NFFT, Fs, detrend,
#          window, noverlap, pad_to, sides, scale_by_freq)

#     # modified here
#     #####################################
#     if minfreq is not None and maxfreq is not None:
#         Pxx = Pxx[(freqs >= minfreq) & (freqs <= maxfreq)]
#         freqs = freqs[(freqs >= minfreq) & (freqs <= maxfreq)]
#     #####################################

#     #Z = 10. * np.log10(Pxx)
#     Z = np.flipud(Pxx)

#     if xextent is None: xextent = 0, np.amax(bins)
#     xmin, xmax = xextent
#     freqs += Fc
#     extent = xmin, xmax, freqs[0], freqs[-1]
#     im = ax.imshow(Z, cmap, extent=extent, **kwargs)
#     ax.axis('auto')

#     return Pxx, freqs, bins, im
def draw_spectrum(spectrum, freq_resolution,min_freq=None, min_idx=None, ref_idxs=None,ref_freqs = None, 
                  show_size = (1200, 500),axis_step_bpm=20,axis_size = 0.4,axis_color= (255, 255, 255),
                  color_list = None):
    """

    Parameters
    ----------
    spectrum: [[fft],[fft],[fft]]
    freq_resolution freq/idx
    min_idx
    ref_lines
    """
    if color_list is None:
        color_list = __draw_spectrum_all_colors
    if min_freq is not None:
        min_idx = int(min_freq/freq_resolution)
    elif min_idx is None:
        return None
    d_size = show_size
    o_size = (len(spectrum[0]),len(spectrum))
    min_freq = min_idx*freq_resolution
    max_freq = (min_idx+len(spectrum[0]))*freq_resolution
    show_cut_freq = np.array([min_freq,max_freq])
    m = np.max(spectrum,1)
    m[m == 0] = 1
    o = np.array(spectrum).T / m
    interp = interp1d(np.linspace(0, 1, len(o)), o* 255, axis=0)
    o=interp(np.linspace(0, 1, d_size[1]))
    o = np.uint8(o)
    pico = cv2.applyColorMap(o, cv2.COLORMAP_VIRIDIS)
    pic = cv2.resize(pico, d_size,interpolation=cv2.INTER_NEAREST)
    if ref_idxs is None:
        ref_idxs = []
    if ref_freqs is not None:
        for ref_freq in ref_freqs:
            ref_freq= np.array(ref_freq)
            ref_idxs.append(ref_freq/freq_resolution-min_idx)

    if ref_idxs is not None:
        for i,ref_idx in enumerate(ref_idxs):
            ref_idx = np.array(ref_idx)
            if len(ref_idx) != o_size[1]:
                warnings.warn("reference and spectrum not aligned")
                if len(ref_idx) > o_size[1]:
                    ref_idx = ref_idx[:o_size[1]]
                else:
                    ref_idx = np.append(ref_idx,np.ones(o_size[1]-len(ref_idx))*ref_idx[-1])
            pts = np.array(
                [np.arange(o_size[1]) / (o_size[1]-1) * (d_size[0]-1),ref_idx / (o_size[0]-1) * (d_size[1]-1)],np.int32)\
                    .T.reshape((-1, 1, 2))
            pic=cv2.polylines(pic,[pts],False,np.array(colors.to_rgb(color_list[i]))[::-1]*255,2)
    # if label is not None:
    #     sz,_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,label_size,thickness=1)
    #     cv2.putText(pic,label,(d_size[0]-sz[0]-10, sz[1]+10),cv2.FONT_HERSHEY_SIMPLEX,label_size,label_color,thickness=1)

    if axis_step_bpm != None:
        axis_step_bpm = int(axis_step_bpm)
        for i in np.arange((int(show_cut_freq[0]*60/10*2)+1)*10/2,(int(show_cut_freq[1]*60/10*2))*10/2,5):
            draw_axis_y = int((i/60/freq_resolution-min_idx)*(d_size[1]-1)/(o_size[0]-1))
            if i%axis_step_bpm==0:
                pic[draw_axis_y, 0:10, :] = axis_color
                pic = cv2.putText(pic, str(int(i)), (0, draw_axis_y), cv2.FONT_HERSHEY_SIMPLEX, axis_size, axis_color,
                                thickness=1)
            elif i%axis_step_bpm==int(axis_step_bpm/2):
                pic[draw_axis_y, 0:5, :] = axis_color
        return pic
    else:
        return pic,[0,1,(min_idx+len(spectrum[0]))*freq_resolution*60,min_idx*freq_resolution*60]
    

