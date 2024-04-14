import cv2
import numpy as np
from matplotlib import colors
__draw_spectrum_all_colors = ['lime', 'tab:blue', 'violet', 'mdeiumturquoise', 'mediumpurple']
from scipy.interpolate import interp1d
import warnings
def draw_spectrum(spectrum, freq_resolution,min_freq=None, min_idx=None, ref_idxs=None,ref_freqs = None, show_size = (1900, 700),color_list = None):
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
        min_idx = min_freq/freq_resolution
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
    pico = cv2.applyColorMap(o, cv2.COLORMAP_HOT)
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
                ref_idx = ref_idx[:o_size[1]]
            pts = np.array(
                [np.arange(o_size[1]) / (o_size[1]-1) * (d_size[0]-1),ref_idx / (o_size[0]-1) * (d_size[1]-1)],np.int32)\
                    .T.reshape((-1, 1, 2))
            pic=cv2.polylines(pic,[pts],False,np.array(colors.to_rgb(color_list[i]))*255,1)

    for i in np.arange((int(show_cut_freq[0]*60/10*2)+1)*10/2,(int(show_cut_freq[1]*60/10*2))*10/2,5):
        draw_axis_y = int((i/60/freq_resolution-min_idx)*(d_size[1]-1)/(o_size[0]-1))
        if i%10==0:
            pic[draw_axis_y, 0:10, :] = (0, 255, 0)
            pic = cv2.putText(pic, str(int(i)), (0, draw_axis_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0),
                              thickness=1)
        else:
            pic[draw_axis_y, 0:5, :] = (255, 255, 255)
    return pic

