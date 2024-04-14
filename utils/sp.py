from scipy import signal
import numpy as np
import math
from collections.abc import Callable
from typing import Any

__draw_spectrum_all_colors = ['lime', 'tab:blue', 'violet', 'mdeiumturquoise', 'mediumpurple']

class Filter:
    def __init__(self, order, cof=(float(0), float('nan')), fs=1):
        self.cof = cof
        fs /= 2  # Nyquist frequency
        if not math.isnan(cof[1]) and cof[0] != float(0):
            self.filter_sos = signal.butter(order, [cof[0] / fs, cof[1] / fs], "bandpass", output='sos')
        elif math.isnan(cof[1]):
            self.filter_sos = signal.butter(order, cof[0] / fs, "highpass", output='sos')
        elif cof[0] == float(0):
            self.filter_sos = signal.butter(order, cof[1] / fs, "lowpass", output='sos')
        else:
            raise

    def filtfilt(self, sig):
        return signal.sosfiltfilt(self.filter_sos, sig)

    def filt(self, sig):
        return signal.sosfilt(self.filter_sos, sig)


class DFT:
    """
    DFT Class providing FFT method.

    Parameters
    ----------
    window : int
        FFT window size. `fft` will append zeros after
        `sig` if `sig` is shorter than `window`.
    roi_freq : (fc-low, fc-high) tuple of floats, optional
        If fc-high is set to -1, the higher cutoff frequency will be set to around fs/2,
        check `roi_freq` after init to retrieve the accurate cut-off frequency value.
    fs : double, optional
        If the default value is passed, then all frequency used in
        this class should be normalized frequency.
    """
    def __toFreq(self, idx):
        return idx * self.freq_resolution

    def __toIdx(self, freq):
        return int(freq / self.freq_resolution)

    def __init__(self, dft_len, roi_freq=(float(0), -1), fs=1):
        self.fs = fs
        self.freq_resolution = fs / dft_len
        self.dft_len = dft_len
        self.roi_freq = roi_freq
        self.roi_idx = [self.__toIdx(roi_freq[0]), int(dft_len / 2) + 1]
        if roi_freq[1] > 0:
            self.roi_idx[1] = self.__toIdx(roi_freq[1])
        else:
            self.roi_freq[1] = self.__toFreq(self.roi_idx[1])

    def fft(self, sig):
        return np.fft.fft(np.append(sig, np.zeros(self.dft_len - len(sig))))[self.roi_idx[0]:self.roi_idx[1]]

    def rfft(self, sig):
        return np.abs(self.fft(sig))

    def get_rfft_idx(self):
        return self.toFreq(np.arange(self.roi_idx[0],self.roi_idx[1]))

    def getFreqIndex(self):
        return np.arange(self.roi_freq[0], self.roi_freq[1], self.freq_resolution)

    def toIdx(self, freq):
        return int(freq / self.freq_resolution) - self.roi_idx[0]

    def toIdxs(self, freq):
        return (np.array(freq) / self.freq_resolution).astype(int) - self.roi_idx[0]

    def toFreq(self, idx):
        if hasattr(idx, "__len__") and len(idx) == 0:
            return []
        return idx * self.freq_resolution + self.roi_freq[0]
    def get_spectrogram(self, sig: np.ndarray, signal_window_len:int, stride:int=1,
                        fft_filter_func: Callable[[np.ndarray], Any]=lambda fft_signal : fft_signal,
                        time_filter_func: Callable[[np.ndarray], Any]=lambda windowed_signal : windowed_signal) -> list[Any]:
        if signal_window_len > self.dft_len:
            return []
        fft_spec = []
        for i in range(0, len(sig) - signal_window_len, stride):
            fft_spec.append(fft_filter_func(self.rfft(time_filter_func(sig[i:i + signal_window_len]))))
        return fft_spec

    def get_spectrogram_l2(self, sig: np.ndarray, signal_window_len:int,l2_coff, stride:int=1,
                        fft_filter_func: Callable[[np.ndarray], Any]=lambda fft_signal : fft_signal,
                        time_filter_func: Callable[[np.ndarray], Any]=lambda windowed_signal : windowed_signal) -> list[Any]:
        if signal_window_len > self.dft_len:
            return []
        fft_spec = []
        sig_l2 = np.interp(np.arange(0, len(sig), 0.5), np.arange(0, len(sig)),sig)
        for i in range(0, len(sig) - signal_window_len, stride):
            fft_spec.append(fft_filter_func(np.abs(self.fft(time_filter_func(sig[i:i + signal_window_len])) +
                            l2_coff* self.fft(time_filter_func(sig_l2[i*2:(i + signal_window_len)*2])))))
        return fft_spec
    def get_spectrogram_f2(self, sig: np.ndarray, signal_window_len:int,l2_coff,l2_fft, stride:int=1,
                        fft_filter_func: Callable[[np.ndarray], Any]=lambda fft_signal : fft_signal,
                        time_filter_func: Callable[[np.ndarray], Any]=lambda windowed_signal : windowed_signal) -> list[Any]:
        if signal_window_len > self.dft_len:
            return []
        fft_spec = []
        for i in range(0, len(sig) - signal_window_len, stride):
            fft_spec.append(fft_filter_func(np.abs(self.fft(time_filter_func(sig[i:i + signal_window_len])) +
                            l2_coff* l2_fft.fft(time_filter_func(sig[i:i + signal_window_len])))))
        return fft_spec


class Spectrogram (DFT):
    def __init__(self, fft_len, sig_win_len, roi_freq=(float(0), -1), fs=1,signal_length=0, stride=1):
        if sig_win_len > fft_len:
            raise Exception('moving window size is larger than fft length')
        super().__init__(self, fft_len, roi_freq, fs)
        self.sig_win_len = sig_win_len
        self.stride = stride
        self.reference_names=[]
        self.reference_specs=[]
        self.reference_colors=[]
        self.signal_length = signal_length
    def to_spectrogram(self,signal,show,color=None,
                        fft_filter_func: Callable[[np.ndarray], Any]=lambda fft_signal : fft_signal,
                        time_filter_func: Callable[[np.ndarray], Any]=lambda windowed_signal : windowed_signal):
        fft_spec = []
        for i in range(0, len(signal) - self.sig_win_len, self.stride):
            fft_spec.append(fft_filter_func(self.rfft(time_filter_func(signal[i:i + self.sig_win_len]))))
        if show:
            from matplotlib import colors
            if color is None:
                color = colors.use(__draw_spectrum_all_colors[len(self.reference)])
            ocv.draw_spectrum(fft_spec,self.freq_resolution,min_idx=self.roi_idx[0],ref_idxs=[ecg_max_idxs] if len(ecg_max_idxs) !=0 else None)
        return fft_spec
    def add_reference(self,signal,name,color=None,
                        fft_filter_func: Callable[[np.ndarray], Any]=lambda fft_signal : fft_signal,
                        time_filter_func: Callable[[np.ndarray], Any]=lambda windowed_signal : windowed_signal):
        if name in self.reference_names:
            raise Exception("name exist")
        self.reference_names.append(name)
        self.reference_specs.append(self.to_spectrogram(self,signal,fft_filter_func,time_filter_func))
        self.reference_colors.append(color)
    def get_reference_spec(self,name):
        idx = self.reference_names.index(name)
        return self.reference_specs[idx]

    def show_spectrogram(self,):
        import opencv as ocv
        ocv.draw_spectrum()

def getSNR(spec, idxs_src,  radius=5):
    spec = np.array(spec)
    spec_len = len(spec[0])
    idxs =idxs_src.copy()
    idxs[idxs < radius] = radius
    idxs[idxs > spec_len - radius - 1] = spec_len - radius - 1
    spec_idx = np.arange(len(spec))
    e_sig = np.zeros(len(spec))
    for i in range(-radius, radius + 1):
        e_sig += spec[spec_idx, idxs + i]
    e_noise = np.sum(spec, 1) - e_sig
    return np.max(spec, 1) / e_noise

def get_spec(sig, win_length, roi_freq=(float(0), float('nan')), fs=1, stride=1):
    fft_spec = []
    sig_fft = DFT(win_length, roi_freq, fs)
    for i in range(0, len(sig) - win_length, stride):
        fft_spec.append(sig_fft.rfft(sig[i:i + win_length]))
    return fft_spec, sig_fft.roi_idx[0]


