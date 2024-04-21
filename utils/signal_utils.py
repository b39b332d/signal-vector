import numpy as np
import utils.sp as sp

def norm_01(sig):
    max_axis = np.argmax(sig.shape)
    out = sig - np.min(sig,axis=max_axis,keepdims = True)
    out/= np.max(out,axis=max_axis,keepdims = True)
    return out

def norm_std(sig):
    max_axis = np.argmax(sig.shape)
    out= sig - np.mean(sig,axis=max_axis,keepdims = True)
    out/= np.std(out,axis=max_axis,keepdims = True)
    return out

def norm_11(sig):
    out = norm_01(sig)
    out -=0.5
    out *=2.0
    return out


def getPos(sig, interp_fs):
    w = []
    w_size = int(interp_fs * 1.6)
    sig_len = len(sig[0])
    out = np.zeros(sig_len)

    for i in range(0, sig_len - w_size):
        S_win = sig[:, i:i + w_size]
        S_win = (S_win.T / np.mean(S_win, 1)).T
        pj_vec = np.array([[0, 1, -1], [-2, 1, 1]])
        Xcomp, Ycomp = pj_vec @ S_win
        sX = np.std(Xcomp)
        sY = np.std(Ycomp)
        alpha = [1, sX / sY]
        pj_bvp = alpha @ pj_vec
        pj_bvp /= np.sqrt(np.sum(pj_bvp ** 2))
        bvp = pj_bvp @ S_win
        out[i:i + w_size] += bvp - np.mean(bvp)
    out /= np.hstack([np.arange(1, w_size + 1), np.ones(sig_len - 2 * w_size) * w_size,
                      np.arange(1, w_size + 1)[::-1]])
    return out


def getDIS(sig, fs, noise=[]):
    e = np.append([0.3, 0.8, 0.5], np.zeros(len(noise)))
    # 去直流归一化
    S = np.vstack([sig, noise]) if len(noise) != 0 else sig
    sig_len = len(sig[0])
    out = np.zeros(sig_len)

    w_size = int(fs * 7)
    for i in range(0, sig_len - w_size):
        S_win = S[:, i:i + w_size]
        vec = e @ np.linalg.pinv(S_win @ S_win.T)
        out[i:i + w_size] += vec @ S_win / np.sqrt(np.sum(vec[:3] ** 2))
    # Output

    out /= np.hstack([np.arange(1, w_size + 1), np.ones(len(out) - 2 * w_size) * w_size,
                      np.arange(1, w_size + 1)[::-1]])
    return out


def lsfilt(sig, noise, wlen):
    sig = np.array(sig)
    noise = np.array(noise)
    if len(sig.shape) == 1:
        sig = np.expand_dims(sig, 1)
    elif sig.shape[0] < 10:
        sig = sig.T
    if len(noise.shape) == 1:
        noise = np.expand_dims(noise, 1)
    elif noise.shape[0] < 10:
        noise = noise.T
    sig_n = np.zeros(sig.shape)
    for i in range(0, len(raw_r) - wlen):
        S_win = sig[i:i + wlen]
        D_win = noise[i:i + wlen]
        # add offset est
        D_win = np.hstack([D_win, np.ones([D_win.shape[0], 1])])

        ls_out = np.linalg.lstsq(D_win, S_win, rcond=None)
        out = S_win - D_win @ ls_out[0]
        # out = s1+s2/np.std(s2)*np.std(s1)
        # out = np.sqrt(s1*s1+s2*s2)
        sig_n[i:i + wlen] += out
    return sig_n.T / wlen


class align_sigs:
    def __init__(self, sig, ref_sig, fs, delta=5, delta_back=None):
        if delta_back == None:
            delta_back = delta
        pad_front = int(delta * fs)
        pad_back = int(delta * fs)
        move_sig = ref_sig[pad_front:-pad_back]
        out = np.convolve(sig,move_sig[::-1], mode='valid')
        max_idx = np.argmax(np.abs(out))
        if out[max_idx] <0:
            self.ref_sig = -ref_sig
        else:
            self.ref_sig = ref_sig

        self.ofs = pad_front - np.argmax(np.abs(out))


    def get_ref(self):
        if self.ofs >= 0:
            return self.ref_sig[self.ofs:]
        else:
            return self.ref_sig[:self.ofs]

    def __call__(self, sig):
        if len(sig.shape) == 1 or sig.shape[0] > sig.shape[1]:
            if self.ofs > 0:
                return sig[:-self.ofs]
            else:
                return sig[-self.ofs:]
        else:
            if self.ofs > 0:
                return sig[:, :-self.ofs]
            else:
                return sig[:, -self.ofs:]

def near_filt(sig, fs, freqs, wlen,delta=0.1):
    out = np.zeros_like(sig)
    for i in range(0, len(sig) - wlen):
        sig_win = sig[i:i + wlen]
        hr_est = freqs[i]
        o = sp.Filter(6, [hr_est - delta, hr_est + delta], fs=fs).filtfilt(sig_win)
        out[i:i + wlen] += o
    out /= np.hstack([np.arange(1, wlen + 1), np.ones(len(out) - 2 * wlen)*wlen,
                          np.arange(1, wlen + 1)[::-1]])
    return out

def get_error(sig1,sig2,fs,w_len,stride):
    dis_arr=[]
    for i in range(0,len(sig1)-w_len,stride):
        sig_win1 = np.append(sig1[i:i+w_len],np.zeros(3000-w_len))
        sig_win2 = np.append(sig2[i:i+w_len],np.zeros(3000-w_len))
        fft1 = np.abs(np.fft.rfft(sig_win1))
        fft2 = np.abs(np.fft.rfft(sig_win2))
        dis = np.argmax(fft1)-np.argmax(fft2)
        dis_arr.append(dis)
    dis_arr = np.array(dis_arr)*fs/3000 *60
    mae = np.mean(np.abs(dis_arr))
    mse = np.mean(dis_arr**2)
    rmse = np.sqrt(mse)
    return mae,mse,rmse