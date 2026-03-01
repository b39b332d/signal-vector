import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import time



'''
Adaptive Multi-Trace Carving for Robust Frequency Tracking in Forensic Applications

Adaptive Multi-Trace Carving (AMTC), a unified approach for  detecting and tracking one or more subtle frequency components
in very low signal-to-noise ratio (SNR) conditions and in pseudo-realtime with low delay. AMTC treats the signal’s time-
frequency representation as the input and identifies all frequency traces with dominating energy through iterative dynamic 
programming and adaptive trace compensation. AMTC considers a long duration of high trace energy as the indicator of frequency 
presence and accurately detects the frequency component.
'''


class AMTC():
    # 阅读论文AMTC，全称 Adaptive Multi-Trace Carving forRobust Frequency Tracking in Forensic Applications， 16页的
    def __init__(self):
        '''
        spectreum : time and frequency graph of the signal
        frequency : frequency axis of spectrum
        ti : time axis of spectreum
        num : how many traces you want to track
        ground_truth: the true heart rate at eatch moment, is a one-dimensional array
        scale: The scale to scale the time-frequency plot
        '''

        self.spec = None
        self.original_spec = None
        self.f = None
        self.t = None
        self.num = 1
        self.gt_hr = None
        self.scale = 3
        self.threshold = 0.1
        self.A = None                     # the state transition matrix, which stores the probabilities of transitions between states 状态转移概率矩阵
        self.show_groundtruth = True      # do you want to show the ground truth in the final spectrum graph      展示真实值
        self.show_traces = True           # do you want to show the teacked traces in the final spectrum graph    展示搜索的的轨迹
        self.show_all_traces = False      # do you want to show all tracked traces in the final spectrum graph or just the final one  展示所有轨迹
        self.gauss_fitting = False        # use gauss function to fit the change between every frequency or just a diff fractional formula  使用高斯分布拟合转移矩阵
        self.roads = []                   # the list to save every trace map(the map is a one-dimensional array as long as ti) after each track  保存搜索到的路径

    # 设置追踪器参数
    def set_parameters(self, spectrum, frequency, ti, num, threshold, scale, ground_truth=None):
        self.spec = spectrum
        self.original_spec = spectrum.copy()
        self.f = frequency
        self.t = ti
        self.num = num
        self.gt_hr = ground_truth   # 默认为None，为None则不绘制真实值
        self.scale = scale
        self.threshold = threshold

    # 计算状态转移概率矩阵
    def status_transform(self, miu=0.0270607, std=0.0266651):
        '''
        diff_threshold: used to limit transitions netween state, when the diffenence between two frequencies is larger than threshold,
                        the probability of the trancition between them is zero.
        miu: the mean of frequency differences use for gauss fitting
        std: the standard deviation of frequency differences use for gauss fitting
        use different frequency as different status

        output:
        A : the state transition matrix
        '''
        # std = 0.2885241053
        # miu = 0.0437

        Mf = max(self.f.shape)
        A = np.zeros((Mf, Mf))
        e = 2.718281828
        sigma = 1 / ((2 * 3.1415926)**0.5 * std)
        diff_threshold = self.threshold
        for i in range(Mf):
            status_i = self.f[i]
            for j in range(Mf):
                status_j = self.f[j]
                if np.abs(status_i - status_j) < diff_threshold:   # 这里设置了一个允许状态转移的最大范围，例如上一时刻心率1Hz，0.2秒后转移到4Hz，这种情况是不考虑的，直接概率置零，减少计算量
                    if self.gauss_fitting:
                        # 高斯拟合状态转移概率
                        A[i, j] = sigma * e**((-(status_i - status_j - miu)**2) / 2)
                    else:
                        # 均匀分布拟合状态转移概率
                        A[i, j] = 1 / (2 * np.abs(status_i - status_j) + 1)
                else:
                    A[i, j] = 0
        return A

    def energy_map(self, PR, p0, la=1.5):
        '''
        PR: the state transition matrix
        p0: initial state probability
        la: λ > 0 is a regularization parameter that controls the smoothness of the resulting trace.

        returns:
        energy: the path energy accumulation graph
        road_map: path record, use to record transition path, it is a two-dimensional array, time on the horizontal axis and frequency on the vertical axis
        '''

        # 这一部分请详细阅读AMTC文章，这里主要就是G矩阵的构建和转移节点记录
        M, N = self.spec.shape
        energy = np.zeros_like(self.spec)
        labda = la
        energy[:, 0] = self.spec[:, 0] + labda * np.log(p0 + 10 ** (-100))
        road_map = np.zeros_like(self.spec)
        for n in range(1, N):
            for m in range(0, M):

                # 仅在非零概率范围内进行追踪
                # pr_index = np.where(PR[:, m] > 0)[0]
                # no_zero = energy[pr_index, n - 1] + labda * np.log(PR[pr_index, m])
                # temp_index = np.argmax(no_zero)
                # M_index = pr_index[temp_index]
                # temp_max = no_zero[temp_index]

                # 在所有的概率范围内进行追踪
                temp = energy[:, n - 1] + labda * np.log(PR[m, :] + 10 ** (-100)).T   # 计算每个状态转移自上一个状态并产生观测的概率，维特比变量，AMTC文章中G矩阵的max{}部分
                M_index = np.argmax(temp)   # 得到最大概率对应的索引
                temp_max = temp[M_index]    # 获取最大概率作为维特比变量
                road_map[m, n] = M_index - m  # 记录状态转移的索引变化量，用于回溯，只需要通过叠加索引变化量便可得到一段完整路径，seam-carving文章中的做法，AMTC中未使用
                energy[m, n] = self.spec[m, n] + temp_max   # 更性此时刻的能量图，即AMTC文章中的G矩阵
        return energy, road_map

    def find_road_amtc(self, energy, PR, la=1.5):

        # AMTC文章中的f^(n)的求解公式
        N = energy.shape[-1]
        road = np.zeros(N).astype(np.uint16)     # the length of a road is as same as the width of the energy
        road[N-1] = np.argmax(energy[:, -1])     # the final of a road is set as the index of max value in the last column of the energy 能量图最后一列的最大能量概率作为回溯起点
        for n in range(N-2, -1, -1):
            temp = energy[:, n] + la * np.log(PR[:, road[n+1]] + 10 ** (-100))   # 根据AMTC文中的f^(n)公式进行回溯
            road[n] = np.argmax(temp)            # 记录某一路径节点在时频图上的列坐标，记频率轴索引，并作为路径信息返回
        return road

    def find_road_seam_carving(self, energy, road_map):
        '''
        Using the path energy accumulation graph and the recorded transition path,
        the trajectory is found according to the principle of maximum path energy

        energy: the path energy accumulation graph
        road_map: path record, from last moment to first moment to find the trace according to the road-map

        output:
        road: the trace found according to the principle of maximum path energy
        '''
        # 阅读seam-carving文章中是如何进行路径map构建与路径回溯的
        N = energy.shape[-1]
        road = np.zeros(N).astype(np.uint16)
        index_temp = np.argmax(energy[:, -1])
        road[-1] = index_temp
        for i in range(N - 1, 0, -1):
            index_temp = index_temp + road_map[int(index_temp), i]
            road[i - 1] = index_temp
        return road

    # 痕迹存在检测
    def trace_exited(f, spec, road):
        M, N = spec.shape
        if len(road) != N:
            return road
        delta_f = 0.15
        delta_n = int(delta_f / np.mean(np.diff(f)))
        F = np.ones_like(f)
        rer = 2
        for n in range(N):
            left_boundary = max(road[n] - delta_n, 0)
            right_boundary = min(road[n] + delta_n, M - 1)
            F[left_boundary:right_boundary + 1] = 0
            fenmu = sum(F * spec[:, n])
            fenzi = sum(F) * spec[road[n], n]
            if fenzi / fenmu < rer:
                road[n] = 0
        return road

    def find_boundary_point(self, spectrum_t, center_f):
        '''
        According to the nearest difference point,
        the frequency range of the center frequency used to attenuate frequency.

        spectrum_t: the frequency and time graph ad time t, is a column vector as high as self.f
        center_f: the center frequency used to attenuate frequency.

        outputs:
        left_boundary: the upper limit of the frequency to be attenuated.
        right_boundary: the lower limit of the frequency to be attenuated.
        '''
        # 基本原理就是在该列频谱中运动频率的左右做一阶差分，求第一个一阶差分零点的频率位置，即为左右波谷点，需要注意范围溢出的问题。
        f_diff = spectrum_t[1:] - spectrum_t[:-1]
        length = len(spectrum_t)
        left = f_diff[: center_f]
        right = f_diff[center_f + 1:]
        left_index = np.where(left < 0)[0]
        right_index = np.where(right > 0)[0]
        if len(left_index) > 0:
            left_boundary = left_index[-1]
        else:
            left_boundary = 0
        if len(right_index) > 0:
            right_boundary = right_index[0] + center_f + 1
        else:
            right_boundary = length - 1

        return left_boundary, right_boundary

    def damping_energy(self, road):
        '''
        The last found trajectory is attenuated on the time-frequency plot.

        road: the last found road

        output:
        the time-frequency graph after attenuating
        '''
        # 根据路径，衰减路径上的能量，反高斯函数曲线进行衰减，原理可以查看AMTC的16页论文，或看吴海鹏毕业论文中轨迹能量衰减谱减法那一部分（一定要看）
        M, N = self.spec.shape
        sigma = np.zeros(N)
        for n in range(N):   #根据时频图的宽度即时间轴，逐列进行轨迹能量衰减
            boundary_left, boundary_right = self.find_boundary_point(spectrum_t=self.spec[:, n], center_f=road[n])   # 每一列都会有对应的运动伪影频率，寻找该频率左右波谷确定衰减宽度
            sum1 = 0
            sum2 = 0
            for m in range(boundary_left, boundary_right + 1):          # 先计算出时频图每一列所对应的反高斯曲线的钟摆宽度sigmma，即论文中公式(7)
                sum1 += self.spec[m, n] * (self.f[m] - self.f[road[n]]) ** 2
                sum2 += self.spec[m, n]
            sigma[n] = sum1 / sum2

        for n in range(N):
            for m in range(M):
                self.spec[m, n] = (1 - np.exp((-(self.f[m] - self.f[road[n]]) ** 2) / (30 * sigma[n]))) * self.spec[m, n] # 将反高斯曲线与该列时频图相乘，完成衰减，AMTC中公式(7)

        return self.spec

    def display_roads(self, roads):
        '''
        show found traces
        roads: the list of found traces

        output:
        spectrum: spectrum with traces

        '''
        spectrum = self.spec.copy()
        R = len(roads)
        if self.show_all_traces:  # 展示所有路径
            for r in range(R):
                road = roads[r]
                N = len(road)
                for i in range(N - 1):
                    x0 = (i - 1) * self.scale
                    y0 = road[i - 1] * self.scale
                    spectrum[y0 - 1: y0 + 1, x0 - 1: x0 + 1] = 0
        else:
            road = roads[-1]
            N = len(road)
            for i in range(N - 1):
                x0 = (i - 1) * self.scale
                y0 = road[i - 1] * self.scale
                spectrum[y0 - 1: y0 + 1, x0 - 1: x0 + 1] = 0

        return spectrum

    def display_hr_gt(self):
        '''
        shoe true heart tare in the spectrum

        output:
        spectrum: spectrum witi heart rate in it
        '''

        spectrum = self.spec
        N = len(self.gt_hr)
        f_index = [np.argmin(abs(hr - self.f)) for hr in self.gt_hr]
        for i in range(1, N - 1):
            x0 = (i - 1) * self.scale
            y0 = f_index[i - 1] * self.scale
            x1 = i * self.scale
            y1 = f_index[i] * self.scale
            spectrum = cv2.line(spectrum, (x0, y0), (x1, y1), color=10, thickness=2)
        return spectrum

    def get_AMTC_HR(self):
        '''
        return the heart rate tracked by AMTC

        output:
        hr_array: two-dimensional array, each row corresponds
                  to the heart rate tracked for each trajectory
        '''

        hr_array = []
        for road in self.roads:
            hr_amtc = np.array([self.f[i] * 60 for i in road])
            #hr_index = np.linspace(0, len(hr_amtc)-1, num=len(hr_amtc), dtype=np.int16)
            hr_array.append(hr_amtc)

        return np.array(hr_array)

    def get_state_PR(self):
        '''
        Returns the state transition matrix
        '''
        return self.A

    def get_roads(self):
        return self.roads

    # 直接绘制追踪结果
    def show(self):
        fmax = max(self.f)
        tmax = max(self.t)
        fnew = np.linspace(0, fmax, self.spec.shape[0])
        tnew = np.linspace(0, tmax, self.spec.shape[1])
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(tnew, fnew, self.spec, cmap='hsv')
        plt.colorbar()
        plt.xticks(fontproperties='Times New Roman', size=15)
        plt.yticks(fontproperties='Times New Roman', size=15)
        plt.xlabel("Time / s", fontproperties='Times New Roman', size=15)
        plt.ylabel("Heart Rate / Hz", fontproperties='Times New Roman', size=15)
        if self.gt_hr is not None:
            hr = self.f[[np.argmin(abs(hr - self.f)) for hr in self.gt_hr]]
            plt.plot(self.t, hr, c='green')
        if self.show_all_traces:
            for road in self.roads:
                plt.plot(self.t, self.f[road], c='black')
        else:
            plt.plot(self.t, self.f[self.roads[-1]], c='black')
        plt.show()

    def run_AMTC(self, lam=0.1):
        '''
        lam: λ > 0 is a regularization parameter that controls the smoothness of the resulting trace.

        outputs:
        tnew: The corresponding time axis after the time-frequency graph size transformation
        fnew: The frequency axis corresponding to the time-frequency plot after size transformation
        self.spec: final spectrum

        '''
        self.A = self.status_transform()    # 定义状态转移概率矩阵
        for i in range(self.num):           # 寻找几条轨迹，循环几次
            pi = self.spec[:, 0]            # 初始状态概率向量
            energy_m, road_map = self.energy_map(PR=self.A, p0=pi, la=lam)         # 频率追踪得到的能量图和节点记录图，仔细阅读AMTC文章
            # self.roads.append(self.find_road_seam_carving(energy_m, road_map))   # seam_carving论文的做法是根据记录的路径路径转移map得到完整路径
            self.roads.append(self.find_road_amtc(energy_m, PR=self.A, la=lam))    # AMTC论文的做法是根据能量图，寻找得到完整路径
            if self.num > 1 and i < self.num - 1:
                self.spec = self.damping_energy(self.roads[i])    # 如果需要寻找多条路径，需要对上次寻找到的路径进行轨迹能量衰减，再开始下面的寻找。

        # 路径搜索已完成，下面就是时频图展示问题了
        self.spec = (self.spec * 100).astype(np.uint16)
        self.spec = cv2.resize(self.spec, dsize=(self.spec.shape[1] * self.scale, self.spec.shape[0] * self.scale))  # 展示尺度变化

        if self.show_groundtruth:     # 展示真实参考心率
            self.spec = self.display_hr_gt()

        if self.show_traces:    # 展示搜索到的路径
            self.spec = self.display_roads(self.roads)

        fmax = max(self.f)
        tmax = max(self.t)
        fnew = np.linspace(0, fmax, self.spec.shape[0])
        tnew = np.linspace(0, tmax, self.spec.shape[1])
        return tnew, fnew, self.spec


def AMTC_method(rppg, hr_gt, fps=30, threshold=0.05, lam=5):
    str_time = time.time()
    # rppg = complte_BVP_sig(rppg, 1, fps).squeeze()
    hmm_f, hmm_t, rppg_spec = signal.stft(rppg, fs=fps, nperseg=200, noverlap=194, nfft=1024)  # rppg stft时频分析
    rppg_spec = np.abs(rppg_spec) / np.max(np.max(np.abs(rppg_spec)))  # 时频图归一化
    f_length = int(300 / 60 / hmm_f[1])

    # 调整gt心率与预测心率时刻长度一致
    # hr_gt = bpmGT / 60
    # TN = len(hmm_t)
    # if len(hr_gt) < TN:
    #     hr_gt = scipy.ndimage.zoom(hr_gt, TN / len(hr_gt), order=3)
    # if len(hr_gt) > TN:
    #     hr_gt = hr_gt[np.linspace(0, len(hr_gt) - 1, num=TN, dtype=np.int16)]

    hr_tracker = AMTC()
    hr_tracker.set_parameters(spectrum=rppg_spec[:f_length, :], frequency=hmm_f[:f_length], ti=hmm_t, num=1,
                           threshold=threshold, ground_truth=hr_gt, scale=1)
    hr_tracker.show_traces = False
    hr_tracker.show_groundtruth = False
    _, _, _ = hr_tracker.run_AMTC(lam=lam)
    # hr_road = hr_tracker.get_roads()[-1]
    # hr_hmm = np.array([hmm_f[i] * 60 for i in hr_road])
    # ti_index = [np.argmin(np.abs(ti - hmm_t)) for ti in ti_gt]
    # hr_hmm_es = hr_hmm[ti_index]
    # ti_hmm_es = hmm_t[ti_index]
    # print('AMTC_SpecSub used: ', time.time() - str_time)

    # hr_tracker.show()
    return hr_tracker.get_AMTC_HR()[0]

def get_amtc_ref(s,fs,d_len):
    s = signal.resample(s,int(len(s)*(30/fs)))
    r = AMTC_method(s,None)
    s = signal.resample(r ,d_len)
    return s/60


if __name__ == "__main__":
    import utils.sp as sp
    s = np.load("temp.npy")
    dft=sp.DFT(60000,[30/60,200/60],300)
    spec = dft.get_spectrogram(s,300*20)
    idxs = dft.spec_get_max_idx(spec)
    freqs = dft.toFreq(idxs)
    plt.plot(get_amtc_ref(s,300,len(freqs)))
    plt.plot(freqs)
    plt.show()
