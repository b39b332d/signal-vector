import numpy as np
from pyampd.ampd import find_peaks
from scipy import signal
def findPeaks(sig, scale):
    return findExtreams(sig, scale)[1]


def findtroughs(sig, scale):
    return findExtreams(sig, scale)[0]

def findExtreams(sig, scale):
    #peak_xs = find_peaks(sig, scale)
    peak_xs = signal.find_peaks(sig,distance=scale)[0]
    #peak_xs = signal.find_peaks_cwt(sig,np.arange(1,10),gap_thresh=scale)

    peak_ids = np.full(len(peak_xs),0)
    troughs_xs = signal.find_peaks(-sig,distance=scale)[0]
    troughs_ids = np.full(len(troughs_xs),1)
    extreams_xs = np.append(np.array([peak_xs,peak_ids]),np.array([troughs_xs,troughs_ids]),1)
    extreams_xs = ((extreams_xs.T)[extreams_xs[0].argsort()]).T
    extreams_xs_append = np.append([[-1],[1]],extreams_xs,1)

    trough_xs  = extreams_xs[0,(extreams_xs_append[1,1:] - extreams_xs_append[1,:-1]) == 1]
    peak_xs  = extreams_xs[0,(extreams_xs_append[1,1:] - extreams_xs_append[1,:-1]) == -1]
    return trough_xs,peak_xs

