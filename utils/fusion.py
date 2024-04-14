import colorsys
from colormath.color_conversions import convert_color
from colormath import color_objects
import numpy as np
import utils.sp as sp
fs = 50
lcut = 36
hcut = 120
def fusion_rgb_red(r,g,b):
    return r
def fusion_rgb_green(r,g,b):
    return g
def fusion_rgb_blue(r,g,b):
    return b

def fusion_hls_hue(r,g,b):
    return np.array([ colorsys.rgb_to_hls(r[i],g[i],b[i])[0] for i in range(len(r))])
def fusion_hls_lightness(r,g,b):
    return np.array([ colorsys.rgb_to_hls(r[i],g[i],b[i])[1] for i in range(len(r))])
def fusion_hls_saturation(r,g,b):
    return np.array([ colorsys.rgb_to_hls(r[i],g[i],b[i])[2] for i in range(len(r))])

def fusion_hsv_hue(r,g,b):
    return np.array([ colorsys.rgb_to_hsv(r[i],g[i],b[i])[0] for i in range(len(r))])
def fusion_hsv_saturation(r,g,b):
    return np.array([ colorsys.rgb_to_hsv(r[i],g[i],b[i])[1] for i in range(len(r))])
def fusion_hsv_brightness(r,g,b):
    return np.array([ colorsys.rgb_to_hsv(r[i],g[i],b[i])[2] for i in range(len(r))])

def fusion_yiq_grey(r,g,b):
    return np.array([ colorsys.rgb_to_yiq(r[i],g[i],b[i])[0] for i in range(len(r))])
def fusion_yiq_I(r,g,b):
    return np.array([ colorsys.rgb_to_yiq(r[i],g[i],b[i])[1] for i in range(len(r))])
def fusion_yiq_Q(r,g,b):
    return np.array([ colorsys.rgb_to_yiq(r[i],g[i],b[i])[2] for i in range(len(r))])



def fusion_lab_L(r,g,b):
    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.LabColor).lab_l for i in range(len(r))])
def fusion_lab_A(r,g,b):
    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.LabColor).lab_a for i in range(len(r))])
def fusion_lab_B(r,g,b):
    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.LabColor).lab_b for i in range(len(r))])



def fusion_lchab_l(r,g,b):
    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.LCHabColor).lch_l for i in range(len(r))])
def fusion_lchab_c(r,g,b):
    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.LCHabColor).lch_c for i in range(len(r))])
def fusion_lchab_h(r,g,b):
    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.LCHabColor).lch_h for i in range(len(r))])


def fusion_lchuv_l(r,g,b):

    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.LCHuvColor).lch_l for i in range(len(r))])
def fusion_lchuv_u(r,g,b):

    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.LCHuvColor).lch_c for i in range(len(r))])
def fusion_lchuv_v(r,g,b):

    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.LCHuvColor).lch_h for i in range(len(r))])


def fusion_luv_l(r,g,b):

    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.LuvColor).luv_l for i in range(len(r))])
def fusion_luv_u(r,g,b):

    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.LuvColor).luv_u for i in range(len(r))])
def fusion_luv_v(r,g,b):

    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.LuvColor).luv_v for i in range(len(r))])

def fusion_xyz_x(r,g,b):

    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.XYZColor).xyz_x  for i in range(len(r))])
def fusion_xyz_y(r,g,b):

    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.XYZColor).xyz_y for i in range(len(r))])
def fusion_xyz_z(r,g,b):

    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.XYZColor).xyz_z for i in range(len(r))])

def fusion_xyY_x(r,g,b):

    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.xyYColor).xyy_x for i in range(len(r))])
def fusion_xyY_y(r,g,b):

    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.xyYColor).xyy_y for i in range(len(r))])
def fusion_xyY_Y(r,g,b):

    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.xyYColor).xyy_Y for i in range(len(r))])

def fusion_cmy_c(r,g,b):

    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.CMYColor).cmy_c for i in range(len(r))])
def fusion_cmy_m(r,g,b):

    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.CMYColor).cmy_m for i in range(len(r))])
def fusion_cmy_y(r,g,b):

    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.CMYColor).cmy_y for i in range(len(r))])

def fusion_ipt_i(r,g,b):

    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.IPTColor).ipt_i for i in range(len(r))])
def fusion_ipt_p(r,g,b):

    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.IPTColor).ipt_p for i in range(len(r))])
def fusion_ipt_t(r,g,b):

    return np.array([convert_color(color_objects.AdobeRGBColor(r[i],g[i],b[i]),color_objects.IPTColor).ipt_t for i in range(len(r))])

def fusion_leastSquare(r,g,b):
    sig_filter = sp.Filter(6, (lcut / 60, hcut / 60), fs)
    raw_sig = np.array([sig_filter.filtfilt(r), sig_filter.filtfilt(g),sig_filter.filtfilt(b)])
    x, y, z = raw_sig
    A_xz = np.vstack((x, np.ones(len(x)))).T
    m_xz, c_xz = np.linalg.lstsq(A_xz, z, rcond=None)[0]
    A_yz = np.vstack((y, np.ones(len(y)))).T
    m_yz, c_yz = np.linalg.lstsq(A_yz, z, rcond=None)[0]
    def lin(z):
        x = (z - c_xz) / m_xz
        y = (z - c_yz) / m_yz
        return x, y
    #zz = np.array([np.mean(z) - np.std(z) * 3, np.mean(z) + np.std(z) * 3])
    xx, yy = lin(1)
    pa = np.array([xx,yy,1])
    #pa,pb = np.array([xx, yy, zz]).T

    return np.sum((np.array([r,g,b]).T) * pa, axis=1) / np.linalg.norm(pa)

def fusion_svd(r,g,b):
    sig_filter = sp.Filter(6, (lcut / 60, hcut / 60), fs)
    raw_sig = np.array([sig_filter.filtfilt(r), sig_filter.filtfilt(g),
                        sig_filter.filtfilt(b)])
    _, _, vv = np.linalg.svd(raw_sig.T)
    return np.sum(np.array([r, g, b]).T * vv[0], axis=1)

def fusion_pca(r,g,b):
    sig_filter = sp.Filter(6, (lcut / 60, hcut / 60), fs)
    raw_sig = np.array([sig_filter.filtfilt(r), sig_filter.filtfilt(g),
                        sig_filter.filtfilt(b)])
    ss, _, _ = np.linalg.svd(raw_sig.T)
    return ss.T[0]

signal_fusion_functions = [fusion_rgb_red,fusion_rgb_green,fusion_rgb_blue,
                           fusion_hls_hue,fusion_hls_lightness,fusion_hls_saturation,
                           fusion_hsv_hue,fusion_hsv_saturation,fusion_hsv_brightness,
                           fusion_yiq_grey,fusion_yiq_I,fusion_yiq_Q,
                           fusion_lab_L,fusion_lab_A,fusion_lab_B,
                           fusion_lchab_l,fusion_lchab_c,fusion_lchab_h,
                           fusion_lchuv_l,fusion_lchuv_u,fusion_lchuv_v,
                           fusion_luv_l,fusion_luv_u,fusion_luv_v,
                           fusion_xyz_x,fusion_xyz_y,fusion_xyz_z,
                           fusion_xyY_x,fusion_xyY_y,fusion_xyY_Y,
                           fusion_cmy_c,fusion_cmy_m,fusion_cmy_y,
                           fusion_ipt_i,fusion_ipt_p,fusion_ipt_t,
                           fusion_leastSquare,fusion_svd,fusion_pca
                           ]
