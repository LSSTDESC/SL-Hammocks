import numpy as np
import sys
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as _spline

import gen_mock_halo

#
# read K-correction data file
#


def set_spline_kcor_sn(btype):

    if (btype < 1) or (btype > 7):
        sys.exit('ERROR: kcor_sn btype out of range')

    rfiles = ['data_Kcor/kc_Bi.dat', 'data_Kcor/kc_Bz.dat', 'data_Kcor/kc_BZ.dat',
              'data_Kcor/kc_BY.dat', 'data_Kcor/kc_BJ.dat', 'data_Kcor/kc_BH_new.dat', 'data_Kcor/kc_BK.dat']

    rfile = rfiles[btype - 1]
    data = np.loadtxt(rfile, comments='#')

    zz = data[:, 0]

    global splines
    splines = [''] * 5

    for i in range(5):
        splines[i] = 'kcor_sn_spline' + '%d' % i
        splines[i] = _spline(zz, data[:, i + 1], k=3)

    return

#
# K_ii(z) for SNe [1: Ia, 2: Ibc, 3: IIP, 4: IIL, 5: IIn]
#


def kcor_calc(z, flag):

    if (flag < 1) or (flag > 5):
        sys.exit('ERROR: kcor_sn flag out of range')

    return splines[flag - 1](z)

#
# SN LF
#


def snrate_ia_func(td, t_ia, cosmo):
    al = 1.08
    eta = 0.04

    tt = t_ia - td

    if tt > 0.0:
        f = (td ** ((-1.0) * al)) / (((ztot(0.0, cosmo) **
                                       (1.0 - al)) - (0.1 ** (1.0 - al))) / (1.0 - al))

        return f * eta * 0.032 * sfr_rate(ttoz(tt, cosmo), cosmo)
    else:
        return 0.0


def snrate_ia(t_ia, cosmo):
    val, err = integrate.quad(snrate_ia_func, 0.1, t_ia, args=(t_ia, cosmo))

    return val


def snrate_sn(z, flag_type, cosmo):

    if flag_type == 1:
        set_spline_snrate_ia(z, cosmo)

        return snrate_ia_spline(z)

    else:
        ftab = np.array([0.23, 0.30, 0.30, 0.02])
        f = ftab[flag_type - 2]

        return f * 0.0132 * sfr_rate(z, cosmo)


def set_spline_snrate_ia(z, cosmo):
    t_ia = ztot(z, cosmo)
    t_ia_max = t_ia.max()
    t_ia_min = t_ia.min()

    t_ia_spl = np.arange(t_ia_max, t_ia_min - 0.2, -0.2)
    z_spl = ttoz(t_ia_spl, cosmo)
    snrate_spl = np.zeros_like(t_ia_spl)
    for i in range(len(snrate_spl)):
        snrate_spl[i] = snrate_ia(t_ia_spl[i], cosmo)

    global snrate_ia_spline
    snrate_ia_spline = _spline(z_spl, snrate_spl, snrate_spl, k=3)

    return


def lf_func_sn(dm, z, flag_type, cosmo):
    snr = snrate_sn(z, flag_type, cosmo)

    mstab = np.array([0.56, 1.39, 1.12, 0.90, 0.92])
    ms = mstab[flag_type - 1]

    return snr * (0.3989 / ms) * np.exp((-0.5) * dm * dm / (ms * ms))

#
# magnitude conversions
#


def mag_peak_sn(flag_type, cosmo):
    mptab = np.array([-19.06, -17.64, -16.60, -17.63, -18.75])
    hh = 5.0 * np.log10(cosmo.H0 / 72.0)

    return mptab[flag_type - 1] + hh


def mtodm_sn(m, z, flag_type, cosmo):
    dismod = 5.0 * np.log10(cosmo.luminosityDistance(z) /
                            (cosmo.H0 / 100.0)) + 25.0
    kcor = kcor_calc(z, flag_type)
    ma = m - dismod - kcor

    return ma - mag_peak_sn(flag_type, cosmo)


def dmtom_sn(dm, z, flag_type, cosmo):
    ma = mag_peak_sn(flag_type, cosmo) + dm
    dismod = 5.0 * np.log10(cosmo.luminosityDistance(z) /
                            (cosmo.H0 / 100.0)) + 25.0
    kcor = kcor_calc(z, flag_type)

    return ma + dismod + kcor

#
# SFR from Hopkins & Beacom, BG IMF
# caution: only for flat LCDM model
#


def sfr_rate(z, cosmo):
    a = 0.0118
    b = 0.08
    c = 3.3
    d = 5.2

    return (a + b * z) * (cosmo.H0 / 100.0) / (1.0 + ((z / c) ** d))


def ztot(z, cosmo):
    s1 = np.sqrt(1.0 + cosmo.Om0 * z * (z * z + 3.0 * z + 3.0))
    s2 = np.sqrt(1.0 - cosmo.Om0)
    s3 = np.sqrt(cosmo.Om0 * (1.0 + z) * (1.0 + z) * (1.0 + z))
    x = np.log((s1 + s2) / s3)
    h0t = 2.0 * x / (3.0 * np.sqrt(1.0 - cosmo.Om0))

    return (977.76 / cosmo.H0) * h0t


def ttoz(t, cosmo):
    x = np.sinh(1.5 * np.sqrt(1.0 - cosmo.Om0) * (t * cosmo.H0 / 977.76))
    a = ((cosmo.Om0 * x * x / (1.0 - cosmo.Om0)) ** (1.0 / 3.0))

    return 1.0 / a - 1.0


#
# for checks
#
if __name__ == '__main__':
    cosmo = gen_mock_halo.init_cosmo()

    set_spline_kcor_sn(1)
    print(sfr_rate(np.array([0.5, 1.0, 1.5]), cosmo))
    print(snrate_sn(np.arange(0.0, 5.0, 0.01), 2, cosmo))
