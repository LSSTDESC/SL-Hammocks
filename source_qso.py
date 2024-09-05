import numpy as np
import sys
from scipy.interpolate import InterpolatedUnivariateSpline as _spline

import gen_mock_halo

#
# read K-correction data file
#


def set_spline_kcor_qso():

    rfile = 'data_Kcor/kcor_richards06.dat'
    data = np.loadtxt(rfile, comments='#')

    zz = data[:, 0]
    kk = data[:, 1]

    global kcor_qso_spline
    kcor_qso_spline = _spline(zz, kk, k=3)

    return

#
# K_ii(z) for QSOs
#


def kcor_ii(z):

    return kcor_qso_spline(z) - kcor_qso_spline(0.0)

#
# QSO LF
#


def lf_func_qso(ma, z, cosmo):
    h70 = cosmo.H0 / 70.0
    phi = (1.83e-6) * h70 * h70 * h70

    alp = np.zeros_like(z) + 2.58
    alp[z < 3.0] = 3.31

    bet = 1.45

    ms = mstar_qso(z, cosmo)

    return phi / (10.0 ** (0.4 * (1.0 - alp) * (ma - ms)) + 10.0 ** (0.4 * (1.0 - bet) * (ma - ms)))

#
# magnitude conversions
#


def mtoma_qso(m, z, cosmo):
    dismod = 5.0 * np.log10(cosmo.luminosityDistance(z) /
                            (cosmo.H0 / 100.0)) + 25.0
    kcor = kcor_ii(z)

    return m - dismod - kcor


def matom_qso(ma, z, cosmo):
    dismod = 5.0 * np.log10(cosmo.luminosityDistance(z) /
                            (cosmo.H0 / 100.0)) + 25.0
    kcor = kcor_ii(z)

    return ma + dismod + kcor


def mstar_qso(z, cosmo):
    zeta = 2.98
    xi = 4.05
    zs = 1.60

    xx = np.exp(0.5 * xi * z) + np.exp(0.5 * xi * zs)
    fz = (np.exp(zeta * z) * (1.0 + np.exp(xi * zs))) / (xx * xx)

    msg = -21.61 + 5.0 * np.log10(cosmo.H0 / 70.0) - 2.5 * np.log10(fz)

    return msg - 0.255 + 0.187


#
# for checks
#
if __name__ == '__main__':
    cosmo = gen_mock_halo.init_cosmo()
    set_spline_kcor_qso()

    print(kcor_ii(0.101))

    # print(mtoma_qso(24.0, 2.5, cosmo))

    print(lf_func_qso(
        np.array([-25.0, -27.0, -29.0, -21.0]), np.array([2.1, 4.4, 0.5, 5.2]), cosmo))
