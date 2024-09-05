import numpy as np
import sys
from colossus.cosmology import cosmology
from scipy.stats import poisson

import gen_mock_halo
import source_qso
import source_sn

#
# make source table
#
# Define a function named make_srctab which takes in five arguments:
# mmax (maximum magnitude of source objects), fov (field of view),
# flag_type_min (minimum type of source object), flag_type_max (maximum type of source object),
# and cosmo (cosmology model)


def make_srctab(mmax, fov, flag_type_min, flag_type_max, cosmo):
    # Define the step sizes for magnitudes and redshifts
    dm = 0.02
    dz = 0.01

    # Create arrays of magnitudes and redshifts using the defined steps
    m = np.arange(14.0 + 0.5 * dm, mmax, dm)
    z = np.arange(0.1, 5.499, dz)

    # Use Numpy's meshgrid to create a grid of all possible combinations of magnitudes and redshifts
    mm, zz = np.meshgrid(m, z)

    # Flatten the meshgrid arrays into 1D arrays
    fmm = mm.flatten()
    fzz = zz.flatten()

    m_all = []
    z_all = []
    f_all = []

    # Loop through each source object type
    for i in range(flag_type_min, flag_type_max + 1):
        # Calculate expected number of source objects per magnitude bin and per redshift bin based on the
        # input field of view and given cosmology model
        nn = fov * dndzdmobs(fmm, fzz, i, cosmo) * dm * dz
        # Generate random Poisson numbers for each source object in each magnitude bin and redshift bin
        # based on the expected number of source objects
        n = np.random.poisson(nn)

        # Identify magnitude, redshift, and number of source objects for bins with at least one object
        cut_n = n[n > 0]
        cut_m = fmm[n > 0]
        cut_z = fzz[n > 0]

        for j in range(len(cut_n)):
            m_all.append([cut_m[j]] * cut_n[j])
            z_all.append([cut_z[j]] * cut_n[j])
            f_all.append([i] * cut_n[j])

    m_tab = np.array([x for xx in m_all for x in xx])
    z_tab = np.array([x for xx in z_all for x in xx])
    f_tab = np.array([x for xx in f_all for x in xx])

    return m_tab, z_tab, f_tab

#
# number count of source dndm_obs*dVdzdOmega. Then this function give the "Number" within the bin of [z,z+dz]&&[mag,mag+dmag]
#


def dndzdmobs(m, z, flag_type, cosmo):
    if flag_type == 0:
        ma = source_qso.mtoma_qso(m, z, cosmo)
        return source_qso.lf_func_qso(ma, z, cosmo) * calc_vol(z, cosmo)
    else:
        dm = source_sn.mtodm_sn(m, z, flag_type, cosmo)
        return source_sn.lf_func_sn(dm, z, flag_type, cosmo) * calc_vol(z, cosmo) / (1.0 + z)
#
# calculate comoving colume element dV/dzdOmega
#


def calc_vol(z, cosmo):
    dis = cosmo.angularDiameterDistance(z) / (cosmo.H0 / 100.0)
    drdz = (2997.92458 / ((1.0 + z) * cosmo.Ez(z))) / (cosmo.H0 / 100.0)

    # 3282.806350011744 is 1rad^2 = 3282.806350011744[deg^2]
    # multiply fov[deg^2] to obtain the expected number to be observed after by KA
    return (dis * dis / 3282.806350011744) * drdz * (1.0 + z) * (1.0 + z) * (1.0 + z)


#
# for checks
#
if __name__ == '__main__':
    cosmo = gen_mock_halo.init_cosmo()

    source_qso.set_spline_kcor_qso()
    source_sn.set_spline_kcor_sn(1)

    # print(dndzdmobs(np.array([22.0, 23.0]), np.array([1.4, 2.7]), 2, cosmo))
    make_srctab(27.0, 20000.0, 0, 0, cosmo)
