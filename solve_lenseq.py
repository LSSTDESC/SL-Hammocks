import numpy as np
import scipy.stats as st

import gen_mock_halo
import global_value as g
import solve_lenseq_glafic

# import solve_lenseq_lenstronomy


def calc_image(lens_par, srcs_par, ein, rt_range, flag_mag, cosmo):
    """
    Calculate the properties of lensed images given lens and source parameters.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param lens_par: Parameters defining the lens model. Each lens parameter set should include
                     mass type (str), mass [M_sol/h], position x [arcsec], y [arcsec], ellipticity, position angle [degree],
                     concentration parameter/scale radius [arcsec], etc.
            e.g., lens_par = [
                        ["anfw", zl, M_h, x_h, y_h, e_h, p_h, con_h, 0.0],
                        ["ahern", zl,M_cen, x_cen, y_cen, e_cen, p_cen, tb_cen, 0.0],
                        ["pert", zl, zs, 0.0, 0.0, gg, tgg, 0.0, kap_pert],
                    ]
    :type  lens_par: list

    :param srcs_par: Source parameters including source redshift and position x, y.
            e.g., srcs_par = [zs, sx, sy]
    :type  srcs_par: list

    :param ein: The Einstein radius in units of arcsec
    :type  ein: float

    :param rt_range: The range factor to define the grid size based on the Einstein radius.
            calculated boxsize is defined by ein*(rt_range + 3.0)
    :type  rt_range: float

    :param flag_mag: Flag to determine which magnitude to be used for magnification bias
    :type  flag_mag: int

    :param cosmo: Cosmological model instance
    :type  cosmo: colossus.cosmology

    Returns
    -----------------------------------------------------------------------------------------------
    out_img: list
        List of image positions, magnitude, and time delay (days) derived from solving the lens equation.

    nim: int
        Number of lensed images found.

    sep: float
        Maximum separation between any two images, only relevant if multiple images are present.

    mag: float
        Magnitude based on the flag_mag parameter

    mag_max: float
        Maximum magnification among all images.

    fr: float
        Flux ratio between the second and first brightest images, applicable if there are 2 or 3 images.

    kapgam: list
        List of convergence (kappa), shears (gamma1 and gamma2) values for each image position, and stellar convergence
    """

    out_img, kapgam = solve_lenseq_glafic.solve_lenseq_glafic(lens_par, srcs_par, ein, rt_range, cosmo)

    if not out_img:
        n = 0.0
        a = []
        return out_img, len(out_img), n, n, n, n, a

    xi = [img[0] for img in out_img]
    yi = [img[1] for img in out_img]
    mi = [abs(img[2]) for img in out_img]

    if len(lens_par) > 3:
        half_distance_sub_host = np.sqrt(lens_par[3][3] ** 2 + lens_par[3][4] ** 2) / 2
        xi, yi, mi, indexi = filter_for_subhalos_imgs(xi, yi, mi, half_distance_sub_host)
        out_img = [out_img[i] for i in indexi]
        kapgam = [kapgam[i] for i in indexi]

    nim = len(xi)
    if nim <= 1:
        n = 0.0
        a = []
        mags = out_img[0][2] if nim == 1 else n
        return out_img, nim, n, mags, mags, n, kapgam

    mi.sort(reverse=True)
    mag_tot = sum(mi)
    mag_max = max(mi)

    si = []
    for i in range(nim - 1):
        for j in range(i + 1, nim):
            si.append(((xi[i] - xi[j]) ** 2) + ((yi[i] - yi[j]) ** 2))
    sep = np.sqrt(max(si))

    if flag_mag > 0:
        ii = min(flag_mag, nim - 1)
        ii = max(ii, 2)
        mag = mi[ii - 1]
    else:
        mag = mag_tot

    fr = mi[1] / mi[0] if nim in [2, 3] else 1.0

    return out_img, nim, sep, mag, mag_max, fr, kapgam


def filter_for_subhalos_imgs(x_img, y_img, mag_img, distance):
    """
    Filter out image that are farther than a half distance between a subhalo and the host halo.
    Remove the image which satisfies,
            |\theta_i^{sub}|>|\theta_{sub2host}|/2

    Parameters
    -----------------------------------------------------------------------------------------------
    :param x_img: List of x-coordinates of images.
    :type  x_img: list of float
    :param y_img: List of y-coordinates of images.
    :type  y_img: list of float
    :param mag_img: List of magnifications of images.
    :type  mag_img: list of float
    :param distance: The threshold distance; a distance between a subhalo and the host halo.
    :type  distance: float

    Returns
    -----------------------------------------------------------------------------------------------
    tuple: x_img_new, y_img_new, mag_img_newfiltered_x_img, filtered_y_img, filtered_mag_img
                (list of float, list of float, list of float)
        A tuple containing the filtered lists of x and y coordinates and magnification within the specified distance.
    """
    # Initialize lists to store the filtered coordinates
    filtered_x_img = []
    filtered_y_img = []
    filtered_mag_img = []
    filtered_index = []
    # Iterate over the coordinate pairs
    for i, (x, y, mag) in enumerate(zip(x_img, y_img, mag_img, strict=True)):
        # Calculate the distance from the origin (0, 0)
        d_img = (x**2 + y**2) ** 0.5

        # If the distance is less than or equal to the threshold, add coordinates to the filtered lists
        if d_img <= distance:
            filtered_x_img.append(x)
            filtered_y_img.append(y)
            filtered_mag_img.append(mag)
            filtered_index.append(i)

    return filtered_x_img, filtered_y_img, filtered_mag_img, filtered_index


def gene_e(n):
    """
    Generate an array of ellipticity values for galaxies following a truncated normal distribution.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param n: The number of ellipticity values to generate.
    :type  n: int

    Returns
    -----------------------------------------------------------------------------------------------
    e: numpy.ndarray
        An array of generated ellipticity values.
    """
    em = 0.3
    se = 0.16
    e = st.truncnorm.rvs((0.0 - em) / se, (0.9 - em) / se, loc=em, scale=se, size=n)
    return e


def gene_e_ang_halo(Mh):
    """
    Generate ellipticity values and position angle for a collection of halos.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param Mh: Array of halo masses in units of M_sol/h
    :type  Mh: numpy.ndarray

    Returns
    -----------------------------------------------------------------------------------------------
    e_h: numpy.ndarray
        An array of generated ellipticity values for the halos.
    p_h: numpy.ndarray
        An array of generated position angle values for the halos in units of degree
    """
    n = len(Mh)
    e = gene_e_halo(Mh)
    p = gene_ang(n)
    return e, p


def gene_e_halo(Mh):
    """
    Generate ellipticity values for a given array of halo masses, based on a fitting function.
    Reference : T. Okabe 2020 Table 3 and Figure 9,  arXiv:2005.11469

    Parameters
    -----------------------------------------------------------------------------------------------
    :param Mh: Array of halo masses in units of M_sol/h
    :type  Mh: numpy.ndarray

    Returns
    -----------------------------------------------------------------------------------------------
    elip: numpy.ndarray
        An array of generated ellipticity values for the halos
    """
    log10Mh = np.log10(Mh)  # log10([Modot/h])
    elip_fit = 0.09427281271709388 * log10Mh - 0.9477721865885471
    se = 0.13
    n = len(Mh)
    elip_fit[elip_fit < 0.233] = 0.233
    elip = st.truncnorm.rvs((0.0 - elip_fit) / se, (0.9 - elip_fit) / se, loc=elip_fit, scale=se, size=n)
    return elip


def gene_ang(n):
    """
    Generate an array of random angles uniformly distributed between -180 and 180 degrees.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param n: The number of random angles to generate.
    :type  n: int

    Returns
    -----------------------------------------------------------------------------------------------
    angles: numpy.ndarray
        An array of `n` random angles, where each angle is in the range [-180, 180) degrees.
    """
    return (np.random.rand(n) - 0.5) * 360.0


def gene_ang_gal(pol_h):
    """
    Generate an array of galaxy position angles with a Gaussian distribution around the input halo position angles.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param pol_h: The input array of halo position angles which the galaxy angles are based on in units of degree
    :type  pol_h: numpy.ndarray or list

    Returns
    -----------------------------------------------------------------------------------------------
    pol_gal: numpy.ndarray
        An array of galaxy position angles with a Gaussian spread centered around the corresponding halo position angles
        in units of degree
    """
    n = len(pol_h)
    sig = 35.4  # Okumura + 2009
    pol_gal = np.random.normal(loc=pol_h, scale=sig, size=n)
    return pol_gal


def gene_gam(z):
    """
    Generate gamma value(s), which is the amplitude of the shear, for a given redshift based on truncated normal distributions.
    Reference: Oguri 2018, arXiv:1807.02584


    Parameters
    -----------------------------------------------------------------------------------------------
    :param z: The redshift at which to generate the gamma value.
    :type  z: float

    Returns
    -----------------------------------------------------------------------------------------------
    gamma: float
        The generated amplitude of the shear, calculated from the quadrature sum of two components
        drawn from truncated normal distributions.
    """
    if z < 1.0:
        sig = 0.023 * z
    else:
        sig = 0.023 + 0.032 * np.log(z)

    g1 = st.truncnorm.rvs(-0.5 / sig, 0.5 / sig, loc=0.0, scale=sig)
    g2 = st.truncnorm.rvs(-0.5 / sig, 0.5 / sig, loc=0.0, scale=sig)

    return np.sqrt(g1 * g1 + g2 * g2)


def set_shear(z):
    """
    Set the shear and its angle for a given redshift by generating values from appropriate distributions.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param z: The redshift at which to set the shear and angle.
    :type  z: float

    Returns
    -----------------------------------------------------------------------------------------------
    tuple: (float, float)
        A tuple containing the shear magnitude 'gg' and the shear angle 'tgg' in units of degree
    """
    gg = gene_gam(z)
    tgg = gene_ang(1)[0]
    return gg, tgg


#
# for checks
#
if __name__ == "__main__":
    cosmo = gen_mock_halo.init_cosmo()
    print(cosmo.Om0, round(1.0 + g.nonflat - cosmo.Om0, 5), g.cosmo_weos, round(cosmo.H0 / 100.0, 5))
