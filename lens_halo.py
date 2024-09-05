import glafic
import numpy as np
from colossus.cosmology import cosmology
from colossus.halo import concentration
from colossus.halo import mass_adv
from colossus.halo import mass_so
from colossus.lss import mass_function
from scipy import interpolate
from scipy import optimize

import gen_mock_halo
import global_value as g
import lens_gals
import lens_subhalo
import solve_lenseq
import source_tab

#
# number counts and properties of lens halo
#


def dNhalodzdlnM_lens(MM_h, z, cosmo):
    """
    Calculate the differential number of halos with respect to redshift,
    the natural logarithm of halo mass, and the solid angle.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param MM_h: Mass (array of masses of float) for which to calculate the mass function in units of M_sol/h
    :type  MM_h: float or ndarray

    :param z: Redshift at which to calculate the volume element and mass function.
    :type  z: float

    :param cosmo: cosmological model instance
    :type  cosmo: colossus.cosmology

    Returns
    -----------------------------------------------------------------------------------------------
    dNdzdlnM: ndarray
        The differential number of halos
        d^3N/dlnM/dz/dOmega [number/deg^2/dlnM[Modot/h]]
    """
    dVdzdOmega = source_tab.calc_vol(z, cosmo)
    hubble3 = (cosmo.H0 / 100.0) ** 3
    # mfunc_so = mass_function.massFunction(MM_h, z, mdef="fof", model="sheth99", q_out="dndlnM") * hubble3
    mfunc_so_200c = mass_function.massFunction(MM_h, z, mdef="200c", model="tinker08", q_out="dndlnM") * hubble3
    return dVdzdOmega * mfunc_so_200c


def concent_m(m, z):
    return concentration.concentration(m, "vir", z, model="diemer19")


def concent_m_w_scatter(m_h, zl, sig):
    """
    Calculate the concentration of host halos with scatter for a given mass and redshift.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param mass: Host halo mass (or array of halo masses) in units of M_sol/h
    :type  mass: float or ndarray

    :param redshift: Redshift at which to calculate the concentration
    :type  redshift: float

    :param sigma: Standard deviation for the log-normal scatter in concentration parameter
    :type  sigma: float

    Returns
    -----------------------------------------------------------------------------------------------
    concentration_with_scatter: ndarray
        Concentration of each halo adjusted by a log-normal scatter
    """
    con_h_mean = concentration.concentration(m_h, "vir", zl, model="diemer19")
    scatter = np.random.lognormal(0.0, sig, len(m_h))
    return con_h_mean * scatter


def halo_properties_200c2vir(mh_200c, zl, sig):
    """
    Calculate halo properties with virial definitions from M200c.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param mh_200c: Halo masses defined at 200 times the critical density in units of M_sol/h
    :type  mh_200c: ndarray

    :param zl: The redshift of the lens.
    :type  zl: float

    :param sig: The standard deviation for lognormal scatter to apply to the concentration.
    :type  sig: float

    Returns
    -----------------------------------------------------------------------------------------------
    mh_vir: ndarray
        Halo masses defined with respect to the virial overdensity in units of M_sol/h

    r_vir: ndarray
        Virial radii corresponding to the virial masses, in units of Mpc/h.

    con_vir: ndarray
        Concentrations corresponding to the virial masses with applied scatter.

    e_h: ndarray
        Ellipticities of the halos.

    p_h: ndarray
        Position angles of the halos, in degrees.
    """
    mh_vir, r_vir_kpc, con_vir = mass_adv.changeMassDefinitionCModel(mh_200c, zl, "200c", "vir")
    con_vir = con_vir * np.random.lognormal(0.0, sig, len(mh_vir))
    r_vir = 1.0e-3 * r_vir_kpc
    e_h, p_h = solve_lenseq.gene_e_ang_halo(mh_vir)
    return mh_vir, r_vir, con_vir, e_h, p_h


def kappa_ext_from_host_halo(tx, ty, zl, zs, mass_host, rs_h, con_h, mcen, rb_cen, e_h, p_h, e_cen, p_cen):
    """
    Calculate the external convergence (kappa_ext) from a host halo and a central galaxy at a given subhalo's
    position

    Parameters
    -----------------------------------------------------------------------------------------------
    :param tx: x-coordinate in the image plane (theta_x) in units of arcsec
    :type  tx: float

    :param ty: y-coordinate in the image plane (theta_y) in units of arcsec
    :type  ty: float

    :param zl: Redshift of the lens.
    :type  zl: float

    :param zs: Redshift of the source.
    :type  zs: float

    :param mass_host: Mass of the host halo in units of M_sol/h, basically this is set to be m_h - m_sh
    :type  mass_host: float

    :param rs_h: Scale radius of the host halo in units of Mpc/h
    :type  rs_h: float

    :param con_h: Concentration of the host halo.
    :type  con_h: float

    :param mcen: Mass of the central galaxy in units of M_sol/h
    :type  mcen: float

    :param rb_cen: Scale radius of the central galaxy in units of Mpc/h
    :type  rb_cen: float

    :param e_h: Ellipticity of the host halo.
    :type  e_h: float

    :param p_h: Position angle of the host halo in degrees.
    :type  p_h: float

    :param e_cen: Ellipticity of the central galaxy
    :type  e_cen: float

    :param p_cen: Position angle of the central galaxy in degrees.
    :type  p_cen: float

    Returns
    -----------------------------------------------------------------------------------------------
    kext_zs: float
        The total external convergence from both the host halo and central galaxy at the given
        coordinates and redshifts.
    """
    cosmo = cosmology.getCurrent()
    convert_t = 1.0 / cosmo.angularDiameterDistance(zl) * 206264.8
    si = np.sin(p_h * np.pi / 180.0)
    co = np.cos(p_h * np.pi / 180.0)
    ddx = (co * tx + si * ty) / convert_t
    ddy = (-si * tx + co * ty) / convert_t
    xyr_sh_hh = np.sqrt(ddx * ddx / (1.0 - e_h) + ddy * ddy * (1.0 - e_h))

    kext_hh_zs = bnorm_nfw(mass_host, con_h, zl, zs, cosmo) * kappa_dl_nfw(xyr_sh_hh / rs_h)
    si = np.sin(p_cen * np.pi / 180.0)
    co = np.cos(p_cen * np.pi / 180.0)
    ddx = (co * tx + si * ty) / convert_t
    ddy = (-si * tx + co * ty) / convert_t
    xyr_sh_cen = np.sqrt(ddx * ddx / (1.0 - e_cen) + ddy * ddy * (1.0 - e_cen))
    kext_cen_zs = bnorm_hern(mcen, rb_cen, zl, zs, cosmo) * kappa_dl_hern(xyr_sh_cen / rb_cen)
    kext_zs = kext_hh_zs + kext_cen_zs
    return kext_zs


def critical_surface_density(zl, zs, cosmo):
    """
    Calculate the critical surface density for lensing given the redshifts of the lens and source.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param lens_redshift: Redshift of the lensing galaxy
    :type  lens_redshift: float

    :param source_redshift: Redshift of the background source galaxy
    :type  source_redshift: float

    :param cosmo: cosmological model instance
    :type  cosmo: colossus.cosmology

    Returns
    -----------------------------------------------------------------------------------------------
    sigma_critical: float
        The critical surface density for lensing in units of mass per area
    """
    rl = cosmo.comovingDistance(z_max=zl)
    rs = cosmo.comovingDistance(z_max=zs)
    dol = (1.0 / (1.0 + zl)) * rl
    dos = (1.0 / (1.0 + zs)) * rs
    dls = (1.0 / (1.0 + zs)) * (rs - rl)
    sig_cr = g.c2_G / 4.0 / np.pi * dos / dol / dls
    return sig_cr


def bnorm_hern(m_g, rb, zl, zs, cosmo):
    """
    Calculate the normalization of the convergence kappa for a Hernquist profile

    Parameters
    -----------------------------------------------------------------------------------------------
    :param m_g: stellar mass in units of M_sol/h
    :type  m_g: float

    :param rb: scale radius of the Hernquist profile
    :type  rb: float

    :param zl: redshift of the lens
    :type  zl: float

    :param zs: redshift of the source
    :type  zs: float

    :param cosmo: cosmological model instance
    :type  cosmo: colossus.cosmology

    Returns
    -----------------------------------------------------------------------------------------------
    b_norm: float
        Normalization of the convergence kappa for a Hernquist mass distribution
    """
    s_cr = critical_surface_density(zl, zs, cosmo)
    b_norm = m_g / 2.0 / np.pi / rb**2 / s_cr
    return b_norm


def mtot_nfw(con):
    """
    Calculate the normalized mass of a Navarro-Frenk-White (NFW) profile within the characteristic
    scale radius.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param con: concentration parameter of the NFW profile
    :type  con: float or ndarray

    Returns
    -----------------------------------------------------------------------------------------------
    mtot: float or ndarray
        The normalized mass within the characteristic scale radius, given the concentration parameter c
    """
    return np.log(1.0 + con) - (con / (1.0 + con))


def F_nfw(uu):
    """
    Calculate the function F(u) for a Navarro-Frenk-White (NFW) and Hernquist profile.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param uu: dimensionless radial distance from the center of the lensing cluster scaled by
               the scale radius
    :type  uu: float or ndarray

    Returns
    -----------------------------------------------------------------------------------------------
    F: float or ndarray
        The value of the function F(u) corresponding to the given dimensionless radial distance uu
    """
    if uu < 1:
        F = (1.0 / (np.sqrt(1.0 - uu ** (2)))) * np.arctanh(np.sqrt(1.0 - uu ** (2)))
    else:
        F = (1.0 / (np.sqrt(uu ** (2) - 1.0))) * np.arctan(np.sqrt(uu ** (2) - 1.0))
    return F


def kappa_dl_nfw(uu):
    """
    Calculate the normalized convergence for a Navarro-Frenk-White (NFW) profile.
    kappa = kappa_dl * bnorm

    Parameters
    -----------------------------------------------------------------------------------------------
    :param uu: dimensionless radial distance from the center of the lensing cluster scaled by
               the scale radius of the NFW profile
    :type  uu: float

    Returns
    -----------------------------------------------------------------------------------------------
    kappa_dl: float
        The calculated normalized convergence (kappa) for the given radial distance uu
    """
    if uu == 1:
        kappa_dl = 0.16667
    else:
        F_u = F_nfw(uu)
        kappa_dl = 1.0 / (2 * (uu ** (2.0) - 1.0)) * (1.0 - F_u)
    return kappa_dl


def kappa_dl_hern(uu):
    """
    Calculate normalized convergence for a Hernquist profile.
    kappa = kappa_dl * bnorm

    Parameters
    -----------------------------------------------------------------------------------------------
    :param uu: dimensionless distance from the center of the lens (impact parameter) scaled by
               the scale radius of the Hernquist profile
    :type  uu: float

    Returns
    -----------------------------------------------------------------------------------------------
    kappa_dl: float
        The calculated normalized convergence for the given impact parameter uu
    """
    if uu == 1:
        kappa_dl = 0.26667
    else:
        F_u = F_nfw(uu)
        kappa_dl = 1.0 / ((uu ** (2.0) - 1.0) ** (2.0)) * (-3.0 + (2.0 + uu ** (2.0)) * F_u)
    return kappa_dl


def bnorm_nfw(Mhalo, con, zl, zs, cosmo):
    r"""
    Calculate the normalization factor of convergence defined as
        \kappa(r)=b_{norm} \kappa_{\mathrm{dl}}(u)
    for an NFW (Navarro-Frenk-White) profile.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param Mhalo: Mass of the dark matter halo in units of M_sol/h
    :type  Mhalo: float

    :param con: Concentration parameter of the halo.
    :type  con: float

    :param zl: Redshift of the lens galaxy/halo.
    :type  zl: float

    :param zs: Redshift of the source galaxy/background galaxy.
    :type  zs: float

    :param cosmo: An instance of the cosmological model to use.
    :type  cosmo: astropy.cosmology object

    Returns
    -----------------------------------------------------------------------------------------------
    :return: The normalization constant for the NFW profile.
    :rtype : float

    Note: This function relies on external functions `critical_surface_density`,
    `mass_so.M_to_R`, `mass_so.deltaVir`, and `mtot_nfw` as well as constants from a module `g`.
    These need to be defined in the same scope or imported properly for the function to work as expected.
    """
    sigma_cr = critical_surface_density(zl, zs, cosmo)
    rvir = 1.0e-3 * mass_so.M_to_R(Mhalo, zl, "vir")
    rs = rvir / con
    rho_s = mass_so.deltaVir(zl) * (cosmo.rho_c(zl) / g.kpc_to_Mpc**3) * con**3 / 3.0 / mtot_nfw(con)
    return 4 * rs * rho_s / sigma_cr


def create_interpolator3(A_values, B_values, C_values, precomputed_grid):
    """
    Creates a 3-dimensional regular grid interpolator.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param A_values: Sample points along the first dimension.
    :type  A_values: array_like

    :param B_values: Sample points along the second dimension.
    :type  B_values: array_like

    :param C_values: Sample points along the third dimension.
    :type  C_values: array_like

    :param precomputed_grid: Values of a function at all points defined by the grid of A_values,
                             B_values, and C_values. The layout of values in this array should
                             match the meshgrid created from (A_values, B_values, C_values).
    :type  precomputed_grid: ndarray

    Returns
    -----------------------------------------------------------------------------------------------
    :return: An interpolator that can be called with an array of points to retrieve interpolated values.
    :rtype : RegularGridInterpolator object

    Note: This function requires the `interpolate` module from `scipy`.
          Make sure to import it before using this function.
    """
    interpolator = interpolate.RegularGridInterpolator((A_values, B_values, C_values), precomputed_grid)
    return interpolator


def func_mag(x, y):
    """
    Calculate the value of magnification at a given position using glafic.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param x: The x-coordinate at which to calculate the image.
    :type  x: float

    :param y: The y-coordinate at which to calculate the image.
    :type  y: float

    Returns
    -----------------------------------------------------------------------------------------------
    :return: the magnification at the specified coordinates.
    :type : float

    """
    img = glafic.calcimage(g.zsmax, x, y, verb=0)
    return 1.0 / np.abs(img[6])


def func_magx_root(x, mag):
    """
    Function to find the solution x to give magnification=mag through optimize.root_scalar()

    Parameters
    -----------------------------------------------------------------------------------------------
    :param x: x-coordinate on the lens plane in units of arcsec
    :type  x: float

    :param mag: The magnification
    :type  mag: float
    """
    return func_mag(x, 0.0) - mag


def func_magy_root(y, mag):
    """
    Function to find the solution y to give magnification=mag through optimize.root_scalar()

    Parameters
    -----------------------------------------------------------------------------------------------
    :param y: y-coordinate on the lens plane in units of arcsec
    :type  y: float

    :param mag: The magnification
    :type  mag: float
    """
    return func_mag(0.0, y) - mag


def func_kap(x, y):
    """
    Calculate the  value of convergence at a given position using glafic.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param x: The x-coordinate at which to calculate the image in units of arcsec
    :type  x: float

    :param y: The y-coordinate at which to calculate the image in units of arcsec
    :type  y: float

    Returns
    -----------------------------------------------------------------------------------------------
    :return: the convergence at the specified coordinates.
    :type : float

    """
    img = glafic.calcimage(g.zsmax, x, y, verb=0)
    return img[3]


def func_kapx_root(x, kap):
    """
    Function to find the solution x to give convergence=kap through optimize.root_scalar()

    Parameters
    -----------------------------------------------------------------------------------------------
    :param x: x-coordinate on the lens plane in units of arcsec
    :type  x: float

    :param kap: The convergence
    :type  kap: float
    """
    return func_kap(x, 0.0) - kap


def func_kapy_root(y, kap):
    """
    Function to find the solution y to give convergence=kap through optimize.root_scalar()

    Parameters
    -----------------------------------------------------------------------------------------------
    :param y: y-coordinate on the lens plane in units of arcsec
    :type  y: float

    :param kap: The convergence
    :type  kap: float
    """
    return func_kap(0.0, y) - kap


def create_interp_bsrc_h(z_min, z_max, Mh_min, Mh_max, cosmo):
    """
    Create an interpolator of the source plane boundary as a function of redshift and halo mass
    for host halos.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param z_min: Minimum redshift value
    :type  z_min: float

    :param z_max: Maximum redshift value
    :type  z_max: float

    :param Mh_min: Minimum halo mass in units of M_sol/h
    :type  Mh_min: float

    :param Mh_max: Maximum halo mass in units of M_sol/h
    :type  Mh_max: float

    :param cosmo: Cosmological class instance
    :type  cosmo: colossus.cosmology

    Returns
    -----------------------------------------------------------------------------------------------
    interp_bsrc_h: callable
        A callable object that can be used to interpolate the source plane boundary values
        given arrays of log halo mass and redshift in units of arcsec
    """
    halo_masses = np.logspace(np.log10(Mh_min), np.log10(Mh_max), 50)
    redshifts = np.linspace(z_min, z_max, 50)

    # derive boundary box size in the source plane
    eh = 0.8
    ph = 0.0
    es = 0.8
    ps = 0.0

    src_y_ar = np.zeros((len(halo_masses), len(redshifts)))

    for i, mh in enumerate(halo_masses):
        for k, z in enumerate(redshifts):
            mcen = (
                lens_gals.stellarmass_halomass(mh / (cosmo.H0 / 100.0), z, g.paramc, g.frac_SM_IMF)
                * 10 ** (g.sig_mcen)
                * (cosmo.H0 / 100.0)
            )
            c = concent_m(mh, z) * 10 ** (g.sig_c)
            tb = lens_gals.galaxy_size(mh, mcen / g.frac_SM_IMF, z, cosmo, model=g.TYPE_GAL_SIZE)
            comega, clambda, cweos, chubble = gen_mock_halo.calc_cosmo_for_glafic(cosmo)
            glafic.init(comega, clambda, cweos, chubble, "out2", -20.0, -20.0, 20.0, 20.0, 0.2, 0.2, 5, verb=0)

            glafic.startup_setnum(2, 0, 0)
            glafic.set_lens(1, "anfw", z, mh, 0.0, 0.0, eh, ph, c, 0.0)
            glafic.set_lens(2, "ahern", z, mcen, 0.0, 0.0, es, ps, tb, 0.0)

            # model_init needs to be done again whenever model parameters are changed
            glafic.model_init(verb=0)

            kap_th = 0.45

            zz = optimize.root_scalar(func_kapy_root, method="brentq", args=(kap_th), bracket=(1.0e-4, 2000.0))
            kap_y = zz.root

            glafic.calcimage(g.zsmax, 0.0, kap_y, verb=0)

            mag_th = 1.5

            zz = optimize.root_scalar(func_magy_root, method="brentq", args=(mag_th), bracket=(kap_y, 1.0e6))
            box_y = zz.root

            img = glafic.calcimage(g.zsmax, 0.0, box_y, verb=0)
            src_y = box_y - img[1]
            src_y_ar[i, k] = src_y
            glafic.quit()
    log10mmh = np.log10(halo_masses)
    interp_bsrc_h = lens_subhalo.create_interpolator(log10mmh, redshifts, src_y_ar)
    np.savez("result/" + g.prefix + "_interp_bsrc_h.npz", log10mmh, redshifts, src_y_ar)
    return interp_bsrc_h


#
# for checks
#
if __name__ == "__main__":
    cosmo = gen_mock_halo.init_cosmo()
    g.paramc, g.params = lens_gals.gals_init()
    g.prefix = "check"
    interp_bsrc_h = create_interp_bsrc_h(0.1, 3.0, 1e11, 3e16, cosmo)
    test = [13.4, 1.2]
    print(interp_bsrc_h(test))
