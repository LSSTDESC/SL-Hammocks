import glafic
import numpy as np
from colossus.halo import mass_so
from colossus.lss import peaks
from scipy import integrate
from scipy import interpolate
from scipy import optimize
from scipy.interpolate import InterpolatedUnivariateSpline as _spline

import gen_mock_halo
import global_value as g
import lens_gals
import lens_halo

#
# number counts and properties of lens subhalo
#

fac_f = 0.5


def deltac_z(z, cosmo):
    """
    Calculate the critical density contrast for spherical collapse at a given redshift in a given cosmology.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param z: The redshift at which to compute the critical density contrast.
    :type  z: float

    :param cosmo: cosmological model instance
    :type  cosmo: colossus.cosmology

    Returns
    -----------------------------------------------------------------------------------------------
    deltac: float
        The critical density contrast for spherical collapse at the specified redshift.
    """
    return (3.0 / 20.0) * ((12.0 * np.pi) ** (2.0 / 3.0)) * (cosmo.Om(z) ** 0.0055) / cosmo.growthFactor(z)


def zf_func(zf, delc_z, sigmaMf, sigmaM, cosmo):
    """
    Equation to calculate the median formation(accretion) redshift zf.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param zf: The formation(accretion) redshift of the subhalo.
    :type  zf: float

    :param delc_z: critical density contrast at redshift z
    :type  delc_z: float

    :param sigmaMf: The variance of the density field smoothed on a scale corresponding to
        the mass multiplying the mass of the host halo by fac_f.
    :type  sigmaMf: float

    :param sigmaM: The variance of the density field smoothed on a scale corresponding to the mass of
               the host halo.
    :type  sigmaM: float

    :param cosmo: cosmological model instance
    :type  cosmo: colossus.cosmology
    """
    return delc_z - deltac_z(zf, cosmo) + 0.974 * np.sqrt((sigmaMf * sigmaMf - sigmaM * sigmaM) / 0.707)


def zf_calc(z, M_h, cosmo):
    """
    Calculate the formation (accretion) redshift of a subhalo given its hostmass and redshift.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param z: The redshift
    :type  z: float

    :param M_h: The mass of the host halo.
    :type  M_h: float

    :param cosmo: cosmological model instance
    :type  cosmo: colossus.cosmology

    Returns
    -----------------------------------------------------------------------------------------------
    z_formation: float
        The estimated formation (accretion) redshift for the subhalo.
    """
    sigmaMf = cosmo.sigma(peaks.lagrangianR(fac_f * M_h))
    sigmaM = cosmo.sigma(peaks.lagrangianR(M_h))
    deltacz = deltac_z(z, cosmo)
    zf = optimize.root_scalar(zf_func, args=(deltacz, sigmaMf, sigmaM, cosmo), method="brentq", bracket=(z, 100.0))

    return zf.root


def mtot_tnfw2(t):
    r"""
    Calculate the enclosed mass of subhalo with a truncated NFW profile.
    correspond to tnfw2_m3d(\infty, t)

    Parameters
    -----------------------------------------------------------------------------------------------
    :param t: The dimensionless parameter related to the truncation radius of the halo.
    :type  t: float

    Returns
    -----------------------------------------------------------------------------------------------
    mtot: float
        The enclosed mass of subhalo with the truncated NFW profile for the given dimensionless parameter.
    """
    t2 = t * t + 1.0
    ff = t * t / (2.0 * t2 * t2 * t2)
    gg = 2.0 * t * t * (t * t - 3.0) * np.log(t) - (3.0 * t * t - 1.0) * (t * t + 1.0 - np.pi * t)

    return ff * gg


def mf_sub_eps(M_shf, z, M_h, zf, cosmo):
    """
    Calculate the mass function subhalo epsilon parameter for a given set of conditions.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param M_shf: The mass(es) of subhalo at formation (accretion) redshift in units of M_sol/h
    :type  M_shf: float or numpy.array

    :param z: The redshift at which the mass function is evaluated.
    :type  z: float

    :param M_h: The mass of the host halo in units of M_sol/h
    :type  M_h: float

    :param zf: The formation redshift of the subhalo.
    :type  zf: float

    :param cosmo: cosmological model instance
    :type  cosmo: colossus.cosmology

    Returns
    -----------------------------------------------------------------------------------------------
    dNeps_dMshf: float or numpy.array
        The derivative of the mass function of subhalo with respect to the subhalo mass at the formation time.
        This can be a scalar or an array depending on whether `sigma_Mshf` is a scalar or an array.
    """
    delc_zf = deltac_z(zf, cosmo)
    delc_z = deltac_z(z, cosmo)

    sigma_Mshf = cosmo.sigma(peaks.lagrangianR(M_shf))
    sigma_Mh = cosmo.sigma(peaks.lagrangianR(M_h))

    dh = 0.005
    dsigma_dMshf = (cosmo.sigma(peaks.lagrangianR(M_shf * (1.0 - dh))) - cosmo.sigma(peaks.lagrangianR(M_shf * (1.0 + dh)))) / (
        2.0 * M_shf * dh
    )

    x = (delc_zf - delc_z) / np.sqrt(2.0 * np.abs(sigma_Mshf * sigma_Mshf - sigma_Mh * sigma_Mh))
    f = (x / np.sqrt(np.pi)) * np.exp(-(1.0) * x * x) / np.abs(sigma_Mshf * sigma_Mshf - sigma_Mh * sigma_Mh)
    ddeltaS_dMshf = 2.0 * sigma_Mshf * dsigma_dMshf

    dNeps_dMshf = f * (M_h / M_shf) * ddeltaS_dMshf

    if sigma_Mshf.size == 1:
        if sigma_Mshf < sigma_Mh:
            dNeps_dMshf = 0.0
    else:
        dNeps_dMshf[sigma_Mshf < sigma_Mh] = 0.0

    return dNeps_dMshf


def sub_m_func(M_sh_ori, M_sh_after, zf, rt):
    """
    Equation to find a solution for tau which is dimensionless truncated radius

    \tau \\equiv \frac{rt}/\frac{rs}

    Parameters
    -----------------------------------------------------------------------------------------------
    :param M_sh_ori: The original mass of the subhalo before tidal effects.
    :type  M_sh_ori: float

    :param M_sh_after: The truncated mass of the subhalo after tidal effects.
    :type  M_sh_after: float

    :param zf: The redshift at which the subhalo formed (accreted)
    :type  zf: float

    :param rt: The truncated radius of the subhal
    :type  rt: float
    """
    # Before tidal effect, this subhalo should be adopted concentration parameter of field halo
    con = lens_halo.concent_m(M_sh_ori, zf)
    rs = 1.0e-3 * mass_so.M_to_R(M_sh_ori, zf, "vir") / con

    return M_sh_ori * mtot_tnfw2(rt / rs) / lens_halo.mtot_nfw(con) - M_sh_after


def t_df(M_h_f, M_shf, z, cosmo):
    """
    Calculate the dynamical friction timescale for a subhalo within a host halo at a given redshift.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param M_h_f: The mass of the host halo muliplied by fac_f in units of M_sol/h
    :type  M_h_f: float

    :param M_shf: The mass of the subhalo at the formation (accretion) time in units of M_sol/h
    :type  M_shf: float

    :param z: The redshift at which to calculate the dynamical friction timescale.
    :type  z: float

    :param cosmo: cosmological model instance
    :type  cosmo: colossus.cosmology

    Returns
    -----------------------------------------------------------------------------------------------
    t_friction: float
        The dynamical friction timescale in Gyr (gigayears) for the subhalo within the host halo.
    """
    hubble = cosmo.H0 * 1.0e-2
    r_vir = 1.0e-3 * mass_so.M_to_R(M_h_f, z, "vir") / hubble
    msc = (1476.6697 * (M_h_f / hubble) / 3.085677581e16) * 1.0e-6
    v_vir = (np.sqrt(msc / r_vir) * (2.99792458e8) / 3.085677581e16) * 1.0e-6
    tdyn = ((r_vir / v_vir) / (3.154e7)) * 1.0e-9

    return 2.0 * (M_h_f / M_shf) * tdyn


def factor_dynamical_friction(M_h, M_shf, z, zf, cosmo):
    """
    Calculate the dynamical friction factor for a subhalo within a host halo between two redshifts.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param M_h: The mass of the host halo at redshift z in units of M_sol/h
    :type  M_h: float

    :param M_shf: The mass of the subhalo at formation redshift zf in units of M_sol/h
    :type  M_shf: float

    :param z: The redshift
    :type  z: float

    :param zf: The redshift at which the subhalo accreted into the host halo.
    :type  zf: float

    :param cosmo: cosmological model instance
    :type  cosmo: colossus.cosmology

    Returns
    -----------------------------------------------------------------------------------------------
    factor: float
        The exponential suppression factor due to dynamical friction experienced by the subhalo.
    """
    if z > zf:
        raise ValueError("Redshift z must be less than or equal to redshift zf")

    dt = cosmo.age(z) - cosmo.age(zf)
    x = dt / t_df(fac_f * M_h, M_shf, zf, cosmo)

    return np.exp((-1.0) * x * x)


def rt_average(M_h_f, M_sh, z):
    r"""
    Calculate the tidal radius for a subhalo within a host halo at a given redshift by
    rt_average = \int dlogx rt_average_integrand

    Parameters
    -----------------------------------------------------------------------------------------------
    :param M_h_f: The mass of the host halo multiplied by fac_f in units of M_sol/h
    :type  M_h_f: float

    :param M_sh: The mass of the subhalo in units of M_sol/h
    :type  M_sh: float

    :param z: The redshift at which to calculate the tidal radius.
    :type  z: float

    Returns
    -----------------------------------------------------------------------------------------------
    rt: float
        The calculated tidal radius for the subhalo within the gravitational potential of the host halo
        in the units of Mpc/h
    """
    con = lens_halo.concent_m(M_h_f, z)
    rs = 1.0e-3 * mass_so.M_to_R(M_h_f, z, "vir") / con
    lx1 = np.log(1.0e-6)
    lx2 = np.log(1.0e4)

    tau = tnfw2_tcalc(con)

    integ, err = integrate.quad(rt_average_integrand, lx1, lx2, args=(M_h_f, tau))

    return rs * integ * (M_sh ** (1.0 / 3.0))


def sub_m_calc(M_h, M_sh_after, zf, log10_M_sh_after2ori_spline):
    """
    Calculate the original mass of a subhalo before tidal stripping and its tidal radius at the time of stripping.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param M_h: The mass of the host halo in units of M_sol/h
    :type  M_h: float

    :param M_sh_after: The mass of the subhalo after stripping in units of M_sol/h
    :type  M_sh_after: float

    :param zf: The formation (accretion) redshift of subhalo
    :type  zf: float

    :param log10_M_sh_after2ori_spline: A spline function for interpolating the logarithm of the
                                  original subhalo mass  in units of M_sol/h as a function
                                  of the logarithm of the subhalo mass after tidal stripping
                                  in units of M_sol/h.
    :type  log10_M_sh_after2ori_spline: scipy.interpolate.UnivariateSpline or similar callable

    Returns
    -----------------------------------------------------------------------------------------------
    - M_sh_ori: float
        The estimated original mass of the subhalo before stripping in units of M_sol/h
    - rt: float
        The calculated tidal radius of the truncated subhalo in units of Mpc/h
    """
    rt = rt_average(fac_f * M_h, M_sh_after, zf)
    M_sh_ori = 10.0 ** log10_M_sh_after2ori_spline(np.log10(M_sh_after))

    return M_sh_ori, rt


def sub_m_calc_setspline(M_h, M_sh_after_ar, zf):
    """
    Create a spline function that relates the mass of subhalos after tidal stripping to their original mass.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param M_h: The mass of the host halo.
    :type  M_h: float

    :param M_sh_after_ar: Array of subhalo masses after tidal stripping.
    :type  M_sh_after_ar: numpy.ndarray

    :param zf: The redshift factor at which the stripping occurs.
    :type  zf: float

    Returns
    -----------------------------------------------------------------------------------------------
    log10M_sh_after2ori_spline: scipy.interpolate.UnivariateSpline
        A spline function that interpolates the logarithm of the original subhalo mass as a function
        of the logarithm of the stripped subhalo mass.
    """
    M_sh_after_min = np.min(M_sh_after_ar) * (10.0 ** (-0.3))
    M_sh_after_max = np.max(M_sh_after_ar) * (10.0**0.301)
    log10M_sh_afters = np.arange(np.log10(M_sh_after_min), np.log10(M_sh_after_max), 0.1)

    r_truncs = rt_average(fac_f * M_h, 10.0**log10M_sh_afters, zf)

    log10M_sh_oris = np.zeros_like(log10M_sh_afters)
    for i in range(len(log10M_sh_afters)):
        M_sh_after = 10.0 ** log10M_sh_afters[i]
        sol = optimize.root_scalar(
            sub_m_func, args=(M_sh_after, zf, r_truncs[i]), method="brentq", bracket=(0.5 * M_sh_after, 100.0 * M_sh_after)
        )
        log10M_sh_oris[i] = np.log10(sol.root)

    return _spline(log10M_sh_afters, log10M_sh_oris, k=5)


def exnum_sh_oguri_w_macc_for_grid(M_h, z, cosmo, Msh_min=1.0e10, n_bins=100):
    """
    Calculate the expected number of subhalos within a host halo for a given mass and redshift,
    considering accretion history and the effects of dynamical friction.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param M_h: The mass of the host halo in units of M_sol/h
    :type  M_h: float

    :param z: The redshift at which the calculation is performed.
    :type  z: float

    :param cosmo: An instance of the cosmological model to use.
    :type  cosmo: colossus.cosmology.Cosmology

    :param Msh_min: Minimum mass of subhalos to consider in the calculation.
    :type  Msh_min: float, optional

    :param n_bins: Number of bins to use when creating the logarithmically spaced subhalo mass array.
    :type  n_bins: int, optional

    Returns
    -----------------------------------------------------------------------------------------------
    - M_sh_after/M_h: Array of the ratios between the subhalo mass after tidal stripping and the host halo mass.
    - M_sh_ori/M_h: Array of the ratios between the original subhalo mass before tidal stripping and the host halo mass.
    - dNeps: Differential expected number of subhalos for each logarithmic interval of M_sh_after_ar[1::]
    """
    M_sh_after_ar = np.logspace(np.log10(Msh_min), np.log10(M_h / 2.0), n_bins)
    zf = zf_calc(z, M_h, cosmo)
    log10M_sh_after2ori_spline = sub_m_calc_setspline(M_h, M_sh_after_ar, zf)
    M_sh_ori_ar, rt = sub_m_calc(M_h, M_sh_after_ar, zf, log10M_sh_after2ori_spline)

    dh = 0.02
    M_sh_orip_ar, rtp = sub_m_calc(M_h, M_sh_after_ar * (1.0 + dh), zf, log10M_sh_after2ori_spline)
    M_sh_orim_ar, rtm = sub_m_calc(M_h, M_sh_after_ar * (1.0 - dh), zf, log10M_sh_after2ori_spline)
    dM_sh_oridM_sh_after = (M_sh_orip_ar - M_sh_orim_ar) / (2.0 * dh * M_sh_after_ar)
    dNepsdM_sh_ori = mf_sub_eps(M_sh_ori_ar, z, M_h, zf, cosmo)
    f_df = factor_dynamical_friction(M_h, M_sh_ori_ar, z, zf, cosmo)
    dNepsdlogM_sh_after = M_sh_after_ar * dNepsdM_sh_ori * dM_sh_oridM_sh_after * f_df
    dNepsdlogM_sh_after = dNepsdlogM_sh_after[1::]
    dlogM_sh_after = np.diff(np.log(M_sh_after_ar), n=1)
    dNeps = dNepsdlogM_sh_after * dlogM_sh_after
    return M_sh_after_ar[1::] / M_h, M_sh_ori_ar[1::] / M_h, dNeps


def concent_m_sub_ando(M_sh_200c_ar, z, cosmo):
    """
    Calculate the concentration parameter array for subhalos given their masses (after tidal stripping)
    and redshift, using the cosmological model provided.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param M_sh_200c_ar: The halo mass(es), M200c, of the subhalos in units of M_sol/h
    :type  M_sh_200c_ar: float or numpy.array

    :param z: The redshift at which to compute the concentration parameters.
    :type  z: float

    :param cosmo: An instance of a cosmological model used for calculations.
    :type  cosmo: colossus.cosmology

    Returns
    -----------------------------------------------------------------------------------------------
    c_vir: float or numpy.array
        The scaled concentration parameter(s) of the subhalos with respect
        to the virial radius.
    """
    ez = cosmo.Ez(z)
    hubble = cosmo.H0 * 1.0e-2

    logM_sh_ar = np.log(M_sh_200c_ar / hubble)
    c200_ar = 94.6609 + logM_sh_ar * (-4.1160 + logM_sh_ar * (0.033747 + logM_sh_ar * (2.0932e-4)))
    if c200_ar.size == 1:
        if c200_ar < 0.1:
            c200_ar = 0.1
    else:
        c200_ar[c200_ar < 0.1] = 0.1

    deltaomega = mass_so.deltaVir(z) * (ez * ez)
    factor = (200.0 / (deltaomega / (ez * ez))) ** (1.0 / 3.0)

    return factor * c200_ar / (ez ** (2.0 / 3.0))


def concentration_subhalo_w_scatter(msh_vir, msh_200c, msh_acc_vir, zl, sig, cosmo):
    """
    Calculate the concentration of subhalos with a scatter applied to the concentration-mass relation.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param msh_vir: Virial masses of subhalos in units of M_sol/h
    :type  msh_vir: ndarray

    :param msh_200c: Masses of subhalos with the definition of M200c in units of M_sol/h
    :type  msh_200c: ndarray

    :param msh_acc_vir: Virial masses of subhalos at the time of accretion in units of M_sol/h
    :type  msh_acc_vir: ndarray

    :param zl: Redshifts corresponding to each subhalo mass.
    :type  zl: float

    :param sig: The standard deviation of the lognormal scatter to be applied to the concentration values.
    :type  sig: float

    :param cosmo: A cosmology instance to calculate concentrations.
    :type  cosmo: Cosmology

    Returns
    -----------------------------------------------------------------------------------------------
    con_sh: ndarray
        Concentrations of subhalos in terms of c_vir = r_vir/r_s with the scatter.
    """
    con_sh_ando_ishiyama = concent_m_sub_ando(msh_200c, zl, cosmo)
    con_sh_diemer19 = lens_halo.concent_m(msh_vir, zl)
    con_sh_ave = np.where(
        con_sh_ando_ishiyama > con_sh_diemer19,
        con_sh_ando_ishiyama,
        con_sh_diemer19,
    )
    cor_con_sh = mass_so.M_to_R(msh_acc_vir, zl, "vir") / mass_so.M_to_R(msh_vir, zl, "vir")
    con_sh = con_sh_ave * cor_con_sh * np.random.lognormal(0.0, sig, len(msh_vir))
    return con_sh


def subhalo_mass_function(mh_vir_ar, zl_ar, min_Msh, interp_dnsh, interp_msh_acc_Mh, length=99):
    """
    Calculate the subhalo mass function and the mass of accreted subhalos for a given halo mass and redshift.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param mh_vir_ar: Array of virial masses of the host halos in units of M_sol/h
    :type  mh_vir_ar: ndarray

    :param zl_ar: Array of lens redshifts corresponding to each host halo virial mass.
    :type  zl_ar: ndarray

    :param min_Msh: The minimum subhalo mass to consider in units of M_sol/h
    :type  min_Msh: float

    :param interp_dnsh: Interpolating function that gives differential number density of subhalos
            as a function of host halo mass and redshift.
    :type  interp_dnsh: callable

    :param interp_msh_acc_Mh: Interpolating function providing the average mass of accreted subhalos
            as a function of host halo mass and redshift.
    :type  interp_msh_acc_Mh: callable

    :param length: Number of bins for subhalo masses between `min_Msh` and half of the host halo virial mass. Default is 99.
    :type  length: int

    Returns
    -----------------------------------------------------------------------------------------------
    mmsh: ndarray
        Log-spaced array of subhalo masses for each bin in units of M_sol/h

    mmsh_acc: ndarray
        Masses of subhalos when they accreted in units of M_sol/h

    NNsh: ndarray
        A 2D array where each element represents the expected number of subhalos in each mass bin for each host halo.
    """
    n_bins = length + 1
    Mhl_zl_tab_vec = np.vstack((np.log10(mh_vir_ar), zl_ar)).T
    dnshp = interp_dnsh(Mhl_zl_tab_vec).reshape(len(mh_vir_ar), length)
    msh_acc_Mh = interp_msh_acc_Mh(Mhl_zl_tab_vec).reshape(len(mh_vir_ar), length)
    dnsh = np.where((msh_acc_Mh > 0.5) | (dnshp < 1.0e-4), 0.0, dnshp)
    mmsh_acc = msh_acc_Mh * mh_vir_ar.reshape(len(mh_vir_ar), -1)
    mmsh = np.logspace(np.log10(min_Msh), np.log10(mh_vir_ar / 2.0), n_bins).T[:, 1:]  # in [Modot/h]
    mmsh = np.round(mmsh, -5)
    NNsh = np.random.poisson(dnsh)
    return mmsh, mmsh_acc, NNsh


def tnfw2_tcalc_func(tau_h, c_h):
    """
    Equation to find tau_h for host halo

    Parameters
    -----------------------------------------------------------------------------------------------
    :param tau_h: The dimensionless truncation scale of a host halo for the truncated NFW profile.
    :type  tau_h: float

    :param c_h: The concentration parameter of a host halo for the NFW profile.
    :type  c_h: float
    """
    return mtot_tnfw2(tau_h) - lens_halo.mtot_nfw(c_h)


def tnfw2_tcalc(c_h):
    """
    Find a solution of tau for truncated NFW profile from concentration parameter

    Parameters
    -----------------------------------------------------------------------------------------------
    :param c_h: The concentration parameter for the NFW profile.
    :type  c_h: float

    Returns
    -----------------------------------------------------------------------------------------------
    tau_h_root: float
        The value of the dimensionless truncation scale (tau_h) that solves the tnfw2_tcalc_func equation
        for the provided concentration parameter c_h.
    """
    sol = optimize.root_scalar(tnfw2_tcalc_func, args=(c_h), method="brentq", bracket=(0.5 * c_h, 10.0 * c_h))
    return sol.root


def enclosed_mass_ratio_bmo(x, tau_h):
    """
    Calculate the spatial distribution of subhalos in truncated NFW halo profile (BMO profile)
    U(r|M,m) multiplied by (4*pi*rs^3 x^4) **due to integrate by dlogx

    Parameters
    -----------------------------------------------------------------------------------------------
    :param x: The dimensionless scaled radius (r/r_s) where the density is to be calculated.
    :type  x: float

    :param tau_h: The dimensionless truncation scale
    :type  tau_h: float

    Returns
    -----------------------------------------------------------------------------------------------
    (4*pi*rs^3*x^3)*U(r|M,m): float
        the spatial distribution of subhalos
    """
    return (
        (x * x / ((1.0 + x) * (1.0 + x)))
        * (tau_h * tau_h / (x * x + tau_h * tau_h))
        * (tau_h * tau_h / (x * x + tau_h * tau_h))
        / mtot_tnfw2(tau_h)
    )


def tnfw2_m3d(x, tau_h):
    """
    Calculate the 3D enclosed mass <x for a truncated NFW halo. (BMO profile with p=2)
    To convert to the dimensional mass, multiply 4pi\rhos rs^3, so that M(<x*rs)=4pi\rhos rs^3* tnfw2_m3d(x,c,t)
    Parameters
    -----------------------------------------------------------------------------------------------
    :param x: dimensionless radius, normalized by the scale radius of NFW profile.
    :type  x: float

    :param tau_h: The dimensionless truncation radius normalized by the scale radius of NFW profile.
    :type  tau_h: float

    Returns
    -----------------------------------------------------------------------------------------------
    dimensionless enclosed mass: float
        The value of the modified 3D mass profile at the specified scaled radius.
    """
    tau_h2 = tau_h * tau_h + 1.0

    ff = tau_h * tau_h / (2.0 * tau_h2 * tau_h2 * tau_h2 * (1.0 + x) * (tau_h * tau_h + x * x))
    gg = tau_h2 * x * (x * (x + 1.0) - tau_h * tau_h * (x - 1.0) * (2.0 + 3.0 * x) - 2.0 * tau_h * tau_h * tau_h * tau_h) + tau_h * (
        x + 1.0
    ) * (tau_h * tau_h + x * x) * (
        2.0 * (3.0 * tau_h * tau_h - 1.0) * np.arctan(x / tau_h)
        + tau_h * (tau_h * tau_h - 3.0) * np.log(tau_h * tau_h * (1.0 + x) * (1.0 + x) / (tau_h * tau_h + x * x))
    )

    return ff * gg


def rt_sub_nom(x, mh, tau_h):
    r"""
    Calculate the normalization of truncation radius of a subhalo within a host halo.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param x: dimensionless radius, normalized by the scale radius of NFW profile.
    :type  x: float

    :param mh: The mass of the host halo in units of M_sol/h
    :type  mh: float

    :param tau_h: The dimensionless truncation radius normalized by the scale radius of NFW profile.
    :type  tau_h: float

    Returns
    -----------------------------------------------------------------------------------------------
    rt_norm: float
        The normalization to calculate average truncation radius of the subhalo.
    """
    # M(<x) for host halo
    M_enclosed_x = mh * tnfw2_m3d(x, tau_h) / mtot_tnfw2(tau_h)

    return x * (1.0 / (3.0 * M_enclosed_x)) ** (1.0 / 3.0)


def rt_average_integrand(logx, mh, tau_h):
    r"""
    Integrand to calculate the average tidal radius function for a subhalo within a host halo.
    rt_average = \int dlogx rt_average_integrand
    Reference: Eq.(B6) of Oguri&Takahashi 2020 	arXiv:2007.01936

    Parameters
    -----------------------------------------------------------------------------------------------
    :param logx: The natural logarithm of the dimensionless distance from the center of the host halo to the subhalo.
    :type  logx: float

    :param mh: The mass of the host halo in units of M_sol/h
    :type  mh: float

    :param tau_h: The dimensionless truncation radius of the host halo normalized by its scale radius.
    :type  tau_h: float

    Returns
    -----------------------------------------------------------------------------------------------
    rt_ave: float
        The average tidal radius of the subhalo at the specified position within the host halo,
        computed as a function of the enclosed mass ratio and the nominal tidal radius.
    """
    x = np.exp(logx)
    return enclosed_mass_ratio_bmo(x, tau_h) * rt_sub_nom(x, mh, tau_h)


def random_points_on_elip_2d(r, e, p, num_points):
    """
    Generate random points on a 2D ellipse.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param r: radius in units of Mpc/h
    :type  r: float

    :param e: ellipticity
    :type  e: float

    :param p: position angle in degrees.
    :type  p: float

    :param num_points: Number of random points to generate.
    :type  num_points: int

    Returns
    -----------------------------------------------------------------------------------------------
    xelip, yelip: tuple of ndarrays in units of Mpc/h
        The projected coordinates of the 'num_points' random points generated on the ellipse surface
        with radius=r.
    """
    z_pre = np.random.rand(num_points)
    z = (z_pre - 0.5) * 2.0
    phi = np.random.rand(num_points) * 2.0 * np.pi
    sqrt_1_minus_z2 = np.sqrt(1.0 - z**2)
    x = sqrt_1_minus_z2 * np.cos(phi) * r * np.sqrt(1.0 - e)
    y = sqrt_1_minus_z2 * np.sin(phi) * r / np.sqrt(1.0 - e)
    pp = p * np.pi / 180.0
    xelip = x * np.cos(pp) - y * np.sin(pp)
    yelip = x * np.sin(pp) + y * np.cos(pp)
    return xelip, yelip


def subhalo_distribute(rvir_h, con_h, e_h, p_h, xfunc, n):
    """
    Distribute subhalos within a host halo

    Parameters
    -----------------------------------------------------------------------------------------------
    :param rvir_h: Virial radius of the host halo in units of Mpc/h
    :type  rvir_h: float

    :param con_h: Concentration parameter of the host halo.
    :type  con_h: float

    :param e_h: Ellipticity of the subhalo distribution.
    :type  e_h: float

    :param p_h: Position angle of the major axis of the subhalo distribution in degrees.
    :type  p_h: float

    :param xfunc: Function used to convert mtot_nfw to x.
    :type  xfunc: interpolate function
    x_ar = np.logspace(-3, 3, 100)
    xfunc = interpolate.interp1d(
            profile_nfw.NFWProfile.mu(x_ar),
            x_ar,
            kind="cubic",
            fill_value="extrapolate",
        )

    :param n: Number of subhalos to distribute.
    :type  n: int

    Returns
    -----------------------------------------------------------------------------------------------
    x_sh_elip, y_sh_elip: tuple of ndarrays in units of Mpc/h
        The projected cordinates of x and y of subhalos distributed within the host halo.
    """
    fnfw = lens_halo.mtot_nfw(con_h)
    u = np.random.rand(n)
    ufnwf = u * fnfw
    x_sol = xfunc(ufnwf)
    x_sol[x_sol < 0.0001] = 0.0001
    radius_sh = rvir_h / con_h * x_sol  # in physical[h^-1 Mpc]
    x_sh_elip, y_sh_elip = random_points_on_elip_2d(radius_sh, e_h, p_h, n)
    return x_sh_elip, y_sh_elip


def precompute_dnsh(Mh_values, zz_values, output_length, cosmo, Msh_min=1e10):
    """
    Precompute the subhalo's original mass before tidal stripping and mass function of subhalos
    within host halos over a grid of host halo masses and redshifts for generate interpolator

    Parameters
    -----------------------------------------------------------------------------------------------
    :param Mh_values: Array of host halo masses for which to compute the subhalo mass function in units of M_sol/h
    :type  Mh_values: ndarray

    :param zz_values: Array of redshifts for which to compute the subhalo mass function.
    :type  zz_values: ndarray

    :param output_length: The size of the subhalo mass function array to be computed for each host mass and redshift.
    :type  output_length: int

    :param cosmo: Cosmological model instance used in the calculation.
    :type  cosmo: colossus.cosmology

    :param Msh_min: Minimum subhalo mass to consider in the calculations (default is 1e10).
    :type  Msh_min: float

    Returns
    -----------------------------------------------------------------------------------------------
    grid_msh_acc_Mh, grid_dnsh: tuple of ndarrays
        Arrays containing the cumulative mass and number functions of subhalos for the specified grid of host halo masses and redshifts.
    """
    grid_msh_acc_Mh = np.zeros((len(Mh_values), len(zz_values), output_length))
    grid_dnsh = np.zeros((len(Mh_values), len(zz_values), output_length))
    for i, mh in enumerate(Mh_values):
        for j, z in enumerate(zz_values):
            msh_Mh, grid_msh_acc_Mh[i, j], grid_dnsh[i, j] = exnum_sh_oguri_w_macc_for_grid(
                mh, z, cosmo, Msh_min=Msh_min, n_bins=output_length + 1
            )

    return grid_msh_acc_Mh, grid_dnsh


def create_interpolator(A_values, B_values, precomputed_grid):
    """
    Create an interpolator for two-dimensional grid data.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param A_values: Sorted array of values for the first dimension over which interpolation is performed.
    :type  A_values: ndarray

    :param B_values: Sorted array of values for the second dimension over which interpolation is performed.
    :type  B_values: ndarray

    :param precomputed_grid: Two-dimensional grid of values that correspond to the points defined by `A_values` and `B_values`.
    :type  precomputed_grid: ndarray

    Returns
    -----------------------------------------------------------------------------------------------
    interpolator: RegularGridInterpolator
        An interpolator object that can be used to perform interpolation on the provided grid.
    """
    interpolator = interpolate.RegularGridInterpolator((A_values, B_values), precomputed_grid)
    return interpolator


def create_interp_bsrc_sh(z_min, z_max, Msh_min, Msh_max, cosmo, n_acc=50):
    """
    Create an interpolator for the boundary box size in the source plane
    as a function of redshift and subhalo mass.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param z_min: The minimum redshift considered for interpolation.
    :type  z_min: float

    :param z_max: The maximum redshift considered for interpolation.
    :type  z_max: float

    :param Msh_min: The minimum mass of subhalos considered for interpolation.
    :type  Msh_min: float

    :param Msh_max: The maximum mass of subhalos considered for interpolation.
    :type  Msh_max: float

    :param cosmo: cosmological model instance providing necessary cosmology methods and parameters.
    :type  cosmo: colossus.cosmology.Cosmology

    :param n_acc: The number of z- and log10Mh- bins when creating the interporator.
    :type  n_acc: int

    Returns
    -----------------------------------------------------------------------------------------------
    interp_bsrc_sh: RegularGridInterpolator
        An interpolator object that can be used to retrieve the boundary box size in the source plane
        for given values of subhalo mass and redshift.
    """
    # this mmsh means mass of subhalos at their accretion time
    Msh_ar = np.logspace(np.log10(Msh_min), np.log10(Msh_max), n_acc)
    z_ar = np.linspace(z_min, z_max, n_acc)

    eh = 0.8
    ph = 0.0
    es = 0.8
    ps = 0.0

    src_y_ar = np.zeros((len(Msh_ar), len(z_ar)))

    for i, msh in enumerate(Msh_ar):
        for k, z_l in enumerate(z_ar):
            c_sh = max(concent_m_sub_ando(msh, z_l, cosmo), lens_halo.concent_m(msh, z_l)) * 10**g.sig_c_sh
            msat = (
                lens_gals.stellarmass_halomass(msh / (cosmo.H0 / 100.0), z_l, g.params, g.frac_SM_IMF)
                * 10 ** (g.sig_msat)
                * (cosmo.H0 / 100.0)
            )

            tb = lens_gals.galaxy_size(msh, msat / g.frac_SM_IMF, z_l, cosmo, model=g.TYPE_GAL_SIZE)
            comega, clambda, cweos, chubble = gen_mock_halo.calc_cosmo_for_glafic(cosmo)
            glafic.init(comega, clambda, cweos, chubble, "out2", -20.0, -20.0, 20.0, 20.0, 0.2, 0.2, 5, verb=0)

            glafic.startup_setnum(2, 0, 0)
            glafic.set_lens(1, "anfw", z_l, msh, 0.0, 0.0, eh, ph, c_sh, 0.0)
            glafic.set_lens(2, "ahern", z_l, msat, 0.0, 0.0, es, ps, tb, 0.0)

            # model_init needs to be done again whenever model parameters are changed
            glafic.model_init(verb=0)

            kap_th = 0.45

            zz = optimize.root_scalar(lens_halo.func_kapy_root, method="brentq", args=(kap_th), bracket=(1.0e-4, 2000.0))
            kap_y = zz.root

            glafic.calcimage(g.zsmax, 0.0, kap_y, verb=0)

            mag_th = 1.5

            zz = optimize.root_scalar(lens_halo.func_magy_root, method="brentq", args=(mag_th), bracket=(kap_y, 1.0e6))
            box_y = zz.root

            img = glafic.calcimage(g.zsmax, 0.0, box_y, verb=0)
            src_y = box_y - img[1]
            src_y_ar[i, k] = src_y
            glafic.quit()

    log10mmsh = np.log10(Msh_ar)
    interp_bsrc_sh = create_interpolator(log10mmsh, z_ar, src_y_ar)
    np.savez("result/" + g.prefix + "_interp_bsrc_sh.npz", log10mmsh, z_ar, src_y_ar)
    return interp_bsrc_sh


def create_interp_dndmsh(z_min, z_max, Mh_min, Mh_max, Msh_min, cosmo, n_acc=30, n_bins=100, output=True):
    """
    Create interpolators for the subhalo mass function and the mass of subhalos at their accretion time
    across a range of redshifts and host halo masses.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param z_min: The minimum redshift considered for interpolation.
    :type  z_min: float

    :param z_max: The maximum redshift considered for interpolation.
    :type  z_max: float

    :param Mh_min: The minimum mass of host halos considered for interpolation in units of M_sol/h
    :type  Mh_min: float

    :param Mh_max: The maximum mass of host halos considered for interpolation in units of M_sol/h
    :type  Mh_max: float

    :param Msh_min: The minimum mass of subhalos to be considered in the mass function in units of M_sol/h
    :type  Msh_min: float

    :param cosmo: cosmological model instance which provides necessary methods and parameters for calculations.
    :type  cosmo: colossus.cosmology.Cosmology

    :param n_acc: The number of z- and log10Mh- bins when creating the interporator.
    :type  n_acc: int

    :param n_bins: The number of bins to use for the histograms when precomputing the mass function.
    :type  n_bins: int

    Returns
    -----------------------------------------------------------------------------------------------
    interp_dnsh: RegularGridInterpolator
        An interpolator object for the subhalo mass function as a function of log10(host halo mass) and redshift.

    interp_msh_acc_Mh: RegularGridInterpolator
        An interpolator object for the mass of subhalos at their accretion time as a function of log10(host halo mass) and redshift.
    """
    zz_int_comp = np.linspace(z_min, z_max, n_acc)
    log10MMh_int_comp = np.linspace(np.log10(Mh_min / 2.0), np.log10(Mh_max), n_acc)
    output_length = n_bins - 1

    # Precompute subhalo mass function & Create the interpolator
    grid_msh_acc_Mhp, grid_dnshp = precompute_dnsh(10**log10MMh_int_comp, zz_int_comp, output_length, cosmo, Msh_min=Msh_min)
    interp_dnsh = create_interpolator(log10MMh_int_comp, zz_int_comp, grid_dnshp)
    interp_msh_acc_Mh = create_interpolator(log10MMh_int_comp, zz_int_comp, grid_msh_acc_Mhp)
    if output:
        np.savez(
            "result/" + g.prefix + "_interp_dnsh_msh_acc_Mh.npz",
            log10MMh_int_comp,
            zz_int_comp,
            grid_dnshp,
            grid_msh_acc_Mhp,
        )

    return interp_dnsh, interp_msh_acc_Mh


#
# for checks
#
if __name__ == "__main__":
    cosmo = gen_mock_halo.init_cosmo()
    g.sig_c_sh = 0.13
    g.sig_msat = 0.3
    g.paramc, g.params = lens_gals.gals_init()
    g.prefix = "check"
    g.TYPE_GAL_SIZE = "oguri20"
    interp_bsrc_sh = create_interp_bsrc_sh(0.1, 3.0, 1e10, 2e15, cosmo)
    test = [13.4, 1.2]
    print(interp_bsrc_sh(test))
