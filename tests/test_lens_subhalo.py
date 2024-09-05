import os
import sys

import numpy as np
import pytest
from colossus.cosmology import cosmology
from colossus.halo import concentration
from colossus.halo import mass_so
from colossus.halo import profile_nfw
from scipy import interpolate

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


def test_deltac_z():
    from lens_subhalo import deltac_z

    cosmo = cosmology.setCosmology("planck18")
    z0 = 0.0
    result_deltac_z0 = deltac_z(z0, cosmo)
    assert np.isclose(result_deltac_z0, 1.67, rtol=0.1), f"Expected value to be close to 1.67 but got {result_deltac_z0}"

    z1 = 1.0
    result_deltac_z1 = deltac_z(z1, cosmo)
    assert result_deltac_z1 > 1.67

    z2 = 2.0
    result_deltac_z2 = deltac_z(z2, cosmo)
    assert result_deltac_z2 > result_deltac_z1


def test_zf_calc():
    from lens_subhalo import zf_calc

    z = 0.5
    M_h = 1e13
    cosmo = cosmology.setCosmology("planck18")
    zf = zf_calc(z, M_h, cosmo)

    assert zf > z


def test_mf_sub_eps():
    from lens_subhalo import mf_sub_eps

    # Mocking the cosmological model instance
    cosmo = cosmology.setCosmology("planck18")

    # Define input parameters
    M_shf = np.array([1e9, 1e10, 1e11, 6e11])
    z = 0.5
    M_h = 1e12
    zf = 1.0

    # Call the function with the mock cosmological model instance
    result_dNeps_dMshf = mf_sub_eps(M_shf, z, M_h, zf, cosmo)

    # Assertion to check if the result matches the expected value
    assert all(result_dNeps_dMshf >= 0)
    assert np.isclose(result_dNeps_dMshf[-1], 0.0), f"Expected value to be close to 0.0 but got {result_dNeps_dMshf[-1]}"


def test_t_df():
    from lens_subhalo import t_df

    # Test parameters
    M_h_f = 1e12  # Host halo mass in M_sol/h
    M_shf = 1e11  # Subhalo mass in M_sol/h
    z = 0.5  # Redshift
    cosmo = cosmology.setCosmology("planck18")

    # Expected value (You would need to calculate this based on the formula or know it from reference)
    expected_value = 26.0  # Hypothetical expected value in Gyr

    # Calculate the dynamical friction timescale using the function
    result = t_df(M_h_f, M_shf, z, cosmo)

    # Assertion to check if the result is close to the expected value
    assert np.isclose(result, expected_value, rtol=1), f"Expected {expected_value}, but got {result}"


def test_factor_dynamical_friction():
    from lens_subhalo import factor_dynamical_friction

    M_h = 1e12
    M_shf = 1e10
    z = 0.5
    zf = 1.0
    cosmo = cosmology.setCosmology("planck18")
    # Call the function to test
    factor = factor_dynamical_friction(M_h, M_shf, z, zf, cosmo)

    # Assertions to validate the output
    assert isinstance(factor, float), "The output should be a float"
    assert 0 <= factor <= 1, "The exponential suppression factor should be between 0 and 1"

    # Test case where z > zf which should raise an error
    z_invalid = 1.5
    with pytest.raises(ValueError):
        factor_dynamical_friction(M_h, M_shf, z_invalid, zf, cosmo)


def test_rt_average():
    from lens_subhalo import rt_average

    # Define test parameters
    M_h_f = 1e12
    M_sh = 1e10
    z = 0.5

    rt = rt_average(M_h_f, M_sh, z)
    con = concentration.concentration(M_h_f, "vir", z, model="diemer15")
    rs = 1.0e-3 * mass_so.M_to_R(M_h_f, z, "vir") / con

    # Assertions to validate the output
    assert isinstance(rt, float), "The output should be a float"
    assert 0 < rt < rs, "The tidal radius should be positive"

    # Edge case: Check behavior with very small subhalo mass
    M_sh_small = 1e5
    rt_small = rt_average(M_h_f, M_sh_small, z)
    assert 0 < rt_small < rs, "The tidal radius with small subhalo mass should be positive"


def test_sub_m_calc():
    from lens_subhalo import sub_m_calc
    from lens_subhalo import sub_m_calc_setspline

    M_h = 1e13
    M_sh_after_ar = np.array([1e9, 1e14])
    zf = 1.0
    log10_M_sh_after2ori_spline = sub_m_calc_setspline(M_h, M_sh_after_ar, zf)
    M_sh_after = 1e10
    M_sh_ori, rt = sub_m_calc(M_h, M_sh_after, zf, log10_M_sh_after2ori_spline)

    assert M_sh_after < M_sh_ori
    assert rt > 0

    # This function calculates for smaller and larger values than the minimum and maximum value
    # in M_sh_after_ar, respectively. But we should confirm whether the output values are valid or not
    M_sh_after = 1e8
    M_sh_ori, rt = sub_m_calc(M_h, M_sh_after, zf, log10_M_sh_after2ori_spline)

    assert M_sh_after < M_sh_ori
    assert rt > 0

    M_sh_after = 1e15
    M_sh_ori, rt = sub_m_calc(M_h, M_sh_after, zf, log10_M_sh_after2ori_spline)

    assert M_sh_after < M_sh_ori
    assert rt > 0


def test_exnum_sh_oguri_w_macc_for_grid():
    from lens_subhalo import exnum_sh_oguri_w_macc_for_grid

    # Test parameters
    M_h = 1e12  # Mass of the host halo in units of M_sol/h
    z = 0.5  # Redshift
    cosmo = cosmology.setCosmology("planck18")
    Msh_min = 1e10
    n_bins = 100

    # Call the function
    M_sh_after_ratio, M_sh_ori_ratio, dNeps = exnum_sh_oguri_w_macc_for_grid(M_h, z, cosmo, Msh_min, n_bins)

    # Assertions to verify the function's output
    assert len(M_sh_after_ratio) == n_bins - 1, "Length of M_sh_after_ratio should be n_bins - 1"
    assert len(M_sh_ori_ratio) == n_bins - 1, "Length of M_sh_ori_ratio should be n_bins - 1"
    assert len(dNeps) == n_bins - 1, "Length of dNeps should be n_bins - 1"

    assert np.all(M_sh_after_ratio >= 0), "All values in M_sh_after_ratio should be positive"
    for r_ori, r_aft in zip(M_sh_ori_ratio, M_sh_after_ratio, strict=True):
        assert r_ori >= r_aft, f"Element {r_aft} in M_sh_after_ratio is not greater than element {r_ori} in M_sh_ori_ratio"

    assert np.all(dNeps >= 0), "All values in dNeps should be non-negative"

    # Additional checks for expected ranges
    assert np.max(M_sh_after_ratio) < 1, "Max value of M_sh_after_ratio should be less than 1"
    # Caution! Max value of M_sh_ori_ratio can be larger than 1 in this function.
    # Such too massive subhalos should be removed in another place.


def test_concent_m_sub_ando():
    from lens_subhalo import concent_m_sub_ando

    # Define a sample cosmological model
    cosmo = cosmology.setCosmology("planck18")

    M_sh_200c_ar_array = np.array([1e11, 1e12, 1e13])  # in units of M_sol/h
    z_array = 1.0
    assert np.all(concent_m_sub_ando(M_sh_200c_ar_array, z_array, cosmo) > 2.0)


def test_concentration_subhalo_w_scatter():
    from lens_subhalo import concentration_subhalo_w_scatter

    # Define a sample cosmological model
    cosmo = cosmology.setCosmology("planck18")

    msh_vir = np.array([1e12, 5e11, 2e12])  # in units of M_sol/h
    msh_200c = np.array([1.2e11, 6e11, 3e12])  # in units of M_sol/h
    msh_acc_vir = np.array([2e12, 1e12, 4e12])  # in units of M_sol/h
    zl = 0.5
    sig = 0.25
    result = concentration_subhalo_w_scatter(msh_vir, msh_200c, msh_acc_vir, zl, sig, cosmo)
    assert len(result) == len(msh_vir), "Output length should match the length of input masses"
    assert all(result > 1), "All concentrations should be positive"


def test_precompute_dnsh():
    from lens_subhalo import precompute_dnsh

    # Sample input data
    Mh_values = np.array([1e11, 1e12, 1e13])  # in units of M_sol/h
    zz_values = np.array([0.0, 0.5, 1.0])
    output_length = 5
    cosmo = cosmology.setCosmology("planck18")

    grid_msh_acc_Mh, grid_dnsh = precompute_dnsh(Mh_values, zz_values, output_length, cosmo)

    # Test shapes of returned arrays
    assert grid_msh_acc_Mh.shape == (len(Mh_values), len(zz_values), output_length), "Shape of grid_msh_acc_Mh is incorrect"
    assert grid_dnsh.shape == (len(Mh_values), len(zz_values), output_length), "Shape of grid_dnsh is incorrect"

    # Test contents of returned arrays
    for i in range(len(Mh_values)):
        for j in range(len(zz_values)):
            assert all(grid_msh_acc_Mh[i, j] > 0), f"Inner shape of grid_msh_acc_Mh at index ({i},{j}) is incorrect"
            assert all(grid_dnsh[i, j] >= 0), f"Inner shape of grid_dnsh at index ({i},{j}) is incorrect"
            assert grid_msh_acc_Mh[i, j].shape == (output_length,), f"Inner shape of grid_msh_acc_Mh at index ({i},{j}) is incorrect"
            assert grid_dnsh[i, j].shape == (output_length,), f"Inner shape of grid_dnsh at index ({i},{j}) is incorrect"


def test_create_interp_dndmsh():
    import global_value as g
    from lens_subhalo import create_interp_dndmsh

    # Sample input data
    z_min = 0.0
    z_max = 1.0
    Mh_min = 1e11  # in units of M_sol/h
    Mh_max = 1e15  # in units of M_sol/h
    Msh_min = 1e8  # in units of M_sol/h
    n_bins = 3
    n_acc = 3
    cosmo = cosmology.setCosmology("planck18")
    g.prefix = "test"

    interp_dnsh, interp_msh_acc_Mh = create_interp_dndmsh(
        z_min, z_max, Mh_min, Mh_max, Msh_min, cosmo, n_acc=n_acc, n_bins=n_bins, output=False
    )

    # Test types of returned interpolators
    assert isinstance(interp_dnsh, interpolate.RegularGridInterpolator), "interp_dnsh should be an instance of RegularGridInterpolator"
    assert isinstance(
        interp_msh_acc_Mh, interpolate.RegularGridInterpolator
    ), "interp_msh_acc_Mh should be an instance of RegularGridInterpolator"

    # Test edge case: minimum and maximum redshift
    edge_points = np.array(
        [[np.log10(Mh_min), z_min], [np.log10(Mh_max), z_max], [((np.log10(Mh_min) + np.log10(Mh_max)) / 2), (z_min + z_max) / 2]]
    )
    edge_dnsh_values = interp_dnsh(edge_points)
    edge_msh_acc_values = interp_msh_acc_Mh(edge_points)

    assert edge_dnsh_values.shape == (
        len(edge_points),
        n_bins - 1,
    ), "Output shape of interp_dnsh for edge cases should match (input points shape, n_bins-1)"
    assert edge_msh_acc_values.shape == (
        len(edge_points),
        n_bins - 1,
    ), "Output shape of interp_msh_acc_Mh for edge cases should match (input points shape, n_bins-1)"


def test_subhalo_mass_function():
    from lens_subhalo import subhalo_mass_function

    # Sample input data
    mh_vir_ar = np.array([1e12, 5e13, 1e14])  # in units of M_sol/h
    zl_ar = np.array([0.5, 0.3, 0.7])
    min_Msh = 1e8
    length = 99

    # Mock interpolation functions
    def mock_interp_dnsh(Mhl_zl_tab_vec):
        return np.ones((len(Mhl_zl_tab_vec), length))

    def mock_interp_msh_acc_Mh(Mhl_zl_tab_vec):
        return np.full((len(Mhl_zl_tab_vec), length), 0.1)

    # Run the function
    mmsh, mmsh_acc, NNsh = subhalo_mass_function(mh_vir_ar, zl_ar, min_Msh, mock_interp_dnsh, mock_interp_msh_acc_Mh, length)

    # Test mmsh dimensions and values
    assert mmsh.shape == (len(mh_vir_ar), length), "mmsh should have shape (number of host halos, number of bins)"
    assert np.all(mmsh > min_Msh), "All mmsh values should be greater than min_Msh"

    # Test mmsh_acc dimensions and values
    assert mmsh_acc.shape == (len(mh_vir_ar), length), "mmsh_acc should have shape (number of host halos, number of bins)"
    assert np.all(mmsh_acc <= mh_vir_ar[:, None]), "All mmsh_acc values should be less than or equal to corresponding host halo masses"

    # Test NNsh dimensions and non-negativity
    assert NNsh.shape == (len(mh_vir_ar), length), "NNsh should have shape (number of host halos, number of bins)"
    assert np.all(NNsh >= 0), "All NNsh values should be non-negative"

    # Test edge case: Single host halo
    mh_vir_single = np.array([1e12])
    zl_single = np.array([0.5])
    mmsh_single, mmsh_acc_single, NNsh_single = subhalo_mass_function(
        mh_vir_single, zl_single, min_Msh, mock_interp_dnsh, mock_interp_msh_acc_Mh, length
    )
    assert mmsh_single.shape == (1, length), "mmsh for single host halo should have shape (1, number of bins)"
    assert mmsh_acc_single.shape == (1, length), "mmsh_acc for single host halo should have shape (1, number of bins)"
    assert NNsh_single.shape == (1, length), "NNsh for single host halo should have shape (1, number of bins)"

    # Test edge case: Minimum subhalo mass equals host halo mass / 2
    min_Msh_edge = mh_vir_ar[0] / 2.0
    mmsh_edge, mmsh_acc_edge, NNsh_edge = subhalo_mass_function(
        mh_vir_ar, zl_ar, min_Msh_edge, mock_interp_dnsh, mock_interp_msh_acc_Mh, length
    )
    assert np.all(mmsh_edge >= min_Msh_edge), "All mmsh values should be greater than or equal to min_Msh_edge"


def test_random_points_on_elip_2d():
    from lens_subhalo import random_points_on_elip_2d

    r = 10.0  # radius in units of Mpc/h
    e = 0.5  # ellipticity
    p = 45.0  # position angle in degrees
    num_points = 1000  # number of random points to generate

    xelip, yelip = random_points_on_elip_2d(r, e, p, num_points)

    # Test the lengths of the returned arrays
    assert len(xelip) == num_points, "Length of xelip is incorrect"
    assert len(yelip) == num_points, "Length of yelip is incorrect"

    # Check that the points exists inside an ellipse by verifying their radii
    # [Notes] This function generates x,y through z-direction projection,
    # so points should not be on an ellipse but inside the ellipse
    a = r * np.sqrt(1.0 - e)
    b = r / np.sqrt(1.0 - e)
    pp = p * np.pi / 180.0
    x_rotated_back = xelip * np.cos(pp) + yelip * np.sin(pp)
    y_rotated_back = -xelip * np.sin(pp) + yelip * np.cos(pp)
    ellipse_equation = (x_rotated_back / a) ** 2 + (y_rotated_back / b) ** 2
    assert np.all(ellipse_equation <= 1.0), "Points do not exist inside the expected ellipse"


def test_subhalo_distribute():
    from lens_subhalo import subhalo_distribute

    rvir_h = 1.0  # Virial radius of the host halo in units of Mpc/h
    con_h = 10.0  # Concentration parameter of the host halo
    e_h = 0.3  # Ellipticity of the subhalo distribution
    p_h = 30.0  # Position angle of the major axis of the subhalo distribution in degrees
    n = 1000  # Number of subhalos to distribute

    x_ar = np.logspace(-3, 3, 100)
    mu_ar = profile_nfw.NFWProfile.mu(x_ar)
    xfunc = interpolate.interp1d(mu_ar, x_ar, kind="cubic", fill_value="extrapolate")

    x_sh_elip, y_sh_elip = subhalo_distribute(rvir_h, con_h, e_h, p_h, xfunc, n)

    # Test the lengths of the returned arrays
    assert len(x_sh_elip) == n, "Length of x_sh_elip is incorrect"
    assert len(y_sh_elip) == n, "Length of y_sh_elip is incorrect"

    # Check that the subhalos are within the virial radius
    subhalo_distances = np.sqrt(x_sh_elip**2 + y_sh_elip**2)
    a = rvir_h * np.sqrt(1.0 - e_h)
    b = rvir_h / np.sqrt(1.0 - e_h)
    pp = p_h * np.pi / 180.0
    x_rotated_back = x_sh_elip * np.cos(pp) + y_sh_elip * np.sin(pp)
    y_rotated_back = -x_sh_elip * np.sin(pp) + y_sh_elip * np.cos(pp)
    subhalo_distances = (x_rotated_back / a) ** 2 + (y_rotated_back / b) ** 2
    assert np.all(subhalo_distances <= rvir_h), "Some subhalos are outside the virial radius"


def test_create_interp_bsrc_sh():
    import global_value as g
    from lens_gals import gals_init
    from lens_subhalo import create_interp_bsrc_sh

    z_min = 0.1
    z_max = 2.0
    # [Notes] Now this function is not available for much smaller Msh_min value than 1e10.
    # This can be fixed and controlled by changing kap_th value in create_interp_bsrc_sh()
    Msh_min = 1e10
    Msh_max = 1e13
    n_acc = 4

    cosmo = cosmology.setCosmology("planck18")
    g.paramc, g.params = gals_init()

    interp_bsrc_sh = create_interp_bsrc_sh(z_min, z_max, Msh_min, Msh_max, cosmo, n_acc)

    # Test the type of the returned interpolator
    assert isinstance(interp_bsrc_sh, interpolate.RegularGridInterpolator), "Returned object is not a RegularGridInterpolator"

    # Test the shape of the interpolation grid
    log10Msh_ar = np.log10(np.logspace(np.log10(Msh_min), np.log10(Msh_max), n_acc))
    z_ar = np.linspace(z_min, z_max, n_acc)
    assert interp_bsrc_sh.grid[0].shape == (n_acc,), "Shape of the mass grid is incorrect"
    assert interp_bsrc_sh.grid[1].shape == (n_acc,), "Shape of the redshift grid is incorrect"

    # Test a few sample points to ensure they are within expected bounds
    sample_points = [(log10Msh_ar[i], z_ar[j]) for i in range(n_acc) for j in range(n_acc)]
    for point in sample_points:
        value = interp_bsrc_sh(point)
        assert value >= 0, f"Interpolated value at {point} is negative"
