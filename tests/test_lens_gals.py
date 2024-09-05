import os
import sys

import numpy as np
import pytest
from colossus.cosmology import cosmology
from scipy import stats

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


def test_galaxy_size():
    from lens_gals import galaxy_size

    # Test the mean galaxy_size with valid arguments to ensure correct behavior for vdW23
    cosmo = cosmology.setCosmology("planck18")
    hubble = cosmo.H0 / 100.0
    mh = 1e13  # Example dark matter halo mass in units of Msun/h
    mstar = 1e11 * hubble  # Example stellar mass in units of Msun/h
    z = 0.75  # Example redshift
    q_out = "rb"  # Output query in units of Mpc/h
    model = "vdW23"  # Model to use

    # Verify the correct result is obtained
    expected_approx_size = 10**0.3 * hubble / 1e3 * 0.551  # from van der Wel et al. 2023 Fig. 10 in units of Mpc/h
    calculated_size = galaxy_size(mh, mstar, z, cosmo, q_out, model, scatter=False)  #  in units of Mpc/h

    assert pytest.approx(calculated_size, rel=0.1) == expected_approx_size, "The galaxy size should match the expected value."

    # Verify the correct scatter is obtained
    n_ar = 1000
    mh_ar = np.array([1e12] * n_ar)
    calculated_size_w_scatters = galaxy_size(mh_ar, mstar, z, cosmo, q_out, model, scatter=True)
    logscatters = np.log(calculated_size_w_scatters / calculated_size)
    # Perform the KS test to see if the logged data follows a normal distribution
    D, p_value = stats.kstest(logscatters, "norm", args=(logscatters.mean(), logscatters.std(ddof=1)))
    # Set the significance level at which you are testing
    alpha = 0.05  # Commonly used significance level
    assert p_value > alpha, f"Sample data does not follow a lognormal distribution (p-value={p_value})"

    # test_galaxy_size_unknown_model
    with pytest.raises(Exception) as excinfo:
        galaxy_size(1e12, 1e11, 0.3, cosmo, model="unknown_model")
    assert "Unknown model" in str(excinfo.value), "An exception with 'Unknown model' message should be raised for unknown models."

    # Test the mean galaxy_size with valid arguments to ensure correct behavior for orugi20 model
    model2 = "oguri20"  # Model to use
    sig_tb = 0.2
    calculated_size_model2 = galaxy_size(mh, mstar, z, cosmo, q_out, model2, scatter=False, sig_tb=sig_tb)  # in units of Mpc/h
    assert 0 < calculated_size_model2 < 0.01, "The galaxy size should be less than 10 kpc"

    # Test the mean galaxy_size with valid arguments to ensure correct behavior for karmakar23 model
    model3 = "karmakar23"
    calculated_size_model3 = galaxy_size(mh, mstar, z, cosmo, q_out, model3, scatter=False)  # in units of Mpc/h
    assert 0 < calculated_size_model3 < 0.01, "The galaxy size should be less than 10 kpc"


def test_stellarmass_halomass():
    from lens_gals import gals_init
    from lens_gals import stellarmass_halomass

    mh_ar = np.logspace(10, 16, 100)  # in units of M_sol/h
    z = 0.5
    paramc, params = gals_init()
    stellar_masses = stellarmass_halomass(mh_ar, z, paramc)  # in units of M_sol/h
    ratio_SMHM = stellar_masses / mh_ar
    index_of_max = np.argmax(ratio_SMHM)
    Mh_at_peak_ratio_SMHM = mh_ar[index_of_max]
    assert pytest.approx(np.log10(Mh_at_peak_ratio_SMHM), rel=0.02) == 12, "The SMHM ratio should be peaked at Mh ~ 1e12 M_sol"
    assert all(0 <= r <= 0.1 for r in ratio_SMHM), "The SMHM conversion rate should be at most 10%"


def test_galaxy_properties_vdw23():
    from lens_gals import galaxy_properties
    from lens_gals import gals_init

    # cosmo = cosmology.setCosmology("planck18")
    paramc, params = gals_init()
    mh_vir = np.array([1e12, 1e13, 1e14])
    zl = 0.5
    p_h = np.array([10, 20, 30])
    frac_SM_IMF = 0.7
    TYPE_GAL_SIZE = "vdW23"
    sig_mg = 0.1

    m_cen, tb_cen, e_cen, p_cen = galaxy_properties(mh_vir, zl, p_h, paramc, frac_SM_IMF, TYPE_GAL_SIZE, sig_mg)
    m_sat, tb_sat, e_sat, p_sat = galaxy_properties(mh_vir, zl, p_h, params, frac_SM_IMF, TYPE_GAL_SIZE, sig_mg)

    assert len(m_cen) == len(tb_cen) == len(e_cen) == len(p_cen) == len(mh_vir)
    assert len(m_sat) == len(tb_sat) == len(e_sat) == len(p_sat) == len(mh_vir)
    assert np.all((m_cen > 0) & (m_cen < mh_vir * 0.1))
    assert np.all((m_sat > 0) & (m_sat < mh_vir * 0.1))
    assert np.all(tb_cen > 0)
    assert np.all(tb_sat > 0)
    assert np.all((e_cen >= 0) & (e_cen <= 1))
    assert np.all((e_sat >= 0) & (e_sat <= 1))
    assert np.all((p_cen >= p_h - 180) & (p_cen <= p_h + 180))
    assert np.all((p_sat >= p_h - 180) & (p_sat <= p_h + 180))


def test_galaxy_properties_oguri20():
    from lens_gals import galaxy_properties
    from lens_gals import gals_init

    # cosmo = cosmology.setCosmology("planck18")
    paramc, params = gals_init()
    mh_vir = np.array([1e12, 1e13, 1e14])
    zl = 0.5
    p_h = np.array([10, 20, 30])
    frac_SM_IMF = 0.7
    TYPE_GAL_SIZE = "oguri20"
    sig_mg = 0.1

    m_cen, tb_cen, e_cen, p_cen = galaxy_properties(mh_vir, zl, p_h, paramc, frac_SM_IMF, TYPE_GAL_SIZE, sig_mg)
    m_sat, tb_sat, e_sat, p_sat = galaxy_properties(mh_vir, zl, p_h, params, frac_SM_IMF, TYPE_GAL_SIZE, sig_mg)

    assert len(m_cen) == len(tb_cen) == len(e_cen) == len(p_cen) == len(mh_vir)
    assert len(m_sat) == len(tb_sat) == len(e_sat) == len(p_sat) == len(mh_vir)
    assert np.all((m_cen > 0) & (m_cen < mh_vir * 0.1))
    assert np.all((m_sat > 0) & (m_sat < mh_vir * 0.1))
    assert np.all(tb_cen > 0)
    assert np.all(tb_sat > 0)
    assert np.all((e_cen >= 0) & (e_cen <= 1))
    assert np.all((e_sat >= 0) & (e_sat <= 1))
    assert np.all((p_cen >= p_h - 180) & (p_cen <= p_h + 180))
    assert np.all((p_sat >= p_h - 180) & (p_sat <= p_h + 180))


def test_galaxy_properties_karmakar23():
    from lens_gals import galaxy_properties
    from lens_gals import gals_init

    # cosmo = cosmology.setCosmology("planck18")
    paramc, params = gals_init()
    mh_vir = np.array([1e12, 1e13, 1e14])
    zl = 0.5
    p_h = np.array([10, 20, 30])
    frac_SM_IMF = 0.7
    TYPE_GAL_SIZE = "karmakar23"
    sig_mg = 0.1

    m_cen, tb_cen, e_cen, p_cen = galaxy_properties(mh_vir, zl, p_h, paramc, frac_SM_IMF, TYPE_GAL_SIZE, sig_mg)
    m_sat, tb_sat, e_sat, p_sat = galaxy_properties(mh_vir, zl, p_h, params, frac_SM_IMF, TYPE_GAL_SIZE, sig_mg)

    assert len(m_cen) == len(tb_cen) == len(e_cen) == len(p_cen) == len(mh_vir)
    assert len(m_sat) == len(tb_sat) == len(e_sat) == len(p_sat) == len(mh_vir)
    assert np.all((m_cen > 0) & (m_cen < mh_vir * 0.1))
    assert np.all((m_sat > 0) & (m_sat < mh_vir * 0.1))
    assert np.all(tb_cen > 0)
    assert np.all(tb_sat > 0)
    assert np.all((e_cen >= 0) & (e_cen <= 1))
    assert np.all((e_sat >= 0) & (e_sat <= 1))
    assert np.all((p_cen >= p_h - 180) & (p_cen <= p_h + 180))
    assert np.all((p_sat >= p_h - 180) & (p_sat <= p_h + 180))
