import os
import sys

import numpy as np
from colossus.cosmology import cosmology

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


def test_calc_vol():
    from source_tab import calc_vol

    z_ar = np.linspace(0, 5, 10)
    cosmo = cosmology.setCosmology("planck18")
    vol_ar = calc_vol(z_ar, cosmo)
    assert all(vol > 0 for vol in vol_ar)


def test_make_srctab():
    from source_sn import set_spline_kcor_sn
    from source_tab import make_srctab

    # Define test parameters
    mmax = 15.0
    fov = 1.0
    flag_type_min = 1
    flag_type_max = 2
    cosmo = cosmology.setCosmology("planck18")
    set_spline_kcor_sn(1)
    # Call the function to test
    m_tab, z_tab, f_tab = make_srctab(mmax, fov, flag_type_min, flag_type_max, cosmo)

    # Assertions to validate the output
    assert isinstance(m_tab, np.ndarray), "m_tab should be a numpy array"
    assert isinstance(z_tab, np.ndarray), "z_tab should be a numpy array"
    assert isinstance(f_tab, np.ndarray), "f_tab should be a numpy array"

    assert len(m_tab) > 0, "m_tab should not be empty"
    assert len(z_tab) > 0, "z_tab should not be empty"
    assert len(f_tab) > 0, "f_tab should not be empty"

    assert m_tab.shape == z_tab.shape == f_tab.shape, "All output arrays should have the same shape"

    # Check values are within expected ranges
    assert np.all(m_tab >= 14.0) and np.all(m_tab < mmax), "Values in m_tab should be within the expected range"
    assert np.all(z_tab >= 0.1) and np.all(z_tab <= 5.499), "Values in z_tab should be within the expected range"
    assert np.all(
        (f_tab >= flag_type_min) & (f_tab <= flag_type_max)
    ), "Values in f_tab should be within the expected range of source object types"


def test_dndzdmobs():
    from source_tab import dndzdmobs

    # Mock dependencies
    class MockSourceQSO:
        @staticmethod
        def mtoma_qso(m, z, cosmo):
            return m + z  # Simple mock implementation

        @staticmethod
        def lf_func_qso(ma, z, cosmo):
            return ma * 0.1  # Simple mock implementation

    class MockSourceSN:
        @staticmethod
        def mtodm_sn(m, z, flag_type, cosmo):
            return m - z  # Simple mock implementation

        @staticmethod
        def lf_func_sn(dm, z, flag_type, cosmo):
            return dm * 0.2  # Simple mock implementation

    def mock_calc_vol(z, cosmo):
        return z * 1000  # Simple mock implementation

    # Replace real modules and functions with mocks
    global source_qso, source_sn, calc_vol
    source_qso = MockSourceQSO()
    source_sn = MockSourceSN()
    calc_vol = mock_calc_vol

    # Define test parameters
    m = np.array([15.0, 16.0, 17.0])
    z = np.array([0.5, 1.0, 1.5])
    flag_type_0 = 0
    flag_type_1 = 1
    cosmo_mock = "mock_cosmo"  # Mock cosmology value

    # Call the function to test for flag_type == 0
    result_qso = dndzdmobs(m, z, flag_type_0, cosmo_mock)
    expected_qso = (m + z) * 0.1 * (z * 1000)
    assert np.allclose(result_qso, expected_qso), "Result for flag_type == 0 is incorrect"

    # Call the function to test for flag_type != 0
    result_sn = dndzdmobs(m, z, flag_type_1, cosmo_mock)
    expected_sn = ((m - z) * 0.2 * (z * 1000)) / (1.0 + z)
    assert np.allclose(result_sn, expected_sn), "Result for flag_type != 0 is incorrect"

    # Clean up the mock
    del source_qso, source_sn, calc_vol
