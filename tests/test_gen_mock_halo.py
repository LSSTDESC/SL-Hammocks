import math
import os
import sys
from io import StringIO

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


def test_run_command_line(monkeypatch):
    import global_value as g
    from gen_mock_halo import run_command_line

    # Mock dependencies and global variables
    class MockSourceQSO:
        @staticmethod
        def set_spline_kcor_qso():
            pass

    class MockSourceSN:
        @staticmethod
        def set_spline_kcor_sn(flag):
            pass

    def mock_init_cosmo(cosmo_model):
        return "cosmo_mock"

    def mock_gals_init(type_smhm):
        return "paramc_mock", "params_mock"

    global source_qso, source_sn, init_cosmo, lens_gals
    source_qso = MockSourceQSO()
    source_sn = MockSourceSN()
    init_cosmo = mock_init_cosmo
    lens_gals = type("lens_gals", (object,), {"gals_init": mock_gals_init})

    # Define test parameters
    argv_test = ["--area=2500.0", "--ilim=24.0", "--zlmax=3.5", "--source=sn", "--prefix=test_sn", "--solver=lenstronomy", "--nworker=4"]

    # Redirect stdout to capture print statements
    captured_output = StringIO()
    monkeypatch.setattr(sys, "stdout", captured_output)

    # Run the function with test arguments
    run_command_line(argv_test)

    # Check printed output
    expected_output = (
        "# area  : 2500.000000\n"
        "# ilim  : 24.000000\n"
        "# zlmax : 3.500000\n"
        "# source: sn\n"
        "# prefix: test_sn\n"
        "# solver: lenstronomy\n"
        "# process: multi(4 core)\n"
    )
    assert captured_output.getvalue() == expected_output, "Printed output is incorrect"

    # Check global parameter values
    assert g.area == 2500.0
    assert g.ilim == 24.0
    assert g.zlmax == 3.5
    assert g.source == "sn"
    assert g.prefix == "test_sn"
    assert g.solver == "lenstronomy"
    assert g.nworker == 4
    assert g.flag_type_min == 1
    assert g.flag_type_max == 5
    assert g.imax == 27.5
    assert g.mlim == 24.0

    # Cosmological parameters check
    assert math.isclose(g.cosmo_omega, 0.3, rel_tol=0.1)
    assert math.isclose(g.cosmo_lambda, 0.7, rel_tol=0.1)
    assert math.isclose(g.cosmo_hubble, 0.7, rel_tol=0.1)

    # Clean up the mock
    del source_qso, source_sn, g, init_cosmo, lens_gals


# Mock class to simulate the cosmological model
class MockCosmo:
    def __init__(self, Om0, H0):
        self.Om0 = Om0
        self.H0 = H0


def test_calc_cosmo_for_glafic():
    from gen_mock_halo import calc_cosmo_for_glafic

    # Define global variable g with necessary attributes
    class MockG:
        nonflat = 0.1
        cosmo_weos = -1.0

    import types

    mock_g = types.ModuleType("g")
    mock_g.nonflat = 0.1
    mock_g.cosmo_weos = -1.0

    # monkeypatch.setattr("g", MockG)

    # Create instances of MockCosmo for different test cases
    cosmo1 = MockCosmo(Om0=0.3, H0=70)
    cosmo2 = MockCosmo(Om0=0.26, H0=67.4)

    # Test case 1
    result = calc_cosmo_for_glafic(cosmo1)
    expected_result = (0.3, 0.7, -1.0, 0.7)
    assert result == expected_result, f"Expected {expected_result}, but got {result}"

    # Test case 2
    result = calc_cosmo_for_glafic(cosmo2)
    expected_result = (0.26, 0.74, -1.0, 0.674)
    assert result == expected_result, f"Expected {expected_result}, but got {result}"
