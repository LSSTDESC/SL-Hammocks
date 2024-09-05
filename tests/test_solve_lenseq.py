import os
import sys

import numpy as np
import pytest
from colossus.cosmology import cosmology

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


def test_calc_image():
    from solve_lenseq import calc_image

    zl = 0.5
    zs = 1.5
    M_h, M_cen = 1e15, 1e13
    e_h, e_cen = 0.5, 0.5
    p_h, p_cen = 0.0, 0.0
    con_h, tb_cen = 10, 1.0
    x_h, y_h, x_cen, y_cen = 0.0, 0.0, 0.0, 0.0
    gg = 0.1
    tgg = 0.0
    kap_pert = 0.0
    sx, sy = 1.0, 1.0
    ein = 22.01587
    rt_range = 4
    flag_mag = 3
    cosmo = cosmology.setCosmology("planck18")
    lens_par = [
        ["anfw", zl, M_h, x_h, y_h, e_h, p_h, con_h, 0.0],
        ["ahern", zl, M_cen, x_cen, y_cen, e_cen, p_cen, tb_cen, 0.0],
        ["pert", zl, zs, 0.0, 0.0, gg, tgg, 0.0, kap_pert],
    ]
    srcs_par = [zs, sx, sy]
    out_img, nim, sep, mag, mag_max, fr, kapgam = calc_image(lens_par, srcs_par, ein, rt_range, flag_mag, cosmo)
    xi_s = [img[0] for img in out_img]
    yi_s = [img[1] for img in out_img]
    mi_s = [abs(img[2]) for img in out_img]
    expected_nim = 4  # In this setup, the central image cannot be recognized by glafic due to accuracy of ein/5.0
    expected_max_xi = 4 * ein
    expected_max_yi = 4 * ein
    expected_min_sep = ein
    expected_max_sep = 4 * ein
    expected_min_mi = 0.0
    expected_min_mag_max = 1.0

    assert all(abs(xi) < expected_max_xi for xi in xi_s)
    assert all(abs(yi) < expected_max_yi for yi in yi_s)
    assert all(mi > expected_min_mi for mi in mi_s)
    assert nim == expected_nim
    assert expected_min_sep < sep < expected_max_sep
    assert mag_max > expected_min_mag_max


def test_filter_for_subhalos_imgs():
    from solve_lenseq import filter_for_subhalos_imgs

    x_img = [1.0, 2.0, 3.0, 4.0]
    y_img = [1.0, 2.0, 3.0, 4.0]
    mag_img = [10.0, 20.0, 30.0, 40.0]
    distance = 3.0 * 2**0.5

    filtered_x_img, filtered_y_img, filtered_mag_img, filtered_index = filter_for_subhalos_imgs(x_img, y_img, mag_img, distance)

    # Expected output
    expected_x_img = [1.0, 2.0, 3.0]
    expected_y_img = [1.0, 2.0, 3.0]
    expected_mag_img = [10.0, 20.0, 30.0]
    expected_index = [0, 1, 2]

    assert filtered_x_img == expected_x_img, f"Expected {expected_x_img} but got {filtered_x_img}"
    assert filtered_y_img == expected_y_img, f"Expected {expected_y_img} but got {filtered_y_img}"
    assert filtered_mag_img == expected_mag_img, f"Expected {expected_mag_img} but got {filtered_mag_img}"
    assert filtered_index == expected_index, f"Expected {expected_index} but got {filtered_index}"


def test_gene_e():
    from solve_lenseq import gene_e

    n = 10
    e_gal = gene_e(n)
    assert all(0 <= elipticily <= 1 for elipticily in e_gal)


def test_gene_e_halo():
    from solve_lenseq import gene_e_halo

    # Test with a small array of halo masses
    Mh_small = np.array([1e12, 2e12, 5e12])  # example masses in units of M_sol/h
    elip_small = gene_e_halo(Mh_small)
    assert isinstance(elip_small, np.ndarray), "Output should be a numpy ndarray"
    assert elip_small.shape == Mh_small.shape, "Output shape should match input shape"
    assert np.all(elip_small >= 0.0), "Ellipticity values should be non-negative"
    assert np.all(elip_small <= 0.9), "Ellipticity values should not exceed 0.9"

    # Test with an edge case: very large mass
    Mh_large = np.array([1e15])  # example large mass in units of M_sol/h
    elip_large = gene_e_halo(Mh_large)
    assert isinstance(elip_large, np.ndarray), "Output should be a numpy ndarray"
    assert elip_large.shape == Mh_large.shape, "Output shape should match input shape"
    assert np.all(elip_large >= 0.0), "Ellipticity values should be non-negative"
    assert np.all(elip_large <= 0.9), "Ellipticity values should not exceed 0.9"

    # Test with an edge case: very small mass
    Mh_very_small = np.array([1e8])  # example very small mass in units of M_sol/h
    elip_very_small = gene_e_halo(Mh_very_small)
    assert isinstance(elip_very_small, np.ndarray), "Output should be a numpy ndarray"
    assert elip_very_small.shape == Mh_very_small.shape, "Output shape should match input shape"
    assert np.all(elip_very_small >= 0.0), "Ellipticity values should be non-negative"
    assert np.all(elip_very_small <= 0.9), "Ellipticity values should not exceed 0.9"


def test_gene_gam():
    from solve_lenseq import gene_gam

    # Test for a small redshift
    z_small = 0.5
    gamma_small = gene_gam(z_small)
    assert isinstance(gamma_small, float), "Output should be a float"
    assert gamma_small >= 0, "Gamma should be non-negative"

    # Test for a large redshift
    z_large = 2.0
    gamma_large = gene_gam(z_large)
    assert isinstance(gamma_large, float), "Output should be a float"
    assert gamma_large >= 0, "Gamma should be non-negative"

    # Test for edge case z=1
    z_edge = 1.0
    gamma_edge = gene_gam(z_edge)
    assert isinstance(gamma_edge, float), "Output should be a float"
    assert gamma_edge >= 0, "Gamma should be non-negative"


def test_gene_ang_gal():
    from solve_lenseq import gene_ang_gal

    p_h = np.zeros(1000)  # in units of degree
    p_gal = gene_ang_gal(p_h)  # in units of degree
    assert all(-180 <= p_angle <= 180 for p_angle in p_gal)
    sigma_p_gal = np.std(p_gal)
    expected_alignment = 35.4  # in units of degree
    assert pytest.approx(sigma_p_gal, rel=0.1) == expected_alignment
