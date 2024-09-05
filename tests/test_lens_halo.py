import os
import sys

import numpy as np
import pytest
from colossus.cosmology import cosmology

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


def test_dNhalodzdlnM_lens():
    from lens_halo import dNhalodzdlnM_lens

    Mh_ar = np.array([1e10, 1e13, 1e16])  # in units of M_sol/h
    z_ar = np.linspace(0, 5, 10)
    cosmo = cosmology.setCosmology("planck18")
    for z in z_ar:
        dN_ar = dNhalodzdlnM_lens(Mh_ar, z, cosmo)
        assert all(num > 0 for num in dN_ar)


def test_concent_m_w_scatter():
    from lens_halo import concent_m_w_scatter

    Mh_ar = np.array([1e10, 1e13, 1e16])  # in units of M_sol/h
    z_ar = np.linspace(0, 5, 5)
    lnsigma = 0.3
    # cosmo = cosmology.setCosmology("planck18")
    for z in z_ar:
        con_ar = concent_m_w_scatter(Mh_ar, z, lnsigma)
        assert all(con >= 1 for con in con_ar)


def test_halo_properties_200c2vir():
    from lens_halo import halo_properties_200c2vir

    mh_200c = np.array([1e12, 1e13, 1e14])
    zl = 0.5
    sig = 0.1

    mh_vir, r_vir, con_vir, e_h, p_h = halo_properties_200c2vir(mh_200c, zl, sig)

    assert len(mh_vir) == len(r_vir) == len(con_vir) == len(e_h) == len(p_h) == len(mh_200c)
    assert np.all((mh_vir / mh_200c > 0.5) & (mh_vir / mh_200c < 2.0))
    assert np.all(r_vir > 0)
    assert np.all(con_vir > 1)
    assert np.all((e_h >= 0) & (e_h <= 1))
    assert np.all((p_h >= -180) & (p_h < 180))


def test_kappa_ext_from_host_halo():
    from lens_halo import kappa_ext_from_host_halo

    tx = 1.0
    ty = 2.0
    zl = 0.5
    zs = 1.0
    mass_host = 1e14
    rs_h = 0.2
    con_h = 5.0
    mcen = 1e12
    rb_cen = 0.1
    e_h = 0.3
    p_h = 45.0
    e_cen = 0.2
    p_cen = 30.0

    kext_zs = kappa_ext_from_host_halo(tx, ty, zl, zs, mass_host, rs_h, con_h, mcen, rb_cen, e_h, p_h, e_cen, p_cen)

    assert isinstance(kext_zs, float)
    assert kext_zs > 0


def test_critical_surface_density():
    from lens_halo import critical_surface_density

    zl = 0.5
    zs = 1.0
    cosmo = cosmology.setCosmology("planck18")

    sigma_critical = critical_surface_density(zl, zs, cosmo)

    assert isinstance(sigma_critical, float)
    assert sigma_critical > 0


def test_bnorm_hern():
    from lens_halo import bnorm_hern

    m_g = 1e12
    zl = 0.5
    zs = 1.0
    rb = 1.0
    cosmo = cosmology.setCosmology("planck18")

    b_hern = bnorm_hern(m_g, rb, zl, zs, cosmo)

    assert isinstance(b_hern, float)
    assert b_hern > 0


def test_bnorm_nfw():
    from lens_halo import bnorm_nfw

    m_h = 1e12
    zl = 0.5
    zs = 1.0
    con = 10.0
    cosmo = cosmology.setCosmology("planck18")

    b_nfw = bnorm_nfw(m_h, con, zl, zs, cosmo)

    assert isinstance(b_nfw, float)
    assert b_nfw > 0


def test_kappa_dl_hern():
    from lens_halo import kappa_dl_hern

    uu = np.array([0, 1, 2])
    for u in uu:
        kappa_dl = kappa_dl_hern(u)
        assert isinstance(kappa_dl, float)
        assert kappa_dl > 0


def test_kappa_dl_nfw():
    from lens_halo import kappa_dl_nfw

    uu = np.array([0, 1, 2])
    for u in uu:
        kappa_dl = kappa_dl_nfw(u)
        assert isinstance(kappa_dl, float)
        assert kappa_dl > 0


def test_create_interp_bsrc_h():
    import global_value as g
    from lens_gals import gals_init
    from lens_halo import create_interp_bsrc_h

    z_min = 0.1
    z_max = 2.0
    Mh_min = 1e11
    Mh_max = 1e15
    cosmo = cosmology.setCosmology("planck18")
    g.paramc, g.params = gals_init()
    g.prefix = "check"
    interp_bsrc_h = create_interp_bsrc_h(z_min, z_max, Mh_min, Mh_max, cosmo)

    assert callable(interp_bsrc_h)
    test = [13.4, 1.2]
    assert interp_bsrc_h(test) == pytest.approx(23.75749906, rel=1e-2)
