import glafic

import gen_mock_halo
import global_value as g


def solve_lenseq_glafic(lens_par, srcs_par, ein, rt_range, cosmo):
    """
    Solve the gravitational lens equation using the GLAFIC software to find image positions and
    convergence/shear values for a given source position, lens parameters, Einstein radius, and cosmology.

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

    :param cosmo: Cosmological model instance
    :type  cosmo: colossus.cosmology

    Returns
    -----------------------------------------------------------------------------------------------
    img_out: list
        List of image positions, magnitude, and time delay (days) derived from solving the lens equation.

    kapgam: list
        List of convergence (kappa), shears (gamma1 and gamma2) values for each image position, and stellar convergence
    """
    comega, clambda, cweos, chubble = gen_mock_halo.calc_cosmo_for_glafic(cosmo)
    glafic.init(
        comega,
        clambda,
        cweos,
        chubble,
        "out",
        (-1.0) * ein * (rt_range + 3.0),
        (-1.0) * ein * (rt_range + 3.0),
        ein * (rt_range + 3.0),
        ein * (rt_range + 3.0),
        ein / 5.0,
        ein / 5.0,
        g.maxlev,
        verb=0,
    )

    glafic.startup_setnum(len(lens_par), 0, 0)
    for i in range(len(lens_par)):
        glafic.set_lens(
            i + 1,
            lens_par[i][0],
            lens_par[i][1],
            lens_par[i][2],
            lens_par[i][3],
            lens_par[i][4],
            lens_par[i][5],
            lens_par[i][6],
            lens_par[i][7],
            lens_par[i][8],
        )
    glafic.model_init(verb=0)

    img_out = glafic.point_solve(srcs_par[0], srcs_par[1], srcs_par[2], verb=0)
    kapgam = []
    for i in range(len(img_out)):
        a = glafic.calcimage(srcs_par[0], img_out[i][0], img_out[i][1])
        kapgam.append([a[3], a[4], a[5]])
    glafic.quit()

    # Output kappa*, convergence only from galaxy components

    glafic.init(
        comega,
        clambda,
        cweos,
        chubble,
        "out",
        (-1.0) * ein * (rt_range + 3.0),
        (-1.0) * ein * (rt_range + 3.0),
        ein * (rt_range + 3.0),
        ein * (rt_range + 3.0),
        ein / 5.0,
        ein / 5.0,
        g.maxlev,
        verb=0,
    )

    glafic.startup_setnum(1, 0, 0)
    j = 1  # to output convergence from cen/sat galaxy
    glafic.set_lens(
        1,
        lens_par[j][0],
        lens_par[j][1],
        lens_par[j][2],
        lens_par[j][3],
        lens_par[j][4],
        lens_par[j][5],
        lens_par[j][6],
        lens_par[j][7],
        lens_par[j][8],
    )
    glafic.model_init(verb=0)

    for i, img_kapgam in enumerate(kapgam):
        a_cen = glafic.calcimage(srcs_par[0], img_out[i][0], img_out[i][1])
        img_kapgam.append(a_cen[3])
    glafic.quit()

    return img_out, kapgam


def calc_ein(zs, lens_par, cosmo):
    """
    Calculate the Einstein radius for a given set of lens parameters and cosmology at specific source redshifts.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param zs: The source redshift at which to compute the Einstein radius.
    :type  zs: float

    :param lens_par: Parameters defining the lens model. Each lens parameter set should include
                     mass type (str), mass [M_sol/h], position x [arcsec], y [arcsec], ellipticity, position angle [degree],
                     concentration parameter/scale radius [arcsec], etc.
            e.g., lens_par = [
                        ["anfw", zl, M_h, x_h, y_h, e_h, p_h, con_h, 0.0],
                        ["ahern", zl,M_cen, x_cen, y_cen, e_cen, p_cen, tb_cen, 0.0],
                        ["pert", zl, zs, 0.0, 0.0, gg, tgg, 0.0, kap_pert],
                    ]
    :type  lens_par: list

    :param cosmo: Cosmological model instance
    :type  cosmo: colossus.cosmology

    Returns
    -----------------------------------------------------------------------------------------------
    ein: float
        The computed Einstein radius
    """
    comega, clambda, cweos, chubble = gen_mock_halo.calc_cosmo_for_glafic(cosmo)
    glafic.init(comega, clambda, cweos, chubble, "out_ein", -20.0, -20.0, 20.0, 20, 0.2, 0.2, g.maxlev, verb=0)

    glafic.startup_setnum(len(lens_par) - 1, 0, 0)
    for i in range(len(lens_par) - 1):
        glafic.set_lens(
            i + 1,
            lens_par[i][0],
            lens_par[i][1],
            lens_par[i][2],
            lens_par[i][3],
            lens_par[i][4],
            lens_par[i][5],
            lens_par[i][6],
            lens_par[i][7],
            lens_par[i][8],
        )
    glafic.model_init(verb=0)

    ein = glafic.calcein2(zs, 0.0, 0.0)
    glafic.quit()

    return ein


#
# for checks
#
if __name__ == "__main__":
    from colossus.cosmology import cosmology

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
    cosmo = cosmology.setCosmology("planck18")
    lens_par = [
        ["anfw", zl, M_h, x_h, y_h, e_h, p_h, con_h, 0.0],
        ["ahern", zl, M_cen, x_cen, y_cen, e_cen, p_cen, tb_cen, 0.0],
        ["pert", zl, zs, 0.0, 0.0, gg, tgg, 0.0, kap_pert],
    ]
    ein_hl_zs = calc_ein(zs, lens_par, cosmo)
    print(ein_hl_zs)
