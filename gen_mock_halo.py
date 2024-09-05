import getopt
import sys
import time
from itertools import chain

import joblib
import numpy as np
from colossus.cosmology import cosmology
from colossus.halo import profile_nfw
from scipy import interpolate

import global_value as g
import lens_gals
import lens_halo
import lens_subhalo
import solve_lenseq
import solve_lenseq_glafic
import source_qso
import source_sn
import source_tab

# ## Definition of functions


def run_command_line(argv):
    """
    Parse the command line arguments and initialize global parameters for mock halo generation.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param argv: The list of arguments passed to the command line.
    :type  argv: list

    Returns
    -----------------------------------------------------------------------------------------------
    None. This function initializes global parameters based on the parsed arguments and does not return
    any value. Exits the program if an invalid option is encountered or help is requested.
    """
    try:
        opts, args = getopt.getopt(
            argv,
            "h",
            [
                "help",
                "area=",
                "ilim=",
                "zlmax=",
                "source=",
                "prefix=",
                "solver=",
                "nworker=",
            ],
        )
    except getopt.GetoptError as err:
        print("Error: wrong option")
        print(str(err))
        sys.exit(0)

    # default values
    area = 2000.0
    ilim = 23.3
    zlmax = 3.0
    source = "qso"
    prefix = "test_qso"
    solver = "glafic"
    nworker = 1

    for o, a in opts:
        if o in ("-h", "--help"):
            print("Example:")
            print("python gen_mock_halo.py --area=20.0 --ilim=23.3 --zlmax=3.0 --source=qso --prefix=test --solver=glafic --nworker=1")
            sys.exit()
        elif o in ("--area"):
            area = float(a)
        elif o in ("--ilim"):
            ilim = float(a)
        elif o in ("--zlmax"):
            zlmax = float(a)
        elif o in ("--source"):
            source = a
        elif o in ("--prefix"):
            prefix = a
        elif o in ("--solver"):
            solver = a
        elif o in ("--nworkers"):
            nworker = int(a)

    print("# area  : %f" % area)
    print("# ilim  : %f" % ilim)
    print("# zlmax : %f" % zlmax)
    print("# source: %s" % source)
    print("# prefix: %s" % prefix)
    print("# solver: %s" % solver)
    if nworker == 1:
        print("# process: single")
    else:
        print("# process: multi(%d core)" % nworker)

    source_qso.set_spline_kcor_qso()
    source_sn.set_spline_kcor_sn(1)

    # if solver == "glafic":
    #     flag_solver = 0
    # elif solver == 'lenstronomy':
    #     flag_solver = 1
    # else:
    #     print("Error: solver should be either glafic or lenstronomy")
    #     sys.exit(0)

    if source == "qso":
        g.flag_type_min = 0
        g.flag_type_max = 0
    elif source == "sn":
        g.flag_type_min = 1
        g.flag_type_max = 5
    else:
        print("Error: source should be either qso or sn")
        sys.exit(0)

    # global parameters

    g.imax = ilim + 3.5
    g.mlim = ilim
    g.area = area
    g.ilim = ilim
    g.zlmax = zlmax
    g.source = source
    g.prefix = prefix
    g.solver = solver
    g.nworker = nworker
    # preparation
    cosmo = init_cosmo(g.COSMO_MODEL)
    g.paramc, g.params = lens_gals.gals_init(g.TYPE_SMHM)

    # cosmological parameters
    comega, clambda, cweos, chubble = calc_cosmo_for_glafic(cosmo)
    g.cosmo_omega = comega
    g.cosmo_lambda = clambda
    g.cosmo_hubble = chubble


def calc_cosmo_for_glafic(cosmo):
    """
    Calculate cosmological parameters for use in the glafic software.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param cosmo: An instance of a cosmological model, providing access to various cosmological parameters.
    :type  cosmo: an instance of a class with attributes Om0 (matter density), H0 (Hubble constant),
                   and possibly other custom attributes used within this function (e.g., g.nonflat).

    Returns
    -----------------------------------------------------------------------------------------------
    tuple(float, float, float, float)
        A tuple containing:
            - cosmo_omega: Current matter density parameter (Omega_m0).
            - cosmo_lambda: Cosmological constant parameter (Lambda), calculated with adjustments if needed.
            - cosmo_weos: Equation of state parameter for dark energy (w).
            - cosmo_hubble: Normalized Hubble constant (H0/100).
    """
    cosmo_omega = cosmo.Om0
    cosmo_lambda = round(1.0 + g.nonflat - cosmo.Om0, 5)
    cosmo_weos = g.cosmo_weos
    cosmo_hubble = round(cosmo.H0 / 100.0, 5)

    return cosmo_omega, cosmo_lambda, cosmo_weos, cosmo_hubble


def init_cosmo(model="planck18"):
    """
    Initialize the cosmological model

    Parameters
    -----------------------------------------------------------------------------------------------
    None

    Returns
    -----------------------------------------------------------------------------------------------
    cosmo: colossus.cosmology.Cosmology
        An instance of the cosmology class initialized with Planck 2018 parameters.
    """
    cosmo = cosmology.setCosmology(model)
    return cosmo


def storage_result(
    lens_par,
    srcs_par,
    ms,
    fs,
    ein,
    frac_sh_trunc,
    result_lens_par,
    result_srcs_par,
    result_imgs_par,
    result_kapg_par,
    result_evnt_par,
    cosmo,
):
    """
    Store the results of gravitational lensing calculations, including lens and source parameters,
    image properties, and event characteristics based on certain criteria.

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

    :param ms: Source original magnitude.
    :type  ms: float

    :param fs: Source type
    :type  fs: int

    :param ein: Einstein radius in units of arcsec
    :type  ein: float

    :param frac_sh_trunc: Fraction the subhalo mass after tidal truncation agaist the mass before tidal truncation.
    :type  frac_sh_trunc: float

    :param result_lens_par: Accumulator list to store output lens parameters.
    :type  result_lens_par: list

    :param result_srcs_par: Accumulator list to store output source parameters.
    :type  result_srcs_par: list

    :param result_imgs_par: Accumulator list to store calculated image parameters.
    :type  result_imgs_par: list

    :param result_kapg_par: Accumulator list to store calculated kappa and gamma values.
    :type  result_kapg_par: list

    :param result_evnt_par: Accumulator list to store parameters defining lensing events.
    :type  result_evnt_par: list

    :param cosmo: Cosmological model instance.
    :type  cosmo: colossus.cosmology.Cosmology

    Returns
    -----------------------------------------------------------------------------------------------
    None: This function does not return any value. It populates the provided accumulator lists with
    results from the gravitational lensing calculations.
    """
    out_img, nim, sep, mag, magmax, fr, kapgam = solve_lenseq.calc_image(lens_par, srcs_par, ein, g.rt_range, g.flag_mag, cosmo)

    flag_out = 0
    if nim > 0:
        mobs = ms - 2.5 * np.log10(mag)
        mmin = ms - 2.5 * np.log10(magmax)

        # events with multiple images
        if nim > 1:
            if mobs <= g.mlim and sep >= g.sepmin and sep <= g.sepmax and fr >= g.frlim:
                flag_out += 1
                evnt_par = [nim, ms, mobs, fr, sep, fs, frac_sh_trunc, ein]

        # events with high magnifications, not necessarily multiply imaged
        if mmin <= g.mlim and magmax >= g.maglim:
            flag_out += 2
            if nim == 1:
                evnt_par = [nim, ms, mobs, 0.0, 0.0, 0.0, frac_sh_trunc, ein]
            else:
                evnt_par = [nim, ms, mobs, fr, sep, fs, frac_sh_trunc, ein]

    if flag_out > 0:
        result_lens_par.append(lens_par)
        result_srcs_par.append(srcs_par)
        imgs_par = out_img
        kapg_par = kapgam
        result_imgs_par.append(imgs_par)
        result_kapg_par.append(kapg_par)
        evnt_par_app = [mmin, flag_out]
        result_evnt_par.append(evnt_par + evnt_par_app)

    return


def lens_judge_d_genmock(fov, zz_ar):
    """
    Generate mock gravitational lensing data based on a field of view and an array of redshifts.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param fov: Field of view in square degrees.
    :type  fov: float

    :param zz_ar: Array of redshift values where lenses are to be judged.
    :type  zz_ar: numpy.ndarray

    Returns
    -----------------------------------------------------------------------------------------------
    tuple
        A tuple containing lists of lens parameters, source parameters, image parameters,
        kappa and gamma parameters, and event parameters generated by the simulation.
    """
    cosmo = init_cosmo()
    result_lens_par = []
    result_srcs_par = []
    result_evnt_par = []
    result_imgs_par = []
    result_kapg_par = []
    # Start for loop-1 of redshift
    for zl in zz_ar:
        zzl = np.array([zl] * len(MMh))
        convert_t = 1.0 / cosmo.angularDiameterDistance(zl) * 206264.8
        zs_min = zl + 0.5 * dz
        # Obtain the expected number of the halo with mass in [M, M+e**dlnM]
        # in the survey area with redshift depth of [z,z+dz]
        # Here, we consider MMh is halo mass with 200c definition
        NNh = fov * lens_halo.dNhalodzdlnM_lens(MMh, zl, cosmo) * dlnMh * dz
        Nh = np.random.poisson(NNh)
        indices = np.nonzero(Nh)[0]
        # Subtract only the halo existing component
        cut_z = zzl[indices]
        cut_Nh = Nh[indices]
        cut_Mh = MMh[indices]
        # lens redshift & halo mass
        zl_tab = np.repeat(cut_z, cut_Nh)
        Mhl_tab = np.repeat(cut_Mh, cut_Nh)
        mh_z_vec = np.vstack((np.log10(Mhl_tab), zl_tab)).T
        box_src = interp_bsrc_h(mh_z_vec)
        nn_src = 4.0 * box_src**2 * d
        n_src = np.random.poisson(nn_src)

        indices2 = np.nonzero(n_src)[0]
        if len(indices2) == 0:
            continue

        # Subtract only haloes with at least one source present behind
        cut_n_src, cut_box_src = n_src[indices2], box_src[indices2]
        zl_tab2 = zl_tab[indices2]
        Mhl_200c_tab2 = Mhl_tab[indices2]

        Mhl_vir_tab2, rvirhl_tab2, conhl_vir_tab2, eliphl_tab2, polarhl_tab2 = lens_halo.halo_properties_200c2vir(
            Mhl_200c_tab2, zl, g.sig_c
        )
        rs_hl_tab2 = rvirhl_tab2 / conhl_vir_tab2

        Mcenl_tab2, tb_cenl_tab2, elipcenl_tab2, polarcenl_tab2 = lens_gals.galaxy_properties(
            Mhl_vir_tab2, zl, polarhl_tab2, g.paramc, g.frac_SM_IMF, g.TYPE_GAL_SIZE, g.sig_mcen, sig_tb=g.sig_tb
        )
        mmsh2, mmsh_acc2, NNsh2 = lens_subhalo.subhalo_mass_function(
            Mhl_vir_tab2, zl_tab2, min_Msh, interp_dnsh, interp_msh_acc_Mh, length=output_length
        )

        # Start: for-loop-2 of host halos
        for j, (mh, mh_200c, mcen, rvir_h, rs_h, con_h, tb_cen, e_h, p_h, e_cen, p_cen) in enumerate(
            zip(
                Mhl_vir_tab2,
                Mhl_200c_tab2,
                Mcenl_tab2,
                rvirhl_tab2,
                rs_hl_tab2,
                conhl_vir_tab2,
                tb_cenl_tab2,
                eliphl_tab2,
                polarhl_tab2,
                elipcenl_tab2,
                polarcenl_tab2,
                strict=True,
            )
        ):
            indices_sh = np.nonzero(NNsh2[j])[0]
            cut_Nsh = NNsh2[j][indices_sh]
            cut_msh = mmsh2[j][indices_sh]
            cut_msh_acc = mmsh_acc2[j][indices_sh]
            # Subhalo mass
            mshl_tab = np.repeat(cut_msh, cut_Nsh)
            msh_accl_tab = np.repeat(cut_msh_acc, cut_Nsh)

            # mshsat_tot = 0
            bsrc_sh = 0
            # Start: If at least one subhalo exists in loop-2 of host halos
            if len(mshl_tab) != 0:
                # Subhalo spatial distribution
                x_sh_elip, y_sh_elip = lens_subhalo.subhalo_distribute(rvir_h, con_h, e_h, p_h, xfunc, len(mshl_tab))
                tx_sh_elip = x_sh_elip * convert_t
                ty_sh_elip = y_sh_elip * convert_t

                mshacc_z_vec = np.vstack((np.log10(msh_accl_tab), np.array([zl] * len(mshl_tab)))).T
                bsrc_sh = interp_bsrc_sh(mshacc_z_vec)

            # Start: for-loop-3 of lensing events for each host halos
            for _ in range(cut_n_src[j]):
                kk = int(n_src_sample * np.random.rand())
                zs = zs_tab[kk]

                if zs > zs_min:
                    ms = m_tab[kk]
                    fs = f_tab[kk]
                    gg, tgg = solve_lenseq.set_shear(zs)
                    kap_pert = 0.0

                    lens_par = [
                        ["anfw", zl, mh, 0.0, 0.0, e_h, p_h, con_h, 0.0],
                        ["ahern", zl, mcen, 0.0, 0.0, e_cen, p_cen, tb_cen, 0.0],
                        ["pert", zl, zs, 0.0, 0.0, gg, tgg, 0.0, kap_pert],
                    ]

                    ein_hl_zs = solve_lenseq_glafic.calc_ein(zs, lens_par, cosmo)

                    if ein_hl_zs > (g.sepmin * 0.2):
                        sx = (np.random.rand() - 0.5) * 2.0 * cut_box_src[j]
                        sy = (np.random.rand() - 0.5) * 2.0 * cut_box_src[j]

                        if np.abs(sx) < (ein_hl_zs * (g.rt_range + 1.0)) and np.abs(sy) < (ein_hl_zs * (g.rt_range + 1.0)):
                            srcs_par = [zs, sx, sy]

                            storage_result(
                                lens_par,
                                srcs_par,
                                ms,
                                fs,
                                ein_hl_zs,
                                g.flag_h,
                                result_lens_par,
                                result_srcs_par,
                                result_imgs_par,
                                result_kapg_par,
                                result_evnt_par,
                                cosmo,
                            )

            # Start: If at least one subhalo exists in loop-2 of host halos
            if len(mshl_tab) != 0:
                rb_cen = tb_cen / convert_t
                box_src_sh = bsrc_sh
                nn_sh_src = 4.0 * box_src_sh**2 * d
                n_sh_src = np.random.poisson(nn_sh_src)
                indices_sh2 = np.nonzero(n_sh_src)[0]

                cut_box_src_sh, cut_n_sh_src = (box_src_sh[indices_sh2], n_sh_src[indices_sh2])
                mshl_vir_tab2, msh_accl_vir_tab2 = (mshl_tab[indices_sh2], msh_accl_tab[indices_sh2])
                mshl_200c_tab2 = mshl_vir_tab2 * mh_200c / mh
                tx_sh_elip2, ty_sh_elip2 = (tx_sh_elip[indices_sh2], ty_sh_elip[indices_sh2])

                con_sh_tab2 = lens_subhalo.concentration_subhalo_w_scatter(
                    mshl_vir_tab2, mshl_200c_tab2, msh_accl_vir_tab2, zl, g.sig_c_sh, cosmo
                )

                elipshl_tab2, polarshl_tab2 = solve_lenseq.gene_e_ang_halo(msh_accl_vir_tab2)
                msatl_tab2, tb_sat_tab2, elipsatl_tab2, polarsatl_tab2 = lens_gals.galaxy_properties(
                    msh_accl_vir_tab2, zl, polarshl_tab2, g.params, g.frac_SM_IMF, g.TYPE_GAL_SIZE, g.sig_msat, sig_tb=g.sig_tb
                )  # TODO check
                # Start: for-loop-3(2) of subhalos in each host halo
                for js, (msh, msh_acc, msat, tx, ty, con_sh, tb_sat, e_sh, p_sh, e_sat, p_sat) in enumerate(
                    zip(
                        mshl_vir_tab2,
                        msh_accl_vir_tab2,
                        msatl_tab2,
                        tx_sh_elip2,
                        ty_sh_elip2,
                        con_sh_tab2,
                        tb_sat_tab2,
                        elipshl_tab2,
                        polarshl_tab2,
                        elipsatl_tab2,
                        polarsatl_tab2,
                        strict=True,
                    )
                ):
                    # Start: for-loop-4 of lensing events of each subhalo in the host halo
                    for _ in range(cut_n_sh_src[js]):
                        kk = int(n_src_sample * np.random.rand())
                        zs = zs_tab[kk]
                        if zs > zs_min:
                            kext_zs = lens_halo.kappa_ext_from_host_halo(
                                tx, ty, zl, zs, mh - msh, rs_h, con_h, mcen, rb_cen, e_h, p_h, e_cen, p_cen
                            )
                            if kext_zs < g.kext_zs_lim:
                                ms = m_tab[kk]
                                fs = f_tab[kk]
                                gg, tgg = solve_lenseq.set_shear(zs)
                                kap_pert = 0.0
                                lens_par = [
                                    ["anfw", zl, msh_acc, 0.0, 0.0, e_sh, p_sh, con_sh, 0.0],
                                    ["ahern", zl, msat, 0.0, 0.0, e_sat, p_sat, tb_sat, 0.0],
                                    # ["anfw", zl, Mhosthl_tab2[j], -tx, -ty, e_h, p_h, con_h, 0.0],
                                    ["anfw", zl, mh - msh, -tx, -ty, e_h, p_h, con_h, 0.0],
                                    ["ahern", zl, mcen, -tx, -ty, e_cen, p_cen, tb_cen, 0.0],
                                    ["pert", zl, zs, 0.0, 0.0, gg, tgg, 0.0, kap_pert],
                                ]
                                ein_shl_zs = solve_lenseq_glafic.calc_ein(zs, lens_par, cosmo)
                                if ein_shl_zs > (g.sepmin * 0.2):
                                    sx = (np.random.rand() - 0.5) * 2.0 * cut_box_src_sh[js]
                                    sy = (np.random.rand() - 0.5) * 2.0 * cut_box_src_sh[js]
                                    if np.abs(sx) < (ein_shl_zs * (g.rt_range + 1.0)) and np.abs(sy) < (ein_shl_zs * (g.rt_range + 1.0)):
                                        srcs_par = [zs, sx, sy]
                                        frac_sh_trunc = msh / msh_acc
                                        storage_result(
                                            lens_par,
                                            srcs_par,
                                            ms,
                                            fs,
                                            ein_shl_zs,
                                            frac_sh_trunc,
                                            result_lens_par,
                                            result_srcs_par,
                                            result_imgs_par,
                                            result_kapg_par,
                                            result_evnt_par,
                                            cosmo,
                                        )
                    # End: for-loop-4 of lensing events of each subhalo in the host halo
                # End: for-loop-3(2) of subhalos in each host halo
            # End: If at least one subhalo exists in loop-2 of host halos
        # End: for-loop-2 of host halos
    # End: for-loop-1 of redshift

    return (
        result_lens_par,
        result_srcs_par,
        result_imgs_par,
        result_kapg_par,
        result_evnt_par,
    )


def dump_result(lens_tup, srcs_tup, imgs_tup, kapg_tup, pars_tup, exec_time, ofile1, ofile2):
    """
    Write simulation results into two specified output files.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param lens_tup: Tuple containing lensing data.
    :type  lens_tup: tuple

    :param srcs_tup: Tuple containing source data.
    :type  srcs_tup: tuple

    :param imgs_tup: Tuple containing image data.
    :type  imgs_tup: tuple

    :param kapg_tup: Tuple containing kappa and gamma data.
    :type  kapg_tup: tuple

    :param pars_tup: Tuple containing various parameters related to the simulations.
    :type  pars_tup: tuple

    :param exec_time: Execution time for the simulation.
    :type  exec_time: float

    :param ofile1: File object for writing detailed lensing data.
    :type  ofile1: file-object

    :param ofile2: File object for writing summary data.
    :type  ofile2: file-object

    Returns
    -----------------------------------------------------------------------------------------------
    None
        This function writes directly to the file objects provided and does not return any value.
    """
    count_fout1 = [i for i, pars in enumerate(pars_tup) if pars[9] == 1]
    count_fout2 = [i for i, pars in enumerate(pars_tup) if pars[9] == 2]
    count_fout3 = [i for i, pars in enumerate(pars_tup) if pars[9] == 3]
    count_sh = [i for i, lens in enumerate(lens_tup) if len(lens) == 5 and (i in count_fout1 or i in count_fout3)]

    out_label = (
        "# [0]nim, [1]mori, [2]mobs, [3]sep, [4]ein, [5]fs, [6]frac_sh_trunc [7]mmin [8]flag_out [9]zs [10]xs [11]ys\n"
        "# [12+8*i]m_type_i, [13+8*i]zl_i, [14+8*i]mass_i, [15+8*i]xl_i, [16+8*i]yl_i\n"
        "# [17+8*i]e_i, [18+8*i]p_i, [19+8*i]param1(con, tb, ...)_i, i=0,1,2,3,4\n"
        "# i=0,1 correspond subhalos and satellite gals if not exist, stored -1, please extract using e.g. np.extract(arr > 0, arr)\n"
        "# i=2,3 correspond host halos and central gals\n"
        "# i=4 corresponds perturbation\n"
        "# m_type: 1 = anfw, 2 = ahern, 3 = pert\n"
        "# observable multi events: %d (subhalo %d), high-mag events: %d\n"
        "# total area, nworkers, and exec time : %3.1f [deg^2], %d process, %5.2f [sec]\n"
        % (
            len(count_fout1) + len(count_fout3),
            len(count_sh),
            len(count_fout2) + len(count_fout3),
            g.area,
            g.nworker,
            exec_time,
        )
    )

    ofile1.write(out_label)

    for j, lens_list in enumerate(lens_tup):
        out_pars = "%d %13e %13e %13e %13e %d %13e %13e %d %8.4f %13e %13e " % (
            pars_tup[j][0],
            pars_tup[j][1],
            pars_tup[j][2],
            pars_tup[j][4],
            pars_tup[j][7],
            pars_tup[j][5],
            pars_tup[j][6],
            pars_tup[j][8],
            pars_tup[j][9],
            srcs_tup[j][0],
            srcs_tup[j][1],
            srcs_tup[j][2],
        )
        ofile1.write(out_pars)

        if len(lens_list) == 3:
            for _ in range(2):
                # subhalo
                out = "%2d %7.4f %13e %13e %13e %13e %13e %13e " % (
                    -1,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                )
                ofile1.write(out)

            for i, lens in enumerate(lens_list):
                # type, zl, mh, x, y, e, p, param1(con, tb, ...)
                out = "%2d %7.4f %13e %13e %13e %13e %13e %13e " % (
                    i,
                    lens[1],
                    lens[2],
                    lens[3],
                    lens[4],
                    lens[5],
                    lens[6],
                    lens[7],
                )
                ofile1.write(out)

        elif len(lens_list) == 5:
            for i, lens in enumerate(lens_list):
                out = "%2d %7.4f %13e %13e %13e %13e %13e %13e " % (
                    i,
                    lens[1],
                    lens[2],
                    lens[3],
                    lens[4],
                    lens[5],
                    lens[6],
                    lens[7],
                )
                ofile1.write(out)

        ofile1.write("\n")

    ofile1.flush()

    out_label = (
        "# num_image x_src y_src, sep, ein, fr, zs, fs, flag_out, frac_sh_trunc\n"
        "# x_img, y_img, mag, delay, kappa, gamma1, gamma2 kappa*\n"
        "# zmin, zmax, log10Mhmin, log10Mhmax, log10mshmin, log10mshmax, n_bins: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %d\n"
        % (
            zmin,
            zmax,
            np.log10(Mhmin),
            np.log10(Mhmax),
            np.log10(min_Msh),
            np.log10(max_Msh),
            n_bins,
        )
    )

    ofile2.write(out_label)
    for j, src in enumerate(srcs_tup):
        # num_image x_src y_src, sep, ein, fr, zs, fs, flag_out, frac_sh_trunc
        out = "%d %13e %13e %13e %13e %13e %13e %d %d %13e\n" % (
            pars_tup[j][0],
            src[1],
            src[2],
            pars_tup[j][4],
            pars_tup[j][7],
            pars_tup[j][3],
            src[0],
            pars_tup[j][5],
            pars_tup[j][9],
            pars_tup[j][6],
        )
        ofile2.write(out)

        for img, kapg in zip(imgs_tup[j], kapg_tup[j], strict=False):
            # x_img, y_img, mag, delay, kappa, gamma1, gamma2 kappa_from_gal
            out = "%13e %13e %13e %13e %13e %13e %13e %13e\n" % (
                img[0],
                img[1],
                img[2],
                img[3],
                kapg[0],
                kapg[1],
                kapg[2],
                kapg[3],
            )
            ofile2.write(out)

    ofile2.flush()

    return


def dump_setup(ofile):
    """
    Dump the configuration parameters to an output file.

    Parameters
    -----------------------------------------------------------------------------------------------
    :param ofile: The output file stream where the configuration is to be written.
    :type  ofile: _io.TextIOWrapper

    Returns
    -----------------------------------------------------------------------------------------------
    None
    """
    for key, value in g.__dict__.items():
        ofile.write(f"{key}: {value}\n")

    ofile.flush()

    return


if __name__ == "__main__":
    run_command_line(sys.argv[1:])

    cosmo = init_cosmo()
    fov = g.area
    m_tab, zs_tab, f_tab = source_tab.make_srctab(g.imax, fov, g.flag_type_min, g.flag_type_max, cosmo)

    d = float(len(m_tab)) / (fov * 3600.0 * 3600.0)
    n_src_sample = float(len(m_tab))

    # setting for calculated redshift of lensing object
    dz = 0.001
    zmin = g.zlmin  # for interp_bsrc_h, zmin cannot be < 0.1 due to the way to create interpolators
    zmax = g.zlmax
    zz_ar = np.arange(zmin, zmax + dz, dz)

    # Create interpolatar of subhalo mf
    min_Msh = 10**g.log10Msh_min  # [Modot/h]
    Mhmin = max(10**g.log10Mh_min, 10 * min_Msh)  # [Modot/h]
    Mhmax = 10**g.log10Mh_max  # [Modot/h]
    dlogMh = 0.001
    MMh = 10 ** np.arange(np.log10(Mhmin), np.log10(Mhmax), dlogMh)  # [Modot/h]
    # colossus.mass_func is given by dn/dlnM where ln is natural logarithmic
    lnMMh = np.log(MMh)
    dlnMh = np.log(10**dlogMh)

    n_split = int(float(fov / g.nworker / 1.0))
    ffov = [float(fov / (n_split * g.nworker))] * (n_split * g.nworker)

    if min(ffov) < 1:
        print("error, each fov is too small")
    else:
        # Set the probability function to distribute subhalos following NFW density profile of host halos
        x_te = np.logspace(-3, 3, 100)
        xfunc = interpolate.interp1d(
            profile_nfw.NFWProfile.mu(x_te),
            x_te,
            kind="cubic",
            fill_value="extrapolate",
        )

        n_bins = 100
        output_length = n_bins - 1
        max_Msh = Mhmax
        interp_dnsh, interp_msh_acc_Mh = lens_subhalo.create_interp_dndmsh(
            0, zmax + 0.1, Mhmin / 2.0, Mhmax * 2.0, min_Msh, cosmo, n_bins=n_bins
        )

        interp_bsrc_h = lens_halo.create_interp_bsrc_h(zmin, zmax + 0.1, Mhmin / 2.0, Mhmax * 2, cosmo)
        interp_bsrc_sh = lens_subhalo.create_interp_bsrc_sh(zmin, zmax + 0.1, min_Msh, max_Msh, cosmo)

        start_time = time.time()

        ofile1 = open("result/" + g.prefix + "_result.dat", "x")
        ofile2 = open("result/" + g.prefix + "_log.dat", "x")
        ofile3 = open("result/" + g.prefix + "_setup.dat", "x")

        result = joblib.Parallel(n_jobs=g.nworker, verbose=1)(joblib.delayed(lens_judge_d_genmock)(fov, zz_ar) for fov in ffov)
        lens_tabp, srcs_tabp, imgs_tabp, kapg_tabp, params_tabp = zip(*result, strict=False)
        src_tup = tuple(chain.from_iterable(srcs_tabp))
        lens_tup = tuple(chain.from_iterable(lens_tabp))
        imgs_tup = tuple(chain.from_iterable(imgs_tabp))
        kapg_tup = tuple(chain.from_iterable(kapg_tabp))
        pars_tup = tuple(chain.from_iterable(params_tabp))

        files = open("result/" + g.prefix + "_srcs_tup.txt", "x")
        filel = open("result/" + g.prefix + "_lens_tup.txt", "x")
        filei = open("result/" + g.prefix + "_imgs_tup.txt", "x")
        filek = open("result/" + g.prefix + "_kapg_tup.txt", "x")
        filep = open("result/" + g.prefix + "_pars_tup.txt", "x")
        for iteml, items, itemi, itemk, itemp in zip(lens_tup, src_tup, imgs_tup, kapg_tup, pars_tup, strict=False):
            filel.write(str(iteml) + "\n")
            files.write(str(items) + "\n")
            filei.write(str(itemi) + "\n")
            filek.write(str(itemk) + "\n")
            filep.write(str(itemp) + "\n")

        files.close()
        filel.close()
        filei.close()
        filek.close()
        filep.close()

        end_time = time.time()
        execution_time = end_time - start_time

        dump_result(lens_tup, src_tup, imgs_tup, kapg_tup, pars_tup, execution_time, ofile1, ofile2)
        dump_setup(ofile3)

        ofile1.close()
        ofile2.close()
        ofile3.close()
