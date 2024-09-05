import getopt
import sys
import time

import joblib
import numpy as np

import gen_mock_halo
import global_value as g
import lens_gals
import lens_halo
import lens_subhalo
import solve_lenseq

# ## Definition of functions


def run_command_line(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["help", "area=", "zlmax=", "log10Mhmin=", "switch_sub=", "prefix=", "nworker="])
    except getopt.GetoptError as err:
        print("Error: wrong option")
        print(str(err))
        sys.exit(0)

    # default values
    area = 20.0
    zlmax = 3.0
    log10Mh_min_pop = 12
    switch_sub = True
    prefix = "test_qso"
    nworker = 1

    for o, a in opts:
        if o in ("-h", "--help"):
            print("Example:")
            print("python gen_pop_halo.py --area=200.0 --zlmax=3.0 --log10Mhmin=12 --switch_sub=True --prefix=test_qso --nworker=1")
            sys.exit()
        elif o in ("--area"):
            area = float(a)
        elif o in ("--zlmax"):
            zlmax = float(a)
        elif o in ("--log10Mhmin"):
            log10Mh_min_pop = float(a)
        elif o in ("--switch_sub"):
            switch_sub = a in ("True", "true", "yes", "Y", "1")
        elif o in ("--prefix"):
            prefix = a
        elif o in ("--nworkers"):
            nworker = int(a)

    print("# area  : %f" % area)
    print("# zlmax : %f" % zlmax)
    print("# log10Mhmin : %f" % log10Mh_min_pop)
    print(f"# switch_sub: {switch_sub}")
    # print('# switch_sub: %s' % switch_sub)
    print("# prefix: %s" % prefix)
    if nworker == 1:
        print("# process: single")
    else:
        print("# process: multi(%d core)" % nworker)

    # global parameters

    g.area = area
    g.zlmax = zlmax
    g.prefix = prefix
    g.nworker = nworker
    g.log10Mh_min_pop = log10Mh_min_pop

    # preparation
    cosmo = gen_mock_halo.init_cosmo(g.COSMO_MODEL)
    g.paramc, g.params = lens_gals.gals_init(g.TYPE_SMHM)

    # cosmological parameters
    comega, clambda, cweos, chubble = gen_mock_halo.calc_cosmo_for_glafic(cosmo)
    g.cosmo_omega = comega
    g.cosmo_lambda = clambda
    g.cosmo_hubble = chubble

    g.switch_sub = switch_sub


def genpop_d(
    fov,
    zz_ar,
):
    cosmo = gen_mock_halo.init_cosmo()
    result_halogal_par = []
    # Start for loop-1 of redshift
    for zz in zz_ar:
        zz2 = np.full(len(MMh), zz)
        NNh = fov * lens_halo.dNhalodzdlnM_lens(MMh, zz, cosmo) * dlnMh * dz
        Nh = np.random.poisson(NNh)
        indices = np.nonzero(Nh)[0]
        if len(indices) == 0:
            continue

        zl_tab = np.repeat(zz2[indices], Nh[indices])
        Mhosthl_200c_tab = np.repeat(MMh[indices], Nh[indices])
        Mhosthl_tab, R_vir_tab, conhl_tab, eliphl_tab, polarhl_tab = lens_halo.halo_properties_200c2vir(Mhosthl_200c_tab, zz, g.sig_c)

        Mcenl_tab, tb_cen_tab, elipcenl_tab, polarcenl_tab = lens_gals.galaxy_properties(
            Mhosthl_tab, zz, polarhl_tab, g.paramc, g.frac_SM_IMF, g.TYPE_GAL_SIZE, g.sig_mcen, sig_tb=g.sig_tb
        )

        if g.switch_sub:
            mmsh, mmsh_acc, NNsh = lens_subhalo.subhalo_mass_function(
                Mhosthl_tab, zl_tab, min_Msh, interp_dnsh, interp_msh_acc_Mh, length=output_length
            )

            # Start: for-loop-2 of host halos
            for j, (mh, mh_200c, zl) in enumerate(zip(Mhosthl_tab, Mhosthl_200c_tab, zl_tab, strict=True)):
                indices_sh = np.nonzero(NNsh[j])[0]
                cut_Nsh = NNsh[j][indices_sh]
                cut_msh = mmsh[j][indices_sh]
                cut_msh_acc = mmsh_acc[j][indices_sh]
                # Subhalo mass
                msh_tab = np.repeat(cut_msh, cut_Nsh)
                msh_acc_tab = np.repeat(cut_msh_acc, cut_Nsh)
                zsub_tab = np.full(len(msh_tab), zz)
                # mshsat_tot = 0
                # Start: If at least one subhalo exists in loop-2 of host halos
                if len(msh_tab) != 0:
                    msh_200c_tab = msh_tab * mh_200c / mh
                    con_sh_tab = lens_subhalo.concentration_subhalo_w_scatter(msh_tab, msh_200c_tab, msh_acc_tab, zl, g.sig_c_sh, cosmo)

                    elipsh_tab, polarsh_tab = solve_lenseq.gene_e_ang_halo(msh_acc_tab)
                    msat_tab, tb_sat_tab, elipsat_tab, polarsat_tab = lens_gals.galaxy_properties(
                        msh_acc_tab, zl, polarsh_tab, g.params, g.frac_SM_IMF, g.TYPE_GAL_SIZE, g.sig_msat, sig_tb=g.sig_tb
                    )  # TODO check
                    halogal_par_mat = np.hstack(
                        (
                            zsub_tab.reshape(-1, 1),
                            msh_tab.reshape(-1, 1),
                            msh_acc_tab.reshape(-1, 1),
                            elipsh_tab.reshape(-1, 1),
                            polarsh_tab.reshape(-1, 1),
                            con_sh_tab.reshape(-1, 1),
                            msat_tab.reshape(-1, 1),
                            elipsat_tab.reshape(-1, 1),
                            polarsat_tab.reshape(-1, 1),
                            tb_sat_tab.reshape(-1, 1),
                        )
                    )

                    result_halogal_par.append(halogal_par_mat)
                # End: If at least one subhalo exists in loop-2 of host halos

        halogal_par_mat = np.hstack(
            (
                zl_tab.reshape(-1, 1),
                Mhosthl_tab.reshape(-1, 1),
                np.zeros_like(Mhosthl_tab).reshape(-1, 1),
                eliphl_tab.reshape(-1, 1),
                polarhl_tab.reshape(-1, 1),
                conhl_tab.reshape(-1, 1),
                Mcenl_tab.reshape(-1, 1),
                elipcenl_tab.reshape(-1, 1),
                polarcenl_tab.reshape(-1, 1),
                tb_cen_tab.reshape(-1, 1),
            )
        )

        result_halogal_par.append(halogal_par_mat)

    return result_halogal_par


def dump_result_in_d_pop(halogals_ar, exec_time, ofile1, k):
    for i in range(len(halogals_ar)):
        k += 1
        out = "%2d %7.4f %13e %13e %13e %13e %13e %13e %13e %13e %13e\n" % (
            k,
            halogals_ar[i][0],
            halogals_ar[i][1],
            halogals_ar[i][2],
            halogals_ar[i][3],
            halogals_ar[i][4],
            halogals_ar[i][5],
            halogals_ar[i][6],
            halogals_ar[i][7],
            halogals_ar[i][8],
            halogals_ar[i][9],
        )
        ofile1.write(out)

    ofile1.flush()

    return k


def dump_result(halogals_ar, exec_time, ofile1):
    out_label = "# [0]index, [1]zl, [2]mass_halo, [3]mass_acc, [4]e_h [5]p_h [6]con., [7]mass_galaxy, [8]e_g, [9]p_g, [10]tb\n"
    ofile1.write(out_label)
    out_label = "# min(zl), max(zl) and min(log10Mh): %.3f, %.3f, %.3f [Msun/h]\n" % (g.zlmin_pop, g.zlmax, g.log10Mh_min_pop)
    ofile1.write(out_label)
    out_label = "# total area, nworkers, and exec time : %.1f [deg^2], %d process, %.2f [sec]\n" % (
        g.area,
        g.nworker,
        exec_time,
    )
    ofile1.write(out_label)

    k = 0
    for i in range(len(halogals_ar)):
        for j in range(len(halogals_ar[i])):
            k += 1
            out = "%2d %7.4f %13e %13e %13e %13e %13e %13e %13e %13e %13e\n" % (
                k,
                halogals_ar[i][j][0],
                halogals_ar[i][j][1],
                halogals_ar[i][j][2],
                halogals_ar[i][j][3],
                halogals_ar[i][j][4],
                halogals_ar[i][j][5],
                halogals_ar[i][j][6],
                halogals_ar[i][j][7],
                halogals_ar[i][j][8],
                halogals_ar[i][j][9],
            )
            ofile1.write(out)

    ofile1.flush()

    return


def dump_setup(ofile):
    for key, value in g.__dict__.items():
        ofile.write(f"{key}: {value}\n")

    ofile.flush()

    return


if __name__ == "__main__":
    run_command_line(sys.argv[1:])

    cosmo = gen_mock_halo.init_cosmo()
    fovtot = g.area

    # setting for calculated redshift of lensing object
    dz = 0.001
    zmin = g.zlmin_pop  # for interp_bsrc_h, zmin cannot be < 0.1 due to the way to create interpolators
    zmax = g.zlmax
    zz_ar = np.arange(zmin, zmax + dz, dz)

    # Create interpolatar of subhalo mf
    min_Msh = 10**g.log10Mh_min_pop / 10.0  # [Modot/h]
    Mhmin = max(10**g.log10Mh_min_pop, 10 * min_Msh)  # [Modot/h]
    Mhmax = 1.0e16  # [Modot/h]
    dlogMh = 0.001
    MMh = 10 ** np.arange(np.log10(Mhmin), np.log10(Mhmax), dlogMh)  # [Modot/h]
    # colossus.mass_func is given by dn/dlnM where ln is natural logarithmic
    lnMMh = np.log(MMh)
    dlnMh = np.log(10**dlogMh)
    print(g.log10Mh_min_pop)

    n_split = float(fovtot / g.nworker / 1.0)
    ffov = [float(fovtot / (g.nworker * 1.0))] * (g.nworker)

    index = 0

    if min(ffov) < 0.01:
        print("error, each fov is too small")
    else:
        start_time = time.time()
        ofile1 = open("result/" + g.prefix + "_result.dat", "x")
        ofile3 = open("result/" + g.prefix + "_setup.dat", "x")
        if g.switch_sub:
            n_bins = 100
            output_length = n_bins - 1
            max_Msh = Mhmax
            interp_dnsh, interp_msh_acc_Mh = lens_subhalo.create_interp_dndmsh(
                0, zmax + 0.1, Mhmin / 2.0, Mhmax * 2.0, min_Msh, cosmo, n_bins=n_bins
            )

        result = joblib.Parallel(n_jobs=g.nworker, verbose=1)(joblib.delayed(genpop_d)(fov, zz_ar) for fov in ffov)
        halogals_ar = np.hstack(result)

        end_time = time.time()
        execution_time = end_time - start_time

        dump_result(halogals_ar, execution_time, ofile1)
        dump_setup(ofile3)

        ofile1.close()
        ofile3.close()
