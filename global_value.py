import sys

import astropy.constants as const

# Definition of global parameters


class _const:
    def __init__(self):
        """
        A class to hold constant values used in cosmological computations.

        Attributes
        -----------------------------------------------------------------------------------------------
        cc : float
            Speed of light in meters per second.
        G : float
            Gravitational constant in m^3 kg^-1 s^-2.
        G_MpcMsun : float
            Gravitational constant in m^2 Mpc Msun^-1 s^-2.
        c2_G : float
            Square of the speed of light divided by the gravitational constant (G) in Mpc Msun^-1.
        kpc_to_Mpc : float
            Conversion factor from kiloparsecs to megaparsecs.
        cosmo_weos : float
            Equation of state parameter for dark energy.
        nonflat : float
            Non-flatness parameter for cosmology.
        zsmax : float
            Maximum redshift considered.
        rt_range : float
            Range for margin of boxsize in glafic calculation
        maxlev : int
            Maximum level for some computational process in glafic calculation
        flag_h : float
            Flag parameter for halo
        kext_zs_lim : float
            limit of external kappa for subhalo to be calculated
        sig_c : float
            Intrinsic lognormal scatter for concentration parameters of host halos
            (https://arxiv.org/abs/astro-ph/0608157)
        sig_c_sh : float
            Same as sig_c, but for subhalo
            (https://arxiv.org/abs/astro-ph/0608157)
        sig_mcen : float
            Intrinsic lognormal scatter for central galaxy mass.
            (https://iopscience.iop.org/article/10.3847/1538-4357/ac4cb4/pdf)
        sig_msat : float
            Sanme as sig_mcen, but for satellite galaxy
            (https://iopscience.iop.org/article/10.3847/1538-4357/ac4cb4/pdf)
        sig_tb : float
            Intrinsic lognormal scatter for effective radius appreared in Hernquist profile
        TYPE_GAL_SIZE : str
            Method to calculate galaxy half-light effective size (or rb in Hernquist profile)
            options: 'vdW23'(JWST-base), 'oguri20'(simple), 'karmakar23'(IllustrisTNG-base), (default 'vdW23')
        frac_SM_IMF : float
            Fraction of Stellar mass-to-light ratio with respect to Chabrier IMF.
            Chabrier: frac_SM_IMF=1.0, Salpeter: =1.715 (default 1.715)
        TYPE_SMHM : str
            Type of fitting formula for the stellar-mass-halo-mass relation.
            options: 'true','true_all','obs', see Berhoozi et al. 2019 Table J1,  default 'true'
        COSMO_MODEL : str
            Cosmological model used for computations,  default 'planck18'
        sepmin : float
            Minimum image separation
        sepmax : float
            Maximum image separation
        frlim : float
            Limit for a fraction value in some computation.
        flag_mag : int
            Flag for magnitude-related computations.
        maglim : float
            Magnitude limit for some selection process.
        zlmin : float
            Minimum lens redshift for generating lensed mocks
        log10Msh_min : float
            Minimum subhalo mass when generating mock catalogs in base-10 logarithmic scale
        log10Mh_min : float
            Minimum host halo mass when generating mock catalogs in base-10 logarithmic scale
        log10Mh_max : float
            Maximum host halo mass when generating mock catalogs in base-10 logarithmic scale

        Methods
        -----------------------------------------------------------------------------------------------
        __setattr__(name, value)
            Prevents reassignment of any attribute after its initial assignment.

        Exceptions
        -----------------------------------------------------------------------------------------------
        ConstError
            Custom exception raised when attempting to rebind a constant value.
        """
        self.cc = const.c.to("m s-1").value
        self.G = const.G.to("m3 kg^-1 s^-2").value
        self.G_MpcMsun = const.G.to("m2 Mpc Msun^-1 s^-2").value
        self.c2_G = self.cc**2 / self.G_MpcMsun
        self.kpc_to_Mpc = 1.0e-3
        self.cosmo_weos = -1.0
        self.nonflat = 0.0
        self.zsmax = 1.0e3
        self.rt_range = 4.0  # Give plenty of surface to measure lensing event
        self.maxlev = 5
        self.flag_h = -1.0
        self.kext_zs_lim = 0.4

        # Scatter of concentration params for halos & gals in log-normal distribution
        self.sig_c = 0.33
        self.sig_c_sh = 0.33
        self.sig_mcen = 0.2
        self.sig_msat = 0.2
        self.sig_tb = 0.46

        self.TYPE_GAL_SIZE = "vdW23"

        self.frac_SM_IMF = 1.715

        self.TYPE_SMHM = "true"

        self.COSMO_MODEL = "planck18"
        self.sepmin = 0.0
        self.sepmax = 100.0
        self.frlim = 0.1
        self.flag_mag = 3
        self.maglim = 3.0
        self.zlmin = 0.1
        self.zlmin_pop = 0.01
        self.log10Msh_min = 10.0
        self.log10Mh_min = 11.0
        self.log10Mh_max = 16.0

    class ConstError(TypeError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't rebind const (%s)" % name)
        self.__dict__[name] = value


sys.modules[__name__] = _const()
