# SL-Hammocks

SL-Hammocks is a Python code to generate mock catalogs of gravitationally lensed quasars and supernovae through a halo-model-based approach. The name is an acronym for Strong Lensing HAlo Model-based MOCK catalogS. 
This is the code that accompanies the project of DESC-PUB-00170, entitled "[A halo model approach for mock catalogs of time-variable strong gravitational lenses](https://confluence.slac.stanford.edu/display/LSSTDESC/DESC-PUB-00170)"(member access only), within LSST DESC.
The code in this repository is based on the code of [gen_mock_mo10p](https://github.com/oguri/gen_mock_mo10p#readme), which was originally developed in [Oguri & Marshall (2010)](https://ui.adsabs.harvard.edu/abs/2010MNRAS.405.2579O/abstract) and updated by e.g., [Oguri (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.480.3842O/abstract), [Lemon et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022arXiv220607714L/abstract), and [Shen et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022arXiv220804979S/abstract).

## Requirement

This code uses [glafic](https://github.com/oguri/glafic2) as a lens equation solver and [Colossus](https://bdiemer.bitbucket.io/colossus/) for computations related to cosmology and dark matter halos. Other required installations include [joblib](https://github.com/joblib/joblib) for parallel computing, [numpy](https://github.com/numpy/numpy) and [scipy](https://github.com/scipy/scipy) for basic numerical computations. These Python modules need to be installed before running this code.


## Examples

The full (1 realization) LSST quasar lens mock catalog is generated by
```
python gen_mock_halo.py --area=20000.0 --ilim=23.3 --zlmax=3.0 --source=qso --prefix=qso_mock --solver=glafic --nworker=1
```
The full (1 realization) LSST supernova lens mock catalog is generated by
```
python gen_mock_halo.py --area=50000.0 --ilim=22.6 --zlmax=2.0 --source=sn --prefix=sne_mock --solver=glafic --nworker=1
```

## Output files

See [output.txt](/result/output.txt) for some explanations.
You can download several examples of mock catalogs that are the output of this code in the [data_public](https://github.com/kta-cosmo/data_public/tree/main/SL_hammocks_catalogs) repository of LSST.

## Licensing, credits and feedback
You are welcome to re-use the code, which is open source and freely available under terms consistent with BSD 3-Clause licensing (see [LICENSE](/LICENSE)) at your own risk; the author shall not take any responsibility for loss or damage caused by the use of this code.
If you use SL-hammocks for any scientific publication, we kindly ask you to cite this [github repository](https://github.com/LSSTDESC/SL-hammocks) and the companion paper:

- [K. T. Abe, et al., arXiv:2411.07509](https://arxiv.org/abs/2411.07509) 

The current lead of [this project](https://confluence.slac.stanford.edu/display/LSSTDESC/DESC-PUB-00170) and the development of SL-hammocks is Katsuya T. Abe (kta-cosmo).
For feedback, please contact the author via github issues or email (ktabecosmology@gmail.com).
