# SL-Hammocks

SL-Hammocks is a Python code to generate mock catalogs of gravitationally lensed quasars and supernovae through a halo-model-based approach. The name is an acronym for Strong Lensing HAlo Model-based MOCK catalogS. This is the code developed as part of an LSST DESC project.

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

See [output.txt](/result/output.txt) for some explanations. You can download several examples of mock catalogs that are the output of this code in the [data_public](https://github.com/LSST-strong-lensing/data_public/tree/main/SL_hammocks_catalogs) repository of LSST.

## Licensing, credits and feedback
You are welcome to re-use the code, which is open source and freely available under terms consistent with BSD 3-Clause licensing (see [LICENSE](/LICENSE)) at your own risk; the author shall not take any responsibility for loss or damage caused by the use of this code.
If you use SL-hammocks or any mock catalogs generated by SL-hammocks for any scientific publication, we kindly ask you to cite the following paper:

- [K. T. Abe, et al., The Open Journal of Astrophysics, 8, 8 (2025)](https://ui.adsabs.harvard.edu/abs/2025OJAp....8E...8A/abstract) 

The current lead of this project and the development of SL-hammocks is Katsuya T. Abe (kta-cosmo). For feedback, please contact the author via github issues or email (ktabecosmology@gmail.com).
