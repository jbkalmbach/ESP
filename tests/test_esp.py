import os
import copy
import unittest
import shutil
import numpy as np
from esp import pcaSED
from esp import nearestNeighborEstimate, gaussianProcessEstimate
from esp import specUtils
from esp.lsst_utils import Bandpass
from esp.lsst_utils import BandpassDict
from esp.lsst_utils import Sed


class testESP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create sample spectra and bandpasses for testing.
        sample_wavelen = np.arange(200, 1500)
        sample_flux = 100. * np.ones(1300)

        # Bandpasses will be two top hat filters
        sb_1 = np.ones(400)
        sb_1[200:] -= 1.0
        sample_bandpass_1 = Bandpass(wavelen=np.arange(500, 900), sb=sb_1)
        sb_2 = np.ones(400)
        sb_2[:200] -= 1.0
        sample_bandpass_2 = Bandpass(wavelen=np.arange(500, 900), sb=sb_2)
        cls.test_bandpass_dict = BandpassDict([sample_bandpass_1,
                                               sample_bandpass_2],
                                              ['a', 'b'])

        # Create SED with mag_filter_1 - mag_filter_2 = 0.2
        sample_sed = Sed()
        sample_sed.setSED(wavelen=sample_wavelen,
                          flambda=sample_flux)
        sample_norm_1_1 = sample_sed.calcFluxNorm(10.1, sample_bandpass_1)
        sample_sed.multiplyFluxNorm(sample_norm_1_1)
        sample_flambda_1_1 = sample_sed.flambda[:500]
        sample_sed.setSED(wavelen=sample_wavelen,
                          flambda=sample_flux)
        sample_norm_1_2 = sample_sed.calcFluxNorm(9.9, sample_bandpass_2)
        sample_sed.multiplyFluxNorm(sample_norm_1_2)
        sample_flambda_1_2 = sample_sed.flambda[500:]
        sample_sed_flambda = np.append(sample_flambda_1_1,
                                       sample_flambda_1_2)
        sample_sed.setSED(wavelen=sample_wavelen, flambda=sample_sed_flambda)
        cls.sample_sed = sample_sed

        # Create SED with mag_filter_1 - mag_filter_2 = -0.2
        sample_sed_2 = Sed()
        sample_sed_2.setSED(wavelen=sample_wavelen,
                            flambda=sample_flux)
        sample_norm_2_1 = sample_sed_2.calcFluxNorm(7.9, sample_bandpass_1)
        sample_sed_2.multiplyFluxNorm(sample_norm_2_1)
        sample_flambda_2_1 = sample_sed_2.flambda[:500]
        sample_sed_2.setSED(wavelen=sample_wavelen,
                            flambda=sample_flux)
        sample_norm_2_2 = sample_sed_2.calcFluxNorm(8.1, sample_bandpass_2)
        sample_sed_2.multiplyFluxNorm(sample_norm_2_2)
        sample_flambda_2_2 = sample_sed_2.flambda[500:]
        sample_sed_2_flambda = np.append(sample_flambda_2_1,
                                         sample_flambda_2_2)
        sample_sed_2.setSED(wavelen=sample_wavelen,
                            flambda=sample_sed_2_flambda)
        cls.sample_sed_2 = sample_sed_2

        # Create SED with mag_filter_1 - mag_filter_2 = 0.0
        sample_sed_3 = Sed()
        sample_sed_3.setSED(wavelen=sample_wavelen,
                            flambda=sample_flux)
        sample_norm_3_1 = sample_sed_3.calcFluxNorm(6., sample_bandpass_1)
        sample_sed_3.multiplyFluxNorm(sample_norm_3_1)
        sample_flambda_3_1 = sample_sed_3.flambda[:500]
        sample_sed_3.setSED(wavelen=sample_wavelen,
                            flambda=sample_flux)
        sample_norm_3_2 = sample_sed_3.calcFluxNorm(6., sample_bandpass_2)
        sample_sed_3.multiplyFluxNorm(sample_norm_3_2)
        sample_flambda_3_2 = sample_sed_3.flambda[500:]
        sample_sed_3_flambda = np.append(sample_flambda_3_1,
                                         sample_flambda_3_2)
        sample_sed_3.setSED(wavelen=sample_wavelen,
                            flambda=sample_sed_3_flambda)
        cls.sample_sed_3 = sample_sed_3

        if os.path.exists('scratch_esp'):
            shutil.rmtree('scratch_esp')

        os.mkdir('scratch_esp')
        sample_spec = np.array([sample_wavelen, sample_sed.flambda])
        np.savetxt('scratch_esp/sample.dat', sample_spec.T,
                   header='Lambda Flux', delimiter=' ')
        sample_spec_2 = np.array([sample_wavelen, sample_sed_2.flambda])
        np.savetxt('scratch_esp/sample_2.dat', sample_spec_2.T,
                   header='Lambda Flux', delimiter=' ')

    def test_nearest_neighbor_predict(self):

        test_pca = pcaSED()
        test_pca.load_full_spectra('scratch_esp')
        test_pca.PCA(2, 249.9, 1300.1)
        test_colors = test_pca.calc_colors(self.test_bandpass_dict, 2)
        np.testing.assert_array_almost_equal(np.sort(test_colors, axis=0),
                                             [[-.2], [.2]])

        test_esp = nearestNeighborEstimate(test_pca, self.test_bandpass_dict,
                                           [[0.0]])
        test_esp_spec = test_esp.nn_predict(2)

        su = specUtils()
        test_spec = test_esp_spec.reconstruct_spectra(2)[0]
        control_spec = su.scale_spectrum(self.sample_sed_3.flambda[50:1101])
        np.testing.assert_array_almost_equal(test_spec, control_spec)

        # Test arguments to KNeighborsRegressor by making weight='distance'

        test_esp_dist = nearestNeighborEstimate(test_pca,
                                                self.test_bandpass_dict,
                                                [[0.1]])
        test_esp_spec = test_esp_dist.nn_predict(2, knr_args={'weights':
                                                              'distance'})
        test_spec_dist = test_esp_spec.reconstruct_spectra(2)[0]

        sed_dist_1 = su.scale_spectrum(self.sample_sed.flambda)
        sed_dist_2 = su.scale_spectrum(self.sample_sed_2.flambda)
        control_spec_dist = sed_dist_1*3. + sed_dist_2
        control_spec_dist = su.scale_spectrum(control_spec_dist[50:1101])
        np.testing.assert_array_almost_equal(test_spec_dist,
                                             control_spec_dist)

    def test_define_kernel(self):

        test_pca = pcaSED()
        test_pca.load_full_spectra('scratch_esp')
        test_pca.PCA(2, 249.9, 1300.1)
        test_colors = test_pca.calc_colors(self.test_bandpass_dict, 2)
        np.testing.assert_array_almost_equal(np.sort(test_colors, axis=0),
                                             [[-.2], [.2]])

        test_gp = gaussianProcessEstimate(test_pca, self.test_bandpass_dict,
                                          [[0.0]])

        test_exp_kernel = test_gp.define_kernel('exp', 1.0, 1.2,
                                                len(test_colors[0]))

        np.testing.assert_array_equal(test_exp_kernel.get_parameter_vector(),
                                      np.log([1.0, 1.2]))

        test_sqexp_kernel = test_gp.define_kernel('sq_exp', 1.5, 1.7,
                                                  len(test_colors[0]))
        np.testing.assert_array_equal(test_sqexp_kernel.get_parameter_vector(),
                                      np.log([1.5, 1.7]))

        test_sqexp_kernel = test_gp.define_kernel('matern_32', 1.1, 1.7,
                                                  len(test_colors[0]))
        np.testing.assert_array_equal(test_sqexp_kernel.get_parameter_vector(),
                                      np.log([1.1, 1.7]))

        test_sqexp_kernel = test_gp.define_kernel('matern_52', 1.5, 1.9,
                                                  len(test_colors[0]))
        np.testing.assert_array_equal(test_sqexp_kernel.get_parameter_vector(),
                                      np.log([1.5, 1.9]))

        with self.assertRaises(Exception):
            test_gp.define_kernel('rational_quadratic', 1.0, 1.0)

    def test_gp_predict(self):

        test_pca = pcaSED()
        test_pca.load_full_spectra('scratch_esp')
        test_pca.PCA(2, 249.9, 1300.1)
        test_colors = test_pca.calc_colors(self.test_bandpass_dict, 2)
        np.testing.assert_array_almost_equal(np.sort(test_colors, axis=0),
                                             [[-.2], [.2]])

        test_gp = gaussianProcessEstimate(test_pca, self.test_bandpass_dict,
                                          [[0.0]])
        test_kernel = test_gp.define_kernel('exp', 1.0, 1.0,
                                            len(test_colors[0]))
        test_gp_spec = test_gp.gp_predict(test_kernel,
                                          self.test_bandpass_dict)

        su = specUtils()
        test_spec = test_gp_spec.reconstruct_spectra(2)[0]
        control_spec = su.scale_spectrum(self.sample_sed_3.flambda[50:1101])
        np.testing.assert_array_almost_equal(test_spec, control_spec)

    @classmethod
    def tearDownClass(cls):

        if os.path.exists('scratch_esp'):
            shutil.rmtree('scratch_esp')


if __name__ == '__main__':
    unittest.main()
