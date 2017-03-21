import os
import unittest
import shutil
import numpy as np
from pca import pcaSED
from esp import nearestNeighborEstimate, gaussianProcessEstimate
from spec_utils import specUtils
from lsst_utils.Bandpass import Bandpass
from lsst_utils.BandpassDict import BandpassDict
from lsst_utils.Sed import Sed


class testESP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        sample_wavelen = np.arange(200, 1500)
        sample_flux = 100. * np.ones(1300)

        sb_1 = np.ones(400)
        sb_1[200:] -= 1.0
        sample_bandpass_1 = Bandpass(wavelen=np.arange(500, 900), sb=sb_1)
        sb_2 = np.ones(400)
        sb_2[:200] -= 1.0
        sample_bandpass_2 = Bandpass(wavelen=np.arange(500, 900), sb=sb_2)
        cls.test_bandpass_dict = BandpassDict([sample_bandpass_1,
                                               sample_bandpass_2],
                                              ['a', 'b'])

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

    def test_define_kernel(self):

        test_pca = pcaSED()
        test_pca.load_full_spectra('scratch_esp')
        test_pca.PCA(2, 249.9, 1300.1)
        test_colors = test_pca.calc_colors(self.test_bandpass_dict, 2)
        np.testing.assert_array_almost_equal(np.sort(test_colors, axis=0),
                                             [[-.2], [.2]])

        test_gp = gaussianProcessEstimate(test_pca, self.test_bandpass_dict,
                                          [[0.0]])

        test_exp_kernel = test_gp.define_kernel('exp', 1.0, 1.0)
        np.testing.assert_array_equal(test_exp_kernel.pars, [1.0, 1.0])
        # George has kernel type 0 for the constant kernel and 3 for exp
        self.assertEqual(test_exp_kernel.k1.kernel_type, 0)
        self.assertEqual(test_exp_kernel.k2.kernel_type, 3)

        test_sq_exp_kernel = test_gp.define_kernel('sq_exp', 1.5, 1.5)
        np.testing.assert_array_equal(test_sq_exp_kernel.pars, [1.5, 1.5])
        # George also has kernel type 4 for squared exp
        self.assertEqual(test_sq_exp_kernel.k1.kernel_type, 0)
        self.assertEqual(test_sq_exp_kernel.k2.kernel_type, 4)

        with self.assertRaises(Exception):
            test_gp.define_kernel('matern', 1.0, 1.0)

    def test_gp_predict(self):

        test_pca = pcaSED()
        test_pca.load_full_spectra('scratch_esp')
        test_pca.PCA(2, 249.9, 1300.1)
        test_colors = test_pca.calc_colors(self.test_bandpass_dict, 2)
        np.testing.assert_array_almost_equal(np.sort(test_colors, axis=0),
                                             [[-.2], [.2]])

        test_gp = gaussianProcessEstimate(test_pca, self.test_bandpass_dict,
                                          [[0.0]])
        test_kernel = test_gp.define_kernel('exp', 1.0, 1.0)
        test_gp_spec = test_gp.gp_predict(test_kernel)

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
