import os
import unittest
import shutil
import numpy as np
from pca import pcaSED
from spec_utils import specUtils
from sklearn.decomposition import PCA as sklPCA


class testPCA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        sample_wavelen = np.arange(200, 1500)
        sample_flux = 100. * np.ones(1300)
        sample_flux[100:650] += np.arange(550)*2.
        sample_flux[650:1200] += np.arange(550)*1100.
        sample_flux[650:1200] -= np.arange(550)*2.

        sample_flux_2 = 100. * np.ones(1300)
        sample_flux_2[100:650] += np.arange(550)*1.
        sample_flux_2[650:1200] += np.arange(550)*550.
        sample_flux_2[650:1200] -= np.arange(550)*1.

        cls.sample_spec = np.array([sample_wavelen, sample_flux])
        cls.sample_spec_2 = np.array([sample_wavelen, sample_flux_2])

        if os.path.exists('scratch'):
            shutil.rmtree('scratch')

        os.mkdir('scratch')
        np.savetxt('scratch/sample.dat', cls.sample_spec.T,
                   header='Lambda Flux', delimiter=' ')
        np.savetxt('scratch/sample_2.dat', cls.sample_spec_2.T,
                   header='Lambda Flux', delimiter=' ')

    def test_load_full_spectra(self):

        test_pca = pcaSED()
        test_pca.load_full_spectra('scratch')

        self.assertItemsEqual(test_pca.spec_list_orig[0].wavelen,
                              self.sample_spec[0])
        self.assertItemsEqual(test_pca.spec_list_orig[1].wavelen,
                              self.sample_spec_2[0])
        self.assertItemsEqual(test_pca.spec_list_orig[0].flambda,
                              self.sample_spec[1])
        self.assertItemsEqual(test_pca.spec_list_orig[1].flambda,
                              self.sample_spec_2[1])
        names = [test_pca.spec_list_orig[0].name,
                 test_pca.spec_list_orig[1].name]
        names.sort()
        self.assertEqual('sample.dat', names[0])
        self.assertEqual('sample_2.dat', names[1])

    def test_PCA(self):

        test_pca = pcaSED()
        test_pca.load_full_spectra('scratch')

        test_pca.PCA(2, 249.9, 1300.1)

        self.assertItemsEqual(test_pca.wavelengths,
                              self.sample_spec[0][50:1101])

        names = test_pca.spec_names
        names.sort()
        self.assertEqual('sample.dat', names[0])
        self.assertEqual('sample_2.dat', names[1])

        test_spec = []
        su = specUtils()
        test_spec.append(su.scale_spectrum(self.sample_spec[1][50:1101]))
        test_spec.append(su.scale_spectrum(self.sample_spec_2[1][50:1101]))
        control_pca = sklPCA()
        control_pca.fit(test_spec)

        np.testing.assert_array_equal(control_pca.components_,
                                      test_pca.eigenspectra)
        np.testing.assert_equal(control_pca.mean_, test_pca.mean_spec)
        control_coeffs = np.array(control_pca.transform(test_spec))
        np.testing.assert_array_equal(control_coeffs, test_pca.coeffs)
        control_evr = control_pca.explained_variance_ratio_
        np.testing.assert_equal(control_evr, test_pca.explained_var)

    @classmethod
    def tearDownClass(cls):

        if os.path.exists('scratch'):
            shutil.rmtree('scratch')


if __name__ == '__main__':
    unittest.main()
