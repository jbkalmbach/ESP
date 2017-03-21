import os
import unittest
import shutil
import numpy as np
from pca import pcaSED
from spec_utils import specUtils
from lsst_utils.Bandpass import Bandpass
from lsst_utils.BandpassDict import BandpassDict
from lsst_utils.Sed import Sed
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

        sb_1 = np.ones(400)
        sb_1[200:] -= 1.0
        sample_bandpass_1 = Bandpass(wavelen=np.arange(500, 900), sb=sb_1)
        sb_2 = np.ones(400)
        sb_2[:200] -= 1.0
        sample_bandpass_2 = Bandpass(wavelen=np.arange(500, 900), sb=sb_2)
        cls.test_bandpass_dict = BandpassDict([sample_bandpass_1,
                                               sample_bandpass_2],
                                              ['a', 'b'])

        os.mkdir('scratch/results')

    def test_load_full_spectra(self):

        test_pca = pcaSED()
        test_pca.load_full_spectra('scratch')

        self.assertItemsEqual(test_pca.spec_list_orig[0].wavelen,
                              self.sample_spec[0])
        self.assertItemsEqual(test_pca.spec_list_orig[1].wavelen,
                              self.sample_spec_2[0])

        names = [test_pca.spec_list_orig[0].name,
                 test_pca.spec_list_orig[1].name]
        names.sort()
        self.assertEqual('sample.dat', names[0])
        self.assertEqual('sample_2.dat', names[1])

        if names[0] == 'sample.dat':
            first = 0
            second = 1
        else:
            first = 1
            second = 0
        self.assertItemsEqual(test_pca.spec_list_orig[first].flambda,
                              self.sample_spec[1])
        self.assertItemsEqual(test_pca.spec_list_orig[second].flambda,
                              self.sample_spec_2[1])

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

    def test_reconstruct_spectra(self):

        test_pca = pcaSED()
        test_pca.load_full_spectra('scratch')

        test_pca.PCA(2, 249.9, 1300.1)
        names = [test_pca.spec_list_orig[0].name,
                 test_pca.spec_list_orig[1].name]
        names.sort()
        if names[0] == 'sample.dat':
            first = 0
            second = 1
        else:
            first = 1
            second = 0

        test_specs = test_pca.reconstruct_spectra(2)
        su = specUtils()
        sample_scaled = su.scale_spectrum(self.sample_spec[1][50:1101])
        np.testing.assert_array_almost_equal(test_specs[first], sample_scaled)
        sample_scaled_2 = su.scale_spectrum(self.sample_spec_2[1][50:1101])
        np.testing.assert_array_almost_equal(test_specs[second],
                                             sample_scaled_2)

    def test_calc_colors(self):

        test_pca = pcaSED()
        test_pca.load_full_spectra('scratch')

        test_pca.PCA(2, 249.9, 1300.1)
        test_colors = test_pca.calc_colors(self.test_bandpass_dict, 2)
        test_sed = Sed()
        test_sed_2 = Sed()
        test_sed.setSED(wavelen=self.sample_spec[0],
                        flambda=self.sample_spec[1])
        test_sed_2.setSED(wavelen=self.sample_spec_2[0],
                          flambda=self.sample_spec_2[1])
        control_mags = self.test_bandpass_dict.magListForSed(test_sed)
        control_mags_2 = self.test_bandpass_dict.magListForSed(test_sed_2)
        control_colors = control_mags[0] - control_mags[1]
        control_colors_2 = control_mags_2[0] - control_mags_2[1]
        control_colors_array = np.array([[control_colors], [control_colors_2]])

        np.testing.assert_array_almost_equal(test_colors, control_colors_array)

    def test_write_output(self):

        test_pca = pcaSED()
        test_pca.load_full_spectra('scratch')

        test_pca.PCA(2, 249.9, 1300.1)
        test_pca.write_output('scratch/results')

        wavelength_array = []
        with open('scratch/results/wavelengths.dat', 'r') as f:
            for line in f:
                wavelength_array.append(np.float(line))

        np.testing.assert_array_equal(wavelength_array, test_pca.wavelengths)

        mean_array = []
        with open('scratch/results/mean_spectrum.dat', 'r') as f:
            for line in f:
                mean_array.append(np.float(line))

        np.testing.assert_array_equal(mean_array, test_pca.mean_spec)

        for spec_name, spec_coeffs in zip(test_pca.spec_names,
                                          test_pca.coeffs):
            spec_array = []
            with open('scratch/results/coeffs/%s.dat' % spec_name, 'r') as f:
                for line in f:
                    spec_array.append(np.float(line))
            np.testing.assert_array_equal(spec_array, spec_coeffs)

        for eig_num, eig_spec in list(enumerate(test_pca.eigenspectra)):
            eig_array = []
            with open('scratch/results/eigenspectra/eigenspectra_%i.dat'
                      % eig_num) as f:
                for line in f:
                    eig_array.append(np.float(line))
            np.testing.assert_array_equal(eig_array, eig_spec)

    @classmethod
    def tearDownClass(cls):

        if os.path.exists('scratch'):
            shutil.rmtree('scratch')


if __name__ == '__main__':
    unittest.main()
