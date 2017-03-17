import unittest
import os
import shutil
import numpy as np
from spec_utils import specUtils


class testSpecUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        sample_wavelen = np.arange(50)
        sample_flux = np.random.uniform(size=50)

        cls.sample_spec = np.array([sample_wavelen, sample_flux])

        if os.path.exists('scratch'):
            shutil.rmtree('scratch')

        os.mkdir('scratch')
        np.savetxt('scratch/sample.dat', cls.sample_spec.T,
                   header='Lambda Flux', delimiter=' ')

    def test_load_spectra(self):

        test_su = specUtils()
        test_spec = test_su.load_spectra('scratch')

        self.assertItemsEqual(self.sample_spec[0], test_spec[0].wavelen)
        self.assertItemsEqual(self.sample_spec[1], test_spec[0].flambda)
        self.assertEqual('sample.dat', test_spec[0].name)

    def test_scale_spectrum(self):

        test_su = specUtils()
        test_spec = test_su.load_spectra('scratch')
        new_flux = test_su.scale_spectrum(test_spec[0].flambda)
        self.assertAlmostEqual(np.sum(new_flux), 1.0)

    @classmethod
    def tearDownClass(cls):

        if os.path.exists('scratch'):
            shutil.rmtree('scratch')


if __name__ == '__main__':

    unittest.main()
