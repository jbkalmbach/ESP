import unittest
import george
import numpy as np
from esp import optimize


class testGPUtils(unittest.TestCase):

    def test_optimize(self):

        np.random.seed(42)

        test_p = 2.0
        test_p_2 = 7.0

        test_kernel = george.kernels.ExpSquaredKernel(test_p)
        test_gp = george.GP(test_kernel)
        t = np.arange(-2, 2, .05)
        y = test_gp.sample(t)

        test_kernel_2 = george.kernels.ExpSquaredKernel(test_p_2)
        test_gp_2 = george.GP(test_kernel_2)
        test_gp_2.compute(t)

        pars, results = optimize(test_gp_2, t, y)

        # Optimize within 5% of original value
        np.testing.assert_allclose(np.exp(pars), test_p, rtol=0.05)
        # Make sure parameters are what we expect from results
        np.testing.assert_equal(pars, results.x)

        test_p_3 = 9.0

        test_kernel_3 = george.kernels.ExpSquaredKernel(test_p_3)
        test_gp_3 = george.GP(test_kernel_3)
        test_gp_3.compute(t)

        # Optimize with alternative method and check that result is within 5%

        kwargs = {'method': 'L-BFGS-B'}
        pars_2, results_2 = optimize(test_gp_3, t, y, **kwargs)

        np.testing.assert_allclose(np.exp(pars_2), test_p, rtol=0.05)
        # Make sure parameters are what we expect from results
        np.testing.assert_equal(pars_2, results_2.x)

if __name__ == '__main__':

    unittest.main()
