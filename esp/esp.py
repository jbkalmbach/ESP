import copy
import george
import numpy as np
from .pca import pcaSED
from .gp_utils import optimize
from sklearn.neighbors import KNeighborsRegressor as knr

__all__ = ["estimateBase", "NearestNeighborEstimate",
           "gaussianProcessEstimate"]


class estimateBase(object):

    """
    Base class for spectra estimation methods.
    """

    def __init__(self, reduced_spec, bandpass_dict, new_colors):

        self.reduced_spec = reduced_spec
        max_comps = len(self.reduced_spec.eigenspectra)
        self.reduced_colors = self.reduced_spec.calc_colors(bandpass_dict,
                                                            max_comps)
        self.new_colors = new_colors

        return


class nearestNeighborEstimate(estimateBase):

    """
    Estimates spectra from colors using the spectra of the nearest neighbors.

    Parameters
    ----------
    reduced_spec: pcaSED object
    An ESP pcaSED object.

    bandpass_dict: bandpass_dict object
    From lsst_utils bandpassDict. Required to calculate colors.

    new_colors: numpy array, [n_objects, n_colors]
    The colors for the objects in the catalog for which you want to estimate
    spectra.
    """

    def nn_predict(self, num_neighbors, knr_args=None):

        """
        Predict the spectra for the given colors using nearest neighbors.

        Parameters
        ----------
        num_neighbors: int
        The number of nearest neighbors to use when estimating the PCA
        coefficients of spectra based upon the input colors.

        knr_args: dict
        A dictionary with additional arguments for scikit-learn's
        k nearest neighbor regression. For instance, the default is a uniform
        weighting but `knr_args=dict(weights='distance')` would change
        to distance weighted regression.

        Returns
        -------
        pred_spec: pcaSED object
        A new pcaSED object with the coefficients of the spectra that
        correspond to the input colors and can be used to generate the
        estimated spectrum.
        """

        default_knr_args = dict(n_neighbors=num_neighbors)

        if knr_args is not None:
            default_knr_args.update(knr_args)
        knr_args = default_knr_args

        neigh = knr(**knr_args)
        neigh.fit(self.reduced_colors, self.reduced_spec.coeffs)
        pred_coeffs = neigh.predict(self.new_colors)

        pred_spec = pcaSED()
        pred_spec.wavelengths = self.reduced_spec.wavelengths
        pred_spec.eigenspectra = self.reduced_spec.eigenspectra
        pred_spec.mean_spec = self.reduced_spec.mean_spec

        pred_spec.coeffs = pred_coeffs

        return pred_spec


class gaussianProcessEstimate(estimateBase):

    """
    Estimates spectra from colors using Gaussian Processes.

    Parameters
    ----------
    reduced_spec: pcaSED object
    An ESP pcaSED object.

    bandpass_dict: bandpass_dict object
    From lsst_utils bandpassDict. Required to calculate colors.

    new_colors: numpy array, [n_objects, n_colors]
    The colors for the objects in the catalog for which you want to estimate
    spectra.
    """

    def define_kernel(self, kernel_type, scale, length, n_dim):

        """
        Define the george kernel object for the Gaussian Processes.

        Parameters
        ----------
        kernel_type: str
        Currently accept 'exp' or 'sq_exp' for an exponential kernel or a
        squared exponential kernel.

        scale: float
        An initial guess for the scale hyperparameter in the kernel. This is
        a constant in front of the exponential or squared exponential of the
        kernel. This will be changed when self.gp_predict optimizes each
        Gaussian Process for the respective PCA coefficients.

        length: float
        An initial guess for the length hyperparameter in the kernel. This
        will be changed when self.gp_predict optimizes each Gaussian Process
        for the respective PCA coefficients.

        n_dim: int
        Number of dimensions needed in the kernel. Should be equal to the
        number of colors being used to optimize.

        Returns
        -------
        """

        self.kernel_type = kernel_type

        if kernel_type == 'exp':
            kernel = scale*george.kernels.ExpKernel(length, ndim=n_dim)
        elif kernel_type == 'sq_exp':
            kernel = scale*george.kernels.ExpSquaredKernel(length, ndim=n_dim)
        elif self.kernel_type == 'matern_32':
            kernel = scale*george.kernels.Matern32Kernel(length, ndim=n_dim)
        elif self.kernel_type == 'matern_52':
            kernel = scale*george.kernels.Matern52Kernel(length, ndim=n_dim)

        else:
            raise Exception("Only currently accept 'exp', 'sq_exp', " +
                            "'matern_32' and 'matern_52' as kernel types.")

        return kernel

    def gp_predict(self, kernel, opt_bandpass_dict, record_params=True):

        """
        Predict the spectra for the given colors using nearest neighbors.

        Parameters
        ----------
        kernel: george kernel object
        The kernel to use in the Gaussian Process regression. Can be created
        using self.define_kernel.

        opt_bandpass_dict: dictionary of bandpass objects
        The dictionary with the bandpasses that will be used to optimize
        the hyperparameters of the Gaussian Processes.

        record_params: boolean, default=True
        If true it will record the log of the optimized hyperparameters
        of the kernel for each PCA coefficient.

        Returns
        -------
        pred_spec: pcaSED object
        A new pcaSED object with the coefficients of the spectra that
        correspond to the input colors and can be used to generate the
        estimated spectrum.
        """

        n_coeffs = len(self.reduced_spec.coeffs[0])
        kernel_copy = copy.deepcopy(kernel)
        max_comps = len(self.reduced_spec.eigenspectra)
        opt_colors = self.reduced_spec.calc_colors(opt_bandpass_dict,
                                                   max_comps)
        n_opt_colors = float(len(opt_colors[0]))
        n_test_colors = float(len(self.new_colors[0]))

        pred_coeffs = []
        pred_var = []
        params = []
        optimized_kernel = None

        for coeff_num in range(n_coeffs):

            if optimized_kernel is None:
                gp_obj = george.GP(kernel_copy)
            else:
                gp_obj = george.GP(kernel_copy)
                optimized_vector = optimized_kernel.get_parameter_vector()
                gp_obj.set_parameter_vector(optimized_vector)

            gp_obj.compute(opt_colors, 0.)

            pars, res = optimize(gp_obj, opt_colors,
                                 self.reduced_spec.coeffs[:, coeff_num])
            kernel_type = self.kernel_type
            optimized_kernel = self.define_kernel(kernel_type,
                                                  (np.exp(pars[0]) *
                                                   (n_opt_colors /
                                                    n_test_colors)),
                                                  np.exp(pars[1]),
                                                  len(self.new_colors[0]))
            gp_obj_opt = george.GP(optimized_kernel)

            gp_obj_opt.compute(self.reduced_colors, 0.)
            mean, cov = gp_obj_opt.predict(self.reduced_spec.coeffs[:,
                                                                    coeff_num],
                                           self.new_colors)
            pred_coeffs.append(mean)
            pred_var.append(np.diag(cov))

            if record_params is True:
                params.append(pars)

        pred_spec = pcaSED()
        pred_spec.wavelengths = self.reduced_spec.wavelengths
        pred_spec.eigenspectra = self.reduced_spec.eigenspectra
        pred_spec.mean_spec = self.reduced_spec.mean_spec

        pred_coeffs = np.array(pred_coeffs)
        pred_spec.coeffs = np.transpose(pred_coeffs)
        pred_spec.var = np.transpose(np.array(pred_var))

        if record_params is True:
            pred_spec.params = params

        return pred_spec
