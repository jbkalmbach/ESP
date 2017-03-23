import copy
import george
import numpy as np
from pca import pcaSED
from sklearn.neighbors import KNeighborsRegressor as knr


class estimateBase(object):

    def __init__(self, reduced_spec, bandpass_dict, new_colors):

        self.reduced_spec = reduced_spec
        max_comps = len(self.reduced_spec.eigenspectra)
        self.reduced_colors = self.reduced_spec.calc_colors(bandpass_dict,
                                                            max_comps)
        self.new_colors = new_colors

        return


class nearestNeighborEstimate(estimateBase):

    def nn_predict(self, num_neighbors, knr_args=None):

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

    def define_kernel(self, kernel_type, length, scale):

        n_dim = len(self.new_colors[0])

        if kernel_type == 'exp':
            kernel = scale*george.kernels.ExpKernel(length, ndim=n_dim)
        elif kernel_type == 'sq_exp':
            kernel = scale*george.kernels.ExpSquaredKernel(length, ndim=n_dim)
        else:
            raise Exception("Only currently accept 'exp' or 'sq_exp' as " +
                            "kernel types.")

        return kernel

    def gp_predict(self, kernel, record_params=True):

        n_coeffs = len(self.reduced_spec.coeffs[0])
        kernel_copy = copy.deepcopy(kernel)

        pred_coeffs = []
        params = []

        for coeff_num in range(n_coeffs):

            gp_obj = george.GP(kernel_copy)
            gp_obj.compute(self.reduced_colors, 0.)

            pars, res = gp_obj.optimize(self.reduced_colors,
                                        self.reduced_spec.coeffs[:, coeff_num])

            mean, cov = gp_obj.predict(self.reduced_spec.coeffs[:, coeff_num],
                                       self.new_colors)
            pred_coeffs.append(mean)

            if record_params is True:
                params.append(pars)

        pred_spec = pcaSED()
        pred_spec.wavelengths = self.reduced_spec.wavelengths
        pred_spec.eigenspectra = self.reduced_spec.eigenspectra
        pred_spec.mean_spec = self.reduced_spec.mean_spec

        pred_coeffs = np.array(pred_coeffs)
        pred_spec.coeffs = np.transpose(pred_coeffs)
        pred_spec.params = params

        return pred_spec
