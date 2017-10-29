import numpy as np
import scipy.optimize as op

__all__ = ["optimize"]


def optimize(gp_obj, x, y, **kwargs):

    """
    Find the maximum likelihood values for the parameters of the Gaussian
    Process Regression.

    Based upon optimization example shown in george documentation:
    http://george.readthedocs.io/en/latest/tutorials/hyper/
    and optimize from george v0.2:
    https://github.com/dfm/george/blob/v0.2.0/george/gp.py

    Parameters
    ----------
    gp_obj: george GP object
    The george object we are using for the Gaussian Process Regression
    """

    op_kwargs = {'method': 'Nelder-Mead'}

    op_kwargs.update(**kwargs)

    def _nll(pars):

        gp_obj.set_parameter_vector(pars)
        ll = gp_obj.log_likelihood(y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    def _grad_nll(pars):

        gp_obj.set_parameter_vector(pars)
        return -gp_obj.grad_log_likelihood(y, quiet=True)

    p0 = gp_obj.get_parameter_vector()
    if op_kwargs['method'] != 'Nelder-Mead':
        op_kwargs['jac'] = _grad_nll
    results = op.minimize(_nll, p0, **op_kwargs)
    gp_obj.set_parameter_vector(results.x)

    return results.x, results
