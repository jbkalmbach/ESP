from builtins import object
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')

__all__ = ["plotUtils"]


class plotUtils(object):
    """
    Contains plotting routines for GP-Spectra code.
    """

    def plot_eigenspectra(self, gpSpec_inst, num_eigen, fig=None):
        """
        Plot the first num_eigen eigenspectra along with the mean spectrum.

        Parameters
        ----------

        gpSpec_inst: gpSpec class instance

        num_eigen: int
        Number of eigenspectra to include in the plot.
        """
        num_plots = num_eigen + 1
        if fig is None:
            fig = plt.figure(figsize=(num_plots*6, 12))

        for i in range(num_plots):
            fig.add_subplot(num_plots, 1, i+1)
            if i == 0:
                flux = gpSpec_inst.mean_spec
                title_str = 'Mean Spectrum'
            else:
                flux = gpSpec_inst.eigenspectra[i-1]
                title_str = 'Eigenspectrum %i' % (i)

            plt.plot(gpSpec_inst.wavelengths, flux)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Scaled Flux')
            plt.title(title_str)
        mpl.rcParams.update({'font.size': 16})
        plt.tight_layout()

        return fig
