from __future__ import print_function
import os
import numpy as np
from .spec_utils import specUtils
from sklearn.decomposition import PCA as sklPCA
from .lsst_utils import Sed

__all__ = ["pcaSED"]


class pcaSED(specUtils):
    """
    Performs Principal Component Analysis on a given library of spectra.

    Also contains methods for input and output of results, reconstructing
    spectra from principal components and calculating the colors of
    reconstructed spectra.

    Attributes
    ----------
    mean_spec: numpy array, [n_bins, n_wavelengths]
        Set by self.PCA. The mean spectrum of the spectral library.

    eigenspectra: numpy array, [n_bins, n_components, n_wavelengths]
        Set by self.PCA. These are the eigenspectra derived from the PCA.

    coeffs: numpy array, [n_bins, n_binMembers, n_components]
        Set by self.PCA. These are the coefficients for each model spectrum
        that are associated with the derived eigenspectra.

    wavelengths: numpy array
        Set by self.PCA. Contains the wavelengths of the eigenspectra.

    spec_names: list
        Set by self.PCA. List of the names of the spectra library in the
        same order that the coefficients are stored.

    spec_list_orig: list
        Set by self.load_full_spectra. Contains the spectra wavelength and
        flux information in its original form so that PCA can be run multiple
        times with different setting without having to reload the spectral
        library.
    """
    def __init__(self):

        # Required for a reduced spectra class
        self.mean_spec = None
        self.eigenspectra = None
        self.coeffs = None
        self.wavelengths = None
        self.spec_names = None

        # Needed to perform PCA
        self.spec_list_orig = None

        return

    def load_full_spectra(self, dir_path):
        """
        Load the full spectra from a directory.

        This will store them in self.spec_list_orig so that it is
        easy to rerun the PCA with different specifications. Spectra files
        should have two columns with left column containing wavelengths
        and the right containing flux.

        Parameters
        ----------
        dir_path: str
        The path to the directory where the spectra files are stored.
        """

        self.spec_list_orig = self.load_spectra(dir_path)
        print('Done loading spectra from file')

        return

    def PCA(self, comps, minWavelen=299., maxWavelen=1200.):
        """
        Perform the PCA.

        Performs the PCA on the loaded spectral library and sets the attributes
        defined in class docstring.

        Parameters
        ----------
        comps: int
            Maximum number of principal components desired.

        minWavelen: float, optional, default = 299.
            Minimum wavelength of spectra to use in creating PCA in units
            that match those in spectral library files.

        maxWavelen: float, optional, default = 1200.
            Maximum wavelength of spectra to use in creating PCA in units
            that match those in spectral library files.
        """

        if self.spec_list_orig is None:
            raise Exception("Need to load spectra. Use load_full_spectra.")

        # Resample the spectra over the desired wavelength range. This will
        # make PCA more accurate where we care about and faster.
        min_wave_x = np.where(self.spec_list_orig[0].wavelen >=
                              minWavelen)[0][0]
        max_wave_x = np.where(self.spec_list_orig[0].wavelen <=
                              maxWavelen)[0][-1]
        wavelen_set = self.spec_list_orig[0].wavelen[min_wave_x:max_wave_x+1]

        self.wavelengths = wavelen_set

        scaled_fluxes = []
        self.spec_names = []

        for spec in self.spec_list_orig:

            # Calculate Mags and save resampled and normalized copies of SEDs
            temp_spec = Sed()
            temp_spec.setSED(wavelen=spec.wavelen, flambda=spec.flambda)
            temp_spec.resampleSED(wavelen_match=wavelen_set)
            temp_spec.scale_flux = self.scale_spectrum(temp_spec.flambda)
            scaled_fluxes.append(temp_spec.scale_flux)
            self.spec_names.append(spec.name)

        """
        Calculate the eigenspectra from each bin. Also, keep the mean spectrum
        for each bin. Then project the model spectra in each bin onto the
        eigenspectra and keep the desired number of principal components.
        """
        spectra_pca = sklPCA(n_components=comps)
        spectra_pca.fit(scaled_fluxes)
        self.mean_spec = spectra_pca.mean_
        self.eigenspectra = spectra_pca.components_
        self.coeffs = np.array(spectra_pca.transform(scaled_fluxes))
        self.exp_var = spectra_pca.explained_variance_ratio_

    def reconstruct_spectra(self, num_comps):
        """
        Reconstruct spectrum using only num_comps principal components.

        Parameters
        ----------
        num_comps: int
        Number of principal components to use to reconstruct spectra.

        Returns
        -------
        reconstructed_specs: numpy array, [# of spectra, # of wavelength]
        The reconstructed spectra.
        """
        reconstructed_specs = self.mean_spec + \
            np.dot(self.coeffs[:, :num_comps],
                   self.eigenspectra[:num_comps])

        for spec_num in range(len(reconstructed_specs)):		
            neg_idx = np.where(reconstructed_specs[spec_num] < 0.)[0]		
            reconstructed_specs[spec_num][neg_idx] = 0.

        return reconstructed_specs

    def calc_colors(self, bandpass_dict, num_comps):
        """
        Calculate the colors using only num_comps principal components.

        Parameters
        ----------
        bandpass_dict: dictionary of bandpass objects
        Dictionary containing the bandpasses to use when calculating colors.

        num_comps: int
        Number of principal components to use to calculate spectra colors.

        Returns
        -------
        reconstructed_colors: numpy array, [# of spectra, # of colors]
        Colors calculated with reconstructed spectra.
        """
        reconstructed_specs = self.reconstruct_spectra(num_comps)

        reconstructed_colors = []
        for spec in reconstructed_specs:
            new_spec = Sed()
            new_spec.setSED(wavelen=self.wavelengths,
                            flambda=spec)
            mags = np.array(bandpass_dict.magListForSed(new_spec))
            colors = [mags[x] - mags[x+1] for x in
                      range(len(bandpass_dict.keys())-1)]
            reconstructed_colors.append(colors)

        return np.array(reconstructed_colors)

    def write_output(self, out_path):
        """
        Write PCA attributes to file.

        This routine will write out the eigenspectra, eigencomponents,
        mean spectrum and wavelength grid to files in a specified output
        directory with a separate folder for each bin.

        Parameters
        ----------
        out_path = str
            Folder where information will be stored.
        """

        np.savetxt(str(out_path + '/wavelengths.dat'),
                   self.wavelengths)

        spec_path = str(out_path + '/eigenspectra')
        os.makedirs(spec_path)
        for spec, spec_num in zip(self.eigenspectra,
                                  range(0, len(self.eigenspectra))):
            np.savetxt(str(spec_path + '/eigenspectra_' +
                           str(spec_num) + '.dat'), spec)

        coeff_path = str(out_path + '/coeffs')
        os.makedirs(coeff_path)
        for spec_name, coeffs in zip(self.spec_names, self.coeffs):
            np.savetxt(str(coeff_path + '/' + spec_name + '.dat'), coeffs)

        np.savetxt(str(out_path + '/mean_spectrum.dat'), self.mean_spec)

    def load_pca_output(self, dir_path):

        """
        This method loads the output from self.write_output.

        Loads in the mean spectrum, wavelengths, eigenspectra, coefficients and
        spectra file names and sets them as attributes as if self.PCA just
        ran.

        Parameters
        ----------
        dir_path: str
            Directory where the PCA output files are found.
        """

        self.wavelengths = np.loadtxt(str(dir_path + '/wavelengths.dat'))
        self.mean_spec = np.loadtxt(str(dir_path + '/mean_spectrum.dat'))

        eigenspectra = []
        spec_path = str(dir_path + '/eigenspectra/')
        for spec_name in sorted(os.listdir(spec_path)):
            eigenspectra.append(np.loadtxt(str(spec_path + spec_name)))
        self.eigenspectra = np.array(eigenspectra)

        coeffs = []
        names = []
        comp_path = str(dir_path + '/coeffs/')
        for comp_file in sorted(os.listdir(comp_path)):
            coeffs.append(np.loadtxt(str(comp_path + comp_file)))
            names.append(comp_file)
        self.coeffs = np.array(coeffs)
        self.spec_names = names

        return
