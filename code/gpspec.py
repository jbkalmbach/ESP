import os
import math
import numpy as np
from sklearn.decomposition import PCA as sklPCA
from lsst_utils.Sed import Sed
from sklearn.neighbors import KNeighborsRegressor as knr


class specUtils(object):
    """Contains utilities for package classes."""

    def load_spectra(self, directory):
        """
        Read in spectra from files in given directory.

        Parameters
        ----------
        directory: str
            Location of spectra files.

        Returns
        -------
        spec_list: list of Sed class instances
            List where each entry is an instance of the Sed class containing
            the information for one model spectrum.
        """
        spec_list = []
        for root, dirs, files in os.walk(directory):
            file_total = len(files)
            file_on = 1
            for name in files:
                if file_on % 100 == 0:
                    print str("File On " + str(file_on) + " out of " +
                              str(file_total))
                file_on += 1
                try:
                    spec = Sed()
                    spec.readSED_flambda(os.path.join(root, name))
                    spec.name = name
                    if math.isnan(spec.flambda[0]) is False:
                        spec_list.append(spec)
                except:
                    continue

        return spec_list

    def scale_spectrum(self, sed_flux):
        """
        Norm spectrum so adds up to 1.

        Parameters
        ----------
        sedFlux: array
            The flux array of an SED.

        Returns
        -------
        norm_spec: array
            The normalized flux array.
        """
        norm = np.sum(sed_flux)
        norm_spec = sed_flux/norm

        return norm_spec


class pcaSED(specUtils):
    """
    When given a directory containing spectra. This class will create sets
    of eigenspectra in bins created based upon the distribution in color-color
    space. It will also provide the given principal components to recreate
    the spectra using the sets of eigenspectra.

    Parameters
    ----------
    directory: str
        Directory where the model spectra are stored.

    bandpassDir: str
        Location of bandpass files. Default is LSST stack's location for LSST
        filters.

    bandpassRoot: str, default: 'total_'
        Root for filenames of bandpasses. Default are LSST Total Bandpasses.

    filters: list, default: ['u', 'g', 'r', 'i', 'z', 'y']
        Name of filters to be used for calculating colors. Default are
        LSST bands.

    Attributes
    ----------
    cluster_centers: array, [n_bins, n_colors]
        Location in color-color space of the bin centers used in grouping the
        model spectra if n_bins > 1.

    meanSpec: array, [n_bins, n_wavelengths]
        The mean spectrum of each bin. Needs to be added back in when
        reconstructing the spectra from
        principal components.

    eigenspectra: array, [n_bins, n_components, n_wavelengths]
        These are the eigenspectra derived from the PCA organized by bin.

    projected: array, [n_bins, n_binMembers, n_components]
        These are the principal components for each model spectrum
        organized by bin.

    binnedNames: list, [n_bins, n_binMembers]
        The names of the model spectra in each bin. Used when writing output.

    temps: array, (only when using blackbodyPCA), [n_bins, n_binMembers]
        The temperatures of the blackbody spectrum divided out of the model
        spectrum and needed in order to reconstruct the spectrum.

    pcaType: str
        'BB' or 'NoBB' depending on which method is used to calculate PCA.
        Used to keep track of what to write to output and when reconstructing
        in other methods.
    """
    def __init__(self):

        # Required for a reduced spectra class
        self.eigenspectra = None
        self.coeffs = None
        self.mean_spec = None
        self.wavelengths = None
        self.spec_names = None

        # Needed to perform PCA
        self.spec_list_orig = None

        return

    def load_full_spectra(self, dir_path):
        """
        Load the full spectra from a directory.

        This will store them in self.spec_list_orig so that it is
        easy to rerun the PCA with different specifications.

        Parameters
        ----------
        dir_path: str
        The path to the directory where the spectra files are stored.
        """

        self.spec_list_orig = self.load_spectra(dir_path)
        print 'Done loading spectra from file'

        return

    def PCA(self, comps, minWavelen=299., maxWavelen=1200.):
        """
        Perform the PCA.

        Parameters
        ----------
        comps: int
            Maximum number of principal components desired.

        minWavelen: float, optional, default = 299.
            Minimum wavelength of spectra to use in creating PCA. Can speed up
            PCA and minimize number of components needed for accuracy in a
            defined range.

        maxWavelen: float, optional, default = 1200.
            Maximum wavelength of spectra to use in creating PCA. Can speed up
            PCA and minimize number of components needed for accuracy in a
            defined range.
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
        self.explained_var = spectra_pca.explained_variance_ratio_

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

        return reconstructed_specs

    def calc_colors(self, bandpass_dict, num_comps):
        """
        Calculate the colors using only num_comps principal components.

        Parameters
        ----------
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
        for spec, specNum in zip(self.eigenspectra,
                                 range(0, len(self.eigenspectra))):
            np.savetxt(str(spec_path + '/eigenspectra_' +
                           str(specNum) + '.dat'), spec)

        coeff_path = str(out_path + '/coeffs')
        os.makedirs(coeff_path)
        for spec_name, coeffs in zip(self.spec_names, self.coeffs):
            np.savetxt(str(coeff_path + '/' + spec_name + '.dat'), coeffs)

        np.savetxt(str(out_path + '/meanSpectrum.dat'), self.mean_spec)

    def load_pca_output(self, dir_path):

        """
        This method loads the output from pcaSED's write_output method.

        Parameters
        ----------
        dir_path: str
            Directory where the PCA output files are found.

        Returns
        -------
        wavelengths: numpy array, [# of wavelength points]
            The wavelengths to which each flux value in the original spectra
            and eigenspectra corresponds.

        mean_spec: numpy array, [# of wavelength points]
            The mean spectrum of the PCA. Originally subtracted off before
            performing PCA.

        eigenspectra: numpy array, [# of components, # of wavelength points]
            The eigenspectra from PCA on the original spectra.

        components: numpy array, [# of spectra, # of principal components]
            The principal components required to reconstruct each spectrum.
        """

        self.wavelengths = np.loadtxt(str(dir_path + '/wavelengths.dat'))
        self.mean_spec = np.loadtxt(str(dir_path + '/meanSpectrum.dat'))

        eigenspectra = []
        spec_path = str(dir_path + '/eigenspectra/')
        for spec_name in os.listdir(spec_path):
            eigenspectra.append(np.loadtxt(str(spec_path + spec_name)))
        self.eigenspectra = np.array(eigenspectra)

        coeffs = []
        names = []
        comp_path = str(dir_path + '/coeffs/')
        for comp_file in os.listdir(comp_path):
            coeffs.append(np.loadtxt(str(comp_path + comp_file)))
            names.append(comp_file)
        self.coeffs = np.array(coeffs)
        self.spec_names = names

        return


class interpBase(object):

    def __init__(self, reduced_spec, bandpass_dict, new_colors):

        self.reduced_spec = reduced_spec
        max_comps = len(self.reduced_spec.eigenspectra)
        self.reduced_colors = self.reduced_spec.calc_colors(bandpass_dict,
                                                            max_comps)
        self.new_colors = new_colors

        return


class nearestNeighborInterp(interpBase):

    def nn_interp(self, num_neighbors, knr_args=None):

        default_knr_args = dict(n_neighbors=num_neighbors)

        if knr_args is not None:
            default_knr_args.update(knr_args)
        knr_args = default_knr_args

        neigh = knr(**knr_args)
        neigh.fit(self.reduced_colors, self.reduced_spec.coeffs)
        new_coeffs = neigh.predict(self.new_colors)

        interp_spec = pcaSED()
        interp_spec.wavelengths = self.reduced_spec.wavelengths
        interp_spec.eigenspectra = self.reduced_spec.eigenspectra
        interp_spec.mean_spec = self.reduced_spec.mean_spec

        interp_spec.coeffs = new_coeffs

        return interp_spec

class gaussianProcessInterp(interpBase):

    def gp_interp(self):

        return
