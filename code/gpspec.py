import os
import math
import numpy as np
from sklearn.decomposition import PCA as sklPCA
from spec_utils.Sed import Sed
from spec_utils.BandpassDict import BandpassDict


class pcaUtils(object):
    """Contains utilities for all other PCA routines."""

    def loadSpectra(self, directory):
        """
        Read in spectra from files in given directory.

        Parameters
        ----------
        directory: str
            Location of spectra files.

        Returns
        -------
        spectraList: list of Sed class instances
            List where each entry is an instance of the Sed class containing
            the information for one model spectrum.
        """
        spectraList = []
        for root, dirs, files in os.walk(directory):
            fileTotal = len(files)
            fileOn = 1
            for name in files:
                if fileOn % 100 == 0:
                    print str("File On " + str(fileOn) + " out of " +
                              str(fileTotal))
                fileOn += 1
                try:
                    spec = Sed()
                    spec.readSED_flambda(os.path.join(root, name))
                    spec.name = name
                    if math.isnan(spec.flambda[0]) is False:
                        spectraList.append(spec)
                except:
                    continue

        return spectraList

    def scaleSpectrum(self, sedFlux):
        """
        Norm spectrum so adds up to 1.

        Parameters
        ----------
        sedFlux: array
            The flux array of an SED.

        Returns
        -------
        normSpec: array
            The normalized flux array.
        """
        norm = np.sum(sedFlux)
        normSpec = sedFlux/norm

        return normSpec


class pcaSED(pcaUtils):
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
    def __init__(self, directory, bandpassDir, bandpassRoot='total_',
                 filters=['u', 'g', 'r', 'i', 'z', 'y']):

        self.spectraListOriginal = self.loadSpectra(directory)
        print 'Done loading spectra from file'

        self.filters = filters
        self.bandpassDict = BandpassDict.loadTotalBandpassesFromFiles(
                                                  bandpassNames=self.filters,
                                                  bandpassDir=bandpassDir,
                                                  bandpassRoot=bandpassRoot)

    def specPCA(self, comps=10, minWavelen=299., maxWavelen=1200.):
        """
        Read in spectra, then calculate the colors.
        Bin the spectra by their colors and then perform the PCA on each bin
        separately.

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
        self.spectraList = []

        # Resample the spectra over the desired wavelength range. This will
        # make PCA more accurate where we care about and faster.
        minWaveIdx = np.where(self.spectraListMaster[0].wavelen >=
                              minWavelen)[0][0]
        maxWaveIdx = np.where(self.spectraListMaster[0].wavelen <=
                              maxWavelen)[0][-1]
        wavelenSet = self.spectraListMaster[0].wavelen[minWaveIdx:maxWaveIdx+1]

        fullMags = []
        scaledFluxes = []

        for spec in self.spectraListOriginal:

            # Calculate Mags and save resampled and normalized copies of SEDs
            tempSpec = Sed()
            tempSpec.setSED(wavelen=spec.wavelen, flambda=spec.flambda)
            tempMags = np.array(self.bandpassDict.magListForSed(tempSpec))
            fullMags.append(tempMags)
            tempSpec.resampleSED(wavelen_match=wavelenSet)
            tempSpec.scaleFlux = self.scaleSpectrum(tempSpec.flambda)
            scaledFluxes.append(tempSpec.scaleFlux)
            tempSpec.name = spec.name
            self.spectraList.append(tempSpec)

        # Get colors from the mags calculated above.
        fullMagsT = np.transpose(np.array(fullMags))
        colorVals = []
        for colorNum in range(0, len(fullMagsT)-1):
            colorVals.append(fullMagsT[colorNum] - fullMagsT[colorNum+1])

        """
        Calculate the eigenspectra from each bin. Also, keep the mean spectrum
        for each bin. Then project the model spectra in each bin onto the
        eigenspectra and keep the desired number of principal components.
        """
        spectraPCA = sklPCA(n_components=comps)
        spectraPCA.fit(scaledFluxes)
        self.meanSpec = spectraPCA.mean_
        self.eigenspectra = spectraPCA.components_
        self.projected = np.array(spectraPCA.transform(scaledFluxes))
        self.explained_var = spectraPCA.explained_variance_ratio_

    def writeOutput(self, outFolder):
        """
        This routine will write out the eigenspectra, eigencomponents,
        mean spectrum and wavelength grid to files in a specified output
        directory with a separate folder for each bin.

        Parameters
        ----------
        outFolder = str
            Folder where information will be stored.
        """

        np.savetxt(str(outFolder + '/wavelengths.dat'),
                   self.spectraList[0].wavelen)

        specPath = str(outFolder + '/eigenspectra')
        os.makedirs(specPath)
        for spec, specNum in zip(self.eigenspectra,
                                 range(0, len(self.eigenspectra))):
            np.savetxt(str(specPath + '/eigenspectra_' +
                           str(specNum) + '.dat'), spec)

        compPath = str(outFolder + '/components')
        os.makedirs(compPath)
        for spec, comps in zip(self.spectraList, self.projected):
            specName = spec.name
            np.savetxt(str(compPath + '/' + specName + '.dat'), comps)

        np.savetxt(str(outFolder + '/meanSpectrum.dat'), self.meanSpec)
