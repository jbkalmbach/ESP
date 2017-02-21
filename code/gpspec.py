import os
import math
import numpy as np
from spec_utils.Sed import Sed


class pcaUtils(object):

    """Contains utilities for all other PCA routines."""

    def loadSpectra(self, directory):

        """Read in spectra from files in given directory.

        Parameters
        ----------
        directory: str
            Location of spectra files.

        Returns
        -------
        spectraList: list of Sed class instances
            List where each entry is an instance of the Sed class containing
            the information for one model spectrum."""

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

        """Norm spectrum so adds up to 1.

        Parameters
        ----------
        sedFlux: array
            The flux array of an SED.

        Returns
        -------
        normSpec: array
            The normalized flux array."""

        norm = np.sum(sedFlux)
        normSpec = sedFlux/norm

        return normSpec


class pcaSED(pcaUtils):

    """When given a directory containing spectra. This class will create sets
    of eigenspectra in bins created based upon the distribution in color-color
    space. It will also provide the given principal components to recreate
    the spectra using the sets of eigenspectra.

    Parameters
    ----------
    directory: str
        Directory where the model spectra are stored.

    filters: list, optional, default: ['u', 'g', 'r', 'i', 'z', 'y']
        Name of filters to be used for calculating colors. Default are
        LSST bands.

    bandpassDir: str, optional, default:
        os.path.join(os.getenv('THROUGHPUTS_DIR'),'baseline')
        Location of bandpass files. Default is LSST stack's location for LSST
        filters.

    bandpassRoot: str, optional, default: 'total_'
        Root for filenames of bandpasses. Default are LSST Total Bandpasses.

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

    def __init__(self, directory, filters=['u', 'g', 'r', 'i', 'z', 'y'],
                 bandpassDir=os.path.join(os.getenv('THROUGHPUTS_DIR'),
                                          'baseline'),
                 bandpassRoot='total_'):

        self.spectraListMaster = self.loadSpectra(directory)
        print 'Done loading spectra from file'

        self.filters = filters
        self.bandpassDict = BandpassDict.loadTotalBandpassesFromFiles(
                                                  bandpassNames=self.filters,
                                                  bandpassDir=bandpassDir,
                                                  bandpassRoot=bandpassRoot)

    def noBlackbodyPCA(self, comps=10, minWavelen=299., maxWavelen=1200.):

        """Read in spectra, then calculate the colors.
        Bin the spectra by their colors and then perform the PCA on each bin separately.

        Parameters
        ----------
        comps: int
            Maximum number of principal components desired.

        minWavelen: float, optional, default = 299.
            Minimum wavelength of spectra to use in creating PCA. Can speed up PCA and minimize number of
            components needed for accuracy in a defined range.

        maxWavelen: float, optional, default = 1200.
            Maximum wavelength of spectra to use in creating PCA. Can speed up PCA and minimize number of
            components needed for accuracy in a defined range.
        """

        self.spectraList = []

        #Resample the spectra over the desired wavelength range. This will make PCA more accurate
        #where we care about and faster.
        minWaveIndex = np.where(self.spectraListMaster[0].wavelen >= minWavelen)[0][0]
        maxWaveIndex = np.where(self.spectraListMaster[0].wavelen <= maxWavelen)[0][-1]
        wavelenSet = self.spectraListMaster[0].wavelen[minWaveIndex:maxWaveIndex+1]

        fullMags = []

        for spec in self.spectraListMaster:

            if spec.name.startswith('k'):
                bbTemp = float(spec.name.split('_')[3].split('.')[0])
            elif spec.name.startswith('l'):
                if spec.name[6] == '.':
                    bbTemp = float(spec.name[3:8])*100.
                else:
                    bbTemp = float(spec.name[3:6])*100.
            else:
                bbTemp = minTemp+1.
            if minTemp <= bbTemp <= maxTemp:
                #Calculate Mags before dividing out blackbody spectrum
                tempSpec = Sed()
                tempSpec2 = Sed()
                #wavelengthArray = np.array([x for x in spec.wavelen])
                #flambdaArray = np.array([x for x in spec.flambda])
                tempSpec.setSED(wavelen = spec.wavelen, flambda = spec.flambda)
                tempSpec2.setSED(wavelen = spec.wavelen, flambda = spec.flambda)
                tempSpec2.resampleSED(wavelen_match=wavelenSet)
                tempMags = np.array(self.bandpassDict.magListForSed(tempSpec))
                fullMags.append(tempMags)
                tempSpec2.scaleFlux = self.scaleSpectrum(tempSpec2.flambda)
                tempSpec2.name = spec.name
                self.spectraList.append(tempSpec2)

        #Get colors from the mags calculated above.
        fullMagsT = np.transpose(np.array(fullMags))
        colorVals = []
        for colorNum in range(0, len(fullMagsT)-1):
            colorVals.append(fullMagsT[colorNum] - fullMagsT[colorNum+1])
        #Use these colors to bin spectra for PCA
        if binCenters is None:
            colorCluster = clusterKM(bins, tol = 1e-6)
            colorCluster.fit(np.transpose(np.array(colorVals)))
        else:
            colorCluster = clusterKM(bins, init = binCenters, n_init=1)
            colorCluster.fit(np.transpose(np.array(colorVals)))
        cutLabel = colorCluster.labels_
        self.cluster_centers = colorCluster.cluster_centers_
        self.binLabel = cutLabel

        #Start populating attributes in proper bins
        binnedSpec = [[] for x in range(0, bins)]
        self.meanSpec = np.zeros((bins, len(self.spectraList[0].wavelen)))
        self.eigenspectra = [[] for x in range(0, bins)]
        self.projected = [[] for x in range(0, bins)]
        self.binnedNames = [[] for x in range(0, bins)]
        self._binnedSpec = [[] for x in range(0, bins)]
        self._explained_var = [[] for x in range(0, bins)]
        for sNum in range(0, len(self.spectraList)):
            binnedSpec[cutLabel[sNum]].append(self.spectraList[sNum].scaleFlux)
            self.binnedNames[cutLabel[sNum]].append(self.spectraList[sNum].name)
        binnedSpec = np.array(binnedSpec)

        """Calculate the eigenspectra from each bin. Also, keep the mean spectrum for each bin. Then
        project the model spectra in each bin onto the eigenspectra and keep the desired number of
        principal components."""
        for binNum in range(0, bins):
            binErrorMsg = 'Bin has fewer members than desired number of components. Try lower number of bins.'
            if len(binnedSpec[binNum]) < compsList[binNum]:
                compsList[binNum] = len(binnedSpec[binNum])
                #raise ValueError(binErrorMsg)
            specPCA = sklPCA(n_components = compsList[binNum])
            specPCA.fit(binnedSpec[binNum])
            self.meanSpec[binNum] = specPCA.mean_
            self.eigenspectra[binNum] = specPCA.components_
            self.projected[binNum] = np.array(specPCA.transform(binnedSpec[binNum]))
            self._binnedSpec[binNum] = np.array(binnedSpec[binNum])
            self._explained_var[binNum] = specPCA.explained_variance_ratio_
