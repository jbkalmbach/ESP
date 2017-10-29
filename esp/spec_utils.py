from __future__ import print_function
from builtins import object
import os
import math
import numpy as np
from .lsst_utils import Sed

__all__ = ["specUtils"]


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
            for name in sorted(files):
                if file_on % 100 == 0:
                    print(str("File On " + str(file_on) + " out of " +
                              str(file_total)))
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
