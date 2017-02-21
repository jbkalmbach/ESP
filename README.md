# GP-Spectra

## Purpose
GP-Spectra is a way to estimate galaxy spectra using Principal Component
Analysis (PCA) and Gaussian Processes. Starting from a set of known spectra and
a set of bandpasses you can predict spectra for other points
in the color space of your bands both interpolating and extrapolating
throughout the color space.

## Setup

## Requirements

### Dependencies

PG-Interp requires the following:

 * numpy
 * scikit-learn

### To Run

From command line (in bash terminal):

    source setup/setup.sh

Now to use from python, just:

    import gpspec

## How to Use
See the example script in the form of an ipython notebook in the examples folder.
