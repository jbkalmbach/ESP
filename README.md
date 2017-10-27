# ESP (Estimating Spectra from Photometry)
[![Build Status](https://travis-ci.org/jbkalmbach/ESP.svg?branch=master)](https://travis-ci.org/jbkalmbach/ESP)[![codecov.io](https://codecov.io/github/jbkalmbach/ESP/coverage.svg?branch=master)](https://codecov.io/github/jbkalmbach/ESP?branch=master)[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)

## Purpose
ESP is a way to estimate galaxy spectra using Principal Component
Analysis (PCA) and Gaussian Processes. Starting from a set of known spectra and
a set of bandpasses you can predict spectra for other points
in the color space of your bands both interpolating and extrapolating
throughout the color space.

## Setup

ESP is pip installable. From the command line run:

    pip install esphot

Now to use from python, just:

    import esp

## How to Use
See the [example jupyter notebook](https://github.com/jbkalmbach/ESP/blob/master/examples/esp_example.ipynb) in the examples folder.
