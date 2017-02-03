# GP-Spectra

## Purpose


## Requirements
In order to run GP-Spectra you will need to install 1) the LSST simulations software
(https://confluence.lsstcorp.org/display/SIM/Catalogs+and+MAF) and 2) scikit-learn. Once those are installed
you can (replacing WORK_DIR with the desired location) add the appropriate form of the following line to your
'.login' file:

    export PYTHONPATH='$WORK_DIR:$PYTHONPATH'

And with the LSST stack installed, once you source the stack you will need to:

    setup sims_photUtils

Now to use from python, just:

    import gpspec

## How to Use
See the example script in the form of an ipython notebook in the examples folder.
