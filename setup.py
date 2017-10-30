from setuptools import setup

setup(
    name="esphot",
    version="0.4.0",
    author="Bryce Kalmbach",
    author_email="brycek@uw.edu",
    url="https://github.com/jbkalmbach/esp",
    packages=["esp", "esp.lsst_utils"],
    description="Estimating Spectra from Photometry",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
    install_requires=["matplotlib", "numpy", "scipy", "sklearn", "george",
                      "future"]
)
