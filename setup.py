import sys
from setuptools import setup, find_packages
import codecs
import os

# Add the path to the directory containing the _information module to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "analytixhero"))

from analytixhero._information import __version__, __desc__, __all__



VERSION = __version__
DESCRIPTION = __desc__

# Long Description From Readme.md
with open("README.md", "r") as ReadMeFile:
    LONG_DESCRIPTION = ReadMeFile.read()


# Setting up
setup(
    name="analytixhero",
    version=VERSION,
    author="Sahib Nanda",
    author_email="<essenbeats@gmail.com>",
    license='BSD',
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    url="https://github.com/thesahibnanda/AnalytiXHero",
    project_urls={
        'Documentation': 'https://github.com/thesahibnanda/AnalytiXHero/blob/main/README.md',
        'Source Code': 'https://github.com/thesahibnanda/AnalytiXHero',
    },
    packages=find_packages(),
    install_requires=[
    "numpy>=1.22.3",
    "pandas>=2.0.0",
    "scikit-learn>=1.0.0",
    "scipy>=1.10.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11",
    "python-dateutil>=2.8.0"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "License :: OSI Approved :: BSD License"
    ]
)