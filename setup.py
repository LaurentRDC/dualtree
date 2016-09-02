from setuptools import setup
from os.path import dirname, join

# Read the license file into a single string
with open(join(dirname(__file__), 'license.txt')) as license_file:
    license = license_file.read().replace('\n', '')

setup(name = 'waveline',
      version = '1.0',
      description = 'Baseline-determination of signals using the multi-level discrete wavelet transform',
      author = 'Laurent P. René de Cotret',
      author_email = 'laurent.renedecotret@mail.mcgill.ca',
      url = 'http://www.physics.mcgill.ca/siwicklab/',
      packages = ['waveline'],
      install_requires = ['numpy', 'PyWavelets'],
      package_data = {'': ['*.txt']},    # Include license.txt and all other .txt files
      license = license
     )