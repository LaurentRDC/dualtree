from setuptools import setup
from os.path import dirname, join

# Read the license file into a single string
with open(join(dirname(__file__), 'license.txt')) as license_file:
    license = license_file.read().replace('\n', '')

setup(name = 'dualtree',
      version = '0.5',
      description = 'Dual-tree complex wavelet transform',
      author = 'Laurent P. René de Cotret',
      author_email = 'laurent.renedecotret@mail.mcgill.ca',
      url = 'http://www.physics.mcgill.ca/siwicklab/',
      packages = ['dualtree', 'dualtree.tests'],
      install_requires = ['numpy', 'PyWavelets'],
      package_data = {'dualtree': ['*.txt'],
                      'dualtree': ['data/*.npz']},    # Include license.txt and all other .txt files
      license = license
     )