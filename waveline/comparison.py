"""
Module concerned with the comparison between the discrete wavelet transform approach and
the dual-tree complex wavelet transform.

The test spectrum is taken from [1].

References
----------
[1] F. Zhao, A. Wang. A background subtraction approach based on complex wavelet transforms in EDXRF
"""
from discrete import baseline, approx_rec
from dualtree import dt_baseline, dt_approx_rec
from dtcwt.numpy.transform1d import Transform1d

import matplotlib.pyplot as plt
import numpy as n
import pywt

i = n.arange(1, 1024, step = 1)

def gaussian(amp, mean, std):
    return amp*n.exp( -( (i - mean)**2 )/std )

BG1 = gaussian(20, 570, 100000)   #background

def reference_spectrum_1():
    """ Returns a test spectrum after [1] Fig. 3 (a) """
    spectrum = n.zeros_like(i, dtype = n.float)
    
    spectrum += gaussian(36, 55, 145)
    spectrum += gaussian(40, 200, 78)
    spectrum += gaussian(8, 225, 78)
    spectrum += gaussian(50, 345, 175)
    spectrum += gaussian(55, 580, 130)
    spectrum += gaussian(23, 610, 90)
    spectrum += gaussian(30, 790, 120)
    spectrum += gaussian(50, 900, 120)
    spectrum += gaussian(15, 950, 80)

    return spectrum

def reference_spectrum_2():
    """ Returns a test spectrum after [1] Fig. 7 (a) """

    spectrum = n.zeros_like(i, dtype = n.float)

    spectrum += gaussian(140, 140, 50)
    spectrum += gaussian(115, 300, 40)
    spectrum += gaussian(78, 485, 30)
    spectrum += gaussian(65, 570, 50)
    spectrum += gaussian(23, 588, 40)
    spectrum += gaussian(25, 640, 50)
    spectrum ++ gaussian(150, 820, 60)
    spectrum += gaussian(80, 930, 25)

    return spectrum

def compare_1():
    spectrum = reference_spectrum_1()
    spectrum += BG1

    plt.plot(i, spectrum - BG1, 'g')
    plt.plot(i, spectrum - approx_rec(array = spectrum, level = 'max', wavelet = 'db3'), 'b')
    plt.plot(i, spectrum - dt_approx_rec(array = spectrum, level = 'max'), 'r')
    plt.show()

def compare_with_dtcwt():
    import dtcwt
    transform = dtcwt.Transform1d()
    vecs_t = transform.forward(spectrum, nlevels = 5)
    rec = transform.inverse(vecs_t)
    

if __name__ == '__main__':
    compare_1()