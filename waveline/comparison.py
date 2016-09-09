"""
Module concerned with the comparison between the discrete wavelet transform approach and
the dual-tree complex wavelet transform.

The test spectrum is taken from [1].

References
----------
[1] F. Zhao, A. Wang. A background subtraction approach based on complex wavelet transforms in EDXRF
"""
from discrete import baseline, approx_rec
from dualtree import dt_baseline, dt_approx_rec, dualtree, idualtree
from dtcwt.numpy.transform1d import Transform1d

import matplotlib.pyplot as plt
import numpy as n
import pywt

i = n.linspace(0, 1024, num = 10240)
BG_REGIONS = [] #[slice(96, 161), slice(390, 526), slice(669, 739), 845, 989]

def gaussian(amp, mean, std):
    return amp*n.exp( -( (i - mean)**2 )/std )

BG1 = gaussian(20, 570, 100000)   #background
#BG1 = n.sin(i/50)

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
    
    plt.plot(i, BG1, 'g')
    plt.plot(i, spectrum + BG1 - approx_rec(array = spectrum + BG1, level = 'max', wavelet = 'db3'), 'b')
    plt.plot(i, spectrum + BG1 - dt_approx_rec(array = spectrum + BG1, level = 'max'), 'r')
    plt.show()

def compare_2():
    spectrum = reference_spectrum_1()
    
    plt.plot(i, spectrum, 'g')
    plt.plot(i, spectrum + BG1 - baseline(array = spectrum + BG1, max_iter = 100, level = 'max', wavelet = 'db8'), 'b')
    plt.plot(i, spectrum + BG1 - dt_baseline(array = spectrum + BG1, max_iter = 100, level = 'max'), 'r')
    plt.show()

def spectrum():
    spectrum = reference_spectrum_1()

    for level in (6,7,8,9, 10):
        plt.plot(i, spectrum + BG1 - dt_baseline(array = spectrum + BG1, max_iter = 100, level = level))
    plt.show()

def compare_brooklyn():
    import matplotlib.pyplot as plt
    level = 5
    wavelet = 'kingsbury99'
    x = n.zeros(shape = (256,), dtype = n.float)
    coeffs = dualtree(x, level = level, wavelet = wavelet)
    coeffs[0][12] = (1/n.sqrt(2)) + 1j*0
    y1 = idualtree(coeffs, wavelet = wavelet)

    coeffs[0][12] = 0 + 1j*(1/n.sqrt(2))
    y2 = idualtree(coeffs)

    amp = n.sqrt(y1**2 + y2**2)
    print(amp.sum())

    i = n.arange(0,256)/256
    plt.plot(i, y1, 'b'); plt.plot(i, y2, 'g'); plt.plot(i, amp, 'k'); plt.show()

if __name__ == '__main__':
    compare_2()