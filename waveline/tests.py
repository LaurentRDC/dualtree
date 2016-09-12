from discrete import approx_rec, baseline, denoise, enhance
from dualtree import dualtree, idualtree, dt_approx_rec, dt_detail_rec
from wavelets import dualtree_wavelet, dt_first_stage, kingsbury99, kingsbury99_fs, qshift, ALL_QSHIFT

import matplotlib.pyplot as plt
import numpy as n
import pywt
import unittest

n.random.seed(23)

##############################################################################
###           DUAL-TREE COMPLEX WAVELET TRANSFORM
##############################################################################

class TestComplexWavelets(unittest.TestCase):

    def test_qshift(self):
        for name in ALL_QSHIFT:
            wavelets = qshift(name)
    
    def test_first_stage(self):
        """ Test of the 1 sample shift """
        array = n.sin(n.arange(0, 10, step = 0.01))
        for wavelet in pywt.wavelist():
            try:
                wav1, wav2 = dt_first_stage(wavelet)
            except(ValueError):  #Invalid wavelet
                continue
            for wav in (wav2, wav2):
                if not n.allclose( array, pywt.waverec(pywt.wavedec(array, wav), wav) ):
                    print(wav)

    #@unittest.expectedFailure
    def test_first_stage_reconstruction(self):
        wav1, wav2 = dt_first_stage('db5')
        array = n.sin(n.arange(0, 10, step = 0.01))
        # Since it is tested that wav1 == Wavelet('db5'), 
        # only test that wav2 is a perfect reconstruction filter

        a, d = pywt.dwt(data = array, wavelet = wav2)
        rec = pywt.idwt(cA = a, cD = d, wavelet = wav2)
        self.assertTrue(n.allclose(array, rec))
    
    def test_kingsbury99_fs(self):
        """ Test for perfect reconstruction """
        array = n.sin(n.arange(0, 10, step = 0.01))
        wav1, wav2 = kingsbury99_fs()
        for wav in (wav1, wav2):
            a, d = pywt.dwt(data = array, wavelet = wav)
            rec = pywt.idwt(cA = a, cD = d, wavelet = wav)
            self.assertTrue(n.allclose(array, rec))
    
    def test_kingsbury99(self):
        """ Test for perfect reconstruction """
        array = n.sin(n.arange(0, 10, step = 0.01))
        wav1, wav2 = kingsbury99()
        for wav in (wav1, wav2):
            a, d = pywt.dwt(data = array, wavelet = wav)
            rec = pywt.idwt(cA = a, cD = d, wavelet = wav)
            self.assertTrue(n.allclose(array, rec))

class TestDualTree(unittest.TestCase):
    
    def test_perfect_reconstruction_level_1(self):
        array = n.sin(n.arange(0, 10, step = 0.01))
        coeffs = dualtree(data = array, level = 1)
        reconstructed = idualtree(coeffs = coeffs)
        self.assertTrue(n.allclose(array, reconstructed))

    def test_perfect_reconstruction_multilevel(self):
        array = n.sin(n.arange(0, 10, step = 0.01))
        coeffs = dualtree(data = array, level = 3)
        reconstructed = idualtree(coeffs = coeffs)
        self.assertTrue(n.allclose(array, reconstructed))
    
    def test_perfect_reconstruction_max_level(self):
        array = n.sin(n.arange(0, 10, step = 0.01))
        coeffs = dualtree(data = array, level = 'max')
        reconstructed = idualtree(coeffs = coeffs)
        self.assertTrue(n.allclose(array, reconstructed))
    
    def test_dt_approx_and_detail_rec(self):
        array = n.sin(n.arange(0, 10, step = 0.01))
        low_freq = dt_approx_rec(array = array, level = 'max')
        high_freq = dt_detail_rec(array = array, level = 'max')
        self.assertTrue(n.allclose(array, low_freq + high_freq))

##############################################################################
###           BASELINE AND COMPANY
##############################################################################

class Test2D(unittest.TestCase):
    def setUp(self):
        self.array = n.zeros(shape = (100, 100), dtype = n.float)

class Test1D(unittest.TestCase):
    def setUp(self):
        self.array = n.zeros(shape = (100,), dtype = n.float)



class TestEdgeCases(object):

    def test_dimensions(self):
        self.assertRaises(Exception, baseline, {'data':n.zeros(shape = (3,3,3), dtype = n.uint), 'max_iter': 10, 'level': 1})

    def test_zero_level(self):
        # Since all function are based on approx_rec, we only need to test level = 0 for approx_rec
        self.assertTrue(n.allclose(self.array, approx_rec(self.array, level = 0, wavelet = 'db1')))

class TestEdgeCases1D(Test1D, TestEdgeCases): pass

class TestEdgeCases2D(Test2D, TestEdgeCases): pass



class TestTrivial(object):

    def test_baseline(self):
        self.assertTrue(n.allclose(self.array, baseline(self.array, max_iter = 10)))

    def test_denoise(self):
        self.assertTrue(n.allclose(self.array, denoise(self.array)))
    
    def test_enhance(self):
        self.assertTrue(n.allclose(self.array, enhance(self.array)))

class TestTrivial1D(Test1D, TestTrivial): pass

class TestTrivial2D(Test2D, TestTrivial): pass



class TestDenoise(object):

    def test_random(self):
        noisy = self.array + 0.05*n.random.random(size = self.array.shape)
        self.assertTrue(n.allclose(self.array, denoise( noisy, level = 'max', wavelet = 'db1' ), atol = 0.05))

class TestDenoise1D(Test1D, TestDenoise): pass

class TestDenoise2D(Test2D, TestDenoise): pass

if __name__ == '__main__':
    unittest.main()