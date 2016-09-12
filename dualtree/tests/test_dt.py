from dualtree import dualtree, idualtree, dt_approx_rec, dt_detail_rec
from wavelets import dualtree_wavelet, dt_first_stage, kingsbury99, kingsbury99_fs, qshift, ALL_QSHIFT

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

if __name__ == '__main__':
    unittest.main()