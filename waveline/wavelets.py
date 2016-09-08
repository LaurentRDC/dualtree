"""
Extension of PyWavelets to complex wavelets suitable for the Dual-Tree Complex Wavelet Transform.
Source: http://www-sigproc.eng.cam.ac.uk/Main/NGK
"""
import numpy as n
from os.path import join, dirname
import pywt
from pywt import Wavelet

DATADIR = join(dirname(__file__), 'data')

ALL_QSHIFT = ('qshift_a', 'qshift_b', 'qshift_c', 'qshift_d')

def _load_from_file(basename, varnames):
    filename = join(DATADIR, basename + '.npz')
    with n.load(filename) as mat:
        try:
            return tuple([mat[k].flatten() for k in varnames])
        except KeyError:
            raise ValueError('Wavelet does not define ({0}) coefficients'.format(', '.join(varnames)))

def dualtree_wavelet(name):
    """
    Returns a complex wavelet suitable for dual-tree cwt from a name.

    Parameters
    ----------
    name : str, {'qshift_06', 'qshift_a', 'qshift_b', 'qshift_c', 'qshift_d', 'kingsbury99'}
    
    Returns
    -------
    real, imag : pywt.Wavelet objects.
    
    Raises
    ------
    ValueError
        If illegal wavelet name.
    """
    if name == 'kingsbury99':
        return kingsbury99()
    
    return qshift(name)

def dt_first_stage(wavelet = 'kingsbury99_fs'):
    """
    Returns two wavelets to be used in the dual-tree complex wavelet transform, at the first stage.

    Parameters
    ----------
    wavelet : str or Wavelet
        Must be a symmetric wavelet.

    Return
    ------
    wav1, wav2 : Wavelet objects

    Raises
    ------
    ValueError
        If input wavelet is not symmetric.
    """
    if wavelet == 'kingsbury99_fs':
        return kingsbury99_fs()

    if not isinstance(wavelet, Wavelet):
        wavelet = Wavelet(wavelet)
    
    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank

    for bank in (dec_lo, dec_hi):
        bank[1:], bank[0] = bank[:-1], 0 #bank[-1]  #Shift by one index
    for bank in (rec_lo, rec_hi):
        bank[0], bank[1:] = bank[-1], bank[:-1]

    wav2 = Wavelet(name = wavelet.name, filter_bank = [dec_lo, dec_hi, rec_lo, rec_hi])
    return wavelet, wav2
    
def qshift(name = 'qshift_a'):
    """
    Returns a complex qshift wavelet by name.

    Parameters
    ----------
    name : str, {'qshift_06', 'qshift_a' (default), 'qshift_b', 'qshift_c', 'qshift_d'}, optional
        Wavelet family name.capitalize
    
    Returns
    -------
    wavelet : pywt.Wavelet object
        Complex wavelet.
    
    Raises
    ------ 
    ValueError 
        If illegal wavelet family name.
    
    Notes
    -----
    Below is a brief description of the qshift wavelets available.

    qshift_06    Quarter Sample Shift Orthogonal (Q-Shift) 10,10 tap filters,
                 (only 6,6 non-zero taps).
    qshift_a     Q-shift 10,10 tap filters,
                 (with 10,10 non-zero taps, unlike qshift_06).
    qshift_b     Q-Shift 14,14 tap filters.
    qshift_c     Q-Shift 16,16 tap filters.
    qshift_d     Q-Shift 18,18 tap filters.
    """
    factor = 1/n.sqrt(2)

    (dec_real_low, dec_imag_low, rec_real_low, rec_imag_low, 
     dec_real_high, dec_imag_high, rec_real_high, rec_imag_high) = _load_from_file(name, varnames = ('h0a', 'h0b', 'g0a', 'g0b', 'h1a', 'h1b', 'g1a', 'g1b'))
    
    real_filter_bank = [dec_real_low, dec_real_high, rec_real_low, rec_real_high]
    imag_filter_bank = [dec_imag_low, dec_imag_high, rec_imag_low, rec_imag_high]

    return pywt.Wavelet(name = 'real:' + name, filter_bank = real_filter_bank), pywt.Wavelet(name = 'imag:' + name, filter_bank = imag_filter_bank)


#############################################################################################
#                           EXAMPLE COMPLEX WAVELETS FROM
#               http://eeweb.poly.edu/iselesni/WaveletSoftware/dt1D.html 
#############################################################################################
def kingsbury99_fs():
    """
    Returns a first-stage complex wavelet as published in Kingsbury 1999. 
    Taken from http://eeweb.poly.edu/iselesni/WaveletSoftware/dt1D.html 
    """
    real_dec_lo = n.array([0, -0.08838834764832, 0.08838834764832, 0.69587998903400,0.69587998903400, 0.08838834764832, -0.08838834764832, 0.01122679215254, 0.01122679215254, 0])
    real_dec_hi = n.array([0, -0.01122679215254, 0.01122679215254, 0.08838834764832, 0.08838834764832, -0.69587998903400, 0.69587998903400, -0.08838834764832, -0.08838834764832, 0])
    real_rec_lo, real_rec_hi = real_dec_lo[::-1], real_dec_hi[::-1]

    imag_dec_lo = n.array([0.01122679215254, 0.01122679215254, -0.08838834764832, 0.08838834764832, 0.69587998903400, 0.69587998903400, 0.08838834764832, -0.08838834764832, 0, 0])
    imag_dec_hi = n.array([0, 0, -0.08838834764832, -0.08838834764832, 0.69587998903400, -0.69587998903400, 0.08838834764832, 0.08838834764832, 0.01122679215254, -0.01122679215254])
    imag_rec_lo, imag_rec_hi = imag_dec_lo[::-1], imag_dec_hi[::-1]

    real_fb = [real_dec_lo, real_dec_hi, real_rec_lo, real_rec_hi]
    imag_fb = [imag_dec_lo, imag_dec_hi, imag_rec_lo, imag_rec_hi]

    return pywt.Wavelet(name = 'real:', filter_bank = real_fb), pywt.Wavelet(name = 'imag:', filter_bank = imag_fb)

def kingsbury99():
    """
    Returns a late-stage complex wavelet as published in Kingsbury 1999. 
    Taken from http://eeweb.poly.edu/iselesni/WaveletSoftware/dt1D.html 
    """
    real_dec_lo = n.array([ 0.03516384000000, 0, -0.08832942000000, 0.23389032000000, 0.76027237000000, 0.58751830000000, 0, -0.11430184000000, 0, 0])
    real_dec_hi = n.array([0, 0, -0.11430184000000, 0, 0.58751830000000, -0.76027237000000, 0.23389032000000, 0.08832942000000, 0, -0.03516384000000])
    real_rec_lo, real_rec_hi = real_dec_lo[::-1], real_dec_hi[::-1]

    imag_dec_lo = n.array([ 0, 0, -0.11430184000000, 0, 0.58751830000000, 0.76027237000000, 0.23389032000000, -0.08832942000000, 0, 0.03516384000000])
    imag_dec_hi = n.array([-0.03516384000000, 0, 0.08832942000000, 0.23389032000000, -0.76027237000000, 0.58751830000000, 0, -0.11430184000000, 0, 0])
    imag_rec_lo, imag_rec_hi = imag_dec_lo[::-1], imag_dec_hi[::-1]

    real_fb = [real_dec_lo, real_dec_hi, real_rec_lo, real_rec_hi]
    imag_fb = [imag_dec_lo, imag_dec_hi, imag_rec_lo, imag_rec_hi]

    return pywt.Wavelet(name = 'real:', filter_bank = real_fb), pywt.Wavelet(name = 'imag:', filter_bank = imag_fb)