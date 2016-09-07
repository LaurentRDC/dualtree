"""
Extension of PyWavelets to complex wavelets suitable for the Dual-Tree Complex Wavelet Transform.
Source: http://www-sigproc.eng.cam.ac.uk/Main/NGK
"""
import numpy as n
from os.path import join, dirname
import pywt

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
    name : str, {'qshift_06', 'qshift_a' (default), 'qshift_b', 'qshift_c', 'qshift_d'}
    
    Returns
    -------
    real, imag : pywt.Wavelet objects.
    
    Raises
    ------
    ValueError
        If illegal wavelet name.
    """
    # This is empty right now, but it might be extended in the future.
    return qshift(name)
    
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