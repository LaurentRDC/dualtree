"""
Extension of PyWavelets to complex wavelets suitable for the Dual-Tree Complex Wavelet Transform.
Source: http://www-sigproc.eng.cam.ac.uk/Main/NGK
"""
from numpy import load
from os.path import join, dirname
import pywt

DATADIR = join(dirname(__file__), 'data')
COEFF_CACHE = {}

def _load_from_file(basename, varnames):
    filename = join(DATADIR, basename + '.npz')

    try:
        mat = COEFF_CACHE[filename]
    except KeyError:
        mat = load(filename)
        COEFF_CACHE[filename] = mat

    try:
        return tuple(mat[k] for k in varnames)
    except KeyError:
        raise ValueError('Wavelet does not define ({0}) coefficients'.format(', '.join(varnames)))

def biort(name = 'near_sym_a'):
    """
    Returns a biorthogonal wavelet by name.

    Parameters
    ----------
    name : str, {'antonini', 'legall', 'near_sym_a' (default), 'near_sym_b'}, optional
        Wavelet family name

    Returns
    -------
    real_wavelet, imag_wavelet : pywt.Wavelet objects
    
    Raises
    ------
    ValueError
        If illegal wavelet family name.
    
    Notes
    -----
    Below is a brief description of the biorthognal wavelets available
    
    antonini       Antonini 9,7 tap filters.
    legall         LeGall 5,3 tap filters.
    near_sym_a     Near-Symmetric 5,7 tap filters.
    near_sym_b     Near-Symmetric 13,19 tap filters.
    """
    factor = 1/n.sqrt(2)
    real_low, imag_low, real_high, imag_high = _load_from_file(name, ('h0o', 'g0o', 'h1o', 'g1o'))
    
    real_filter_bank = [real_low, real_high, imag_low**(-1), imag_high**(-1)]
    real_filter_bank = [factor*filt for filt in real_filter_bank]

    imag_filter_bank = [imag_low, imag_high, real_low**(-1), real_high**(-1)]
    imag_filter_bank = [factor*filt for filt in imag_filter_bank]

    return pywt.Wavelet(name = 'real:'+name, filter_bank = real_filter_bank), pywt.Wavelet(name = 'imag:'+name, filter_bank = imag_filter_bank)
    
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
    (real_low_a, real_low_b, imag_low_a, imag_low_b, 
     real_high_a, real_high_b, imag_high_a, imag_high_b) = _load_from_file(name, ('h0a', 'h0b', 'g0a', 'g0b', 'h1a', 'h1b', 'g1a', 'g1b'))

# vim:sw=4:sts=4:et