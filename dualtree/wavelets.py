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
ALL_COMPLEX_WAV = ('kingsbury99',) + ALL_QSHIFT
ALL_FIRST_STAGE = ('kingsbury99_fs')

def _load_from_file(basename, varnames):
    filename = join(DATADIR, basename + '.npz')
    with n.load(filename) as mat:
        try:
            return tuple([mat[k].flatten() for k in varnames])
        except KeyError:
            raise ValueError('Wavelet does not define ({0}) coefficients'.format(', '.join(varnames)))

def circular_shift(signal, i):
    """ 
    Circular shift of a signal by i samples.
    
    Parameters
    ----------
    signal : array-like, ndim 1

    i : int
        Number of samples by which to shift. Positive numbers
        shift to the right, while negative numbers shift to the left.
    
    Returns
    -------
    shifted : ndarray, ndim 1

    Examples
    --------
    >>> import numpy as n
    >>> arr = n.zeros( shape = (10, ) )
    >>> arr[5] = 1
    >>> arr
    array([ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.])
    >>> circular_shift(arr, -1)
    array([ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.])
    >>> circular_shift(arr, 5)  # loops around
    array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
    """
    extent = n.arange(0, len(signal))
    return signal[ n.mod(extent - i, len(signal))]

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
        Wavelet to be shifted for first-stage use. Can be any wavelet in pywt.wavelist() except for wavelets
        in the 'dmey' family.

    Return
    ------
    wav1, wav2 : Wavelet objects

    Raises
    ------
    ValueError
        If invalid first stage wavelet.
    """
    # Special case, preshifted
    if wavelet == 'kingsbury99_fs':
        return kingsbury99_fs()

    if not isinstance(wavelet, Wavelet):
        wavelet = Wavelet(wavelet)
    
    if wavelet.name == 'dmey':
        raise ValueError('{} is an invalid first stage wavelet.'.format(wavelet.name))
    
    # extend filter bank with zeros
    filter_bank = [n.array(f, copy = True) for f in wavelet.filter_bank]
    for filt in filter_bank:
        extended = n.zeros( shape = (filt.shape[0] + 2,), dtype = n.float)
        extended[1:-1] = filt
        filt = extended

    # Shift deconstruction filters to one side, and reconstruction
    # to the other side
    shifted_fb = [n.array(f, copy = True) for f in wavelet.filter_bank]
    for filt in shifted_fb[::2]:    #Deconstruction filters
        filt = circular_shift(filt, 1)
    for filt in shifted_fb[2::]:    # Reconstruction filters
        filt = circular_shift(filt, -1)
    
    return Wavelet(name = wavelet.name, filter_bank = filter_bank), Wavelet(name = wavelet.name, filter_bank = shifted_fb)
    
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