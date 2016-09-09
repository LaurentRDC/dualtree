
import numpy as n
import pywt
from scipy.signal import upfirdn
import wavelets

DEFAULT_WAV = wavelets.kingsbury99()[0]

def dwt(data, level, wavelet = DEFAULT_WAV):
    """
    Multi-level dicrete wavelet transform.

    Parameters
    ----------

    Returns
    -------
    [cA_n, cD_n, cD_n-1, ..., cD2, cD1] : list
        Ordered list of coefficients arrays
        where `n` denotes the level of decomposition. The first element
        (`cA_n`) of the result is approximation coefficients array and the
        following elements (`cD_n` - `cD_1`) are details coefficients arrays.

    """
    lo = data
    coeffs = list()
    for i in range(level):
        lo, hi = analysis(signal = lo, low_pass = wavelet.dec_lo, high_pass = wavelet.dec_hi)
        coeffs.append(hi)
    coeffs.append(lo)
    return list(reversed(coeffs))

def idwt(coeffs, wavelet = DEFAULT_WAV):
    """
    Inverse multi-level discrete wavelet transform

    Returns
    -------
    signal : ndarray
    """
    signal = coeffs[0]
    for hi_coeffs in coeffs[1:]:
        signal = synthesis(lo_coeffs = signal, hi_coeffs = hi_coeffs, low_pass = wavelet.rec_lo, high_pass = wavelet.rec_hi)
    return signal 


def analysis(signal, low_pass, high_pass):
    """
    Signal decomposition from wavelet filter banks

    Parameters
    ----------
    signal : array-like, ndim 1

    low_pass : array-like, ndim 1

    high_pass : array-like, ndim 1
    
    Returns
    -------
    lo, hi : ndarrays
        Low-frequency and high-frequency coefficients
    """
    signal = n.asarray(signal, dtype = n.float)
    low_pass, high_pass = n.asarray(low_pass, dtype = n.float), n.asarray(high_pass, dtype = n.float)

    # Is this really necessary?
    # signal = circular_shift(signal, len(low_pass)/2)

    lo = upfirdn(h = low_pass, x = signal, up = 1, down = 2)
    hi = upfirdn(h = high_pass, x = signal, up = 1, down = 2)

    # Adjust to size len(signal)/2
    # upfirdn's final length seems to be len(signal)/2 + len(lo).
    excess_lo = len(lo) - len(signal)/2
    excess_hi = len(hi) - len(signal)/2
    return lo[excess_lo/2:-(excess_lo/2 + 1)], hi[excess_hi/2:-(excess_hi/2 + 1)]

def synthesis(lo_coeffs, hi_coeffs, low_pass, high_pass):
    """
    Signal recomposition from wavelet filter banks

    Parameters
    ----------
    lo_coeffs : array-like, ndim 1

    hi_coeffs : array-like, ndim 1

    low_pass : array-like, ndim 1

    high_pass : array-like, ndim 1
    
    Returns
    -------
    signal : ndarrays
        Low-frequency and high-frequency coefficients
    """
    lo = upfirdn(h = low_pass, x = lo_coeffs, up = 2, down = 1) 
    hi = upfirdn(h = high_pass, x = hi_coeffs, up = 2, down = 1)

    # Adjust to size 2*coeffs
    # upfirdn's final length seems to be len(signal)/2 + len(lo).
    excess_lo = len(lo) - 2*len(lo_coeffs)
    excess_hi = len(hi) - 2*len(hi_coeffs)
    return lo[excess_lo/2:-(excess_lo/2 + 1)] + hi[excess_hi/2:-(excess_hi/2) + 1]

def circular_shift(signal, i):
    """ Circular shift of a signal by i samples."""
    n = n.arange(0, len(signal))
    n = n.mod(n - i, len(signal))
    return signal[n]