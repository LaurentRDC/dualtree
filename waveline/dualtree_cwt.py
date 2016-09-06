import numpy as n
from pywt import dwt, idwt

def dtcwt(data, wavelet, mode = 'symmetric', axis = -1):
    """
    Single level Dual-Tree Complex Wavelet Transform.
    Parameters
    ----------
    data : array_like
        Input signal
    wavelet : Wavelet object or name
        Wavelet to use
    mode : str, optional
        Signal extension mode, see Modes
    axis: int, optional
        Axis over which to compute the DWT. If not given, the
        last axis is used.

    Returns
    -------
    (cA, cD) : tuple of ndarrays, dtype complex
        Approximate and Detail coefficients. 
    """
    # TODO: check that the wavelet is a complex wavelet of the right form.
    # TODO: create a real wavelet object and an imaginary wavelet object
    real_wavelet, imag_wavelet = biort()
    return dwt(data = data, wavelet = real_wavelet, mode = mode, axis = axis) 
         + dwt(data = data, wavelet = imag_wavelet, mode = mode, axis = axis)

def idtcwt(cA, cD, wavelet, mode='symmetric', axis=-1):
    """
    Single level Inverse Dual-Tree Complex Wavelet Transform.

    Parameters
    ----------
    cA : array_like or None
        Complex approximation coefficients.  If None, will be set to array of zeros
        with same shape as `cD`.
    cD : array_like or None
        Complex detail coefficients.  If None, will be set to array of zeros
        with same shape as `cA`.
    wavelet : Wavelet object or name
        Wavelet to use
    mode : str, optional (default: 'symmetric')
        Signal extension mode, see Modes
    axis: int, optional
        Axis over which to compute the inverse DWT. If not given, the
        last axis is used.

    Returns
    -------
    rec: array_like
        Single level reconstruction of signal from given coefficients.
    """
    # TODO: check that the wavelet is a complex wavelet of the right form.
    # TODO: create a real wavelet object and an imaginary wavelet object
    real_wavelet = wavelet
    imag_wavelet = wavelet
    approx_coeffs = n.asarray(cA, dtype = n.complex)
    detail_coeffs = n.asarrat(cD, dtype = n.complex)
    return 0.5*(idwt(n.real(approx_coeffs), n.real(detail_coeffs), wavelet = real_wavelet, mode = mode, axis = axis) 
              + idwt(n.imag(approx_coeffs), n.imag(detail_coeffs), wavelet = imag_wavelet, mode = mode, axis = axis))