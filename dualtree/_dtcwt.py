"""
Dual-tree complex wavelet transform (DTCWT) module.

Author: Laurent P. René de Cotret
"""
import numpy as n
from pywt import dwt, idwt, dwt_max_level
from ._wavelets import dualtree_wavelet, dualtree_first_stage

__all__ = ['dualtree', 'idualtree', 'dualtree_max_level', 'approx_rec', 'detail_rec']

DEFAULT_MODE = 'constant'
DEFAULT_FIRST_STAGE = 'sym4'
DEFAULT_CMP_WAV = 'qshift3'

#TODO: extend to 2D
def dualtree(data, first_stage = DEFAULT_FIRST_STAGE, wavelet = DEFAULT_CMP_WAV, level = 'max', mode = DEFAULT_MODE):
    """
    1D dual-tree complex wavelet transform, implemented from [1].

    Parameters
    ----------
    data: array_like
        Input data. Only 1D arrays are supported for now. 2D support is planned.
    first_stage : str, optional
        Wavelet to use for the first stage. See dualtree.ALL_FIRST_STAGE for a list of suitable arguments
    wavelet : str, optional
        Wavelet to use in stages > 1. Must be appropriate for the dual-tree complex wavelet transform.
        See dualtree.ALL_COMPLEX_WAV for possible
    level : int or 'max', optional
        Decomposition level (must be >= 0). If level is 'max' (default) then it
        will be calculated using the ``dwt_max_level`` function.
    mode : str, optional
        Signal extension mode, see pywt.Modes. Default is 'constant'.

    Returns
    -------
    [cA_n, cD_n, cD_n-1, ..., cD2, cD1] : list
        Ordered list of coefficients arrays
        where `n` denotes the level of decomposition. The first element
        (`cA_n`) of the result is approximation coefficients array and the
        following elements (`cD_n` - `cD_1`) are details coefficients arrays.
    
    Raises
    ------
    NotImplementedError
        If input array has dimension 2
    ValueError
        Raised if the input array has dimension > 2
        Raised if level < 0
    
    Notes
    -----
    The implementation uses two tricks presented in [1]:
        `` Different first-stage wavelet ``
            The first level of the dual-tree complex wavelet transform involves a combo of shifted wavelets.
        
        `` Swapping of filters at each stage ``
            At each level > 1, the filters (separated into real and imaginary trees) are swapped.
    
    References
    ----------
    [1] Selesnick, I. W. et al. 'The Dual-tree Complex Wavelet Transform', IEEE Signal Processing Magazine pp. 123 - 151, November 2005.
    """
    data = n.asarray(data, dtype = n.float)/n.sqrt(2)
    if data.ndim == 2:
        raise NotImplementedError('Dual-tree complex wavelet transform is not yet implemented for 2D arrays.')
    elif data.ndim > 2:
        raise ValueError('Dual-tree complex wavelet transform is not supported for {}D arrays'.format(array.ndim))

    real_wavelet, imag_wavelet = dualtree_wavelet(wavelet)
    real_first, imag_first = dualtree_first_stage(first_stage)

    if level == 'max':
        level = dualtree_max_level(data = data, first_stage = first_stage, wavelet = wavelet)
    elif level < 0:
        raise ValueError('Invalid level value {}. Must be a nonnegative integer.'.format(level))
    elif level == 0:
        return [data]
    
    real_coeffs = _single_tree_analysis(data = data, first_stage = real_first, wavelet = (real_wavelet, imag_wavelet), level = level, mode = mode, axis = -1)
    imag_coeffs = _single_tree_analysis(data = data, first_stage = imag_first, wavelet = (imag_wavelet, real_wavelet), level = level, mode = mode, axis = -1)

    # Combine coefficients into complex form
    return [real + 1j*imag for real, imag in zip(real_coeffs, imag_coeffs)]

def idualtree(coeffs, first_stage = DEFAULT_FIRST_STAGE, wavelet = DEFAULT_CMP_WAV, mode = DEFAULT_MODE):
    """
    1D inverse dual-tree complex wavelet transform implemented from [1].

    Parameters
    ----------
    coeffs : array_like
        Coefficients list [cAn, cDn, cDn-1, ..., cD2, cD1]
    first_stage : str, optional
        Wavelet to use for the first stage. See dualtree.ALL_FIRST_STAGE for a list of possible arguments.
    wavelet : str, optional
        Wavelet to use in stages > 1. Must be appropriate for the dual-tree complex wavelet transform.
        See dualtree.ALL_COMPLEX_WAV for possible arguments.
    mode : str, optional
        Signal extension mode, see Modes.
    
    Returns
    -------
    reconstructed : ndarray

    Raises
    ------
    ValueError
        If the input coefficients are too few
    
    Notes
    -----
    The implementation uses two tricks presented in [1]:
        `` Different first-stage wavelet ``
            The first level of the dual-tree complex wavelet transform involves a combo of shifted wavelets.
        
        `` Swapping of filters at each stage ``
            At each level > 1, the filters (separated into real and imaginary trees) are swapped.
    
    References
    ----------
    [1] Selesnick, I. W. et al. 'The Dual-tree Complex Wavelet Transform', IEEE Signal Processing Magazine pp. 123 - 151, November 2005.
    """
    if len(coeffs) < 1:
        raise ValueError(
            "Coefficient list too short (minimum 1 array required).")
    elif len(coeffs) == 1: # level 0 inverse transform
        real, imag = coeffs[0], coeffs[0]
    else:
        real_wavelet, imag_wavelet = dualtree_wavelet(wavelet)
        real_first, imag_first = dualtree_first_stage(first_stage)

        real = _single_tree_synthesis(coeffs = [coeff.real for coeff in coeffs], first_stage = real_first, wavelet = (real_wavelet, imag_wavelet), mode = mode, axis = -1)
        imag = _single_tree_synthesis(coeffs = [coeff.imag for coeff in coeffs], first_stage = imag_first, wavelet = (imag_wavelet, real_wavelet), mode = mode, axis = -1)
    
    return n.sqrt(2)*(real + imag)/2

def approx_rec(array, level, first_stage = DEFAULT_FIRST_STAGE, wavelet = DEFAULT_CMP_WAV, mask = None):
    """
    Approximate reconstruction of a signal/image using the dual-tree approach.
    
    Parameters
    ----------
    array : array-like
        Array to be decomposed. Currently, only 1D arrays are supported. 2D support is planned.
    level : int or 'max'
        Decomposition level. A higher level will result in a coarser approximation of
        the input array. If the level is higher than the maximum possible decomposition level,
        the maximum level is used.
        If 'max', the maximum possible decomposition level is used.
    first_stage : str, optional
        First-stage wavelet to use. See dualtree.ALL_FIRST_STAGE for possible arguments.
    wavelet : str, optional
        Complex wavelet to use in late stages. See dualtree.ALL_COMPLEX_WAV for possible arguments.
    mask : ndarray or None, optional.
        Same shape as array. Must evaluate to True where data is invalid.
        If None (default), a trivial mask is used.
            
    Returns
    -------
    reconstructed : ndarray
        Approximated reconstruction of the input array.
    
    Raises
    ------    
    ValueError
        If input array has dimension > 2
    NotImplementedError
        If input array has dimension 2 
    """
    coeffs = dualtree(data = array, first_stage = first_stage, wavelet = wavelet, level = level, mode = DEFAULT_MODE)
    app_coeffs, det_coeffs = coeffs[0], coeffs[1:]
    
    det_coeffs = [n.zeros_like(det, dtype = n.complex) for det in det_coeffs]
    reconstructed = idualtree(coeffs = [app_coeffs] + det_coeffs, first_stage = first_stage, wavelet = wavelet, mode = DEFAULT_MODE)
    return n.resize(reconstructed, new_shape = array.shape)

def detail_rec(array, level, first_stage = DEFAULT_FIRST_STAGE, wavelet = DEFAULT_CMP_WAV, mask = None):
    """
    Detail reconstruction of a signal/image using the dual-tree approach.
    
    Parameters
    ----------
    array : array-like
        Array to be decomposed. Currently, only 1D arrays are supported. 2D support is planned.
    level : int or 'max'
        Decomposition level. 
        If None, the maximum possible decomposition level is used.
    wavelet : str or Wavelet object
        Can be any argument accepted by PyWavelet.Wavelet, e.g. 'db10'
    mask : ndarray
        Same shape as array. Must evaluate to True where data is invalid.
            
    Returns
    -------
    reconstructed : ndarray
        Approximated reconstruction of the input array.
    
    Raises
    ------    
    ValueError
        If input array has dimension > 2
    NotImplementedError
        If input array has dimension 2

    See Also
    --------
    approx_rec 
    """
    coeffs = dualtree(data = array, first_stage = first_stage, wavelet = wavelet, level = level, mode = DEFAULT_MODE)
    app_coeffs = n.zeros_like(coeffs[0], dtype = n.complex) 
    reconstructed = idualtree(coeffs = [app_coeffs] + coeffs[1:], first_stage = first_stage, wavelet = wavelet, mode = DEFAULT_MODE)
    return n.resize(reconstructed, new_shape = array.shape)

def dualtree_max_level(data, first_stage, wavelet):
    """
    Returns the maximum decomposition level from the dual-tree complex wavelet transform.

    Parameters
    ----------
    data : ndarray
        Input data. Can be of any dimension.
    first_stage : str
        Wavelet used in the first stage of the dual-tree cwt. See pywt.wavelist() for suitable arguments.
    wavelet : str
        Dual-tree complex wavelet to use. Argument must be supported by dualtree_wavelet
    
    Returns
    -------
    max_level : int
    """
    real_wavelet, imag_wavelet = dualtree_wavelet(wavelet)
    return dwt_max_level(data_len = min(data.shape), filter_len = max([real_wavelet.dec_len, imag_wavelet.dec_len]))

##############################################################################
#           SINGLE TREE OF THE TRANSFORM

def _altern_wavelet(wavelets, level = 1):
    """ Generator yielding alternative wavelets for swapping at each stage. """
    while True:
        yield wavelets[(level + 1) % 2]
        yield wavelets[level % 2]

def _single_tree_analysis(data, first_stage, wavelet, level, mode, axis = -1):
    """
    Single tree of the forward dual-tree complex wavelet transform.

    Parameters
    ----------
    data : ndarray, ndim 1

    first_stage : Wavelet object

    wavelet : 2-tuple of Wavelet object

    level : int

    mode : str

    axis : int, optional
        Axis over which to compute. Default is last axis

    Returns
    -------
    [cA_n, cD_n, cD_n-1, ..., cD2, cD1] : list
        Ordered list of coefficients arrays
        where `n` denotes the level of decomposition. The first element
        (`cA_n`) of the result is approximation coefficients array and the
        following elements (`cD_n` - `cD_1`) are details coefficients arrays.
    """
    approx, first_detail = dwt(data = data, wavelet = first_stage, mode = mode)    
    coeffs_list = list()
    for i, wav in zip(range(level - 1), _altern_wavelet(wavelet)):
        approx, detail = dwt(data = approx, wavelet = wav, mode = mode, axis = axis)
        coeffs_list.append(detail)
    
    # Format list ot be compatible to PyWavelet's format. See pywt.wavedec documentation.
    coeffs_list.append(approx)
    coeffs_list.reverse()
    coeffs_list.append(first_detail)

    return coeffs_list

def _single_tree_synthesis(coeffs, first_stage, wavelet, mode, axis = -1):
    """
    Single tree of the inverse dual-tree complex wavelet transform.

    Parameters
    ----------
    coeffs : list

    first_stage : Wavelet object

    wavelet : 2-tuple of Wavelet object

    mode : str

    axis : int, optional
        Axis over which to compute. Default is last axis

    Returns
    -------
    reconstructed : ndarray, ndim 1
    """
    # Determine the level except first stage:
    # The order of wavelets depends on whether
    # the level is even or odd.
    level = len(coeffs) - 2    
    late_stage_coeffs, first_stage_detail = coeffs[:-1], coeffs[-1]

    # late stage reconstruction
    approx, detail_coeffs = late_stage_coeffs[0], late_stage_coeffs[1:]
    for detail, wav in zip(detail_coeffs, _altern_wavelet(wavelet, level)):
        if len(approx) == len(detail) + 1:  # As done in pywt.wavedec
            approx = approx[:-1]
        approx = idwt(cA = approx, cD = detail, wavelet = wav, mode = mode)
    
    if len(approx) == len(first_stage_detail) + 1:
        approx = approx[:-1]
    
    return idwt(cA = approx, cD = first_stage_detail, wavelet = first_stage, mode = mode, axis = axis)