
import numpy as n
from pywt import dwt, idwt, swt, iswt, wavedec, waverec, dwt_max_level, Wavelet
from warnings import warn
from wavelets import dualtree_wavelet, dt_first_stage

__all__ = ['dualtree', 'idualtree', 'dt_max_level', 'dt_approx_rec', 'dt_detail_rec', 'dt_baseline']

EXTENSION_MODE = 'constant'

DEFAULT_FIRST_STAGE = 'db8' #'kingsbury99_fs'
DEFAULT_CMP_WAV = 'qshift_a'

def dt_baseline(array, max_iter, level = 'max', first_stage = DEFAULT_FIRST_STAGE, wavelet = DEFAULT_CMP_WAV, background_regions = [], mask = None):
    """
    Iterative method of baseline determination modified from [1]. This function handles
    both 1D curves and 2D images.
    
    Parameters
    ----------
    array : ndarray, shape (M,N)
        Data with background. Can be either 1D signal or 2D array.
    max_iter : int
        Number of iterations to perform.
    level : int or None, optional
        Decomposition level. A higher level will result in a coarser approximation of
        the input signal (read: a lower frequency baseline). If None (default), the maximum level
        possible is used.
    wavelet : PyWavelet.Wavelet object or str, optional
        Wavelet with which to perform the algorithm. See PyWavelet documentation
        for available values. Default is 'sym6'.
    background_regions : list, optional
        Indices of the array values that are known to be purely background. Depending
        on the dimensions of array, the format is different:
        
        ``array.ndim == 1``
          background_regions is a list of ints (indices) or slices
          E.g. >>> background_regions = [0, 7, 122, slice(534, 1000)]
          
        ``array.ndim == 2``
          background_regions is a list of tuples of ints (indices) or tuples of slices
          E.g. >>> background_regions = [(14, 19), (42, 99), (slice(59, 82), slice(81,23))]
         
        Default is empty list.
    
    mask : ndarray, dtype bool, optional
        Mask array that evaluates to True for pixels that are invalid. Useful to determine which pixels are masked
        by a beam block.
    
    Returns
    -------
    baseline : ndarray, shape (M,N)
        Baseline of the input array.
    
    Raises
    ------
    ValueError
        If input array is neither 1D nor 2D.
    """
    array = n.asarray(array, dtype = n.float)
    if mask is None:
        mask = n.zeros_like(array, dtype = n.bool)
    
    signal = n.copy(array)
    background = n.zeros_like(array, dtype = n.float)
    for i in range(max_iter):
        
        # Make sure the background values are equal to the original signal values in the
        # background regions
        for index in background_regions:
            signal[index] = array[index]
        
        # Wavelet reconstruction using approximation coefficients
        background = dt_approx_rec(array = signal, level = level, first_stage = first_stage, wavelet = wavelet, mask = mask)
        
        # Modify the signal so it cannot be more than the background
        # This reduces the influence of the peaks in the wavelet decomposition
        signal[signal > background] = background[signal > background]
    
    # The background should be identically 0 where the data points are invalid
    background[mask] = 0  
    return background

def dt_approx_rec(array, level, first_stage = DEFAULT_FIRST_STAGE, wavelet = DEFAULT_CMP_WAV, mask = None):
    """
    Approximate reconstruction of a signal/image using the dual-tree approach.
    
    Parameters
    ----------
    array : array-like
        Array to be decomposed. Currently, only 1D and 2D arrays are supported.
    level : int or 'max'
        Decomposition level. A higher level will result in a coarser approximation of
        the input array. If the level is higher than the maximum possible decomposition level,
        the maximum level is used.
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
        If input array has dimension > 1 
    """
    array = n.asarray(array, dtype = n.float)
    
    dim = array.ndim
    if dim > 2:
        raise ValueError('Signal dimensions {} larger than 2 is not supported.'.format(dim))
    elif dim == 2:
        raise NotImplementedError('Only 1D signals are currently supported.')
            
    # By now, we are sure that the decomposition level will be supported.
    # Decompose the signal using the multilevel discrete wavelet transform
    coeffs = dualtree(data = array, first_stage = first_stage, wavelet = wavelet, level = level, mode = EXTENSION_MODE)
    app_coeffs, det_coeffs = coeffs[0], coeffs[1:]
    
    # Replace detail coefficients by 0 + 0*1j; keep the correct length so that the
    # reconstructed signal has the same size as the (possibly upsampled) signal
    # The structure of coefficients depends on the dimensionality
    det_coeffs = [n.zeros_like(det, dtype = n.complex) for det in det_coeffs]
    reconstructed = idualtree(coeffs = [app_coeffs] + det_coeffs, first_stage = first_stage, wavelet = wavelet, mode = EXTENSION_MODE)
    return n.resize(reconstructed, new_shape = array.shape)

def dt_detail_rec(array, level, first_stage = DEFAULT_FIRST_STAGE, wavelet = DEFAULT_CMP_WAV, mask = None):
    """
    Detail reconstruction of a signal/image using the dual-tree approach.
    
    Parameters
    ----------
    array : array-like
        Array to be decomposed. Currently, only 1D and 2D arrays are supported.
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
        If input array has dimension > 1 
    """
    array = n.asarray(array, dtype = n.float)
    
    dim = array.ndim
    if dim > 2:
        raise ValueError('Signal dimensions {} larger than 2 is not supported.'.format(dim))
    elif dim == 2:
        raise NotImplementedError('Only 1D signals are currently supported.')

    coeffs = dualtree(data = array, first_stage = first_stage, wavelet = wavelet, level = level, mode = EXTENSION_MODE)
    app_coeffs = n.zeros_like(coeffs[0], dtype = n.complex) 

    reconstructed = idualtree(coeffs = [app_coeffs] + coeffs[1:], first_stage = first_stage, wavelet = wavelet, mode = EXTENSION_MODE)
    return n.resize(reconstructed, new_shape = array.shape)

def dt_max_level(data, first_stage, wavelet):
    """
    Returns the maximum decomposition level from the dual-tree cwt.

    Parameters
    ----------
    data : array-like
        Input data. Can be of any dimension.
    first_stage : str or Wavelet object
        Wavelet used in the first stage of the dual-tree cwt. See pywt.wavelist() for suitable arguments.
    wavelet : str
        Dual-tree complex wavelet to use. Argument must be supported by dualtree_wavelet
    
    Returns
    -------
    max_level : int
    """
    data = n.asarray(data)
    real_wavelet, imag_wavelet = dualtree_wavelet(wavelet)
    
    filter_len = max([real_wavelet.dec_len, imag_wavelet.dec_len])
    return dwt_max_level(data_len = min(data.shape), filter_len = filter_len)

#TODO: extend to 2D
def dualtree(data, first_stage = DEFAULT_FIRST_STAGE, wavelet = DEFAULT_CMP_WAV, level = 'max', mode = EXTENSION_MODE):
    """
    Multi-level 1D dual-tree complex wavelet transform.

    Parameters
    ----------
    data: array_like
        Input data
    first_stage : str, optional
        Wavelet to use for the first stage. See pywt.wavelist() for a list of suitable arguments
    wavelet : str, optional
        Wavelet to use. Must be appropriate for the dual-tree complex wavelet transform.
        Default is 'qshift_a'.
    level : int or 'max', optional
        Decomposition level (must be >= 0). If level is 'max' (default) then it
        will be calculated using the ``dwt_max_level`` function.
    mode : str, optional
        Signal extension mode, see Modes (default: 'symmetric')

    Returns
    -------
    [cA_n, cD_n, cD_n-1, ..., cD2, cD1] : list
        Ordered list of coefficients arrays
        where `n` denotes the level of decomposition. The first element
        (`cA_n`) of the result is approximation coefficients array and the
        following elements (`cD_n` - `cD_1`) are details coefficients arrays.
    
    Raises
    ------
    ValueError
        If level is a nonnegative integer
    """
    data = n.asarray(data, dtype = n.float)/n.sqrt(2)

    real_wavelet, imag_wavelet = dualtree_wavelet(wavelet)
    real_first, imag_first = dt_first_stage(first_stage)

    if level == 'max':
        level = dt_max_level(data = data, first_stage = first_stage, wavelet = wavelet)
    elif level < 0:
        raise ValueError('Invalid level value {}. Must be a nonnegative integer.'.format(level))
    elif level == 0:
        return data

    #Separate computation trees
    real_coeffs = _single_tree_analysis(data = data, first_stage = real_first, wavelet = real_wavelet, level = level, mode = mode)
    imag_coeffs = _single_tree_analysis(data = data, first_stage = imag_first, wavelet = imag_wavelet, level = level, mode = mode)

    # Combine coefficients into complex form
    coeffs_list = list()
    for real, imag in zip(real_coeffs, imag_coeffs):
        coeffs_list.append(real + 1j*imag)
    return coeffs_list

def idualtree(coeffs, first_stage = DEFAULT_FIRST_STAGE, wavelet = DEFAULT_CMP_WAV, mode = EXTENSION_MODE):
    """
    Multilevel 1D inverse dual-tree complex wavelet transform.

    Parameters
    ----------
    coeffs : array_like
        Coefficients list [cAn, cDn, cDn-1, ..., cD2, cD1]
    first_stage : str, optional
        Wavelet to use for the first stage. See pywt.wavelist() for a list of possible arguments.
    wavelet : str, optional
        Wavelet to use. Must be appropriate for the dual-tree complex wavelet transform.
        Default is 'qshift_a'.
    mode : str, optional
        Signal extension mode, see Modes (default: 'symmetric')
    
    Returns
    -------
    reconstructed : ndarray

    Raises
    ------
    ValueError 
        If the input coefficients are too few
    """
    real_wavelet, imag_wavelet = dualtree_wavelet(wavelet)
    real_first, imag_first = dt_first_stage(first_stage)

    if len(coeffs) < 1:
        raise ValueError(
            "Coefficient list too short (minimum 1 array required).")
    elif len(coeffs) == 1: # level 0 transform
        return coeffs[0]

    # Parallel trees:
    real = _single_tree_synthesis(coeffs = [n.real(coeff) for coeff in coeffs], first_stage = real_first, wavelet = real_wavelet, mode = mode)
    imag = _single_tree_synthesis(coeffs = [n.imag(coeff) for coeff in coeffs], first_stage = imag_first, wavelet = imag_wavelet, mode = mode)
    
    return n.sqrt(2)*(real + imag)/2

def _single_tree_analysis(data, first_stage, wavelet, level, mode):
    """
    Single tree of the forward dual-tree complex wavelet transform.

    Parameters
    ----------
    data : ndarray, ndim 1

    first_stage : Wavelet object

    wavelet : Wavelet object

    level : int

    mode : str

    Returns
    -------
    [cA_n, cD_n, cD_n-1, ..., cD2, cD1] : list
        Ordered list of coefficients arrays
        where `n` denotes the level of decomposition. The first element
        (`cA_n`) of the result is approximation coefficients array and the
        following elements (`cD_n` - `cD_1`) are details coefficients arrays.
    """    
    approx, detail = dwt(data = data, wavelet = wavelet, mode = mode)
    coeffs = wavedec(data = approx, wavelet = wavelet, mode = mode, level = level - 1)
    coeffs.append(detail)
    return coeffs

def _single_tree_synthesis(coeffs, first_stage, wavelet, mode):
    """
    Single tree of the inverse dual-tree complex wavelet transform.

    Parameters
    ----------
    coeffs : list

    first_stage : Wavelet object

    wavelet : Wavelet object

    mode : str

    Returns
    -------
    reconstructed : ndarray, ndim 1
    """
    late_stage_coeffs, first_stage_detail = coeffs[:-1], coeffs[-1]
    late_synthesis = waverec(coeffs = late_stage_coeffs, wavelet = wavelet, mode = mode)

    if len(late_synthesis) == len(first_stage_detail) + 1:
        late_synthesis = late_synthesis[:-1]
    
    return idwt(cA = late_synthesis, cD = first_stage_detail, wavelet = wavelet, mode = mode)