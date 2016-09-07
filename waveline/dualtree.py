from wavelets import dualtree_wavelet
import numpy as n
from pywt import dwt, idwt, wavedec, waverec, dwt_max_level, Wavelet
from warnings import warn

__all__ = ['dtanalysis', 'dtsynthesis', 'dt_max_level', 'dt_approx_rec', 'dt_baseline']

EXTENSION_MODE = 'constant'

def dt_baseline(array, max_iter, level = 'max', first_stage = 'bior5.5', wavelet = 'qshift_a', background_regions = [], mask = None):
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

def dt_approx_rec(array, level, first_stage = 'bior5.5', wavelet = 'qshift_a', mask = None):
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
    original_array = n.copy(array)
    
    # Choose deconstruction and reconstruction functions based on dimensionality
    dim = array.ndim
    if dim > 2:
        raise ValueError('Signal dimensions {} larger than 2 is not supported.'.format(dim))
    elif dim == 2:
        raise NotImplementedError('Only 1D signals are currently supported.')

    if not isinstance(first_stage, Wavelet):
        first_stage = Wavelet(first_stage)
    
    max_level = dt_max_level(data = array, first_stage = first_stage, wavelet = wavelet)
    if level == 'max':
        level = max_level
    elif level > max_level:
        warn('Input level {} higher than maximum {}. Maximum level used.'.format(level, max_level), UserWarning)
        level = max_level
            
    # By now, we are sure that the decomposition level will be supported.
    # Decompose the signal using the multilevel discrete wavelet transform
    coeffs = dtanalysis(data = array, first_stage = first_stage, wavelet = wavelet, level = level, mode = EXTENSION_MODE)
    app_coeffs, det_coeffs = coeffs[0], coeffs[1:]
    
    # Replace detail coefficients by 0 + 0*1j; keep the correct length so that the
    # reconstructed signal has the same size as the (possibly upsampled) signal
    # The structure of coefficients depends on the dimensionality
    zeroed = list()
    if dim == 1:
        for detail in det_coeffs:
            zeroed.append( n.zeros_like(detail, dtype = n.complex) )
    elif dim == 2:
        for detail_tuples in det_coeffs:
            cHn, cVn, cDn = detail_tuples       # See PyWavelet.wavedec2 documentation for coefficients structure.
            zeroed.append( (n.zeros_like(cHn, dtype = n.complex), n.zeros_like(cVn, dtype = n.complex), n.zeros_like(cDn, dtype = n.complex)) )
        
    # Reconstruct signal
    reconstructed = dtsynthesis(coeffs = [app_coeffs] + zeroed, first_stage = first_stage, wavelet = wavelet, mode = EXTENSION_MODE)
    
    # Adjust size of reconstructed signal so that it is the same size as input
    if reconstructed.size == original_array.size:
        return reconstructed
        
    elif original_array.size < reconstructed.size:
        if dim == 1:
            return reconstructed[:original_array.shape[0]]
        elif dim == 2:
            return reconstructed[:original_array.shape[0], :original_array.shape[1]]
        
    elif original_array.size > reconstructed.size:
        extended_reconstructed = n.zeros_like(original_array, dtype = original_array.dtype)        
        if dim == 1:
            extended_reconstructed[:reconstructed.shape[0]] = reconstructed
        elif dim == 2:
            extended_reconstructed[:reconstructed.shape[0], :reconstructed.shape[1]] = reconstructed
        return extended_reconstructed

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
    if not isinstance(first_stage, Wavelet):
        first_stage = Wavelet(first_stage)
    
    filter_len = max([real_wavelet.dec_len, imag_wavelet.dec_len, first_stage.dec_len])
    return dwt_max_level(data_len = min(data.shape),
                         filter_len = max([real_wavelet.dec_len, imag_wavelet.dec_len, first_stage.dec_len]))

#TODO: extend to 2D
def dtanalysis(data, first_stage = 'bior5.5', wavelet = 'qshift_a', level = 'max', mode = 'symmetric'):
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
    """
    data = n.asarray(data)

    real_wavelet, imag_wavelet = dualtree_wavelet(wavelet)
    if not isinstance(first_stage, Wavelet):
        first_stage = Wavelet(first_stage)

    if level == 'max':
        level = dt_max_level(data = data, first_stage = first_stage, wavelet = wavelet)
    elif level < 0:
        raise ValueError('Invalid level value {}. Must be a nonnegative integer.'.format(level))
    elif level == 0:
        return data
    elif level == 1:
        return dwt(data = data, wavelet = first_stage, mode = mode)

    # First stage:
    a, d = dwt(data = data, wavelet = first_stage, mode = mode)

    #parallel trees: real and imag
    real_coeffs = wavedec(data = a, wavelet = real_wavelet, mode = mode, level = level - 1)
    imag_coeffs = wavedec(data = a, wavelet = imag_wavelet, mode = mode, level = level - 1)

    # Combine coefficients into complex form
    coeffs_list = list()
    for real, imag in zip(real_coeffs, imag_coeffs):
        coeffs_list.append(real + 1j*imag)
    coeffs_list.append(d)   # From first stage
    return coeffs_list

def dtsynthesis(coeffs, first_stage = 'bior5.5', wavelet = 'qshift_a', mode = 'symmetric'):
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
        If the input coefficients are too few / not 
    """
    if len(coeffs) < 1:
        raise ValueError(
            "Coefficient list too short (minimum 1 array required).")
    elif len(coeffs) == 1: # level 0 transform
        return coeffs[0]
    elif len(coeffs) == 2: # Level 1 transform: first stage only
        a, d = coeffs
        return idwt(cA = a, cD = d, wavelet = first_stage, mode = mode)

    # Parallel trees:
    real_wavelet, imag_wavelet = dualtree_wavelet(wavelet)
    real_coeffs = [n.real(coeff) for coeff in coeffs[:-1]]  # Last coeff is reserved for first stage
    imag_coeffs = [n.imag(coeff) for coeff in coeffs[:-1]]
    real_synt = waverec(coeffs = real_coeffs, wavelet = real_wavelet, mode = mode)
    imag_synt = waverec(coeffs = imag_coeffs, wavelet = imag_wavelet, mode = mode)

    # First stage
    d = coeffs[-1]
    if len(real_synt) == len(d) + 1:
        real_synt = real_synt[:-1]
    if len(imag_synt == len(d)) + 1:
        imag_synt = imag_synt[:-1]
    
    real = idwt(cA = real_synt, cD = d, wavelet = first_stage, mode = mode)
    imag = idwt(cA = imag_synt, cD = coeffs[-1], wavelet = first_stage, mode = mode)
    return 0.5*(real + imag)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from discrete import approx_rec, baseline

    x = n.arange(0, 10, step = 0.01)
    signal = n.cos(2*n.pi*x)
    background = n.ones_like(x)

   # plt.plot(x, signal, 'r')
    plt.plot(x, dt_baseline(array = signal + background, max_iter = 10, level = 'max', first_stage = 'bior3.7'), 'b')
    plt.plot(x, baseline(array = signal + background, max_iter = 10, level = 'max', wavelet = 'bior3.7'), 'g')
    plt.show()