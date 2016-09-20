# dualtree

Python package, based on PyWavelets, implementing Nick Kingsbury's dual-tree complex wavelet transform.

## Contents

From the package docstrings:

* dualtree, idualtree
    Dual-tree complex wavelet transform (and its inverse) implemented using PyWavelets. Implementation
    tricks from [1], such as first stage filtering and filter swapping at later stages, are also
    included.

* approx_rec, detail_rec
    Decomposition and recomposition of signals using only approximate or detail coefficients.

* dualtree_max_level
    Maximal decomposition level of the dual-tree complex wavelet transform.

* baseline
    Baseline determination of signals using the dual-tree complex wavelet transform. Modified algorithm
    from [2].

* denoise
    Denoising of signals using the dual-tree complex wavelet transform.

* dualtree_wavelet
    Pair of real and imaginary wavelet that forms a complex wavelet appropriate for the dual-tree
    complex wavelet transform.

* dualtree_first_stage
    Pair of real and imaginary wavelet, shifted by one sample with respect to one another, forming a complex
    wavelet appropriate for first-stage filtering during the dual-tree complex wavelet transform.

## Example

We start with the base:
    from dualtree import dualtre, idualtree, baseline
    import matplotlib.pyplot as plt

Here is an example of getting the dual-tree complex wavelet transform coefficients from real data:
    signal = n.load('~\data\diffraction.npy')                           # Included example electron diffraction data
    coeffs = dualtree(signal, wavelet = 'qshift3', level = 4)
    reconstructed = idualtree(coeffs = coeffs, wavelet = 'qshift3')     # level is inferred
    n.allclose(signal, reconstructed)                                   # Check perfect reconstruction

Example of algorithm (baseline-determination) on a NumPy array:
    signal = n.load('~\data\diffraction.npy')
    background = baseline(signal, wavelet = 'qshift3', max_iter = 100)  # Might not be optimal parameters
    plt.plot(signal, '.k', background, '.r')

## TODO

There are a few things left to do before hitting version 1.0:

1. Implementation beyond 1D: the dualtree and idualtree functions have to be implemented for 2D images.

2. Algorithms: a more robust denoising algorithm should be implemented.

## References

1. Selesnick, I. W. et al. 'The Dual-tree Complex Wavelet Transform', IEEE Signal Processing Magazine pp. 123 - 151, November 2005.

2. Galloway et al. 'An Iterative Algorithm for Background Removal in Spectroscopy by Wavelet Transforms', Applied Spectroscopy pp. 1370 - 1376, September 2009.
