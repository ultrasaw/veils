#!/usr/bin/env python3
"""
Test the difference between numpy.fft and scipy.fft
"""

import numpy as np
import scipy.fft


def test_fft_libraries():
    """Test if numpy.fft and scipy.fft give different results."""
    # Test data
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    
    print("=== FFT Library Comparison ===")
    print(f"Input: {x}")
    
    # Test regular FFT
    numpy_fft = np.fft.fft(x)
    scipy_fft = scipy.fft.fft(x)
    
    print(f"NumPy FFT: {numpy_fft}")
    print(f"SciPy FFT: {scipy_fft}")
    print(f"FFT difference: {np.max(np.abs(numpy_fft - scipy_fft))}")
    
    # Test RFFT
    numpy_rfft = np.fft.rfft(x)
    scipy_rfft = scipy.fft.rfft(x)
    
    print(f"NumPy RFFT: {numpy_rfft}")
    print(f"SciPy RFFT: {scipy_rfft}")
    print(f"RFFT difference: {np.max(np.abs(numpy_rfft - scipy_rfft))}")
    
    # Test with different lengths
    for n in [8, 10, 12]:
        numpy_rfft_n = np.fft.rfft(x, n=n)
        scipy_rfft_n = scipy.fft.rfft(x, n=n)
        diff = np.max(np.abs(numpy_rfft_n - scipy_rfft_n))
        print(f"RFFT(n={n}) difference: {diff}")


if __name__ == "__main__":
    test_fft_libraries()

