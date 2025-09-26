#!/usr/bin/env python3
"""
Minimal debug to isolate the STFT difference.
"""

import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from standalone_stft import StandaloneSTFT


def test_single_windowed_fft():
    """Test a single windowed FFT operation."""
    # Create simple test data
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    win = gaussian(len(x), std=len(x)/8, sym=True)
    
    print("=== Single Windowed FFT Test ===")
    print(f"Signal: {x}")
    print(f"Window: {win}")
    
    # Create STFT objects
    scipy_stft = ShortTimeFFT(win, hop=1, fs=1.0)
    standalone_stft = StandaloneSTFT(win, hop=1, fs=1.0)
    
    # Apply window and conjugate (as done in STFT)
    x_windowed = x * win.conj()
    print(f"Windowed signal: {x_windowed}")
    
    # Test FFT directly
    scipy_fft_result = scipy_stft._fft_func(x_windowed)
    standalone_fft_result = standalone_stft._fft_func(x_windowed)
    
    print(f"Scipy FFT result: {scipy_fft_result}")
    print(f"Standalone FFT result: {standalone_fft_result}")
    print(f"Difference: {np.abs(scipy_fft_result - standalone_fft_result)}")
    print(f"Max difference: {np.max(np.abs(scipy_fft_result - standalone_fft_result))}")
    
    # Test numpy rfft directly
    numpy_rfft = np.fft.rfft(x_windowed, n=len(x))
    print(f"NumPy rfft result: {numpy_rfft}")
    print(f"Scipy vs NumPy: {np.max(np.abs(scipy_fft_result - numpy_rfft))}")
    print(f"Standalone vs NumPy: {np.max(np.abs(standalone_fft_result - numpy_rfft))}")


def test_stft_single_slice():
    """Test STFT with a single slice to isolate the issue."""
    # Create a signal that will produce exactly one slice
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    win = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # Rectangular window for simplicity
    
    print("\n=== Single STFT Slice Test ===")
    print(f"Signal: {x}")
    print(f"Window: {win}")
    
    # Create STFT objects
    scipy_stft = ShortTimeFFT(win, hop=1, fs=1.0)
    standalone_stft = StandaloneSTFT(win, hop=1, fs=1.0)
    
    print(f"Scipy p_range: {scipy_stft.p_range(len(x))}")
    print(f"Standalone p_range: {standalone_stft.p_range(len(x))}")
    
    # Get STFT
    S_scipy = scipy_stft.stft(x)
    S_standalone = standalone_stft.stft(x)
    
    print(f"Scipy STFT shape: {S_scipy.shape}")
    print(f"Standalone STFT shape: {S_standalone.shape}")
    
    # Compare each slice
    for i in range(min(S_scipy.shape[1], S_standalone.shape[1])):
        diff = np.abs(S_scipy[:, i] - S_standalone[:, i])
        print(f"Slice {i}: max diff = {np.max(diff):.2e}")
        if np.max(diff) > 1e-10:
            print(f"  Scipy: {S_scipy[:, i]}")
            print(f"  Standalone: {S_standalone[:, i]}")
            print(f"  Diff: {diff}")


def test_slice_extraction():
    """Test the slice extraction process in detail."""
    print("\n=== Slice Extraction Detailed Test ===")
    
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    win = np.ones(3)  # Simple 3-point window
    hop = 2
    
    scipy_stft = ShortTimeFFT(win, hop=hop, fs=1.0)
    standalone_stft = StandaloneSTFT(win, hop=hop, fs=1.0)
    
    print(f"Signal: {x}")
    print(f"Window: {win}")
    print(f"Hop: {hop}")
    
    # Get slice ranges
    scipy_p0, scipy_p1 = scipy_stft.p_range(len(x))
    standalone_p0, standalone_p1 = standalone_stft.p_range(len(x))
    
    print(f"Scipy p_range: ({scipy_p0}, {scipy_p1})")
    print(f"Standalone p_range: ({standalone_p0}, {standalone_p1})")
    
    # Extract slices manually and compare
    print("\nManual slice extraction:")
    for p in range(scipy_p0, min(scipy_p0 + 3, scipy_p1)):  # First 3 slices
        # Calculate slice position
        k_center = p * hop
        k_start = k_center - scipy_stft.m_num_mid
        k_end = k_start + scipy_stft.m_num
        
        print(f"Slice {p}: center={k_center}, range=[{k_start}:{k_end}]")
        
        # Extract slice with padding
        if k_start >= 0 and k_end <= len(x):
            slice_data = x[k_start:k_end]
        else:
            slice_data = np.zeros(scipy_stft.m_num)
            valid_start = max(0, k_start)
            valid_end = min(len(x), k_end)
            if valid_end > valid_start:
                slice_start = valid_start - k_start
                slice_end = slice_start + (valid_end - valid_start)
                slice_data[slice_start:slice_end] = x[valid_start:valid_end]
        
        print(f"  Manual slice: {slice_data}")
        
        # Apply window and FFT
        windowed = slice_data * win.conj()
        fft_result = np.fft.rfft(windowed, n=len(win))
        print(f"  Windowed: {windowed}")
        print(f"  FFT: {fft_result}")


if __name__ == "__main__":
    test_single_windowed_fft()
    test_stft_single_slice()
    test_slice_extraction()

