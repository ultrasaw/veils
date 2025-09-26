#!/usr/bin/env python3
"""
Debug scipy's internal STFT processing step by step.
"""

import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
import scipy.fft as fft_lib


def debug_scipy_stft_step_by_step():
    """Debug scipy STFT step by step to understand the exact processing."""
    # Simple test case
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    win = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # Rectangular window
    
    print("=== Step-by-Step Scipy STFT Debug ===")
    print(f"Signal: {x}")
    print(f"Window: {win}")
    
    # Create scipy STFT
    scipy_stft = ShortTimeFFT(win, hop=1, fs=1.0)
    
    print(f"Basic properties:")
    print(f"  m_num: {scipy_stft.m_num}")
    print(f"  m_num_mid: {scipy_stft.m_num_mid}")
    print(f"  hop: {scipy_stft.hop}")
    print(f"  mfft: {scipy_stft.mfft}")
    print(f"  fft_mode: {scipy_stft.fft_mode}")
    print(f"  phase_shift: {scipy_stft.phase_shift}")
    
    # Get p_range
    p0, p1 = scipy_stft.p_range(len(x))
    print(f"  p_range: ({p0}, {p1})")
    
    # Manual slice extraction and processing
    print(f"\nManual slice processing:")
    for p in range(p0, min(p0 + 3, p1)):  # First 3 slices
        print(f"\nSlice {p}:")
        
        # Calculate slice position (replicating scipy's logic)
        k_center = p * scipy_stft.hop
        k_start = k_center - scipy_stft.m_num_mid
        k_end = k_start + scipy_stft.m_num
        
        print(f"  Position: center={k_center}, range=[{k_start}:{k_end}]")
        
        # Extract slice with padding (simplified)
        if k_start >= 0 and k_end <= len(x):
            slice_data = x[k_start:k_end].copy()
        else:
            slice_data = np.zeros(scipy_stft.m_num, dtype=x.dtype)
            valid_start = max(0, k_start)
            valid_end = min(len(x), k_end)
            if valid_end > valid_start:
                slice_start = valid_start - k_start
                slice_end = slice_start + (valid_end - valid_start)
                slice_data[slice_start:slice_end] = x[valid_start:valid_end]
        
        print(f"  Extracted slice: {slice_data}")
        
        # Apply window and conjugate (as done in STFT)
        windowed = slice_data * scipy_stft.win.conj()
        print(f"  Windowed: {windowed}")
        
        # Test different FFT approaches
        manual_fft = fft_lib.rfft(windowed, n=scipy_stft.mfft)
        scipy_internal_fft = scipy_stft._fft_func(windowed)
        numpy_fft = np.fft.rfft(windowed, n=scipy_stft.mfft)
        
        print(f"  Manual scipy.fft.rfft: {manual_fft}")
        print(f"  Scipy internal _fft_func: {scipy_internal_fft}")
        print(f"  NumPy rfft: {numpy_fft}")
        
        print(f"  Diff (manual vs internal): {np.abs(manual_fft - scipy_internal_fft)}")
        print(f"  Diff (numpy vs internal): {np.abs(numpy_fft - scipy_internal_fft)}")
    
    # Get full STFT for comparison
    print(f"\nFull STFT comparison:")
    S_scipy = scipy_stft.stft(x)
    print(f"Scipy STFT shape: {S_scipy.shape}")
    print(f"First slice from full STFT: {S_scipy[:, 0]}")


def test_axis_handling():
    """Test if axis handling affects the results."""
    print("\n=== Axis Handling Test ===")
    
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Test different axis specifications
    result_default = fft_lib.rfft(x)
    result_axis_neg1 = fft_lib.rfft(x, axis=-1)
    result_axis_0 = fft_lib.rfft(x, axis=0)
    
    print(f"Input: {x}")
    print(f"Default: {result_default}")
    print(f"axis=-1: {result_axis_neg1}")
    print(f"axis=0: {result_axis_0}")
    
    # Test with 2D array
    x_2d = x.reshape(1, -1)
    result_2d_default = fft_lib.rfft(x_2d)
    result_2d_axis_neg1 = fft_lib.rfft(x_2d, axis=-1)
    result_2d_axis_1 = fft_lib.rfft(x_2d, axis=1)
    
    print(f"\n2D Input: {x_2d}")
    print(f"2D Default: {result_2d_default}")
    print(f"2D axis=-1: {result_2d_axis_neg1}")
    print(f"2D axis=1: {result_2d_axis_1}")


if __name__ == "__main__":
    debug_scipy_stft_step_by_step()
    test_axis_handling()

