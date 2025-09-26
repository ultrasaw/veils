#!/usr/bin/env python3
"""
Detailed debug script to find exact differences in STFT implementations.
"""

import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from standalone_stft import StandaloneSTFT


def debug_single_slice():
    """Debug a single STFT slice to understand the differences."""
    # Simple test signal
    n_samples = 100
    x = np.sin(2 * np.pi * 5 * np.arange(n_samples) / 100.0)  # 5 Hz sine wave
    
    # Window parameters
    window_length = 20
    hop_length = 5
    fs = 100.0
    
    # Create window
    win = gaussian(window_length, std=window_length/8, sym=True)
    
    print("=== Single Slice Debug ===")
    print(f"Signal: 5 Hz sine wave, {n_samples} samples")
    print(f"Window: Gaussian, {window_length} samples")
    print(f"Hop: {hop_length} samples")
    print(f"Sampling rate: {fs} Hz")
    print()
    
    # Create both implementations
    scipy_stft = ShortTimeFFT(win, hop=hop_length, fs=fs)
    standalone_stft = StandaloneSTFT(win, hop=hop_length, fs=fs)
    
    # Compare basic properties
    print("Basic properties comparison:")
    props = ['m_num', 'm_num_mid', 'hop', 'fs', 'mfft', 'f_pts', 'p_min']
    for prop in props:
        scipy_val = getattr(scipy_stft, prop)
        standalone_val = getattr(standalone_stft, prop)
        match = "✓" if scipy_val == standalone_val else "✗"
        print(f"  {prop}: scipy={scipy_val}, standalone={standalone_val} {match}")
    
    print(f"  p_max: scipy={scipy_stft.p_max(n_samples)}, standalone={standalone_stft.p_max(n_samples)}")
    print()
    
    # Compare windows
    print("Window comparison:")
    win_diff = np.max(np.abs(scipy_stft.win - standalone_stft.win))
    print(f"  Window max difference: {win_diff}")
    
    dual_win_diff = np.max(np.abs(scipy_stft.dual_win - standalone_stft.dual_win))
    print(f"  Dual window max difference: {dual_win_diff}")
    print()
    
    # Get STFT
    S_scipy = scipy_stft.stft(x)
    S_standalone = standalone_stft.stft(x)
    
    print(f"STFT shapes: scipy={S_scipy.shape}, standalone={S_standalone.shape}")
    
    # Compare first few slices in detail
    print("\nFirst 3 slices comparison:")
    for i in range(min(3, S_scipy.shape[1])):
        slice_diff = np.abs(S_scipy[:, i] - S_standalone[:, i])
        print(f"  Slice {i}: max diff = {np.max(slice_diff):.2e}, mean diff = {np.mean(slice_diff):.2e}")
        
        if np.max(slice_diff) > 1e-10:
            print(f"    Scipy slice {i} (first 5): {S_scipy[:5, i]}")
            print(f"    Standalone slice {i} (first 5): {S_standalone[:5, i]}")
            print(f"    Difference (first 5): {slice_diff[:5]}")
    
    return scipy_stft, standalone_stft, S_scipy, S_standalone, x


def debug_fft_implementation():
    """Debug the FFT implementation specifically."""
    print("\n=== FFT Implementation Debug ===")
    
    # Simple test data
    x = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    win = np.ones(len(x))  # Rectangular window for simplicity
    
    # Create standalone STFT
    standalone_stft = StandaloneSTFT(win, hop=1, fs=1.0)
    
    # Test windowed signal
    x_windowed = x * win.conj()
    
    # Compare FFT implementations
    scipy_fft = np.fft.fft(x_windowed)
    standalone_fft = standalone_stft._fft_func(x_windowed)
    
    print(f"Input signal: {x}")
    print(f"Windowed signal: {x_windowed}")
    print(f"NumPy FFT: {scipy_fft}")
    print(f"Standalone FFT: {standalone_fft}")
    print(f"FFT difference: {np.abs(scipy_fft[:len(standalone_fft)] - standalone_fft)}")
    
    # Test with different fft_mode
    for mode in ['onesided', 'twosided', 'centered']:
        standalone_stft._fft_mode = mode
        result = standalone_stft._fft_func(x_windowed)
        print(f"Mode {mode}: shape={result.shape}, first 3 values={result[:3]}")


def debug_slice_extraction():
    """Debug the slice extraction process."""
    print("\n=== Slice Extraction Debug ===")
    
    # Simple signal
    x = np.arange(20, dtype=float)
    win = np.ones(5)
    hop = 2
    
    scipy_stft = ShortTimeFFT(win, hop=hop, fs=1.0)
    standalone_stft = StandaloneSTFT(win, hop=hop, fs=1.0)
    
    print(f"Signal: {x}")
    print(f"Window length: {len(win)}")
    print(f"Hop: {hop}")
    
    # Get p_range for both
    scipy_p0, scipy_p1 = scipy_stft.p_range(len(x))
    standalone_p0, standalone_p1 = standalone_stft.p_range(len(x))
    
    print(f"Scipy p_range: ({scipy_p0}, {scipy_p1})")
    print(f"Standalone p_range: ({standalone_p0}, {standalone_p1})")
    
    # Compare slice extraction
    print("\nSlice extraction comparison:")
    scipy_slices = []
    standalone_slices = []
    
    # Get slices from scipy (we need to replicate the internal logic)
    for p in range(scipy_p0, min(scipy_p0 + 5, scipy_p1)):  # First 5 slices
        k_center = p * hop
        k_start = k_center - scipy_stft.m_num_mid
        k_end = k_start + scipy_stft.m_num
        
        if k_start >= 0 and k_end <= len(x):
            scipy_slice = x[k_start:k_end]
        else:
            # This is where padding happens - scipy has complex logic
            scipy_slice = np.zeros(scipy_stft.m_num)
            valid_start = max(0, k_start)
            valid_end = min(len(x), k_end)
            if valid_end > valid_start:
                slice_start = valid_start - k_start
                slice_end = slice_start + (valid_end - valid_start)
                scipy_slice[slice_start:slice_end] = x[valid_start:valid_end]
        
        scipy_slices.append(scipy_slice)
        print(f"  Scipy slice {p}: center={k_center}, range=[{k_start}:{k_end}], data={scipy_slice}")
    
    # Get slices from standalone
    for i, slice_data in enumerate(standalone_stft._x_slices(x, 0, standalone_p0, min(standalone_p0 + 5, standalone_p1))):
        standalone_slices.append(slice_data)
        print(f"  Standalone slice {standalone_p0 + i}: data={slice_data}")
        
        if i < len(scipy_slices):
            diff = np.abs(scipy_slices[i] - slice_data)
            print(f"    Difference: {diff} (max: {np.max(diff)})")


if __name__ == "__main__":
    debug_fft_implementation()
    debug_slice_extraction()
    scipy_stft, standalone_stft, S_scipy, S_standalone, x = debug_single_slice()

