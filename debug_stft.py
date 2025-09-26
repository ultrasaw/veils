#!/usr/bin/env python3
"""
Debug script to understand STFT parameter differences.
"""

import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from standalone_stft import StandaloneSTFT


def debug_stft_parameters():
    """Debug STFT parameter calculations."""
    # Create test signal
    n_samples = 1000
    x = np.random.randn(n_samples)
    
    # Window parameters
    window_length = 50
    hop_length = 10
    fs = 100.0
    
    # Create window
    win = gaussian(window_length, std=window_length/8, sym=True)
    
    print("=== STFT Parameter Comparison ===")
    print(f"Signal length: {n_samples}")
    print(f"Window length: {window_length}")
    print(f"Hop length: {hop_length}")
    print(f"Sampling frequency: {fs} Hz")
    print()
    
    # Scipy version
    scipy_stft = ShortTimeFFT(win, hop=hop_length, fs=fs)
    print("Scipy ShortTimeFFT:")
    print(f"  m_num: {scipy_stft.m_num}")
    print(f"  m_num_mid: {scipy_stft.m_num_mid}")
    print(f"  hop: {scipy_stft.hop}")
    print(f"  p_min: {scipy_stft.p_min}")
    print(f"  p_max(n): {scipy_stft.p_max(n_samples)}")
    print(f"  p_num(n): {scipy_stft.p_num(n_samples)}")
    print(f"  k_min: {scipy_stft.k_min}")
    print(f"  f_pts: {scipy_stft.f_pts}")
    
    # Test STFT
    S_scipy = scipy_stft.stft(x)
    print(f"  STFT shape: {S_scipy.shape}")
    print()
    
    # Standalone version
    standalone_stft = StandaloneSTFT(win, hop=hop_length, fs=fs)
    print("Standalone STFT:")
    print(f"  m_num: {standalone_stft.m_num}")
    print(f"  m_num_mid: {standalone_stft.m_num_mid}")
    print(f"  hop: {standalone_stft.hop}")
    print(f"  p_min: {standalone_stft.p_min}")
    print(f"  p_max(n): {standalone_stft.p_max(n_samples)}")
    print(f"  p_num(n): {standalone_stft.p_num(n_samples)}")
    print(f"  k_min: {standalone_stft.k_min}")
    print(f"  f_pts: {standalone_stft.f_pts}")
    
    # Test STFT
    S_standalone = standalone_stft.stft(x)
    print(f"  STFT shape: {S_standalone.shape}")
    print()
    
    # Compare p_range
    print("p_range comparison:")
    scipy_p0, scipy_p1 = scipy_stft.p_range(n_samples)
    standalone_p0, standalone_p1 = standalone_stft.p_range(n_samples)
    print(f"  Scipy p_range: ({scipy_p0}, {scipy_p1})")
    print(f"  Standalone p_range: ({standalone_p0}, {standalone_p1})")
    print()
    
    # Let's manually check the calculation
    print("Manual calculations:")
    print(f"  (n - m_num + m_num_mid) // hop + 1 = ({n_samples} - {window_length} + {window_length//2}) // {hop_length} + 1")
    manual_p_max = (n_samples - window_length + window_length//2) // hop_length + 1
    print(f"  = {manual_p_max}")
    print(f"  p_num = p_max - p_min = {manual_p_max} - {standalone_stft.p_min} = {manual_p_max - standalone_stft.p_min}")


if __name__ == "__main__":
    debug_stft_parameters()

