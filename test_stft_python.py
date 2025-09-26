#!/usr/bin/env python3
"""
Test script for the standalone STFT implementation using random walk data.
This will generate test data and save results for comparison with Rust implementation.
"""

import numpy as np
import json
from standalone_stft import StandaloneSTFT

def generate_random_walk(n_samples: int, seed: int = 42) -> np.ndarray:
    """Generate a random walk signal for testing."""
    np.random.seed(seed)
    steps = np.random.randn(n_samples)
    return np.cumsum(steps)

def create_hann_window(length: int) -> np.ndarray:
    """Create a Hann window."""
    n = np.arange(length)
    return 0.5 * (1 - np.cos(2 * np.pi * n / (length - 1)))

def main():
    # Test parameters
    n_samples = 1000
    window_length = 256
    hop_length = 64
    fs = 1000.0
    
    print("Generating test data...")
    
    # Generate random walk signal
    signal = generate_random_walk(n_samples)
    
    # Create Hann window
    window = create_hann_window(window_length)
    
    print(f"Signal length: {len(signal)}")
    print(f"Window length: {len(window)}")
    print(f"Hop length: {hop_length}")
    print(f"Sampling rate: {fs}")
    
    # Create STFT object
    stft = StandaloneSTFT(
        win=window,
        hop=hop_length,
        fs=fs,
        fft_mode='onesided'
    )
    
    print(f"STFT properties:")
    print(f"  m_num: {stft.m_num}")
    print(f"  m_num_mid: {stft.m_num_mid}")
    print(f"  f_pts: {stft.f_pts}")
    print(f"  p_min: {stft.p_min}")
    print(f"  p_max: {stft.p_max(n_samples)}")
    print(f"  invertible: {stft.invertible}")
    
    # Perform STFT
    print("\nPerforming STFT...")
    S = stft.stft(signal)
    print(f"STFT shape: {S.shape}")
    
    # Perform ISTFT
    print("Performing ISTFT...")
    reconstructed = stft.istft(S)
    print(f"Reconstructed signal length: {len(reconstructed)}")
    
    # Calculate reconstruction error
    # We need to align the signals properly for comparison
    min_len = min(len(signal), len(reconstructed))
    error = np.mean(np.abs(signal[:min_len] - reconstructed[:min_len]))
    relative_error = error / np.mean(np.abs(signal[:min_len]))
    
    print(f"\nReconstruction error:")
    print(f"  Absolute error: {error:.2e}")
    print(f"  Relative error: {relative_error:.2e}")
    
    # Save test data for Rust comparison
    test_data = {
        'signal': signal.tolist(),
        'window': window.tolist(),
        'hop_length': hop_length,
        'fs': fs,
        'stft_real': S.real.tolist(),
        'stft_imag': S.imag.tolist(),
        'reconstructed': reconstructed.tolist(),
        'stft_properties': {
            'm_num': stft.m_num,
            'm_num_mid': stft.m_num_mid,
            'f_pts': stft.f_pts,
            'p_min': stft.p_min,
            'p_max': stft.p_max(n_samples),
            'mfft': stft.mfft
        }
    }
    
    with open('test_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\nTest data saved to test_data.json")
    print(f"Python STFT test completed successfully!")

if __name__ == "__main__":
    main()
