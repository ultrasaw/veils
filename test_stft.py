#!/usr/bin/env python3
"""
Test script to compare standalone STFT implementation with scipy's version.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from standalone_stft import StandaloneSTFT


def create_random_walk_signal(n_samples: int = 1000, fs: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Create a random walk signal with some structure."""
    np.random.seed(42)  # For reproducibility
    
    t = np.arange(n_samples) / fs
    
    # Create a signal with multiple components
    # 1. Random walk component
    random_walk = np.cumsum(np.random.randn(n_samples)) * 0.1
    
    # 2. Sinusoidal components with time-varying frequency
    f1 = 5 + 2 * np.sin(2 * np.pi * 0.1 * t)  # Varying frequency around 5 Hz
    f2 = 15 + np.sin(2 * np.pi * 0.05 * t)    # Varying frequency around 15 Hz
    
    sine1 = np.sin(2 * np.pi * np.cumsum(f1) / fs)
    sine2 = 0.5 * np.sin(2 * np.pi * np.cumsum(f2) / fs)
    
    # 3. Some noise
    noise = 0.1 * np.random.randn(n_samples)
    
    # Combine all components
    signal = random_walk + sine1 + sine2 + noise
    
    return t, signal


def test_stft_implementations():
    """Test both STFT implementations and compare results."""
    print("Creating test signal...")
    t, x = create_random_walk_signal(n_samples=1000, fs=100.0)
    
    # Window parameters
    window_length = 50
    hop_length = 10
    fs = 100.0
    
    # Create Gaussian window
    win = gaussian(window_length, std=window_length/8, sym=True)
    
    print(f"Signal length: {len(x)}")
    print(f"Window length: {window_length}")
    print(f"Hop length: {hop_length}")
    print(f"Sampling frequency: {fs} Hz")
    
    # Test scipy implementation
    print("\nTesting scipy ShortTimeFFT...")
    scipy_stft = ShortTimeFFT(win, hop=hop_length, fs=fs, mfft=None)
    print(f"Scipy STFT invertible: {scipy_stft.invertible}")
    
    S_scipy = scipy_stft.stft(x)
    x_reconstructed_scipy = scipy_stft.istft(S_scipy, k1=len(x))
    
    print(f"Scipy STFT shape: {S_scipy.shape}")
    print(f"Scipy reconstruction error: {np.mean(np.abs(x - x_reconstructed_scipy)):.2e}")
    print(f"Scipy max reconstruction error: {np.max(np.abs(x - x_reconstructed_scipy)):.2e}")
    
    # Test standalone implementation
    print("\nTesting standalone STFT...")
    standalone_stft = StandaloneSTFT(win, hop=hop_length, fs=fs, mfft=None)
    print(f"Standalone STFT invertible: {standalone_stft.invertible}")
    
    S_standalone = standalone_stft.stft(x)
    x_reconstructed_standalone = standalone_stft.istft(S_standalone, k1=len(x))
    
    print(f"Standalone STFT shape: {S_standalone.shape}")
    print(f"Standalone reconstruction error: {np.mean(np.abs(x - x_reconstructed_standalone)):.2e}")
    print(f"Standalone max reconstruction error: {np.max(np.abs(x - x_reconstructed_standalone)):.2e}")
    
    # Compare the two implementations
    print("\nComparing implementations...")
    stft_diff = np.abs(S_scipy - S_standalone)
    print(f"STFT difference (mean): {np.mean(stft_diff):.2e}")
    print(f"STFT difference (max): {np.max(stft_diff):.2e}")
    
    recon_diff = np.abs(x_reconstructed_scipy - x_reconstructed_standalone)
    print(f"Reconstruction difference (mean): {np.mean(recon_diff):.2e}")
    print(f"Reconstruction difference (max): {np.max(recon_diff):.2e}")
    
    # Check if they're close enough
    stft_close = np.allclose(S_scipy, S_standalone, rtol=1e-10, atol=1e-12)
    recon_close = np.allclose(x_reconstructed_scipy, x_reconstructed_standalone, rtol=1e-10, atol=1e-12)
    
    print(f"\nSTFT results are close: {stft_close}")
    print(f"Reconstruction results are close: {recon_close}")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Original signal
    axes[0, 0].plot(t, x)
    axes[0, 0].set_title('Original Signal')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    
    # STFT magnitude (scipy)
    t_stft = scipy_stft.t(len(x))
    f_stft = scipy_stft.f
    im1 = axes[0, 1].imshow(np.abs(S_scipy), origin='lower', aspect='auto',
                           extent=[t_stft[0], t_stft[-1], f_stft[0], f_stft[-1]])
    axes[0, 1].set_title('Scipy STFT Magnitude')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # STFT magnitude (standalone)
    t_stft_standalone = standalone_stft.t(len(x))
    f_stft_standalone = standalone_stft.f()
    im2 = axes[1, 0].imshow(np.abs(S_standalone), origin='lower', aspect='auto',
                           extent=[t_stft_standalone[0], t_stft_standalone[-1], 
                                  f_stft_standalone[0], f_stft_standalone[-1]])
    axes[1, 0].set_title('Standalone STFT Magnitude')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Reconstruction comparison
    axes[1, 1].plot(t, x, label='Original', alpha=0.7)
    axes[1, 1].plot(t, x_reconstructed_scipy, label='Scipy Reconstruction', alpha=0.7)
    axes[1, 1].plot(t, x_reconstructed_standalone, label='Standalone Reconstruction', alpha=0.7)
    axes[1, 1].set_title('Signal Reconstruction Comparison')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('stft_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'stft_comparison.png'")
    
    return {
        'scipy_stft': S_scipy,
        'standalone_stft': S_standalone,
        'scipy_recon': x_reconstructed_scipy,
        'standalone_recon': x_reconstructed_standalone,
        'original': x,
        'stft_close': stft_close,
        'recon_close': recon_close
    }


if __name__ == "__main__":
    results = test_stft_implementations()
    
    if results['stft_close'] and results['recon_close']:
        print("\n✅ SUCCESS: Standalone STFT implementation matches scipy!")
    else:
        print("\n❌ FAILURE: Standalone STFT implementation differs from scipy")
        print("This might be due to implementation differences in edge cases or numerical precision.")

