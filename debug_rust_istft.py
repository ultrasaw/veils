#!/usr/bin/env python3
"""
Create a simple debug case for Rust ISTFT debugging.
"""

import numpy as np
import json
from standalone_stft import StandaloneSTFT

def create_simple_debug_case():
    """Create the simplest possible test case."""
    
    # Very simple signal - just a few non-zero values
    signal = np.zeros(64)
    signal[32] = 1.0  # Single impulse in the middle
    
    # Simple window
    window_length = 16
    n = np.arange(window_length)
    window = 0.5 * (1 - np.cos(2 * np.pi * n / (window_length - 1)))
    
    hop = 4
    fs = 64.0
    
    print("=== Simple Debug Case ===")
    print(f"Signal: {signal.nonzero()[0]} (impulse at index 32)")
    print(f"Window length: {window_length}")
    print(f"Hop: {hop}")
    print(f"FS: {fs}")
    
    # Create STFT
    stft = StandaloneSTFT(win=window, hop=hop, fs=fs, fft_mode='onesided')
    
    print(f"\nSTFT Properties:")
    print(f"  m_num: {stft.m_num}")
    print(f"  f_pts: {stft.f_pts}")
    print(f"  p_min: {stft.p_min}")
    print(f"  p_max: {stft.p_max(len(signal))}")
    
    # Forward STFT
    S = stft.stft(signal)
    print(f"\nSTFT shape: {S.shape}")
    
    # Check which time slices have non-zero energy
    energy_per_slice = np.sum(np.abs(S)**2, axis=0)
    nonzero_slices = np.where(energy_per_slice > 1e-10)[0]
    print(f"Non-zero time slices: {nonzero_slices}")
    
    # ISTFT
    reconstructed = stft.istft(S)
    print(f"Reconstructed length: {len(reconstructed)}")
    
    # Check reconstruction quality
    min_len = min(len(signal), len(reconstructed))
    error = np.mean(np.abs(signal[:min_len] - reconstructed[:min_len]))
    print(f"Reconstruction error: {error:.2e}")
    
    # Find where the reconstructed impulse is
    recon_peak_idx = np.argmax(np.abs(reconstructed))
    recon_peak_val = reconstructed[recon_peak_idx]
    print(f"Original impulse: index 32, value 1.0")
    print(f"Reconstructed peak: index {recon_peak_idx}, value {recon_peak_val:.6f}")
    
    # Save debug data
    debug_data = {
        'signal': signal.tolist(),
        'window': window.tolist(),
        'hop': hop,
        'fs': fs,
        'stft_real': S.real.tolist(),
        'stft_imag': S.imag.tolist(),
        'reconstructed': reconstructed.tolist(),
        'properties': {
            'm_num': stft.m_num,
            'f_pts': stft.f_pts,
            'p_min': stft.p_min,
            'p_max': stft.p_max(len(signal)),
            'mfft': stft.mfft
        },
        'analysis': {
            'original_impulse_idx': 32,
            'reconstructed_peak_idx': int(recon_peak_idx),
            'reconstructed_peak_val': float(recon_peak_val),
            'reconstruction_error': float(error),
            'nonzero_time_slices': nonzero_slices.tolist()
        }
    }
    
    with open('simple_debug_case.json', 'w') as f:
        json.dump(debug_data, f, indent=2)
    
    print(f"\nDebug data saved to simple_debug_case.json")
    
    if error < 1e-10:
        print("✅ Perfect reconstruction!")
    else:
        print("❌ Reconstruction has errors")
        
    return debug_data

if __name__ == "__main__":
    create_simple_debug_case()
