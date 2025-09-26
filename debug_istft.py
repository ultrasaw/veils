#!/usr/bin/env python3
"""
Debug script to understand the exact ISTFT behavior in Python
and generate reference data for Rust debugging.
"""

import numpy as np
import json
from standalone_stft import StandaloneSTFT

def debug_istft_step_by_step():
    """Debug ISTFT step by step to understand the exact logic."""
    
    # Simple test case - impulse signal
    signal = np.zeros(100)
    signal[50] = 1.0  # Single impulse
    
    # Create STFT
    # Create Hann window manually
    window_length = 32
    n = np.arange(window_length)
    window = 0.5 * (1 - np.cos(2 * np.pi * n / (window_length - 1)))
    hop = 8
    fs = 100.0
    
    stft = StandaloneSTFT(win=window, hop=hop, fs=fs, fft_mode='onesided')
    
    print("=== STFT Properties ===")
    print(f"m_num: {stft.m_num}")
    print(f"m_num_mid: {stft.m_num_mid}")
    print(f"f_pts: {stft.f_pts}")
    print(f"p_min: {stft.p_min}")
    print(f"p_max: {stft.p_max(len(signal))}")
    print(f"hop: {stft.hop}")
    
    # Forward STFT
    S = stft.stft(signal)
    print(f"\n=== STFT Result ===")
    print(f"S.shape: {S.shape}")
    print(f"S format: (f_pts={S.shape[0]}, time_slices={S.shape[1]})")
    
    # Manual ISTFT with debugging
    print(f"\n=== ISTFT Debug ===")
    
    # Parameters from Python ISTFT
    k0 = 0
    k1 = None
    
    n_min = stft.m_num - stft.m_num_mid
    q_num = stft.p_num(n_min)
    q_max = S.shape[1] + stft.p_min
    k_max = (q_max - 1) * stft.hop + stft.m_num - stft.m_num_mid
    
    if k1 is None:
        k1 = k_max
    
    print(f"n_min: {n_min}")
    print(f"q_num: {q_num}")
    print(f"q_max: {q_max}")
    print(f"k_max: {k_max}")
    print(f"k0: {k0}, k1: {k1}")
    
    q0 = (k0 // stft.hop + stft.p_min if k0 >= 0 else k0 // stft.hop)
    q1 = min(stft.p_max(k1), q_max)
    
    print(f"q0: {q0}, q1: {q1}")
    
    num_pts = k1 - k0
    x = np.zeros(num_pts, dtype=float if stft.onesided_fft else complex)
    
    print(f"num_pts: {num_pts}")
    print(f"Reconstruction array shape: {x.shape}")
    
    # Debug each time slice
    debug_data = {
        'signal': signal.tolist(),
        'window': window.tolist(),
        'stft_real': S.real.tolist(),
        'stft_imag': S.imag.tolist(),
        'parameters': {
            'm_num': stft.m_num,
            'm_num_mid': stft.m_num_mid,
            'f_pts': stft.f_pts,
            'p_min': stft.p_min,
            'p_max': stft.p_max(len(signal)),
            'hop': stft.hop,
            'k0': k0,
            'k1': k1,
            'q0': q0,
            'q1': q1,
            'num_pts': num_pts
        },
        'time_slices': []
    }
    
    for q_ in range(q0, q1):
        if q_ - stft.p_min >= S.shape[1]:
            break
        
        print(f"\n--- Time slice q_={q_} ---")
        
        # Get STFT slice (this is the key!)
        stft_slice = S[:, q_ - stft.p_min]  # Shape: (f_pts,)
        print(f"STFT slice shape: {stft_slice.shape}")
        print(f"STFT slice (first 5): {stft_slice[:5]}")
        
        # Apply IFFT
        xs_raw = stft._ifft_func(stft_slice)
        print(f"IFFT result shape: {xs_raw.shape}")
        print(f"IFFT result (first 5): {xs_raw[:5]}")
        
        # Apply dual window
        xs = xs_raw * stft.dual_win
        print(f"After dual window (first 5): {xs[:5]}")
        
        # Calculate indices
        i0 = q_ * stft.hop - stft.m_num_mid
        i1 = min(i0 + stft.m_num, num_pts + k0)
        j0, j1 = 0, i1 - i0
        
        if i0 < k0:
            j0 += k0 - i0
            i0 = k0
        
        if i1 > k0 + num_pts:
            j1 -= (i1 - k0 - num_pts)
            i1 = k0 + num_pts
        
        print(f"Indices: i0={i0}, i1={i1}, j0={j0}, j1={j1}")
        print(f"Target range in x: [{i0-k0}:{i1-k0}]")
        print(f"Source range in xs: [{j0}:{j1}]")
        
        # Store debug data
        slice_data = {
            'q': q_,
            'stft_slice_real': stft_slice.real.tolist(),
            'stft_slice_imag': stft_slice.imag.tolist(),
            'ifft_result_real': xs_raw.real.tolist(),
            'ifft_result_imag': xs_raw.imag.tolist(),
            'dual_windowed_real': xs.real.tolist(),
            'dual_windowed_imag': xs.imag.tolist(),
            'indices': {'i0': i0, 'i1': i1, 'j0': j0, 'j1': j1},
            'target_range': [i0-k0, i1-k0],
            'source_range': [j0, j1]
        }
        debug_data['time_slices'].append(slice_data)
        
        # Apply to reconstruction
        if i0 < i1 and j0 < j1:
            if stft.onesided_fft:
                x[i0-k0:i1-k0] += xs[j0:j1].real
            else:
                x[i0-k0:i1-k0] += xs[j0:j1]
            
            print(f"Added to reconstruction: {xs[j0:j1].real if stft.onesided_fft else xs[j0:j1]}")
    
    # Compare with official ISTFT
    official_reconstruction = stft.istft(S)
    
    print(f"\n=== Final Comparison ===")
    print(f"Manual reconstruction length: {len(x)}")
    print(f"Official reconstruction length: {len(official_reconstruction)}")
    
    # Align lengths for comparison
    min_len = min(len(x), len(official_reconstruction))
    manual_aligned = x[:min_len]
    official_aligned = official_reconstruction[:min_len]
    
    diff = np.abs(manual_aligned - official_aligned)
    print(f"Max difference: {np.max(diff)}")
    print(f"Mean difference: {np.mean(diff)}")
    
    if np.max(diff) < 1e-10:
        print("✅ Manual ISTFT matches official implementation!")
    else:
        print("❌ Manual ISTFT differs from official implementation")
        print(f"Manual (first 10): {manual_aligned[:10]}")
        print(f"Official (first 10): {official_aligned[:10]}")
    
    # Save debug data
    debug_data['manual_reconstruction'] = x.tolist()
    debug_data['official_reconstruction'] = official_reconstruction.tolist()
    debug_data['dual_window'] = stft.dual_win.tolist()
    
    with open('istft_debug_data.json', 'w') as f:
        json.dump(debug_data, f, indent=2)
    
    print(f"\nDebug data saved to istft_debug_data.json")
    
    return debug_data

if __name__ == "__main__":
    debug_istft_step_by_step()
