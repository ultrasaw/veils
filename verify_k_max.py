#!/usr/bin/env python3
"""
Verify k_max property implementation against scipy.
"""

import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from standalone_stft import StandaloneSTFT


def verify_k_max_property():
    """Verify k_max property matches scipy exactly."""
    print("="*50)
    print("VERIFYING k_max PROPERTY")
    print("="*50)
    
    # Test with different signal lengths and parameters
    test_cases = [
        (100, 20, 5),   # signal_len, window_len, hop
        (500, 50, 10),
        (1000, 30, 15),
        (200, 40, 8)
    ]
    
    for signal_len, window_len, hop in test_cases:
        print(f"\nTest: signal_len={signal_len}, window_len={window_len}, hop={hop}")
        
        # Create window and STFT objects
        win = gaussian(window_len, std=window_len/8, sym=True)
        scipy_stft = ShortTimeFFT(win, hop=hop, fs=100.0)
        standalone_stft = StandaloneSTFT(win, hop=hop, fs=100.0)
        
        # Test k_max
        scipy_k_max = scipy_stft.k_max(signal_len)
        standalone_k_max = standalone_stft.k_max(signal_len)
        
        match = "✓" if scipy_k_max == standalone_k_max else "✗"
        print(f"  k_max({signal_len}): scipy={scipy_k_max}, standalone={standalone_k_max} {match}")
        
        # Also test other properties for completeness
        scipy_k_min = scipy_stft.k_min
        standalone_k_min = standalone_stft.k_min
        k_min_match = "✓" if scipy_k_min == standalone_k_min else "✗"
        print(f"  k_min: scipy={scipy_k_min}, standalone={standalone_k_min} {k_min_match}")
        
        scipy_p_max = scipy_stft.p_max(signal_len)
        standalone_p_max = standalone_stft.p_max(signal_len)
        p_max_match = "✓" if scipy_p_max == standalone_p_max else "✗"
        print(f"  p_max({signal_len}): scipy={scipy_p_max}, standalone={standalone_p_max} {p_max_match}")
    
    print("\n" + "="*50)
    print("k_max PROPERTY VERIFICATION COMPLETE")
    print("="*50)


if __name__ == "__main__":
    verify_k_max_property()

