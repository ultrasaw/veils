#!/usr/bin/env python3
"""
Final Python-Rust STFT Comparison Script

This script performs comprehensive 1:1 comparison between the Python standalone_stft.py
implementation and the Rust lib.rs implementation. It's designed to run in a clean
Docker environment to avoid dependency issues.
"""

import json
import subprocess
import tempfile
import os
from pathlib import Path
import sys

# Import our standalone STFT implementation
from standalone_stft import StandaloneSTFT
import numpy as np

def run_rust_stft(signal, window, hop_length, fs=1.0, fft_mode='onesided'):
    """Run Rust STFT implementation and return results."""
    
    # Create temporary input file
    test_data = {
        'signal': signal.tolist() if hasattr(signal, 'tolist') else list(signal),
        'window': window.tolist() if hasattr(window, 'tolist') else list(window),
        'hop_length': hop_length,
        'fs': fs,
        'fft_mode': fft_mode
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        input_file = f.name
    
    try:
        # Run Rust binary
        result = subprocess.run([
            'cargo', 'run', '--bin', 'python_rust_comparison_helper', '--', input_file
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode != 0:
            raise RuntimeError(f"Rust execution failed: {result.stderr}")
        
        # Parse output
        output_data = json.loads(result.stdout)
        return output_data
        
    finally:
        os.unlink(input_file)

def compare_complex_matrices(py_matrix, rust_matrix, tolerance=1e-12):
    """Compare complex matrices between Python and Rust."""
    
    if len(py_matrix) != len(rust_matrix):
        return False, f"Frequency dimension mismatch: {len(py_matrix)} vs {len(rust_matrix)}"
    
    max_diff = 0.0
    total_diff_sq = 0.0
    count = 0
    
    for f in range(len(py_matrix)):
        if len(py_matrix[f]) != len(rust_matrix[f]):
            return False, f"Time dimension mismatch at freq {f}: {len(py_matrix[f])} vs {len(rust_matrix[f])}"
        
        for t in range(len(py_matrix[f])):
            py_val = py_matrix[f][t]
            rust_val = complex(rust_matrix[f][t]['real'], rust_matrix[f][t]['imag'])
            
            diff = abs(py_val - rust_val)
            max_diff = max(max_diff, diff)
            total_diff_sq += diff * diff
            count += 1
    
    mse = total_diff_sq / count if count > 0 else 0.0
    passed = max_diff < tolerance
    
    return passed, {
        'max_difference': float(max_diff),
        'mse': float(mse),
        'rmse': float(mse ** 0.5),
        'total_comparisons': int(count)
    }

def compare_real_arrays(py_array, rust_array, tolerance=1e-12):
    """Compare real arrays between Python and Rust."""
    
    min_len = min(len(py_array), len(rust_array))
    
    max_diff = 0.0
    total_diff_sq = 0.0
    
    for i in range(min_len):
        diff = abs(py_array[i] - rust_array[i])
        max_diff = max(max_diff, diff)
        total_diff_sq += diff * diff
    
    mse = total_diff_sq / min_len if min_len > 0 else 0.0
    passed = max_diff < tolerance
    
    return passed, {
        'max_difference': float(max_diff),
        'mse': float(mse),
        'rmse': float(mse ** 0.5),
        'compared_length': int(min_len)
    }

def test_signal_configuration(signal_name, signal, window_name, window, 
                            hop_length, fft_mode='onesided', fs=1.0):
    """Test a specific signal-window configuration."""
    
    print(f"\n--- {signal_name} + {window_name} (hop={hop_length}, mode={fft_mode}) ---")
    
    try:
        # Python implementation
        python_stft = StandaloneSTFT(
            win=window, 
            hop=hop_length, 
            fs=fs, 
            fft_mode=fft_mode
        )
        
        py_stft_result = python_stft.stft(signal)
        py_reconstruction = python_stft.istft(py_stft_result)
        
        print(f"  Python STFT shape: {len(py_stft_result)} x {len(py_stft_result[0])}")
        print(f"  Python reconstruction length: {len(py_reconstruction)}")
        
        # Rust implementation
        rust_result = run_rust_stft(signal, window, hop_length, fs, fft_mode)
        rust_stft_result = rust_result['stft']
        rust_reconstruction = np.array(rust_result['istft'])
        
        print(f"  Rust STFT shape: {len(rust_stft_result)} x {len(rust_stft_result[0])}")
        print(f"  Rust reconstruction length: {len(rust_reconstruction)}")
        
        # Compare STFT coefficients
        stft_match, stft_details = compare_complex_matrices(py_stft_result, rust_stft_result)
        
        # Compare reconstructions
        recon_match, recon_details = compare_real_arrays(py_reconstruction, rust_reconstruction)
        
        # Calculate reconstruction errors
        min_len = min(len(signal), len(py_reconstruction))
        py_recon_error = np.mean(np.abs(signal[:min_len] - py_reconstruction[:min_len]))
        
        min_len = min(len(signal), len(rust_reconstruction))
        rust_recon_error = np.mean(np.abs(signal[:min_len] - rust_reconstruction[:min_len]))
        
        # Assessment
        perfect_py = py_recon_error < 1e-10
        perfect_rust = rust_recon_error < 1e-10
        implementations_match = stft_match and recon_match
        
        overall_success = perfect_py and perfect_rust and implementations_match
        
        # Results
        status = "✅ PASS" if overall_success else "❌ FAIL"
        print(f"  {status}")
        print(f"    STFT coefficients match: {'✅' if stft_match else '❌'} (max diff: {stft_details['max_difference']:.2e})")
        print(f"    Python reconstruction: {'✅' if perfect_py else '❌'} (error: {py_recon_error:.2e})")
        print(f"    Rust reconstruction: {'✅' if perfect_rust else '❌'} (error: {rust_recon_error:.2e})")
        print(f"    Cross-implementation: {'✅' if recon_match else '❌'} (max diff: {recon_details['max_difference']:.2e})")
        
        return {
            'signal_name': signal_name,
            'window_name': window_name,
            'hop_length': int(hop_length),
            'fft_mode': fft_mode,
            'overall_success': bool(overall_success),
            'stft_match': bool(stft_match),
            'perfect_py': bool(perfect_py),
            'perfect_rust': bool(perfect_rust),
            'implementations_match': bool(implementations_match),
            'stft_details': stft_details,
            'recon_details': recon_details,
            'py_recon_error': float(py_recon_error),
            'rust_recon_error': float(rust_recon_error)
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {str(e)}")
        return {
            'signal_name': signal_name,
            'window_name': window_name,
            'hop_length': int(hop_length),
            'fft_mode': fft_mode,
            'overall_success': False,
            'error': str(e)
        }

def create_test_data():
    """Create comprehensive test data."""
    
    np.random.seed(42)  # For reproducible results
    
    signals = {}
    windows = {}
    
    # Test signals
    signals['reference'] = np.array([
        0.49671415, -0.1382643, 0.64768854, 1.52302986, -0.23415337,
        -0.23413696, 1.57921282, 0.76743473, -0.46947439, 0.54256004,
        -0.46341769, -0.46572975, 0.24196227, -1.91328024, -1.72491783,
        -0.56228753, -1.01283112, 0.31424733, -0.90802408, -1.4123037,
        1.46564877, -0.2257763, 0.0675282, -1.42474819, -0.54438272,
        0.11092259, -1.15099358, 0.37569802, -0.60063869, -0.29169375
    ])
    
    t = np.linspace(0, 2*np.pi, 64)
    signals['sine_5hz'] = np.sin(5 * t)
    signals['chirp'] = np.sin(2 * np.pi * t * t / 4)
    
    impulse = np.zeros(64)
    impulse[32] = 1.0
    signals['impulse'] = impulse
    
    signals['noise'] = np.random.randn(64) * 0.5
    signals['multi_freq'] = np.sin(2 * t) + 0.5 * np.sin(7 * t) + 0.3 * np.sin(13 * t)
    
    # Test windows
    for length in [8, 15, 16, 32]:
        windows[f'hann_{length}'] = np.array([
            0.5 * (1 - np.cos(2 * np.pi * i / length)) 
            for i in range(length)
        ])
    
    windows['rect_16'] = np.ones(16)
    windows['hamming_16'] = np.array([
        0.54 - 0.46 * np.cos(2 * np.pi * i / 16) 
        for i in range(16)
    ])
    
    return signals, windows

def main():
    """Run comprehensive Python-Rust STFT comparison."""
    
    print("🔬 Final Python-Rust STFT Implementation Comparison")
    print("=" * 65)
    
    # Create test data
    print("\n📊 Generating test data...")
    signals, windows = create_test_data()
    
    print(f"Created {len(signals)} test signals and {len(windows)} test windows")
    
    # Test configurations
    test_configs = [
        # (signal_name, window_name, hop_length, fft_mode)
        ('reference', 'hann_15', 8, 'onesided'),
        ('sine_5hz', 'hann_16', 4, 'onesided'),
        ('sine_5hz', 'hann_16', 4, 'twosided'),
        ('sine_5hz', 'hann_16', 4, 'centered'),
        ('chirp', 'hann_32', 8, 'onesided'),
        ('impulse', 'hann_16', 4, 'onesided'),
        ('noise', 'hamming_16', 8, 'onesided'),
        ('multi_freq', 'hann_16', 4, 'onesided'),
    ]
    
    # Run tests
    print(f"\n🧪 Running {len(test_configs)} test configurations...")
    results = []
    
    for signal_name, window_name, hop_length, fft_mode in test_configs:
        if signal_name in signals and window_name in windows:
            signal = signals[signal_name]
            window = windows[window_name]
            
            # Skip if signal is too short for window
            if len(signal) < len(window):
                print(f"\n--- Skipping {signal_name} + {window_name}: signal too short ---")
                continue
                
            result = test_signal_configuration(
                signal_name, signal, window_name, window, hop_length, fft_mode
            )
            results.append(result)
    
    # Summary
    print(f"\n\n📋 COMPREHENSIVE SUMMARY")
    print("=" * 35)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.get('overall_success', False))
    failed_tests = total_tests - passed_tests
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests} ✅")
    print(f"Failed: {failed_tests} ❌")
    print(f"Success rate: {100 * passed_tests / total_tests:.1f}%")
    
    # Detailed analysis
    stft_matches = sum(1 for r in results if r.get('stft_match', False))
    perfect_py = sum(1 for r in results if r.get('perfect_py', False))
    perfect_rust = sum(1 for r in results if r.get('perfect_rust', False))
    impl_matches = sum(1 for r in results if r.get('implementations_match', False))
    
    print(f"\nDetailed Analysis:")
    print(f"  STFT coefficients match: {stft_matches}/{total_tests} ✅")
    print(f"  Python perfect reconstruction: {perfect_py}/{total_tests} ✅")
    print(f"  Rust perfect reconstruction: {perfect_rust}/{total_tests} ✅")
    print(f"  Implementation consistency: {impl_matches}/{total_tests} ✅")
    
    if failed_tests > 0:
        print(f"\n❌ Failed tests:")
        for result in results:
            if not result.get('overall_success', False):
                print(f"  - {result['signal_name']} + {result['window_name']} "
                      f"(hop={result['hop_length']}, mode={result['fft_mode']})")
                if 'error' in result:
                    print(f"    Error: {result['error']}")
    
    # Save detailed results
    output_file = 'final_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_tests': int(total_tests),
                'passed_tests': int(passed_tests),
                'failed_tests': int(failed_tests),
                'success_rate': float(passed_tests / total_tests if total_tests > 0 else 0),
                'stft_matches': int(stft_matches),
                'perfect_py': int(perfect_py),
                'perfect_rust': int(perfect_rust),
                'impl_matches': int(impl_matches)
            },
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {output_file}")
    
    # Final verdict
    if failed_tests == 0:
        print(f"\n🎉 PERFECT SUCCESS!")
        print(f"   Python and Rust STFT implementations are 1:1 compatible.")
        print(f"   All tests passed with perfect reconstruction and coefficient matching.")
        return 0
    else:
        print(f"\n⚠️  PARTIAL SUCCESS")
        print(f"   {passed_tests}/{total_tests} tests passed.")
        print(f"   Check detailed results for more information.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
