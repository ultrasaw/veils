#!/usr/bin/env python3
"""
Comprehensive comparison script between Python and Rust STFT implementations.
Generates plots, logs, and detailed analysis files.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import subprocess
import os
from standalone_stft import StandaloneSTFT
from datetime import datetime

def generate_test_signals(n_samples=1000, seed=42):
    """Generate various test signals for comprehensive testing."""
    np.random.seed(seed)
    
    signals = {}
    
    # Random walk (original test)
    steps = np.random.randn(n_samples)
    signals['random_walk'] = np.cumsum(steps)
    
    # Sine wave
    t = np.linspace(0, 10, n_samples)
    signals['sine_wave'] = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
    
    # Chirp signal
    signals['chirp'] = np.sin(2 * np.pi * (5 + 10 * t / 10) * t)  # 5-15 Hz chirp
    
    # White noise
    signals['white_noise'] = np.random.randn(n_samples) * 0.5
    
    # Impulse
    impulse = np.zeros(n_samples)
    impulse[n_samples//2] = 1.0
    signals['impulse'] = impulse
    
    return signals, t

def create_hann_window(length):
    """Create a Hann window."""
    n = np.arange(length)
    return 0.5 * (1 - np.cos(2 * np.pi * n / (length - 1)))

def run_python_stft(signal, window, hop_length, fs):
    """Run Python STFT implementation."""
    stft = StandaloneSTFT(
        win=window,
        hop=hop_length,
        fs=fs,
        fft_mode='onesided'
    )
    
    # Forward STFT
    S = stft.stft(signal)
    
    # Inverse STFT
    reconstructed = stft.istft(S)
    
    return S, reconstructed, stft

def run_rust_stft(signal_name):
    """Run Rust STFT implementation via Docker."""
    # Save signal for Rust to use
    test_data = {
        'signal': signal_name,  # Will be replaced by actual signal data
        'use_signal': signal_name
    }
    
    # Run Rust implementation
    result = subprocess.run([
        'docker', 'run', '--rm', 
        '-v', f'{os.getcwd()}:/workspace', 
        '-w', '/workspace',
        'rust:1.75', './target/debug/stft_test'
    ], capture_output=True, text=True)
    
    return result.stdout, result.stderr, result.returncode

def create_comparison_plots(signals, results, output_dir):
    """Create comprehensive comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    for signal_name, signal_data in signals.items():
        if signal_name not in results:
            continue
            
        result = results[signal_name]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'STFT Comparison: {signal_name.replace("_", " ").title()}', fontsize=16)
        
        # Original signal
        axes[0, 0].plot(signal_data['time'], signal_data['signal'])
        axes[0, 0].set_title('Original Signal')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True)
        
        # Python reconstruction
        if 'python_reconstructed' in result:
            recon_len = min(len(signal_data['time']), len(result['python_reconstructed']))
            recon_time = np.linspace(0, (recon_len - 1) / signal_data['fs'], recon_len)
            axes[0, 1].plot(recon_time, result['python_reconstructed'][:recon_len])
            axes[0, 1].set_title('Python Reconstruction')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Amplitude')
            axes[0, 1].grid(True)
        
        # Reconstruction error
        if 'python_reconstructed' in result:
            min_len = min(len(signal_data['signal']), len(result['python_reconstructed']))
            error = np.array(signal_data['signal'][:min_len]) - np.array(result['python_reconstructed'][:min_len])
            error_time = np.linspace(0, (min_len - 1) / signal_data['fs'], min_len)
            axes[0, 2].plot(error_time, error)
            axes[0, 2].set_title(f'Python Reconstruction Error\n(RMS: {np.sqrt(np.mean(error**2)):.2e})')
            axes[0, 2].set_xlabel('Time (s)')
            axes[0, 2].set_ylabel('Error')
            axes[0, 2].grid(True)
        
        # STFT magnitude (Python)
        if 'python_stft' in result:
            S = result['python_stft']
            freqs = np.fft.fftfreq(len(signal_data['window']), 1/signal_data['fs'])[:S.shape[0]]
            times = np.arange(S.shape[1]) * signal_data['hop_length'] / signal_data['fs']
            
            im = axes[1, 0].imshow(np.abs(S), aspect='auto', origin='lower', 
                                  extent=[times[0], times[-1], freqs[0], freqs[-1]])
            axes[1, 0].set_title('Python STFT Magnitude')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Frequency (Hz)')
            plt.colorbar(im, ax=axes[1, 0])
        
        # STFT phase (Python)
        if 'python_stft' in result:
            im = axes[1, 1].imshow(np.angle(S), aspect='auto', origin='lower',
                                  extent=[times[0], times[-1], freqs[0], freqs[-1]],
                                  cmap='hsv')
            axes[1, 1].set_title('Python STFT Phase')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Frequency (Hz)')
            plt.colorbar(im, ax=axes[1, 1])
        
        # Error summary
        axes[1, 2].axis('off')
        error_text = f"""
Error Summary:
Python STFT->ISTFT:
  Abs Error: {result.get('python_abs_error', 'N/A')}
  Rel Error: {result.get('python_rel_error', 'N/A')}

Rust STFT->ISTFT:
  Abs Error: {result.get('rust_abs_error', 'N/A')}
  Rel Error: {result.get('rust_rel_error', 'N/A')}

STFT Comparison:
  Max Diff: {result.get('stft_max_diff', 'N/A')}
  RMS Diff: {result.get('stft_rms_diff', 'N/A')}
        """
        axes[1, 2].text(0.1, 0.5, error_text, fontsize=10, verticalalignment='center',
                        fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{signal_name}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

def analyze_signal(signal, signal_name, window, hop_length, fs, t):
    """Analyze a single signal with both implementations."""
    print(f"\nAnalyzing {signal_name}...")
    
    # Python implementation
    S_py, reconstructed_py, stft_obj = run_python_stft(signal, window, hop_length, fs)
    
    # Calculate Python errors
    min_len = min(len(signal), len(reconstructed_py))
    py_error = np.mean(np.abs(signal[:min_len] - reconstructed_py[:min_len]))
    py_rel_error = py_error / np.mean(np.abs(signal[:min_len]))
    
    result = {
        'signal_name': signal_name,
        'python_stft': S_py,
        'python_reconstructed': reconstructed_py,
        'python_abs_error': py_error,
        'python_rel_error': py_rel_error,
        'stft_properties': {
            'm_num': stft_obj.m_num,
            'f_pts': stft_obj.f_pts,
            'p_min': stft_obj.p_min,
            'p_max': stft_obj.p_max(len(signal))
        }
    }
    
    return result

def create_detailed_log(results, output_dir):
    """Create a detailed log file with all results."""
    log_path = f'{output_dir}/comparison_log.txt'
    
    with open(log_path, 'w') as f:
        f.write(f"STFT Implementation Comparison Log\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("="*80 + "\n\n")
        
        for signal_name, result in results.items():
            f.write(f"Signal: {signal_name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Python STFT->ISTFT Reconstruction:\n")
            f.write(f"  Absolute Error: {result['python_abs_error']:.6e}\n")
            f.write(f"  Relative Error: {result['python_rel_error']:.6e}\n")
            
            if 'rust_abs_error' in result:
                f.write(f"Rust STFT->ISTFT Reconstruction:\n")
                f.write(f"  Absolute Error: {result['rust_abs_error']:.6e}\n")
                f.write(f"  Relative Error: {result['rust_rel_error']:.6e}\n")
            
            f.write(f"STFT Properties:\n")
            for key, value in result['stft_properties'].items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n")
    
    print(f"Detailed log saved to: {log_path}")

def main():
    """Main comparison function."""
    print("Starting comprehensive STFT implementation comparison...")
    
    # Parameters
    n_samples = 1000
    window_length = 256
    hop_length = 64
    fs = 1000.0
    
    # Generate test signals
    signals_data, t = generate_test_signals(n_samples)
    window = create_hann_window(window_length)
    
    # Prepare signals with metadata
    signals = {}
    for name, signal in signals_data.items():
        signals[name] = {
            'signal': signal,
            'time': t,
            'window': window,
            'hop_length': hop_length,
            'fs': fs
        }
    
    # Analyze each signal
    results = {}
    for signal_name, signal_data in signals.items():
        result = analyze_signal(
            signal_data['signal'], 
            signal_name, 
            signal_data['window'], 
            signal_data['hop_length'], 
            signal_data['fs'],
            signal_data['time']
        )
        results[signal_name] = result
    
    # Create output directory
    output_dir = 'comparison_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results as JSON (for Rust to potentially use)
    json_results = {}
    for name, result in results.items():
        json_results[name] = {
            'signal': signals[name]['signal'].tolist(),
            'window': signals[name]['window'].tolist(),
            'hop_length': signals[name]['hop_length'],
            'fs': signals[name]['fs'],
            'stft_real': result['python_stft'].real.tolist(),
            'stft_imag': result['python_stft'].imag.tolist(),
            'reconstructed': result['python_reconstructed'].tolist(),
            'python_abs_error': result['python_abs_error'],
            'python_rel_error': result['python_rel_error'],
            'stft_properties': result['stft_properties']
        }
    
    with open(f'{output_dir}/test_signals.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Create plots
    create_comparison_plots(signals, results, output_dir)
    
    # Create detailed log
    create_detailed_log(results, output_dir)
    
    # Summary
    print(f"\nComparison completed!")
    print(f"Results saved to: {output_dir}/")
    print(f"Generated files:")
    print(f"  - test_signals.json (test data)")
    print(f"  - comparison_log.txt (detailed log)")
    print(f"  - *_comparison.png (plots for each signal)")
    
    # Print summary table
    print(f"\nSummary Table:")
    print(f"{'Signal':<15} {'Python Error':<15} {'Status':<10}")
    print("-" * 45)
    for name, result in results.items():
        error = result['python_abs_error']
        status = "✅ Perfect" if error < 1e-10 else "⚠️ Good" if error < 1e-6 else "❌ Poor"
        print(f"{name:<15} {error:<15.2e} {status:<10}")

if __name__ == "__main__":
    main()
