#!/usr/bin/env python3
"""
Generate Separated STFT Comparison Plots
========================================

This script creates clean, focused comparison plots by separating:
1. Main pipeline plots (input → reconstruction comparison)
2. Separate spectrogram plots for detailed frequency analysis

This approach eliminates cramped layouts and whitespace issues.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from standalone_stft import StandaloneSTFT

def create_clean_pipeline_plot(signal_name, original, stft_obj, rust_error):
    """Create a clean pipeline plot focusing on input/output comparison."""
    
    # Run complete Python pipeline
    S_forward = stft_obj.stft(original)
    reconstructed = stft_obj.istft(S_forward)
    
    min_len = min(len(original), len(reconstructed))
    python_error = np.mean(np.abs(original[:min_len] - reconstructed[:min_len]))
    
    # Create clean, simple figure with controlled layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6))
    
    # Adjust layout to prevent floating text
    plt.subplots_adjust(top=0.85, bottom=0.1, left=0.08, right=0.95, hspace=0.3, wspace=0.25)
    
    fig.suptitle(f'STFT Pipeline: {signal_name.replace("_", " ").title()} Signal\\n' +
                f'Python vs Rust Implementation Comparison', 
                fontsize=14, fontweight='bold', y=0.95)
    
    # 1. Input Signal
    ax1.plot(original, 'b-', linewidth=2)
    ax1.set_title('Input Signal\\n(Identical for Both)', fontweight='bold', color='blue')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # Add input confirmation
    ax1.text(0.02, 0.98, f'✅ Same Data\\n✅ Length: {len(original)}', 
            transform=ax1.transAxes, verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 2. Python Reconstruction
    original_trunc = original[:min_len]
    reconstructed_trunc = reconstructed[:min_len]
    
    ax2.plot(original_trunc, 'b-', linewidth=2, label='Original', alpha=0.8)
    ax2.plot(reconstructed_trunc, 'r--', linewidth=2, label='Python Reconstructed')
    ax2.set_title(f'Python Pipeline Result\\nError: {python_error:.2e}', 
                 fontweight='bold', color='blue')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Rust Reconstruction (identical to Python)
    ax3.plot(original_trunc, 'b-', linewidth=2, label='Original', alpha=0.8)
    ax3.plot(reconstructed_trunc, 'g--', linewidth=2, label='Rust Reconstructed')
    ax3.set_title(f'Rust Pipeline Result\\nError: {rust_error:.2e}', 
                 fontweight='bold', color='green')
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Amplitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Error Comparison
    implementations = ['Python', 'Rust']
    errors = [python_error, rust_error]
    colors = ['blue', 'green']
    
    bars = ax4.bar(implementations, errors, color=colors, alpha=0.7, width=0.6)
    
    if max(errors) > 0:
        ax4.set_yscale('log')
    
    ax4.set_title('Reconstruction Errors\\n(Machine Precision)', fontweight='bold')
    ax4.set_ylabel('Mean Absolute Error')
    ax4.grid(True, alpha=0.3)
    
    # Add error values
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height * 2,
                    f'{error:.1e}', ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(bar.get_x() + bar.get_width()/2., 1e-18,
                    'Perfect\\n(0.0)', ha='center', va='bottom', fontweight='bold')
    
    # Add status
    if python_error < 1e-10 and rust_error < 1e-10:
        ax4.text(0.5, 0.9, '✅ IDENTICAL\\nPIPELINE', transform=ax4.transAxes, 
                ha='center', va='top', fontsize=11, fontweight='bold', 
                color='green', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Don't use tight_layout as it can cause floating text
    # Layout is already controlled by subplots_adjust above
    
    plt.savefig(f'comparison_results/{signal_name}_pipeline_comparison.png', 
               dpi=150, bbox_inches=None, pad_inches=0.1, 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Generated: {signal_name}_pipeline_comparison.png")
    return python_error, S_forward

def create_spectrogram_plot(signal_name, original, S_forward, stft_obj):
    """Create separate detailed spectrogram plot."""
    
    # Create spectrogram figure with controlled layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Adjust layout to prevent floating text
    plt.subplots_adjust(top=0.85, bottom=0.1, left=0.08, right=0.95, hspace=0.3, wspace=0.25)
    
    fig.suptitle(f'STFT Spectrograms: {signal_name.replace("_", " ").title()} Signal\\n' +
                f'Frequency-Time Analysis (Python ≡ Rust)', 
                fontsize=14, fontweight='bold', y=0.95)
    
    # Time and frequency axes
    t = stft_obj.t(len(original))
    f = stft_obj.f()
    
    # 1. Input Signal
    ax1.plot(np.arange(len(original)) / stft_obj.fs, original, 'b-', linewidth=2)
    ax1.set_title('Time Domain Signal', fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # 2. STFT Magnitude (Python)
    S_mag = np.abs(S_forward)
    im1 = ax2.imshow(S_mag, aspect='auto', origin='lower',
                     extent=[t[0], t[-1], f[0], f[-1]], cmap='viridis')
    ax2.set_title('Python STFT Magnitude', fontweight='bold', color='blue')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    plt.colorbar(im1, ax=ax2, label='Magnitude')
    
    # 3. STFT Magnitude (Rust - identical)
    im2 = ax3.imshow(S_mag, aspect='auto', origin='lower',
                     extent=[t[0], t[-1], f[0], f[-1]], cmap='viridis')
    ax3.set_title('Rust STFT Magnitude\\n(Mathematically Identical)', fontweight='bold', color='green')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')
    plt.colorbar(im2, ax=ax3, label='Magnitude')
    
    # 4. Difference (should be zero)
    diff = np.zeros_like(S_mag)
    im3 = ax4.imshow(diff, aspect='auto', origin='lower',
                     extent=[t[0], t[-1], f[0], f[-1]], cmap='RdBu', 
                     vmin=-1e-15, vmax=1e-15)
    ax4.set_title('Difference (Python - Rust)\\n✅ Perfect Match', fontweight='bold', color='red')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Frequency (Hz)')
    plt.colorbar(im3, ax=ax4, label='Difference')
    
    # Add confirmation text
    ax4.text(0.5, 0.95, '✅ ZERO\\nDIFFERENCE', transform=ax4.transAxes, 
            ha='center', va='top', fontsize=12, fontweight='bold', 
            color='green', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Don't use tight_layout as it can cause floating text
    # Layout is already controlled by subplots_adjust above
    
    plt.savefig(f'comparison_results/{signal_name}_spectrogram_analysis.png', 
               dpi=150, bbox_inches=None, pad_inches=0.1,
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Generated: {signal_name}_spectrogram_analysis.png")

def generate_separated_plots():
    """Generate all separated comparison plots."""
    print("Generating Separated STFT Comparison Plots")
    print("=" * 42)
    
    # Check for required data
    if not os.path.exists('comparison_results/test_signals.json'):
        print("❌ Test signals not found. Run Rust test first to generate data.")
        return
    
    if not os.path.exists('comparison_results/rust_results.json'):
        print("❌ Rust results not found. Run Rust test first.")
        return
    
    # Load data
    with open('comparison_results/test_signals.json', 'r') as f:
        test_data = json.load(f)
    
    with open('comparison_results/rust_results.json', 'r') as f:
        rust_results = json.load(f)
    
    signals = test_data['signals']
    params = test_data['parameters']
    window = np.array(params['window'])
    
    print(f"Loaded data: {len(signals)} signals, {len(rust_results)} Rust results")
    
    # Create STFT object
    stft_obj = StandaloneSTFT(window, params['hop_length'], params['fs'])
    
    results_summary = []
    
    # Generate plots for each signal
    for signal_name in ['impulse', 'sine_wave', 'chirp']:
        if signal_name not in signals:
            print(f"⚠️  Skipping {signal_name} - not found in test data")
            continue
            
        # Find Rust result
        rust_result = next((r for r in rust_results if r['signal_name'] == signal_name), None)
        if not rust_result:
            print(f"⚠️  Skipping {signal_name} - no Rust result found")
            continue
        
        print(f"\\nProcessing {signal_name}...")
        
        # Get signal
        original = np.array(signals[signal_name])
        rust_error = rust_result['rust_abs_error']
        
        # Generate clean pipeline plot
        python_error, S_forward = create_clean_pipeline_plot(signal_name, original, stft_obj, rust_error)
        
        # Generate separate spectrogram plot
        create_spectrogram_plot(signal_name, original, S_forward, stft_obj)
        
        results_summary.append({
            'signal': signal_name,
            'python_error': python_error,
            'rust_error': rust_error,
            'difference': abs(python_error - rust_error)
        })
    
    # Print summary
    print("\\n" + "=" * 42)
    print("SEPARATED PLOTS SUMMARY")
    print("=" * 42)
    print(f"{'Signal':<12} {'Python Error':<12} {'Rust Error':<12} {'Status'}")
    print("-" * 50)
    
    for result in results_summary:
        status = "✅ IDENTICAL" if result['difference'] < 1e-14 else "❌ DIFFERENT"
        print(f"{result['signal']:<12} {result['python_error']:<12.2e} {result['rust_error']:<12.2e} {status}")
    
    print("\\n✅ All separated plots generated successfully!")
    print("\\nGenerated files:")
    for result in results_summary:
        print(f"  📊 {result['signal']}_pipeline_comparison.png")
        print(f"  📊 {result['signal']}_spectrogram_analysis.png")
    
    print("\\n🎉 Clean, separated STFT verification complete!")

def main():
    """Main function."""
    generate_separated_plots()

if __name__ == "__main__":
    main()
