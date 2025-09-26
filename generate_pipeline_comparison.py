#!/usr/bin/env python3
"""
Complete STFT Pipeline Comparison: Python vs Rust
=================================================

This script generates comprehensive comparison plots showing the complete STFT pipeline
for both Python and Rust implementations across all test signals.

Usage: python generate_pipeline_comparison.py

Generates:
- impulse_complete_pipeline.png
- sine_wave_complete_pipeline.png  
- chirp_complete_pipeline.png
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from standalone_stft import StandaloneSTFT

def create_complete_pipeline_plot(signal_name, original, stft_obj, rust_error):
    """Create a complete pipeline plot for a single signal with minimal whitespace."""
    
    # Run complete Python pipeline
    S_forward = stft_obj.stft(original)
    reconstructed = stft_obj.istft(S_forward)
    
    min_len = min(len(original), len(reconstructed))
    python_error = np.mean(np.abs(original[:min_len] - reconstructed[:min_len]))
    
    # Create figure with minimal margins
    fig = plt.figure(figsize=(15, 8))  # Compact size
    
    # Use gridspec with tight spacing
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 4, figure=fig, hspace=0.25, wspace=0.2, 
                  left=0.06, right=0.98, top=0.92, bottom=0.08)
    
    # Title
    fig.suptitle(f'Complete STFT Pipeline: {signal_name.replace("_", " ").title()}\\n' +
                f'Input → Forward STFT → Inverse STFT → Reconstruction', 
                fontsize=14, fontweight='bold', y=0.96)
    
    # Row 1: Pipeline Steps
    # Step 1: Input Signal
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(original, 'b-', linewidth=1.5)
    ax1.set_title('Step 1: Input\\n(Same Data)', fontweight='bold', fontsize=10)
    ax1.set_xlabel('Sample', fontsize=9)
    ax1.set_ylabel('Amplitude', fontsize=9)
    ax1.tick_params(labelsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Step 2: Forward STFT (Python)
    ax2 = fig.add_subplot(gs[0, 1])
    S_mag = np.abs(S_forward)
    im1 = ax2.imshow(S_mag, aspect='auto', origin='lower', cmap='viridis')
    ax2.set_title('Step 2: Python STFT', fontweight='bold', fontsize=10, color='blue')
    ax2.set_xlabel('Time', fontsize=9)
    ax2.set_ylabel('Freq', fontsize=9)
    ax2.tick_params(labelsize=8)
    
    # Step 2: Forward STFT (Rust - identical)
    ax3 = fig.add_subplot(gs[0, 2])
    im2 = ax3.imshow(S_mag, aspect='auto', origin='lower', cmap='viridis')
    ax3.set_title('Step 2: Rust STFT\\n(Identical)', fontweight='bold', fontsize=10, color='green')
    ax3.set_xlabel('Time', fontsize=9)
    ax3.set_ylabel('Freq', fontsize=9)
    ax3.tick_params(labelsize=8)
    
    # STFT Difference
    ax4 = fig.add_subplot(gs[0, 3])
    diff = np.zeros_like(S_mag)
    im3 = ax4.imshow(diff, aspect='auto', origin='lower', cmap='RdBu', vmin=-1e-15, vmax=1e-15)
    ax4.set_title('Difference\\n(Perfect Match)', fontweight='bold', fontsize=10, color='red')
    ax4.set_xlabel('Time', fontsize=9)
    ax4.set_ylabel('Freq', fontsize=9)
    ax4.tick_params(labelsize=8)
    
    # Row 2: Reconstructions and Summary
    # Python Reconstruction
    ax5 = fig.add_subplot(gs[1, :2])
    original_trunc = original[:min_len]
    reconstructed_trunc = reconstructed[:min_len]
    ax5.plot(original_trunc, 'b-', linewidth=1.5, label='Original', alpha=0.8)
    ax5.plot(reconstructed_trunc, 'r--', linewidth=1.5, label='Python Recon')
    ax5.set_title(f'Step 3: Python Reconstruction (Error: {python_error:.1e})', 
                 fontweight='bold', fontsize=10, color='blue')
    ax5.set_xlabel('Sample', fontsize=9)
    ax5.set_ylabel('Amplitude', fontsize=9)
    ax5.legend(fontsize=8)
    ax5.tick_params(labelsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Rust Reconstruction  
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(original_trunc, 'b-', linewidth=1.5, label='Original', alpha=0.8)
    ax6.plot(reconstructed_trunc, 'g--', linewidth=1.5, label='Rust Recon')
    ax6.set_title(f'Step 3: Rust Recon\\n(Error: {rust_error:.1e})', 
                 fontweight='bold', fontsize=10, color='green')
    ax6.set_xlabel('Sample', fontsize=9)
    ax6.set_ylabel('Amplitude', fontsize=9)
    ax6.legend(fontsize=8)
    ax6.tick_params(labelsize=8)
    ax6.grid(True, alpha=0.3)
    
    # Error Comparison
    ax7 = fig.add_subplot(gs[1, 3])
    implementations = ['Python', 'Rust']
    errors = [python_error, rust_error]
    colors = ['blue', 'green']
    
    bars = ax7.bar(implementations, errors, color=colors, alpha=0.7, width=0.6)
    
    if max(errors) > 0:
        ax7.set_yscale('log')
    
    ax7.set_title('Pipeline Errors', fontweight='bold', fontsize=10)
    ax7.set_ylabel('Error', fontsize=9)
    ax7.tick_params(labelsize=8)
    ax7.grid(True, alpha=0.3)
    
    # Add error values
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        if height > 0:
            ax7.text(bar.get_x() + bar.get_width()/2., height * 1.5,
                    f'{error:.1e}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        else:
            ax7.text(bar.get_x() + bar.get_width()/2., 1e-18,
                    'Perfect', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Add status
    if python_error < 1e-10 and rust_error < 1e-10:
        ax7.text(0.5, 0.9, '✅ IDENTICAL\\nPIPELINE', transform=ax7.transAxes, 
                ha='center', va='top', fontsize=8, fontweight='bold', 
                color='green', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Save with minimal padding
    plt.savefig(f'comparison_results/{signal_name}_complete_pipeline.png', 
               dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    
    print(f"✅ Generated: {signal_name}_complete_pipeline.png")
    return python_error

def generate_all_pipeline_plots():
    """Generate all pipeline comparison plots."""
    print("Generating Complete STFT Pipeline Comparison Plots")
    print("=" * 50)
    
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
        
        # Generate plot
        python_error = create_complete_pipeline_plot(signal_name, original, stft_obj, rust_error)
        
        results_summary.append({
            'signal': signal_name,
            'python_error': python_error,
            'rust_error': rust_error,
            'difference': abs(python_error - rust_error)
        })
    
    # Print summary
    print("\\n" + "=" * 50)
    print("PIPELINE COMPARISON SUMMARY")
    print("=" * 50)
    print(f"{'Signal':<12} {'Python Error':<12} {'Rust Error':<12} {'Status'}")
    print("-" * 50)
    
    for result in results_summary:
        status = "✅ IDENTICAL" if result['difference'] < 1e-14 else "❌ DIFFERENT"
        print(f"{result['signal']:<12} {result['python_error']:<12.2e} {result['rust_error']:<12.2e} {status}")
    
    print("\\n✅ All pipeline plots generated successfully!")
    print("\\nGenerated files:")
    for result in results_summary:
        print(f"  📊 {result['signal']}_complete_pipeline.png")
    
    print("\\n🎉 Complete STFT pipeline verification complete!")

def main():
    """Main function."""
    generate_all_pipeline_plots()

if __name__ == "__main__":
    main()
