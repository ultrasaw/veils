#!/bin/bash

echo "🚀 Starting Comprehensive STFT Implementation Comparison"
echo "========================================================"

# Clean up previous results
echo "🧹 Cleaning up previous results..."
rm -rf comparison_results/
mkdir -p comparison_results/

# Step 1: Run Python analysis and generate test data
echo "🐍 Running Python analysis..."
docker run --rm -v $(pwd):/workspace -w /workspace python:3.11 bash -c "
    pip install numpy scipy matplotlib > /dev/null 2>&1
    python compare_implementations.py
"

if [ $? -ne 0 ]; then
    echo "❌ Python analysis failed"
    exit 1
fi

echo "✅ Python analysis completed"

# Step 2: Build Rust implementation
echo "🦀 Building Rust implementation..."
docker run --rm -v $(pwd):/workspace -w /workspace rust:1.75 cargo build --bin comprehensive_test --release

if [ $? -ne 0 ]; then
    echo "❌ Rust build failed"
    exit 1
fi

echo "✅ Rust build completed"

# Step 3: Run Rust comprehensive test
echo "🔬 Running Rust comprehensive test..."
docker run --rm -v $(pwd):/workspace -w /workspace rust:1.75 ./target/release/comprehensive_test > comparison_results/rust_output.log 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Rust test failed"
    cat comparison_results/rust_output.log
    exit 1
fi

echo "✅ Rust test completed"

# Step 4: Generate final comparison report
echo "📊 Generating final comparison report..."
docker run --rm -v $(pwd):/workspace -w /workspace python:3.11 bash -c "
    pip install numpy matplotlib > /dev/null 2>&1
    python -c \"
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Load results
with open('comparison_results/rust_results.json', 'r') as f:
    rust_results = json.load(f)

# Create summary plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Reconstruction errors
signals = [r['signal_name'] for r in rust_results]
rust_errors = [r['rust_abs_error'] for r in rust_results]
python_errors = [r['python_abs_error'] for r in rust_results]

x = np.arange(len(signals))
width = 0.35

bars1 = ax1.bar(x - width/2, rust_errors, width, label='Rust', alpha=0.8)
bars2 = ax1.bar(x + width/2, python_errors, width, label='Python', alpha=0.8)

ax1.set_xlabel('Test Signals')
ax1.set_ylabel('Absolute Error')
ax1.set_title('Reconstruction Error Comparison')
ax1.set_yscale('log')
ax1.set_xticks(x)
ax1.set_xticklabels([s.replace('_', ' ').title() for s in signals], rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: STFT matching errors
stft_errors = [r['stft_match_error'] for r in rust_results]
istft_errors = [r['istft_cross_check_error'] for r in rust_results]

bars3 = ax2.bar(x - width/2, stft_errors, width, label='STFT Match', alpha=0.8)
bars4 = ax2.bar(x + width/2, istft_errors, width, label='ISTFT Cross-check', alpha=0.8)

ax2.set_xlabel('Test Signals')
ax2.set_ylabel('Error')
ax2.set_title('Implementation Matching Errors')
ax2.set_yscale('log')
ax2.set_xticks(x)
ax2.set_xticklabels([s.replace('_', ' ').title() for s in signals], rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_results/summary_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Generate final report
report = f'''
STFT Implementation Comparison Report
Generated: {datetime.now().isoformat()}
==========================================

SUMMARY:
--------
Total signals tested: {len(rust_results)}

Rust Implementation Results:
'''

perfect_rust = sum(1 for r in rust_results if r['rust_abs_error'] < 1e-10)
good_rust = sum(1 for r in rust_results if r['rust_abs_error'] < 1e-6)
perfect_stft = sum(1 for r in rust_results if r['stft_match_error'] < 1e-10)
perfect_istft = sum(1 for r in rust_results if r['istft_cross_check_error'] < 1e-10)

report += f'''
- Perfect reconstruction (< 1e-10): {perfect_rust}/{len(rust_results)}
- Good reconstruction (< 1e-6): {good_rust}/{len(rust_results)}
- Perfect STFT match: {perfect_stft}/{len(rust_results)}
- Perfect ISTFT cross-check: {perfect_istft}/{len(rust_results)}

DETAILED RESULTS:
-----------------
'''

for r in rust_results:
    status = '✅ PERFECT' if r['rust_abs_error'] < 1e-10 else '⚠️ GOOD' if r['rust_abs_error'] < 1e-6 else '❌ POOR'
    report += f'''
{r['signal_name'].upper()}:
  Status: {status}
  Rust reconstruction error: {r['rust_abs_error']:.2e}
  Python reconstruction error: {r['python_abs_error']:.2e}
  STFT matching error: {r['stft_match_error']:.2e}
  ISTFT cross-check error: {r['istft_cross_check_error']:.2e}
'''

with open('comparison_results/final_report.txt', 'w') as f:
    f.write(report)

print('Final report generated!')
\"
"

echo "✅ Final report generated"

# Step 5: Display results
echo ""
echo "📋 COMPARISON RESULTS SUMMARY"
echo "============================="

# Show Rust output
echo "🦀 Rust Test Output:"
cat comparison_results/rust_output.log

echo ""
echo "📁 Generated Files:"
echo "  📊 comparison_results/summary_comparison.png - Visual comparison"
echo "  📄 comparison_results/final_report.txt - Detailed report"
echo "  🔍 comparison_results/*_comparison.png - Individual signal plots"
echo "  📋 comparison_results/comparison_log.txt - Python analysis log"
echo "  🦀 comparison_results/rust_results.json - Rust test results"
echo "  🐍 comparison_results/test_signals.json - Test signal data"

echo ""
echo "🎯 Quick Assessment:"
if [ -f comparison_results/rust_results.json ]; then
    # Quick assessment using basic tools
    perfect_count=$(grep -o '"rust_abs_error":[^,]*' comparison_results/rust_results.json | awk -F: '{print $2}' | awk '$1 < 1e-10' | wc -l)
    total_count=$(grep -c '"signal_name"' comparison_results/rust_results.json)
    
    if [ "$perfect_count" -eq "$total_count" ]; then
        echo "🎉 ALL SIGNALS PERFECTLY RECONSTRUCTED!"
    elif [ "$perfect_count" -gt 0 ]; then
        echo "✅ $perfect_count/$total_count signals perfectly reconstructed"
    else
        echo "⚠️ No perfect reconstructions - check detailed results"
    fi
else
    echo "❌ Could not assess results"
fi

echo ""
echo "🔍 View results with:"
echo "  cat comparison_results/final_report.txt"
echo "  open comparison_results/summary_comparison.png"
