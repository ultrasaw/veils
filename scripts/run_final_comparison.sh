#!/bin/bash

echo "🔬 Final Python-Rust STFT Comparison (Docker)"
echo "=============================================="

echo "🐳 Building comparison container..."
docker build -f docker/python_rust_comparison.Dockerfile -t veils-comparison .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "🧪 Running comprehensive comparison tests..."
docker run --rm -v $(pwd):/workspace -w /workspace veils-comparison

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Comparison completed successfully!"
    echo "   Check final_comparison_results.json for detailed results."
else
    echo ""
    echo "⚠️  Comparison completed with issues."
    echo "   Check final_comparison_results.json for detailed results."
    exit 1
fi
