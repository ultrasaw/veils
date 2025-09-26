#!/bin/bash

# Complete STFT Comparison using Docker Containers
# ================================================

echo "🐳 Complete STFT Implementation Comparison (Docker)"
echo "=================================================="
echo "Running Rust and Python implementations in separate containers"
echo ""

# Create comparison results directory
mkdir -p comparison_results

# Step 1: Build Docker images
echo "📦 Building Docker images..."
echo "Building Rust container..."
docker build -f Dockerfile.rust -t spectrust-rust .

echo "Building Python container..."
docker build -f Dockerfile.python -t spectrust-python .

echo ""

# Step 2: Run Rust container first to generate test data and results
echo "🦀 Running Rust STFT implementation..."
docker run --rm -v $(pwd)/comparison_results:/workspace/comparison_results spectrust-rust

if [ $? -ne 0 ]; then
    echo "❌ Rust container failed!"
    exit 1
fi

echo "✅ Rust container completed successfully"
echo ""

# Step 3: Run Python container to process results and generate plots
echo "🐍 Running Python STFT implementation and plotting..."
docker run --rm -v $(pwd)/comparison_results:/workspace/comparison_results spectrust-python

if [ $? -ne 0 ]; then
    echo "❌ Python container failed!"
    exit 1
fi

echo ""
echo "🎉 Docker comparison pipeline completed successfully!"
echo ""
echo "Generated files in comparison_results/:"
ls -la comparison_results/*.png comparison_results/*.txt 2>/dev/null || echo "No files generated"

echo ""
echo "✅ Complete containerized STFT verification finished!"
