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
docker build -f docker/rust.Dockerfile -t veils-rust .

echo "Building Python container..."
docker build -f docker/py.Dockerfile -t veils-python .

echo ""

# Step 2: Run Python container to generate data
echo "📊 Generating test data..."
docker run --rm -v $(pwd)/comparison_results:/workspace/comparison_results veils-python python run_full_comparison_docker.py --generate-data

if [ $? -ne 0 ]; then
    echo "❌ Data generation failed!"
    exit 1
fi

echo "✅ Test data generated successfully"
echo ""


# Step 3: Run Rust container
echo "🦀 Running Rust STFT implementation..."
docker run --rm -v $(pwd)/comparison_results:/workspace/comparison_results veils-rust

if [ $? -ne 0 ]; then
    echo "❌ Rust container failed!"
    exit 1
fi

echo "✅ Rust container completed successfully"
echo ""

# Step 4: Run Python container to process results and generate plots
echo "🐍 Running Python STFT implementation and plotting..."
docker run --rm -v $(pwd)/comparison_results:/workspace/comparison_results veils-python python run_full_comparison_docker.py --run-comparison

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
