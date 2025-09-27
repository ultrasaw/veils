#!/bin/bash

echo "🦀 Running Comprehensive Veils Tests in Docker"
echo "================================================="

echo "Building comprehensive test container..."
docker build -f docker/code_quality.Dockerfile -t veils-comprehensive-test .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "Running comprehensive tests..."
docker run --rm -v /home/gio/Documents/_projects/spectrust:/workspace -w /workspace veils-comprehensive-test sh -c "
    echo '📋 Building library...' &&
    cargo build --verbose && 
    echo '🧪 Running unit tests...' &&
    cargo test --verbose && 
    echo '🔗 Running integration tests...' &&
    cargo test --test integration_tests --verbose && 
    echo '📚 Testing with all features...' &&
    cargo test --all-features --verbose && 
    echo '📖 Building documentation...' &&
    cargo doc --no-deps --all-features &&
    echo '✅ All comprehensive tests completed successfully!'
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 All comprehensive tests passed!"
else
    echo "❌ Comprehensive tests failed"
    exit 1
fi
