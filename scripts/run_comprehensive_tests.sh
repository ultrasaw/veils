#!/bin/bash

echo "🦀 Running Comprehensive Veils Tests in Docker"
echo "================================================="

echo "Building comprehensive test container..."
docker build -f docker/unified_test.Dockerfile -t veils-unified-test .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "Building code quality test container..."
docker build -f docker/code_quality.Dockerfile -t veils-code-quality .

if [ $? -ne 0 ]; then
    echo "❌ Code quality Docker build failed!"
    exit 1
fi

echo "Running code quality tests..."
docker run --rm -v $(pwd):/workspace -w /workspace veils-code-quality sh -c "
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
    echo '✅ All code quality tests completed successfully!'
"

if [ $? -ne 0 ]; then
    echo "❌ Code quality tests failed"
    exit 1
fi

echo ""
echo "Running unified comprehensive STFT tests..."
docker run --rm -v $(pwd):/workspace -w /workspace veils-unified-test

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 All comprehensive tests passed!"
    echo "   - Code quality: ✅"
    echo "   - Unit tests: ✅"
    echo "   - Integration tests: ✅"
    echo "   - STFT coefficient tests: ✅"
    echo "   - Reconstruction tests: ✅"
else
    echo "❌ Comprehensive STFT tests failed"
    exit 1
fi
