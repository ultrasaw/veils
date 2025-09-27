#!/bin/bash

# Test the Rust crate in a clean Docker environment
# This ensures compatibility and that the crate works independently

echo "🦀 Testing Veils STFT Crate in Docker"
echo "========================================"

# Build the test image
echo "📦 Building Docker test environment..."
docker build -f docker/create_test.Dockerfile -t veils-crate-test .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✅ Docker build successful!"

# Run the tests
echo ""
echo "🧪 Running crate tests in clean environment..."
docker run --rm veils-crate-test

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 All crate tests passed!"
    echo "✅ Ready for publishing to crates.io"
    echo ""
    echo "To publish:"
    echo "  1. Update authors/repository in Cargo.toml"
    echo "  2. Run: cargo publish"
else
    echo "❌ Crate tests failed in Docker"
    exit 1
fi
