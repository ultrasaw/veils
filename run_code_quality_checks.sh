#!/bin/bash

echo "Building code quality container..."
docker build -f docker/code_quality.Dockerfile -t spectrust-code-quality .

echo "Running checks..."
docker run --rm -v /home/gio/Documents/_projects/spectrust:/workspace -w /workspace spectrust-code-quality sh -c "cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings"
