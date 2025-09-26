# Dockerfile for Python STFT implementation and plotting
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies
RUN pip install --no-cache-dir numpy scipy matplotlib

# Create output directory
RUN mkdir -p comparison_results

# Copy Python files
COPY standalone_stft.py ./
COPY run_full_comparison_docker.py ./
