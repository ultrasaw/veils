# Dockerfile for unified comprehensive STFT testing
FROM rust:1.89

# Install required dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy Rust project files
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/

# Build the unified comprehensive test binary
RUN cargo build --bin unified_comprehensive_test

# Create output directory
RUN mkdir -p comparison_results

# Set entrypoint to run the unified comprehensive test
ENTRYPOINT ["./target/debug/unified_comprehensive_test"]
