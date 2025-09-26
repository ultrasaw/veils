# Dockerfile for Rust STFT implementation
FROM rust:1.75

# Set working directory
WORKDIR /workspace

# Copy Rust project files
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/

# Build the Rust binary
RUN cargo build --release --bin shared_data_test

# Create output directory
RUN mkdir -p comparison_results

# Set entrypoint to run the Rust test
ENTRYPOINT ["./target/release/shared_data_test"]
