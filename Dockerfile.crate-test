# Dockerfile for testing the Rust crate in a clean environment
FROM rust:1.89

# Set working directory
WORKDIR /workspace

# Copy only the files needed for the crate (excluding binaries)
COPY Cargo.toml ./
COPY src/lib.rs ./src/
COPY tests/ ./tests/
COPY examples/ ./examples/
COPY README.md ./
COPY LICENSE ./

# Install dependencies and run tests
RUN cargo test --lib --tests --examples --verbose

# Test documentation build
RUN cargo doc --no-deps --all-features

# Test publish (dry run)
RUN cargo publish --dry-run

# Run example
RUN cargo run --example basic_usage

CMD ["echo", "✅ All crate tests passed in Docker!"]
