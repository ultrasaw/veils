FROM rust:1.89

# Install rustfmt and clippy
RUN rustup component add rustfmt
RUN rustup component add clippy

# Set working directory
WORKDIR /workspace
