# Veils: Rust STFT Implementation

This repository contains an implementation of the stripped-down `scipy.signal.ShortTimeFFT` class in Python and Rust, with comprehensive verification of ~1:1 accuracy between the implementations.

## Files

### Core Implementations
- `standalone_stft.py` - Python STFT reference implementation
- `src/lib.rs` - Rust STFT implementation
- `src/bin/shared_data_test.rs` - Rust test binary for pipeline verification

### Test Scripts (Local Development)
- `run_code_quality_checks.sh` - **Fast quality checks** (formatting + clippy) → matches CI `code-quality` job
- `run_comprehensive_tests.sh` - **Full test suite** (build + tests + docs) → matches CI `comprehensive-test` job  
- `run_crate_tests.sh` - **Publishing readiness** (clean env + publish dry-run) → matches CI `publish-check` job
- `run_docker_comparison.sh` - **Python vs Rust comparison** (complete containerized pipeline)

### Docker Infrastructure
- `docker/code_quality.Dockerfile` - Base Rust environment with fmt/clippy for quality checks
- `docker/rust.Dockerfile` - Rust STFT implementation container for comparisons
- `docker/py.Dockerfile` - Python STFT implementation and plotting container
- `docker/create_test.Dockerfile` - Clean environment for crate publishing tests
- `run_full_comparison_docker.py` - Python script for data generation and comparison

## Local Testing Scripts → CI Jobs Mapping

| Local Script | CI Job | Purpose |
|-------------|---------|---------|
| `./run_code_quality_checks.sh` | `code-quality` | Format + lint checks |
| `./run_comprehensive_tests.sh` | `comprehensive-test` | Build + all tests + docs |
| `./run_crate_tests.sh` | `publish-check` | Clean env + publish dry-run |
| `./run_docker_comparison.sh` | `sanity-check` | Python vs Rust verification |

## Usage

### Local Development Testing

Use docker and do small incremental changes; run test scripts to pinpoint and fix issues, or when working on new features.

```bash
# Quick quality check (matches CI code-quality job)
./run_code_quality_checks.sh

# Full test suite (matches CI comprehensive testing)  
./run_comprehensive_tests.sh

# Publishing readiness (matches CI publish check)
./run_crate_tests.sh

# Python vs Rust comparison (matches CI sanity check)
./run_docker_comparison.sh
```

The `run_docker_comparison.sh` script:
1. **Builds** both Rust and Python Docker images
2. **Runs Python container** to generate test data
3. **Runs Rust container** to generate STFT results
4. **Runs Python container again** to create plots and reports
5. **Generates** all comparison files in `comparison_results/`

### Manual Docker Pipeline (Alternative)

```bash
# 1. Build images
docker build -f docker/rust.Dockerfile -t veils-rust .
docker build -f docker/py.Dockerfile -t veils-python .

# 2. Run Python container to generate data
docker run --rm -v $(pwd)/comparison_results:/workspace/comparison_results veils-python python run_full_comparison_docker.py --generate-data

# 3. Run Rust container
docker run --rm -v $(pwd)/comparison_results:/workspace/comparison_results veils-rust

# 4. Run Python container to generate plots and report
docker run --rm -v $(pwd)/comparison_results:/workspace/comparison_results veils-python python run_full_comparison_docker.py --run-comparison
```

### Generated Files

**Pipeline Comparisons** (Input → Reconstruction):
- `impulse_pipeline_comparison.png`
- `sine_wave_pipeline_comparison.png`
- `chirp_pipeline_comparison.png`

**Spectrogram Analysis** (Frequency-Time Details):
- `impulse_spectrogram_analysis.png`
- `sine_wave_spectrogram_analysis.png`
- `chirp_spectrogram_analysis.png`

**Summary Report**:
- `full_comparison_report.txt`

## Verification Results

| Signal Type | Python Error | Rust Error |
|-------------|--------------|------------|
| **Impulse** | 0.00e+00 | 0.00e+00 |
| **Sine Wave** | 4.57e-17 | 6.19e-17 |
| **Chirp** | 5.86e-17 | 6.79e-17 |
