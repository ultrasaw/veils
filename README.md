# Veils: Rust STFT Implementation

This repository contains an implementation of the stripped-down `scipy.signal.ShortTimeFFT` class in Python and Rust, with verification of ~1:1 accuracy between the implementations.

## Files

### Core Implementations
- `python/standalone_stft.py` - Python STFT reference implementation
- `src/lib.rs` - Rust STFT implementation
- `src/bin/shared_data_test.rs` - Rust test binary for pipeline verification

### Test Scripts (Local Development)
- `scripts/run_code_quality_checks.sh` - **Fast quality checks** (formatting + clippy) → matches CI `code-quality` job
- `scripts/run_comprehensive_tests.sh` - **Full test suite** (build + tests + docs + unified STFT tests) → matches CI `comprehensive-test` job  
- `scripts/run_crate_tests.sh` - **Publishing readiness** (clean env + publish dry-run) → matches CI `publish-check` job
- `scripts/run_final_comparison.sh` - **Python-Rust compatibility** (1:1 STFT coefficient and reconstruction verification)

### Docker Infrastructure
- `docker/code_quality.Dockerfile` - Base Rust environment with fmt/clippy for quality checks
- `docker/unified_test.Dockerfile` - Unified comprehensive STFT testing container
- `docker/python_rust_comparison.Dockerfile` - Python-Rust compatibility testing container
- `docker/create_test.Dockerfile` - Clean environment for crate publishing tests
- `python/final_comparison.py` - Python script for comprehensive Python-Rust comparison

## Local Testing Scripts → CI Jobs Mapping

| Local Script | CI Job | Purpose |
|-------------|---------|---------|
| `scripts/run_code_quality_checks.sh` | `code-quality` | Format + lint checks |
| `scripts/run_comprehensive_tests.sh` | `comprehensive-test` | Build + all tests + docs + unified STFT tests |
| `scripts/run_crate_tests.sh` | `publish-check` | Clean env + publish dry-run |
| `scripts/run_final_comparison.sh` | `python-rust-comparison` | Python-Rust 1:1 compatibility verification |

## Usage

### Local Development Testing

Use docker and do small incremental changes; run test scripts to pinpoint and fix issues, or when working on new features.

```bash
# Quick quality check (matches CI code-quality job)
bash scripts/run_code_quality_checks.sh

# Full test suite (matches CI comprehensive testing)  
bash scripts/run_comprehensive_tests.sh

# Publishing readiness (matches CI publish check)
bash scripts/run_crate_tests.sh

# Python-Rust compatibility verification (matches CI python-rust-comparison job)
bash scripts/run_final_comparison.sh
```

The `scripts/run_final_comparison.sh` script:
1. **Builds** Python-Rust comparison Docker container
2. **Runs comprehensive 1:1 compatibility tests** between Python and Rust implementations  
3. **Verifies perfect reconstruction** and coefficient accuracy across multiple signal types
4. **Generates** detailed comparison results in `final_comparison_results.json`

### Manual Docker Pipeline (Alternative)

```bash
# 1. Build and run Python-Rust comparison
docker build -f docker/python_rust_comparison.Dockerfile -t veils-comparison .
docker run --rm -v $(pwd):/workspace -w /workspace veils-comparison

# 2. Build and run comprehensive tests
docker build -f docker/unified_test.Dockerfile -t veils-unified-test .
docker run --rm -v $(pwd)/comparison_results:/workspace/comparison_results veils-unified-test

# 3. Build and run crate tests
docker build -f docker/create_test.Dockerfile -t veils-crate-test .
docker run --rm veils-crate-test
```
