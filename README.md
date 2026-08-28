# Veils

Veils is a Rust implementation of the one-dimensional, real-valued subset of
[`scipy.signal.ShortTimeFFT`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html).
Its coefficients and reconstruction are tested directly against SciPy 1.18.1.

## Supported Compatibility

- Real `f64` signals and windows
- `onesided`, `twosided`, `centered`, and magnitude-scaled `onesided2X` FFTs
- Configurable hop size, sampling frequency, FFT length, dual window, phase
  shift, STFT slice range, and reconstruction range
- Zero padding at signal boundaries
- Frequency-by-time STFT output and canonical dual-window reconstruction

Veils does not currently implement multidimensional axes, complex-valued
signals/windows, SciPy's detrending API, non-zero padding modes, or a general
magnitude/PSD scaling API. `onesided2X` corresponds to magnitude scaling when
the supplied window has already been magnitude-scaled.

## Usage

```rust
use veils::StandaloneSTFT;

let window: Vec<f64> = (0..16)
    .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / 16.0).cos()))
    .collect();
let signal: Vec<f64> = (0..64)
    .map(|i| (2.0 * std::f64::consts::PI * 5.0 * i as f64 / 1000.0).sin())
    .collect();

let mut stft = StandaloneSTFT::new(
    window,
    4,
    1000.0,
    Some("onesided"),
    None,
    None,
    None,
)?;
let coefficients = stft.stft(&signal, None, None, None)?;
let reconstructed = stft.istft(&coefficients, Some(0), Some(signal.len() as i32))?;

# Ok::<(), String>(())
```

## Development

The Python compatibility environment is pinned in `requirements.txt`. Run the
direct SciPy comparison in Docker with:

```bash
bash scripts/run_final_comparison.sh
```

Rust checks are available through:

```bash
bash scripts/run_code_quality_checks.sh
bash scripts/run_comprehensive_tests.sh
bash scripts/run_crate_tests.sh
```

## Upstream Provenance

The implementation is adapted from the algorithms in SciPy's
`scipy/signal/_short_time_fft.py`. Compatibility is currently verified against
the SciPy `v1.18.1` release. SciPy is distributed under the BSD 3-Clause
license included in `LICENSE-SCIPY`; Veils' original code is distributed under
the MIT license included in `LICENSE`.
