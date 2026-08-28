# Changelog

## 0.1.4 - 2026-08-28

Version 0.1.3 was not published because its tag name was reserved by a failed
immutable GitHub release draft.

- Validate compatibility directly against `scipy.signal.ShortTimeFFT` 1.18.1.
- Match SciPy validation for sampling frequency, FFT length, phase shift, STFT
  slice bounds, and ISTFT reconstruction bounds.
- Correct centered and even-length two-sided frequency axes.
- Support negative phase shifts without unsigned rotation overflow.
- Document the supported SciPy subset and upstream attribution.
