#!/usr/bin/env python3
"""Compare the Rust implementation directly with SciPy ShortTimeFFT."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import ShortTimeFFT, windows


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "final_comparison_results.json"
EXPECTED_SCIPY = "1.18.1"
RTOL = 1e-12
ATOL = 1e-12


def run_rust(case: dict, rust_window: np.ndarray, dual_win: np.ndarray | None) -> dict:
    payload = {
        "signal": case["signal"].tolist(),
        "window": rust_window.tolist(),
        "hop_length": case["hop"],
        "fs": case["fs"],
        "fft_mode": case["fft_mode"],
        "mfft": case["mfft"],
        "dual_win": None if dual_win is None else dual_win.tolist(),
        "phase_shift": case["phase_shift"],
        "k1": len(case["signal"]),
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as file:
        json.dump(payload, file)
        input_path = Path(file.name)

    try:
        result = subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "--bin",
                "python_rust_comparison_helper",
                "--",
                str(input_path),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)
    finally:
        input_path.unlink()


def decode_stft(values: list[list[dict[str, float]]]) -> np.ndarray:
    return np.asarray(
        [[complex(value["real"], value["imag"]) for value in row] for row in values],
        dtype=np.complex128,
    )


def metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    difference = np.abs(expected - actual)
    return {
        "max_abs": float(np.max(difference, initial=0.0)),
        "rmse": float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0,
    }


def create_cases() -> list[dict]:
    fs = 1000.0
    sample_count = 65
    time = np.arange(sample_count, dtype=np.float64) / fs
    impulse = np.zeros(sample_count)
    impulse[32] = 1.0
    sine = np.sin(2 * np.pi * 73.0 * time)
    noise = np.random.default_rng(42).standard_normal(sample_count)

    return [
        dict(name="rectangular-even", signal=impulse, window=np.ones(8), hop=8,
             fs=fs, fft_mode="onesided", mfft=8, phase_shift=0),
        dict(name="odd-padded-negative-phase", signal=noise,
             window=windows.hann(15, sym=False), hop=5, fs=fs,
             fft_mode="onesided", mfft=21, phase_shift=-7),
        dict(name="twosided-padded", signal=sine, window=windows.hann(16, sym=False),
             hop=4, fs=fs, fft_mode="twosided", mfft=24, phase_shift=3),
        dict(name="centered-odd-mfft", signal=noise,
             window=windows.hamming(16, sym=False), hop=8, fs=fs,
             fft_mode="centered", mfft=17, phase_shift=-16),
        dict(name="onesided2x-magnitude", signal=sine,
             window=windows.hann(16, sym=False), hop=4, fs=fs,
             fft_mode="onesided2X", mfft=20, phase_shift=0,
             scale_to="magnitude"),
        dict(name="explicit-dual-window", signal=noise,
             window=windows.hamming(15, sym=False), hop=5, fs=fs,
             fft_mode="onesided", mfft=18, phase_shift=4,
             explicit_dual=True),
    ]


def run_case(case: dict) -> dict:
    scale_to = case.get("scale_to")
    initial = ShortTimeFFT(
        case["window"],
        case["hop"],
        case["fs"],
        fft_mode=case["fft_mode"],
        mfft=case["mfft"],
        scale_to=scale_to,
        phase_shift=case["phase_shift"],
    )

    # onesided2X requires scaling in SciPy. Passing the effective windows to
    # Rust tests the transform without adding a second scaling API.
    rust_window = initial.win.copy()
    dual_win = initial.dual_win.copy() if case.get("explicit_dual") or scale_to else None
    oracle = ShortTimeFFT(
        rust_window,
        case["hop"],
        case["fs"],
        fft_mode=case["fft_mode"],
        mfft=case["mfft"],
        dual_win=dual_win,
        phase_shift=case["phase_shift"],
    ) if case["fft_mode"] != "onesided2X" else initial

    scipy_stft = oracle.stft(case["signal"])
    scipy_istft = oracle.istft(scipy_stft, k0=0, k1=len(case["signal"]))
    rust = run_rust(case, rust_window, dual_win)
    rust_stft = decode_stft(rust["stft"])
    rust_istft = np.asarray(rust["istft"], dtype=np.float64)

    expected_properties = {
        "m_num": oracle.m_num,
        "f_pts": oracle.f_pts,
        "p_min": oracle.p_min,
        "p_max": oracle.p_max(len(case["signal"])),
        "mfft": oracle.mfft,
        "hop": oracle.hop,
        "fs": oracle.fs,
    }
    if rust["properties"] != expected_properties:
        raise AssertionError(
            f"property mismatch: expected {expected_properties}, got {rust['properties']}"
        )
    if rust_stft.shape != scipy_stft.shape:
        raise AssertionError(
            f"STFT shape mismatch: expected {scipy_stft.shape}, got {rust_stft.shape}"
        )
    if rust_istft.shape != scipy_istft.shape:
        raise AssertionError(
            f"ISTFT shape mismatch: expected {scipy_istft.shape}, got {rust_istft.shape}"
        )

    np.testing.assert_allclose(rust_stft, scipy_stft, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(rust_istft, scipy_istft.real, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(rust_istft, case["signal"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(rust["frequencies"], oracle.f, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        rust["times"], oracle.t(len(case["signal"])), rtol=RTOL, atol=ATOL
    )

    return {
        "name": case["name"],
        "passed": True,
        "shape": list(scipy_stft.shape),
        "stft": metrics(scipy_stft, rust_stft),
        "istft": metrics(scipy_istft.real, rust_istft),
    }


def main() -> int:
    if scipy.__version__ != EXPECTED_SCIPY:
        raise RuntimeError(
            f"SciPy {EXPECTED_SCIP} is required, found {scipy.__version__}"
        )

    results = []
    for case in create_cases():
        try:
            result = run_case(case)
            print(f"PASS {case['name']}: max STFT diff {result['stft']['max_abs']:.3e}")
            results.append(result)
        except Exception as error:
            print(f"FAIL {case['name']}: {error}", file=sys.stderr)
            results.append({"name": case["name"], "passed": False, "error": str(error)})

    passed = sum(result["passed"] for result in results)
    report = {
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "summary": {"total": len(results), "passed": passed, "failed": len(results) - passed},
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{passed}/{len(results)} compatibility cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
