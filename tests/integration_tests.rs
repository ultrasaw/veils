use num_complex::Complex;
use spectrust::StandaloneSTFT;

/// Convert STFT result from [time][freq] to [freq][time] format for ISTFT
fn transpose_stft(stft_result: &[Vec<Complex<f64>>]) -> Vec<Vec<Complex<f64>>> {
    if stft_result.is_empty() || stft_result[0].is_empty() {
        return vec![];
    }

    let time_slices = stft_result.len();
    let freq_bins = stft_result[0].len();
    let mut transposed = vec![vec![Complex::new(0.0, 0.0); time_slices]; freq_bins];

    for (t, row) in stft_result.iter().enumerate() {
        for (f, value) in row.iter().enumerate() {
            transposed[f][t] = *value;
        }
    }

    transposed
}

/// Create a Hann window of given length
fn hann_window(length: usize) -> Vec<f64> {
    (0..length)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (length - 1) as f64).cos()))
        .collect()
}

/// Test data structure for reproducible tests
#[derive(Debug)]
struct TestSignal {
    name: &'static str,
    data: Vec<f64>,
    expected_error: f64,
}

/// Generate the three standard test signals used for validation
fn generate_test_signals() -> Vec<TestSignal> {
    vec![
        // 1. Impulse signal - most critical test for STFT accuracy
        TestSignal {
            name: "impulse",
            data: {
                let mut impulse = vec![0.0; 64];
                impulse[32] = 1.0;
                impulse
            },
            expected_error: 0.0, // Should be perfect
        },
        // 2. Sine wave - tests harmonic content
        TestSignal {
            name: "sine_wave",
            data: (0..64)
                .map(|i| (2.0 * std::f64::consts::PI * 5.0 * i as f64 / 1000.0).sin())
                .collect(),
            expected_error: 1e-15, // Machine precision
        },
        // 3. Chirp signal - tests time-varying frequency content
        TestSignal {
            name: "chirp",
            data: {
                let t: Vec<f64> = (0..64).map(|i| i as f64 / 1000.0).collect();
                t.iter()
                    .map(|&t_val| (2.0 * std::f64::consts::PI * (5.0 + 10.0 * t_val) * t_val).sin())
                    .collect()
            },
            expected_error: 1e-15, // Machine precision
        },
    ]
}

#[test]
fn test_stft_creation() {
    let window = hann_window(16);
    let stft = StandaloneSTFT::new(
        window,
        4,      // hop length
        1000.0, // sampling rate
        Some("onesided"),
        None, // mfft
        None, // dual_win
        None, // phase_shift
    );

    assert!(stft.is_ok(), "STFT creation should succeed");

    let stft = stft.unwrap();
    assert_eq!(stft.hop(), 4);
    assert_eq!(stft.fs(), 1000.0);
    assert_eq!(stft.mfft(), 16);
}

#[test]
fn test_stft_invalid_parameters() {
    // Empty window should fail
    let result = StandaloneSTFT::new(vec![], 4, 1000.0, Some("onesided"), None, None, None);
    assert!(result.is_err());

    // Zero hop should fail
    let window = hann_window(16);
    let result = StandaloneSTFT::new(
        window,
        0, // invalid hop
        1000.0,
        Some("onesided"),
        None,
        None,
        None,
    );
    assert!(result.is_err());
}

#[test]
fn test_fft_mode_parsing() {
    let window = hann_window(16);

    // Test all valid FFT modes
    let modes = ["twosided", "centered", "onesided", "onesided2X"];
    for mode in &modes {
        let stft = StandaloneSTFT::new(window.clone(), 4, 1000.0, Some(mode), None, None, None);
        assert!(stft.is_ok(), "Mode {} should be valid", mode);
    }

    // Test invalid mode
    let stft = StandaloneSTFT::new(window, 4, 1000.0, Some("invalid_mode"), None, None, None);
    assert!(stft.is_err());
}

#[test]
fn test_perfect_reconstruction_impulse() {
    let window = hann_window(16);
    let mut stft = StandaloneSTFT::new(
        window,
        4,      // hop length
        1000.0, // sampling rate
        Some("onesided"),
        None,
        None,
        None,
    )
    .unwrap();

    // Create impulse signal
    let mut signal = vec![0.0; 64];
    signal[32] = 1.0;

    // Forward STFT
    let stft_result = stft.stft(&signal, None, None, None).unwrap();

    // Transpose for ISTFT (convert [time][freq] to [freq][time])
    let stft_transposed = transpose_stft(&stft_result);

    // Inverse STFT
    let reconstructed = stft.istft(&stft_transposed, None, None).unwrap();

    // Check perfect reconstruction
    let min_len = signal.len().min(reconstructed.len());
    let error: f64 = signal[..min_len]
        .iter()
        .zip(reconstructed[..min_len].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    assert!(
        error < 1e-14,
        "Impulse reconstruction error {} should be < 1e-14",
        error
    );
}

#[test]
fn test_perfect_reconstruction_sine_wave() {
    let window = hann_window(16);
    let mut stft =
        StandaloneSTFT::new(window, 4, 1000.0, Some("onesided"), None, None, None).unwrap();

    // Create sine wave
    let signal: Vec<f64> = (0..64)
        .map(|i| (2.0 * std::f64::consts::PI * 5.0 * i as f64 / 1000.0).sin())
        .collect();

    // Forward STFT
    let stft_result = stft.stft(&signal, None, None, None).unwrap();

    // Transpose for ISTFT (convert [time][freq] to [freq][time])
    let stft_transposed = transpose_stft(&stft_result);

    // Inverse STFT
    let reconstructed = stft.istft(&stft_transposed, None, None).unwrap();

    // Check reconstruction accuracy
    let min_len = signal.len().min(reconstructed.len());
    let error: f64 = signal[..min_len]
        .iter()
        .zip(reconstructed[..min_len].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    assert!(
        error < 1e-15,
        "Sine wave reconstruction error {} should be < 1e-15",
        error
    );
}

#[test]
fn test_perfect_reconstruction_chirp() {
    let window = hann_window(16);
    let mut stft =
        StandaloneSTFT::new(window, 4, 1000.0, Some("onesided"), None, None, None).unwrap();

    // Create chirp signal
    let t: Vec<f64> = (0..64).map(|i| i as f64 / 1000.0).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&t_val| (2.0 * std::f64::consts::PI * (5.0 + 10.0 * t_val) * t_val).sin())
        .collect();

    // Forward STFT
    let stft_result = stft.stft(&signal, None, None, None).unwrap();

    // Transpose for ISTFT (convert [time][freq] to [freq][time])
    let stft_transposed = transpose_stft(&stft_result);

    // Inverse STFT
    let reconstructed = stft.istft(&stft_transposed, None, None).unwrap();

    // Check reconstruction accuracy
    let min_len = signal.len().min(reconstructed.len());
    let error: f64 = signal[..min_len]
        .iter()
        .zip(reconstructed[..min_len].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    assert!(
        error < 1e-15,
        "Chirp reconstruction error {} should be < 1e-15",
        error
    );
}

/// Comprehensive test using all three standard test signals
#[test]
fn test_all_signals_perfect_reconstruction() {
    let test_signals = generate_test_signals();
    let window = hann_window(16);

    for test_signal in test_signals {
        println!("Testing signal: {}", test_signal.name);

        let mut stft = StandaloneSTFT::new(
            window.clone(),
            4,
            1000.0,
            Some("onesided"),
            None,
            None,
            None,
        )
        .unwrap();

        // Forward STFT
        let stft_result = stft.stft(&test_signal.data, None, None, None).unwrap();

        // Verify STFT dimensions
        assert!(!stft_result.is_empty(), "STFT result should not be empty");
        assert!(
            !stft_result[0].is_empty(),
            "STFT frequency bins should not be empty"
        );

        // Transpose for ISTFT (convert [time][freq] to [freq][time])
        let stft_transposed = transpose_stft(&stft_result);

        // Inverse STFT
        let reconstructed = stft.istft(&stft_transposed, None, None).unwrap();

        // Check reconstruction accuracy
        let min_len = test_signal.data.len().min(reconstructed.len());
        let error: f64 = test_signal.data[..min_len]
            .iter()
            .zip(reconstructed[..min_len].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        println!("  Reconstruction error: {:.2e}", error);
        assert!(
            error <= test_signal.expected_error,
            "Signal {} reconstruction error {:.2e} exceeds expected {:.2e}",
            test_signal.name,
            error,
            test_signal.expected_error
        );
    }
}

#[test]
fn test_different_fft_modes() {
    let window = hann_window(16);
    let signal: Vec<f64> = (0..64)
        .map(|i| (2.0 * std::f64::consts::PI * 5.0 * i as f64 / 1000.0).sin())
        .collect();

    let modes = ["twosided", "centered", "onesided", "onesided2X"];

    for mode in &modes {
        println!("Testing FFT mode: {}", mode);

        let mut stft =
            StandaloneSTFT::new(window.clone(), 4, 1000.0, Some(mode), None, None, None).unwrap();

        // Forward STFT
        let stft_result = stft.stft(&signal, None, None, None).unwrap();

        // Verify dimensions based on mode (format: [time_slices][freq_bins])
        let expected_freq_bins = match *mode {
            "onesided" | "onesided2X" => 16 / 2 + 1, // 9 bins
            _ => 16,                                 // Full spectrum
        };

        assert_eq!(
            stft_result[0].len(),
            expected_freq_bins,
            "Mode {} should have {} frequency bins",
            mode,
            expected_freq_bins
        );

        // Transpose for ISTFT (convert [time][freq] to [freq][time])
        let stft_transposed = transpose_stft(&stft_result);

        // Inverse STFT
        let reconstructed = stft.istft(&stft_transposed, None, None).unwrap();

        // Check reconstruction
        let min_len = signal.len().min(reconstructed.len());
        let error: f64 = signal[..min_len]
            .iter()
            .zip(reconstructed[..min_len].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        println!("  Reconstruction error: {:.2e}", error);
        assert!(
            error < 1e-14,
            "Mode {} reconstruction error should be < 1e-14",
            mode
        );
    }
}

#[test]
fn test_stft_properties() {
    let window = hann_window(16);
    let stft = StandaloneSTFT::new(window, 4, 1000.0, Some("onesided"), None, None, None).unwrap();

    // Test basic properties
    assert_eq!(stft.hop(), 4);
    assert_eq!(stft.fs(), 1000.0);
    assert_eq!(stft.mfft(), 16);
    assert_eq!(stft.m_num(), 16);
    assert_eq!(stft.m_num_mid(), 8);
    assert_eq!(stft.f_pts(), 9); // onesided: mfft/2 + 1
    assert!(stft.onesided_fft());

    // Test frequency axis
    let freqs = stft.f();
    assert_eq!(freqs.len(), 9);
    assert_eq!(freqs[0], 0.0); // DC component
    assert!((freqs[8] - 500.0).abs() < 1e-10); // Nyquist frequency
}

#[test]
fn test_time_axis() {
    let window = hann_window(16);
    let stft = StandaloneSTFT::new(window, 4, 1000.0, Some("onesided"), None, None, None).unwrap();

    let signal_length = 64;
    let time_axis = stft.t(signal_length, None, None, None).unwrap();

    assert!(!time_axis.is_empty());
    assert_eq!(time_axis[0], stft.p_min() as f64 * stft.delta_t());
}

/// Test that ensures the same input always produces the same output (deterministic)
#[test]
fn test_deterministic_behavior() {
    let window = hann_window(16);
    let signal: Vec<f64> = (0..64)
        .map(|i| (2.0 * std::f64::consts::PI * 5.0 * i as f64 / 1000.0).sin())
        .collect();

    // Run STFT multiple times
    let mut results = Vec::new();
    for _ in 0..3 {
        let stft = StandaloneSTFT::new(
            window.clone(),
            4,
            1000.0,
            Some("onesided"),
            None,
            None,
            None,
        )
        .unwrap();

        let stft_result = stft.stft(&signal, None, None, None).unwrap();
        results.push(stft_result);
    }

    // Verify all results are identical
    for i in 1..results.len() {
        assert_eq!(results[0].len(), results[i].len());
        for (slice1, slice2) in results[0].iter().zip(results[i].iter()) {
            assert_eq!(slice1.len(), slice2.len());
            for (val1, val2) in slice1.iter().zip(slice2.iter()) {
                assert!((val1.re - val2.re).abs() < 1e-15);
                assert!((val1.im - val2.im).abs() < 1e-15);
            }
        }
    }
}
