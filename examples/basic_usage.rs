// '''WARNING 100% AI generated file'''
use num_complex::Complex;
use veils::StandaloneSTFT;

/// Create a Hann window of given length (sym=False like scipy)
fn hann_window(length: usize) -> Vec<f64> {
    (0..length)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / length as f64).cos()))
        .collect()
}

/// Expected scipy coefficients for the reference signal (from instructions.md)
fn get_expected_scipy_coefficients() -> Vec<Vec<Complex<f64>>> {
    vec![
        // Freq bin 0 (DC)
        vec![
            Complex::new(2.16686938, 0.0),
            Complex::new(0.41507935, 0.0),
            Complex::new(-4.76289176, 0.0),
            Complex::new(-2.7497881, 0.0),
            Complex::new(-0.35620841, 0.0),
        ],
        // Freq bin 1
        vec![
            Complex::new(0.92149425, -1.42790582),
            Complex::new(1.45518835, 2.23154074),
            Complex::new(-3.19990722, -1.40842161),
            Complex::new(-2.04897287, 0.55131399),
            Complex::new(-0.01891355, -0.33500474),
        ],
        // Freq bin 2
        vec![
            Complex::new(-0.30760012, -0.98739154),
            Complex::new(0.71763657, 0.78718444),
            Complex::new(-0.05623971, -0.92249341),
            Complex::new(-1.36664598, -0.21423326),
            Complex::new(0.28685536, -0.05804994),
        ],
    ]
}

/// Create the reference test signal from instructions.md
fn create_reference_signal() -> Vec<f64> {
    // Using the exact same random seed and signal as in instructions.md
    // np.random.seed(42); np.random.randn(30)
    vec![
        0.49671415, -0.1382643, 0.64768854, 1.52302986, -0.23415337, -0.23413696,
        1.57921282, 0.76743473, -0.46947439, 0.54256004, -0.46341769, -0.46572975,
        0.24196227, -1.91328024, -1.72491783, -0.56228753, -1.01283112, 0.31424733,
        -0.90802408, -1.4123037, 1.46564877, -0.2257763, 0.0675282, -1.42474819,
        -0.54438272, 0.11092259, -1.15099358, 0.37569802, -0.60063869, -0.29169375
    ]
}

/// Test coefficient accuracy against scipy reference
fn test_coefficient_accuracy(stft_result: &[Vec<Complex<f64>>]) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Coefficient Accuracy Test ===");
    
    let expected = get_expected_scipy_coefficients();
    let mut max_diff = 0.0f64;
    let mut total_mse = 0.0f64;
    let mut count = 0;
    
    println!("Comparing first 3 frequency bins:");
    
    for freq_bin in 0..3.min(stft_result.len()) {
        println!("\nFreq bin {}:", freq_bin);
        
        let rust_real: Vec<f64> = stft_result[freq_bin].iter().map(|c| c.re).collect();
        let rust_imag: Vec<f64> = stft_result[freq_bin].iter().map(|c| c.im).collect();
        let expected_real: Vec<f64> = expected[freq_bin].iter().map(|c| c.re).collect();
        let expected_imag: Vec<f64> = expected[freq_bin].iter().map(|c| c.im).collect();
        
        println!("  Rust real:     {:?}", rust_real);
        println!("  Expected real: {:?}", expected_real);
        println!("  Rust imag:     {:?}", rust_imag);
        println!("  Expected imag: {:?}", expected_imag);
        
        for time_slice in 0..5.min(stft_result[freq_bin].len()) {
            let rust_val = stft_result[freq_bin][time_slice];
            let expected_val = expected[freq_bin][time_slice];
            let diff = (rust_val - expected_val).norm();
            max_diff = max_diff.max(diff);
            total_mse += diff * diff;
            count += 1;
            
            if diff > 1e-10 {
                println!("  Large diff at [{}, {}]: rust={:?}, expected={:?}, diff={:.2e}", 
                       freq_bin, time_slice, rust_val, expected_val, diff);
            }
        }
    }
    
    let mse = total_mse / count as f64;
    
    println!("\nAccuracy Results:");
    println!("  Max coefficient difference: {:.2e}", max_diff);
    println!("  MSE: {:.2e}", mse);
    
    if mse < 1e-12 {
        println!("  ✅ PERFECT: Coefficients match scipy exactly!");
        Ok(())
    } else if mse < 1e-6 {
        println!("  ✅ GOOD: Coefficients are very close to scipy");
        Ok(())
    } else {
        println!("  ❌ POOR: Significant differences from scipy");
        Err("Coefficient accuracy test failed".into())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Veils Basic Usage Example");
    println!("============================");

    // 0. FIRST: Test with reference signal from instructions.md for scipy compatibility
    println!("\n🔬 SCIPY COMPATIBILITY TEST");
    println!("Testing with reference signal from instructions.md...");
    
    let reference_signal = create_reference_signal();
    let reference_window = hann_window(15); // nperseg = 15 from instructions
    
    let mut reference_stft = StandaloneSTFT::new(
        reference_window,
        8,                // hop = 8 from instructions (nperseg - noverlap = 15 - 7)
        1.0,              // fs = 1.0 from instructions
        Some("onesided"), // fft_mode = 'onesided' from instructions
        None,             // mfft (defaults to window length = 15)
        None,             // dual_win (computed automatically)
        None,             // phase_shift (defaults to 0)
    )?;
    
    println!("Reference STFT parameters:");
    println!("  Window length: {}", reference_stft.m_num());
    println!("  Hop length: {}", reference_stft.hop());
    println!("  FFT length: {}", reference_stft.mfft());
    println!("  Frequency bins: {}", reference_stft.f_pts());
    
    // Compute STFT on reference signal
    let reference_stft_result = reference_stft.stft(&reference_signal, None, None, None)?;
    println!("Reference STFT shape: {} frequency bins × {} time slices", 
             reference_stft_result.len(), reference_stft_result[0].len());
    
    // Test coefficient accuracy against scipy
    test_coefficient_accuracy(&reference_stft_result)?;
    
    // Test reconstruction
    let reference_reconstructed = reference_stft.istft(&reference_stft_result, None, None)?;
    let min_len = reference_signal.len().min(reference_reconstructed.len());
    let reconstruction_mse: f64 = reference_signal[..min_len]
        .iter()
        .zip(reference_reconstructed[..min_len].iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>() / min_len as f64;
    
    println!("\nReconstruction Test:");
    println!("  Original length: {}", reference_signal.len());
    println!("  Reconstructed length: {}", reference_reconstructed.len());
    println!("  Reconstruction MSE: {:.2e}", reconstruction_mse);
    
    if reconstruction_mse < 1e-30 {
        println!("  ✅ PERFECT: Reconstruction is mathematically perfect!");
    } else if reconstruction_mse < 1e-12 {
        println!("  ✅ EXCELLENT: Reconstruction is near-perfect!");
    } else {
        println!("  ❌ POOR: Reconstruction has significant errors");
    }

    // 1. Create a window function (Hann window) for general examples
    let window = hann_window(16);
    println!("\n📊 GENERAL EXAMPLES");
    println!("Created Hann window of length {}", window.len());

    // 2. Create STFT instance
    let mut stft = StandaloneSTFT::new(
        window,
        4,                // hop length
        1000.0,           // sampling rate (Hz)
        Some("onesided"), // FFT mode
        None,             // mfft (defaults to window length)
        None,             // dual_win (computed automatically)
        None,             // phase_shift
    )?;

    println!("STFT parameters:");
    println!("  Hop length: {}", stft.hop());
    println!("  Sampling rate: {} Hz", stft.fs());
    println!("  FFT length: {}", stft.mfft());
    println!("  Frequency bins: {}", stft.f_pts());

    // 3. Generate test signals
    let signals = vec![
        ("Impulse", generate_impulse(64)),
        ("Sine Wave (5 Hz)", generate_sine_wave(64, 5.0, 1000.0)),
        ("Chirp (5-15 Hz)", generate_chirp(64, 5.0, 15.0, 1000.0)),
    ];

    // 4. Process each signal
    for (name, signal) in signals {
        println!("\nProcessing: {}", name);
        println!("Signal length: {}", signal.len());

        // Forward STFT
        let stft_result = stft.stft(&signal, None, None, None)?;
        println!(
            "STFT shape: {} frequency bins × {} time slices",
            stft_result.len(),
            stft_result[0].len()
        );

        // STFT now returns correct format [freq][time], no transpose needed
        // Inverse STFT (perfect reconstruction)
        let reconstructed = stft.istft(&stft_result, None, None)?;
        println!("Reconstructed signal length: {}", reconstructed.len());

        // Calculate reconstruction error
        let min_len = signal.len().min(reconstructed.len());
        let max_error = signal[..min_len]
            .iter()
            .zip(reconstructed[..min_len].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        let mean_error = signal[..min_len]
            .iter()
            .zip(reconstructed[..min_len].iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / min_len as f64;

        println!("Reconstruction quality:");
        println!("  Max error: {:.2e}", max_error);
        println!("  Mean error: {:.2e}", mean_error);

        if max_error < 1e-14 {
            println!("  ✅ Perfect reconstruction (machine precision)");
        } else if max_error < 1e-10 {
            println!("  ✅ Excellent reconstruction");
        } else {
            println!("  ⚠️  Reconstruction error above expected threshold");
        }
    }

    // 5. Demonstrate frequency and time axes
    println!("\nFrequency and Time Axes:");
    let freqs = stft.f();
    println!("Frequency bins: {:?}", freqs);

    let time_axis = stft.t(64, None, None, None)?;
    println!(
        "Time samples (first 5): {:?}",
        &time_axis[..5.min(time_axis.len())]
    );

    println!("\n🎉 Veils demonstration complete!");
    Ok(())
}

/// Generate an impulse signal
fn generate_impulse(length: usize) -> Vec<f64> {
    let mut signal = vec![0.0; length];
    if length > 0 {
        signal[length / 2] = 1.0;
    }
    signal
}

/// Generate a sine wave
fn generate_sine_wave(length: usize, frequency: f64, sample_rate: f64) -> Vec<f64> {
    (0..length)
        .map(|i| (2.0 * std::f64::consts::PI * frequency * i as f64 / sample_rate).sin())
        .collect()
}

/// Generate a chirp signal (frequency sweep)
fn generate_chirp(length: usize, f0: f64, f1: f64, sample_rate: f64) -> Vec<f64> {
    let duration = length as f64 / sample_rate;
    (0..length)
        .map(|i| {
            let t = i as f64 / sample_rate;
            let freq = f0 + (f1 - f0) * t / duration;
            (2.0 * std::f64::consts::PI * freq * t).sin()
        })
        .collect()
}
