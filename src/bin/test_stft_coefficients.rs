use veils::StandaloneSTFT;
use num_complex::Complex;

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

/// Test multiple signal types and configurations
fn test_signal_variations() {
    println!("\n=== Testing Signal Variations ===");
    
    let test_cases = vec![
        ("DC Signal", vec![1.0; 30]),
        ("Sine Wave", (0..30).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 30.0).sin()).collect()),
        ("Impulse", {
            let mut imp = vec![0.0; 30];
            imp[15] = 1.0;
            imp
        }),
    ];
    
    for (name, signal) in test_cases {
        println!("\nTesting {}", name);
        
        let window: Vec<f64> = (0..15)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / 15.0).cos()))
            .collect();
        
        let mut stft = StandaloneSTFT::new(
            window, 8, 1.0, Some("onesided"), None, None, None
        ).unwrap();
        
        match stft.stft(&signal, None, None, None) {
            Ok(stft_result) => {
                match stft.istft(&stft_result, None, None) {
                    Ok(reconstructed) => {
                        let min_len = signal.len().min(reconstructed.len());
                        let mse: f64 = signal[..min_len]
                            .iter()
                            .zip(reconstructed[..min_len].iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>() / min_len as f64;
                        
                        println!("  STFT shape: {} x {}", stft_result.len(), stft_result[0].len());
                        println!("  Reconstruction MSE: {:.2e}", mse);
                        
                        if mse < 1e-25 {
                            println!("  ✅ PERFECT reconstruction");
                        } else if mse < 1e-12 {
                            println!("  ✅ EXCELLENT reconstruction");
                        } else {
                            println!("  ❌ POOR reconstruction");
                        }
                    }
                    Err(e) => println!("  ❌ ISTFT failed: {}", e),
                }
            }
            Err(e) => println!("  ❌ STFT failed: {}", e),
        }
    }
}

/// Test different FFT modes
fn test_fft_modes() {
    println!("\n=== Testing FFT Modes ===");
    
    let signal: Vec<f64> = (0..30).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 30.0).sin()).collect();
    let window: Vec<f64> = (0..15)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / 15.0).cos()))
        .collect();
    
    let modes = vec!["onesided", "twosided", "centered"];
    
    for mode in modes {
        println!("\nTesting FFT mode: {}", mode);
        
        let mut stft = StandaloneSTFT::new(
            window.clone(), 8, 1.0, Some(mode), None, None, None
        ).unwrap();
        
        match stft.stft(&signal, None, None, None) {
            Ok(stft_result) => {
                println!("  STFT shape: {} x {}", stft_result.len(), stft_result[0].len());
                println!("  Frequency bins: {}", stft.f_pts());
                
                match stft.istft(&stft_result, None, None) {
                    Ok(reconstructed) => {
                        let min_len = signal.len().min(reconstructed.len());
                        let mse: f64 = signal[..min_len]
                            .iter()
                            .zip(reconstructed[..min_len].iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>() / min_len as f64;
                        
                        println!("  Reconstruction MSE: {:.2e}", mse);
                        
                        if mse < 1e-25 {
                            println!("  ✅ PERFECT");
                        } else if mse < 1e-12 {
                            println!("  ✅ EXCELLENT");
                        } else {
                            println!("  ❌ POOR");
                        }
                    }
                    Err(e) => println!("  ❌ ISTFT failed: {}", e),
                }
            }
            Err(e) => println!("  ❌ STFT failed: {}", e),
        }
    }
}

fn main() {
    println!("=== Comprehensive STFT Coefficient Testing ===");
    
    // Test 1: Reference signal from instructions.md
    println!("\n🔬 REFERENCE SIGNAL TEST (Instructions.md)");
    
    // Create the exact test signal from instructions.md
    let signal = vec![
        0.49671415, -0.1382643, 0.64768854, 1.52302986, -0.23415337, -0.23413696,
        1.57921282, 0.76743473, -0.46947439, 0.54256004, -0.46341769, -0.46572975,
        0.24196227, -1.91328024, -1.72491783, -0.56228753, -1.01283112, 0.31424733,
        -0.90802408, -1.4123037, 1.46564877, -0.2257763, 0.0675282, -1.42474819,
        -0.54438272, 0.11092259, -1.15099358, 0.37569802, -0.60063869, -0.29169375
    ];
    
    // Create Hann window (sym=False)
    let window: Vec<f64> = (0..15)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / 15.0).cos()))
        .collect();
    
    println!("Signal length: {}", signal.len());
    println!("Window length: {}", window.len());
    
    // Create STFT instance
    let mut stft = StandaloneSTFT::new(
        window,
        8,      // hop length
        1.0,    // sampling rate
        Some("onesided"),
        None,   // mfft (defaults to window length)
        None,   // dual_win (computed automatically)
        None,   // phase_shift
    ).unwrap();
    
    println!("STFT parameters:");
    println!("  f_pts: {}", stft.f_pts());
    println!("  hop: {}", stft.hop());
    println!("  mfft: {}", stft.mfft());
    
    // Compute STFT
    match stft.stft(&signal, None, None, None) {
        Ok(stft_result) => {
            println!("STFT computed successfully");
            println!("STFT shape: {} x {}", stft_result.len(), stft_result[0].len());
            
            // Print intermediate coefficients for comparison with scipy
            println!("\nRust STFT coefficients:");
            for freq_bin in 0..3.min(stft_result.len()) {
                let real_parts: Vec<f64> = stft_result[freq_bin].iter().map(|c| c.re).collect();
                let imag_parts: Vec<f64> = stft_result[freq_bin].iter().map(|c| c.im).collect();
                println!("Real part freq bin {}: {:?}", freq_bin, real_parts);
                println!("Imag part freq bin {}: {:?}", freq_bin, imag_parts);
            }
            
            // Use expected results from helper function
            let expected_results = get_expected_scipy_coefficients();
            
            println!("\nExpected scipy coefficients:");
            for freq_bin in 0..3 {
                let real_parts: Vec<f64> = expected_results[freq_bin].iter().map(|c| c.re).collect();
                let imag_parts: Vec<f64> = expected_results[freq_bin].iter().map(|c| c.im).collect();
                println!("Real part freq bin {}: {:?}", freq_bin, real_parts);
                println!("Imag part freq bin {}: {:?}", freq_bin, imag_parts);
            }
            
            // Calculate differences
            println!("\nCoefficient differences:");
            let mut max_diff: f64 = 0.0;
            let mut total_mse: f64 = 0.0;
            let mut count = 0;
            
            for freq_bin in 0..3 {
                for time_slice in 0..5 {
                    if freq_bin < stft_result.len() && time_slice < stft_result[freq_bin].len() {
                        let rust_val = stft_result[freq_bin][time_slice];
                        let expected_val = expected_results[freq_bin][time_slice];
                        let diff = (rust_val - expected_val).norm();
                        max_diff = max_diff.max(diff);
                        total_mse += diff * diff;
                        count += 1;
                        
                        if diff > 0.1 {
                            println!("Large diff at [{}, {}]: rust={:?}, expected={:?}, diff={:.6}", 
                                   freq_bin, time_slice, rust_val, expected_val, diff);
                        }
                    }
                }
            }
            
            let mse = total_mse / count as f64;
            println!("Max coefficient difference: {:.6}", max_diff);
            println!("MSE: {:.6}", mse);
            
            // Test round-trip reconstruction
            let mut stft_mut = stft;
            match stft_mut.istft(&stft_result, None, None) {
                Ok(reconstructed) => {
                    println!("\nISTFT computed successfully");
                    println!("Reconstructed length: {}", reconstructed.len());
                    
                    // Calculate reconstruction error
                    let min_len = signal.len().min(reconstructed.len());
                    let mut reconstruction_mse = 0.0;
                    for i in 0..min_len {
                        let diff = signal[i] - reconstructed[i];
                        reconstruction_mse += diff * diff;
                    }
                    reconstruction_mse /= min_len as f64;
                    
                    println!("Reconstruction MSE: {:.2e}", reconstruction_mse);
                    
                    if mse < 1e-12 {
                        println!("✅ SUCCESS: Coefficients match scipy within tolerance");
                    } else {
                        println!("❌ FAILURE: Coefficients differ from scipy (MSE: {:.2e})", mse);
                    }
                    
                    if reconstruction_mse < 1e-12 {
                        println!("✅ SUCCESS: Perfect reconstruction");
                    } else {
                        println!("❌ FAILURE: Poor reconstruction (MSE: {:.2e})", reconstruction_mse);
                    }
                }
                Err(e) => {
                    println!("❌ ISTFT failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("❌ STFT failed: {}", e);
        }
    }
    
    // Run additional tests
    test_signal_variations();
    test_fft_modes();
    
    println!("\n🎉 Comprehensive testing complete!");
}
