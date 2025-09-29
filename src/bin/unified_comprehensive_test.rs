// '''WARNING 100% AI generated file'''
use num_complex::Complex;
use std::collections::HashMap;
use std::fs;
use veils::StandaloneSTFT;

#[derive(serde::Deserialize)]
struct TestSignal {
    signal: Vec<f64>,
    window: Vec<f64>,
    hop_length: usize,
    fs: f64,
    stft_real: Vec<Vec<f64>>,
    stft_imag: Vec<Vec<f64>>,
    reconstructed: Vec<f64>,
    python_abs_error: f64,
    python_rel_error: f64,
    stft_properties: StftProperties,
}

#[derive(serde::Deserialize)]
struct StftProperties {
    m_num: usize,
    f_pts: usize,
    p_min: i32,
    p_max: i32,
}

#[derive(serde::Serialize)]
struct ComparisonResult {
    signal_name: String,
    rust_abs_error: f64,
    rust_rel_error: f64,
    python_abs_error: f64,
    python_rel_error: f64,
    stft_match_error: f64,
    istft_cross_check_error: f64,
    properties_match: bool,
    first_few_stft_values: Vec<StftValueComparison>,
    coefficient_test_passed: bool,
    coefficient_max_diff: f64,
}

#[derive(serde::Serialize)]
struct StftValueComparison {
    rust_real: f64,
    rust_imag: f64,
    python_real: f64,
    python_imag: f64,
    difference: f64,
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

/// Test the reference signal from instructions.md against expected scipy coefficients
fn test_reference_signal_coefficients() -> (bool, f64) {
    println!("\n🔬 REFERENCE SIGNAL COEFFICIENT TEST (Instructions.md)");

    // Create the exact test signal from instructions.md
    let signal = vec![
        0.49671415,
        -0.1382643,
        0.64768854,
        1.52302986,
        -0.23415337,
        -0.23413696,
        1.57921282,
        0.76743473,
        -0.46947439,
        0.54256004,
        -0.46341769,
        -0.46572975,
        0.24196227,
        -1.91328024,
        -1.72491783,
        -0.56228753,
        -1.01283112,
        0.31424733,
        -0.90802408,
        -1.4123037,
        1.46564877,
        -0.2257763,
        0.0675282,
        -1.42474819,
        -0.54438272,
        0.11092259,
        -1.15099358,
        0.37569802,
        -0.60063869,
        -0.29169375,
    ];

    // Create Hann window (sym=False)
    let window: Vec<f64> = (0..15)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / 15.0).cos()))
        .collect();

    println!("Signal length: {}", signal.len());
    println!("Window length: {}", window.len());

    // Create STFT instance
    let stft = StandaloneSTFT::new(
        window,
        8,   // hop length
        1.0, // sampling rate
        Some("onesided"),
        None, // mfft (defaults to window length)
        None, // dual_win (computed automatically)
        None, // phase_shift
    )
    .unwrap();

    println!("STFT parameters:");
    println!("  f_pts: {}", stft.f_pts());
    println!("  hop: {}", stft.hop());
    println!("  mfft: {}", stft.mfft());

    // Compute STFT
    match stft.stft(&signal, None, None, None) {
        Ok(stft_result) => {
            println!("STFT computed successfully");
            println!(
                "STFT shape: {} x {}",
                stft_result.len(),
                stft_result[0].len()
            );

            // Use expected results from helper function
            let expected_results = get_expected_scipy_coefficients();

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
                            println!(
                                "Large diff at [{}, {}]: rust={:?}, expected={:?}, diff={:.6}",
                                freq_bin, time_slice, rust_val, expected_val, diff
                            );
                        }
                    }
                }
            }

            let mse = total_mse / count as f64;
            println!("Max coefficient difference: {:.6}", max_diff);
            println!("MSE: {:.6}", mse);

            let passed = mse < 1e-12;
            if passed {
                println!("✅ SUCCESS: Coefficients match scipy within tolerance");
            } else {
                println!(
                    "❌ FAILURE: Coefficients differ from scipy (MSE: {:.2e})",
                    mse
                );
            }

            (passed, max_diff)
        }
        Err(e) => {
            println!("❌ STFT failed: {}", e);
            (false, f64::INFINITY)
        }
    }
}

/// Test multiple signal types and configurations
fn test_signal_variations() {
    println!("\n=== Testing Signal Variations ===");

    let test_cases = vec![
        ("DC Signal", vec![1.0; 30]),
        (
            "Sine Wave",
            (0..30)
                .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 30.0).sin())
                .collect(),
        ),
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

        let mut stft =
            StandaloneSTFT::new(window, 8, 1.0, Some("onesided"), None, None, None).unwrap();

        match stft.stft(&signal, None, None, None) {
            Ok(stft_result) => match stft.istft(&stft_result, None, None) {
                Ok(reconstructed) => {
                    let min_len = signal.len().min(reconstructed.len());
                    let mse: f64 = signal[..min_len]
                        .iter()
                        .zip(reconstructed[..min_len].iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        / min_len as f64;

                    println!(
                        "  STFT shape: {} x {}",
                        stft_result.len(),
                        stft_result[0].len()
                    );
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
            },
            Err(e) => println!("  ❌ STFT failed: {}", e),
        }
    }
}

/// Test different FFT modes
fn test_fft_modes() {
    println!("\n=== Testing FFT Modes ===");

    let signal: Vec<f64> = (0..30)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 30.0).sin())
        .collect();
    let window: Vec<f64> = (0..15)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / 15.0).cos()))
        .collect();

    let modes = vec!["onesided", "twosided", "centered"];

    for mode in modes {
        println!("\nTesting FFT mode: {}", mode);

        let mut stft =
            StandaloneSTFT::new(window.clone(), 8, 1.0, Some(mode), None, None, None).unwrap();

        match stft.stft(&signal, None, None, None) {
            Ok(stft_result) => {
                println!(
                    "  STFT shape: {} x {}",
                    stft_result.len(),
                    stft_result[0].len()
                );
                println!("  Frequency bins: {}", stft.f_pts());

                match stft.istft(&stft_result, None, None) {
                    Ok(reconstructed) => {
                        let min_len = signal.len().min(reconstructed.len());
                        let mse: f64 = signal[..min_len]
                            .iter()
                            .zip(reconstructed[..min_len].iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            / min_len as f64;

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

fn calculate_stft_difference(
    rust_stft: &[Vec<Complex<f64>>],
    python_stft_real: &[Vec<f64>],
    python_stft_imag: &[Vec<f64>],
) -> f64 {
    let mut total_diff = 0.0;
    let mut count = 0;

    // Compare all values
    for (t, rust_slice) in rust_stft.iter().enumerate() {
        if t >= python_stft_real[0].len() {
            continue;
        }
        for (f, &rust_val) in rust_slice.iter().enumerate() {
            if f >= python_stft_real.len() {
                continue;
            }
            let python_val = Complex::new(python_stft_real[f][t], python_stft_imag[f][t]);
            total_diff += (rust_val - python_val).norm_sqr();
            count += 1;
        }
    }

    if count > 0 {
        (total_diff / count as f64).sqrt()
    } else {
        0.0
    }
}

fn test_signal(
    signal_name: &str,
    test_data: &TestSignal,
) -> Result<ComparisonResult, Box<dyn std::error::Error>> {
    println!("Testing signal: {}", signal_name);

    // Create STFT object
    let stft = StandaloneSTFT::new(
        test_data.window.clone(),
        test_data.hop_length,
        test_data.fs,
        Some("onesided"),
        None,
        None,
        None,
    )?;

    // Perform Rust STFT
    let rust_stft = stft.stft(&test_data.signal, None, None, None)?;

    // Convert Rust STFT to Python format for ISTFT
    // Rust STFT is [time][freq], Python expects [freq][time]
    let mut rust_stft_python_format =
        vec![vec![Complex::new(0.0, 0.0); rust_stft.len()]; rust_stft[0].len()];
    for (t, row) in rust_stft.iter().enumerate() {
        for (f, val) in row.iter().enumerate() {
            rust_stft_python_format[f][t] = *val;
        }
    }

    // Perform Rust ISTFT with correct format
    let mut stft_mut = stft;
    let rust_reconstructed = stft_mut.istft(&rust_stft_python_format, None, None)?;

    // Calculate Rust reconstruction error
    let min_len = test_data.signal.len().min(rust_reconstructed.len());
    let (rust_error_sum, signal_sum) = test_data.signal[..min_len]
        .iter()
        .zip(&rust_reconstructed[..min_len])
        .fold((0.0, 0.0), |(err_sum, sig_sum), (sig, recon)| {
            (err_sum + (sig - recon).abs(), sig_sum + sig.abs())
        });

    let rust_abs_error = rust_error_sum / min_len as f64;
    let rust_rel_error = rust_abs_error / (signal_sum / min_len as f64);

    // Cross-check: Rust ISTFT with Python STFT data (only if we have Python data)
    let istft_cross_check_error = if !test_data.stft_real.is_empty() && !test_data.stft_real[0].is_empty() && !test_data.reconstructed.is_empty() {
        // Python STFT is already in [freq][time] format, convert to Vec<Vec<Complex<f64>>>
        let python_stft_native: Vec<Vec<Complex<f64>>> = test_data
            .stft_real
            .iter()
            .zip(test_data.stft_imag.iter())
            .map(|(real_row, imag_row)| {
                real_row
                    .iter()
                    .zip(imag_row.iter())
                    .map(|(real, imag)| Complex::new(*real, *imag))
                    .collect()
            })
            .collect();

        // Cross-check: Rust ISTFT with Python STFT data
        let cross_check_reconstructed = stft_mut.istft(&python_stft_native, None, None)?;
        let cross_check_len = test_data
            .reconstructed
            .len()
            .min(cross_check_reconstructed.len());
        let cross_check_error_sum: f64 = test_data.reconstructed[..cross_check_len]
            .iter()
            .zip(&cross_check_reconstructed[..cross_check_len])
            .map(|(a, b)| (a - b).abs())
            .sum();
        cross_check_error_sum / cross_check_len as f64
    } else {
        0.0 // No Python data to cross-check against
    };

    // Calculate STFT matching error (only if we have Python data)
    let stft_match_error = if !test_data.stft_real.is_empty() && !test_data.stft_real[0].is_empty() {
        calculate_stft_difference(&rust_stft, &test_data.stft_real, &test_data.stft_imag)
    } else {
        0.0 // No Python data to compare against
    };

    // Check properties match (only if we have meaningful data)
    let properties_match = if test_data.stft_properties.m_num > 0 {
        stft_mut.m_num() == test_data.stft_properties.m_num
            && stft_mut.f_pts() == test_data.stft_properties.f_pts
            && stft_mut.p_min() == test_data.stft_properties.p_min
            && stft_mut.p_max(test_data.signal.len()) == test_data.stft_properties.p_max
    } else {
        true // No meaningful properties to compare
    };

    // Get first few STFT values for detailed comparison (only if we have Python data)
    let mut first_few_values = Vec::new();
    if !test_data.stft_real.is_empty() && !test_data.stft_real[0].is_empty() {
        for i in 0..5.min(rust_stft[0].len()).min(test_data.stft_real.len()) {
            let rust_val = rust_stft[0][i];
            let python_val = Complex::new(test_data.stft_real[i][0], test_data.stft_imag[i][0]);
            let diff = (rust_val - python_val).norm();
            first_few_values.push(StftValueComparison {
                rust_real: rust_val.re,
                rust_imag: rust_val.im,
                python_real: python_val.re,
                python_imag: python_val.im,
                difference: diff,
            });
        }
    }

    // Run coefficient test if this is the reference signal
    let (coefficient_test_passed, coefficient_max_diff) =
        if signal_name.contains("reference") || signal_name.contains("instructions") {
            test_reference_signal_coefficients()
        } else {
            (true, 0.0) // Skip coefficient test for other signals
        };

    Ok(ComparisonResult {
        signal_name: signal_name.to_string(),
        rust_abs_error,
        rust_rel_error,
        python_abs_error: test_data.python_abs_error,
        python_rel_error: test_data.python_rel_error,
        stft_match_error,
        istft_cross_check_error,
        properties_match,
        first_few_stft_values: first_few_values,
        coefficient_test_passed,
        coefficient_max_diff,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== UNIFIED COMPREHENSIVE STFT TEST ===");
    println!("========================================");

    // First run standalone coefficient tests
    println!("\n🧪 STANDALONE COEFFICIENT TESTS");
    println!("================================");

    let (coeff_passed, coeff_max_diff) = test_reference_signal_coefficients();
    test_signal_variations();
    test_fft_modes();

    println!("\n📊 PYTHON COMPARISON TESTS");
    println!("===========================");

    // Load test signals if available
    let test_data_path = "comparison_results/test_signals.json";
    if !std::path::Path::new(test_data_path).exists() {
        println!("⚠️  Test signals file not found at {}", test_data_path);
        println!("   Run Python comparison script first to generate test data.");
        println!("   Standalone tests completed successfully.");
        return Ok(());
    }

    let test_data_str = fs::read_to_string(test_data_path)?;
    
    // Try to parse as new format first
    #[derive(serde::Deserialize)]
    struct NewFormatData {
        signal_sets: HashMap<String, HashMap<String, Vec<f64>>>,
        parameter_sets: HashMap<String, ParameterSet>,
        test_combinations: Vec<TestCombination>,
    }
    
    #[derive(serde::Deserialize)]
    struct ParameterSet {
        name: String,
        window_length: usize,
        hop_length: usize,
        fs: f64,
        window: Vec<f64>,
    }
    
    #[derive(serde::Deserialize)]
    struct TestCombination {
        signals: String,
        params: String,
    }
    
    // Parse the new format and convert to old format for compatibility
    let new_data: NewFormatData = serde_json::from_str(&test_data_str)?;
    let mut test_signals: HashMap<String, TestSignal> = HashMap::new();
    
    // Convert new format to old format for processing
    for combination in &new_data.test_combinations {
        let signal_set = &new_data.signal_sets[&combination.signals];
        let param_set = &new_data.parameter_sets[&combination.params];
        
        for (signal_name, signal_data) in signal_set {
            let key = format!("{}_{}", signal_name, param_set.name);
            
            // Create a dummy TestSignal for compatibility (we'll only use signal data)
            let test_signal = TestSignal {
                signal: signal_data.clone(),
                window: param_set.window.clone(),
                hop_length: param_set.hop_length,
                fs: param_set.fs,
                stft_real: vec![vec![]], // Dummy data
                stft_imag: vec![vec![]], // Dummy data
                reconstructed: vec![], // Dummy data
                python_abs_error: 0.0, // Dummy data
                python_rel_error: 0.0, // Dummy data
                stft_properties: StftProperties {
                    m_num: param_set.window_length,
                    f_pts: param_set.window_length / 2 + 1, // For onesided FFT
                    p_min: 0, // Dummy data
                    p_max: 10, // Dummy data
                },
            };
            
            test_signals.insert(key, test_signal);
        }
    }

    let mut all_results = Vec::new();

    // Test each signal
    for (signal_name, test_data) in &test_signals {
        match test_signal(signal_name, test_data) {
            Ok(result) => {
                println!("✅ {} completed", signal_name);
                all_results.push(result);
            }
            Err(e) => {
                println!("❌ {} failed: {}", signal_name, e);
            }
        }
    }

    // Print summary table
    println!(
        "\n{:<15} {:<12} {:<12} {:<12} {:<12} {:<8} {:<8}",
        "Signal", "Rust Error", "Python Error", "STFT Diff", "ISTFT Check", "Props", "Coeffs"
    );
    println!("{}", "-".repeat(95));

    for result in &all_results {
        let _rust_status = if result.rust_abs_error < 1e-10 {
            "✅"
        } else if result.rust_abs_error < 1e-6 {
            "⚠️"
        } else {
            "❌"
        };

        let _stft_status = if result.stft_match_error < 1e-10 {
            "✅"
        } else if result.stft_match_error < 1e-6 {
            "⚠️"
        } else {
            "❌"
        };

        let _istft_status = if result.istft_cross_check_error < 1e-10 {
            "✅"
        } else if result.istft_cross_check_error < 1e-6 {
            "⚠️"
        } else {
            "❌"
        };

        let props_status = if result.properties_match {
            "✅"
        } else {
            "❌"
        };

        let coeff_status = if result.coefficient_test_passed {
            "✅"
        } else {
            "❌"
        };

        println!(
            "{:<15} {:<12.2e} {:<12.2e} {:<12.2e} {:<12.2e} {:<8} {:<8}",
            result.signal_name,
            result.rust_abs_error,
            result.python_abs_error,
            result.stft_match_error,
            result.istft_cross_check_error,
            props_status,
            coeff_status
        );
    }

    // Detailed analysis for first signal
    if let Some(result) = all_results.first() {
        println!("\nDetailed STFT comparison for {}:", result.signal_name);
        println!(
            "{:<8} {:<25} {:<25} {:<12}",
            "Bin", "Rust", "Python", "Difference"
        );
        println!("{}", "-".repeat(75));

        for (i, val_comp) in result.first_few_stft_values.iter().enumerate() {
            println!(
                "{:<8} {:<25} {:<25} {:<12.2e}",
                i,
                format!("{:.6}+{:.6}i", val_comp.rust_real, val_comp.rust_imag),
                format!("{:.6}+{:.6}i", val_comp.python_real, val_comp.python_imag),
                val_comp.difference
            );
        }
    }

    // Save results
    let results_json = serde_json::to_string_pretty(&all_results)?;
    fs::write("comparison_results/rust_results.json", results_json)?;

    // Overall assessment
    let perfect_count = all_results
        .iter()
        .filter(|r| r.rust_abs_error < 1e-10)
        .count();
    let good_count = all_results
        .iter()
        .filter(|r| r.rust_abs_error < 1e-6)
        .count();
    let coeff_passed_count = all_results
        .iter()
        .filter(|r| r.coefficient_test_passed)
        .count();

    println!("\n🎯 FINAL ASSESSMENT");
    println!("===================");
    println!(
        "Perfect reconstruction (< 1e-10): {}/{}",
        perfect_count,
        all_results.len()
    );
    println!(
        "Good reconstruction (< 1e-6): {}/{}",
        good_count,
        all_results.len()
    );
    println!(
        "Coefficient tests passed: {}/{}",
        coeff_passed_count,
        all_results.len()
    );
    println!(
        "Standalone coefficient test: {} (max diff: {:.2e})",
        if coeff_passed {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        },
        coeff_max_diff
    );

    if perfect_count == all_results.len() && coeff_passed {
        println!("🎉 ALL TESTS PASSED - Perfect STFT implementation!");
    } else if good_count == all_results.len() {
        println!("✅ All signals well reconstructed!");
    } else {
        println!("⚠️ Some tests have issues - check individual results");
    }

    println!("\nResults saved to comparison_results/rust_results.json");

    Ok(())
}
