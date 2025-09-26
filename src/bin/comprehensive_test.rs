use spectrust::StandaloneSTFT;
use num_complex::Complex;
use serde_json;
use std::fs;
use std::collections::HashMap;

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
}

#[derive(serde::Serialize)]
struct StftValueComparison {
    rust_real: f64,
    rust_imag: f64,
    python_real: f64,
    python_imag: f64,
    difference: f64,
}

fn calculate_stft_difference(rust_stft: &[Vec<Complex<f64>>], python_stft_real: &[Vec<f64>], python_stft_imag: &[Vec<f64>]) -> f64 {
    let mut total_diff = 0.0;
    let mut count = 0;
    
    // Compare all values
    for t in 0..rust_stft.len().min(python_stft_real[0].len()) {
        for f in 0..rust_stft[0].len().min(python_stft_real.len()) {
            let rust_val = rust_stft[t][f];
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

fn test_signal(signal_name: &str, test_data: &TestSignal) -> Result<ComparisonResult, Box<dyn std::error::Error>> {
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
    let mut rust_stft_python_format = vec![vec![Complex::new(0.0, 0.0); rust_stft.len()]; rust_stft[0].len()];
    for t in 0..rust_stft.len() {
        for f in 0..rust_stft[0].len() {
            rust_stft_python_format[f][t] = rust_stft[t][f];
        }
    }
    
    // Perform Rust ISTFT with correct format
    let mut stft_mut = stft;
    let rust_reconstructed = stft_mut.istft(&rust_stft_python_format, None, None)?;
    
    // Calculate Rust reconstruction error
    let min_len = test_data.signal.len().min(rust_reconstructed.len());
    let mut rust_error_sum = 0.0;
    let mut signal_sum = 0.0;
    
    for i in 0..min_len {
        rust_error_sum += (test_data.signal[i] - rust_reconstructed[i]).abs();
        signal_sum += test_data.signal[i].abs();
    }
    
    let rust_abs_error = rust_error_sum / min_len as f64;
    let rust_rel_error = rust_abs_error / (signal_sum / min_len as f64);
    
    // Python STFT is already in [freq][time] format, convert to Vec<Vec<Complex<f64>>>
    let mut python_stft_native = Vec::new();
    for f in 0..test_data.stft_real.len() {
        let mut freq_slice = Vec::new();
        for t in 0..test_data.stft_real[0].len() {
            freq_slice.push(Complex::new(test_data.stft_real[f][t], test_data.stft_imag[f][t]));
        }
        python_stft_native.push(freq_slice);
    }
    
    // Cross-check: Rust ISTFT with Python STFT data
    let cross_check_reconstructed = stft_mut.istft(&python_stft_native, None, None)?;
    let mut cross_check_error_sum = 0.0;
    let cross_check_len = test_data.reconstructed.len().min(cross_check_reconstructed.len());
    
    for i in 0..cross_check_len {
        cross_check_error_sum += (test_data.reconstructed[i] - cross_check_reconstructed[i]).abs();
    }
    let istft_cross_check_error = cross_check_error_sum / cross_check_len as f64;
    
    // Calculate STFT matching error
    let stft_match_error = calculate_stft_difference(&rust_stft, &test_data.stft_real, &test_data.stft_imag);
    
    // Check properties match
    let properties_match = 
        stft_mut.m_num() == test_data.stft_properties.m_num &&
        stft_mut.f_pts() == test_data.stft_properties.f_pts &&
        stft_mut.p_min() == test_data.stft_properties.p_min &&
        stft_mut.p_max(test_data.signal.len()) == test_data.stft_properties.p_max;
    
    // Get first few STFT values for detailed comparison
    let mut first_few_values = Vec::new();
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
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Comprehensive STFT Implementation Test");
    println!("=====================================");
    
    // Load test signals
    let test_data_str = fs::read_to_string("comparison_results/test_signals.json")
        .expect("Could not read test_signals.json - run Python comparison script first");
    
    let test_signals: HashMap<String, TestSignal> = serde_json::from_str(&test_data_str)?;
    
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
    println!("\n{:<15} {:<12} {:<12} {:<12} {:<12} {:<8}", 
             "Signal", "Rust Error", "Python Error", "STFT Diff", "ISTFT Check", "Props");
    println!("{}", "-".repeat(85));
    
    for result in &all_results {
        let rust_status = if result.rust_abs_error < 1e-10 { "✅" } 
                         else if result.rust_abs_error < 1e-6 { "⚠️" } 
                         else { "❌" };
        
        let stft_status = if result.stft_match_error < 1e-10 { "✅" } 
                         else if result.stft_match_error < 1e-6 { "⚠️" } 
                         else { "❌" };
        
        let istft_status = if result.istft_cross_check_error < 1e-10 { "✅" } 
                          else if result.istft_cross_check_error < 1e-6 { "⚠️" } 
                          else { "❌" };
        
        let props_status = if result.properties_match { "✅" } else { "❌" };
        
        println!("{:<15} {:<12.2e} {:<12.2e} {:<12.2e} {:<12.2e} {:<8}", 
                 result.signal_name,
                 result.rust_abs_error,
                 result.python_abs_error,
                 result.stft_match_error,
                 result.istft_cross_check_error,
                 props_status);
    }
    
    // Detailed analysis for first signal
    if let Some(result) = all_results.first() {
        println!("\nDetailed STFT comparison for {}:", result.signal_name);
        println!("{:<8} {:<25} {:<25} {:<12}", "Bin", "Rust", "Python", "Difference");
        println!("{}", "-".repeat(75));
        
        for (i, val_comp) in result.first_few_stft_values.iter().enumerate() {
            println!("{:<8} {:<25} {:<25} {:<12.2e}", 
                     i, 
                     format!("{:.6}+{:.6}i", val_comp.rust_real, val_comp.rust_imag),
                     format!("{:.6}+{:.6}i", val_comp.python_real, val_comp.python_imag),
                     val_comp.difference);
        }
    }
    
    // Save results
    let results_json = serde_json::to_string_pretty(&all_results)?;
    fs::write("comparison_results/rust_results.json", results_json)?;
    
    // Overall assessment
    let perfect_count = all_results.iter().filter(|r| r.rust_abs_error < 1e-10).count();
    let good_count = all_results.iter().filter(|r| r.rust_abs_error < 1e-6).count();
    
    println!("\nOverall Assessment:");
    println!("Perfect reconstruction (< 1e-10): {}/{}", perfect_count, all_results.len());
    println!("Good reconstruction (< 1e-6): {}/{}", good_count, all_results.len());
    
    if perfect_count == all_results.len() {
        println!("🎉 All signals perfectly reconstructed!");
    } else if good_count == all_results.len() {
        println!("✅ All signals well reconstructed!");
    } else {
        println!("⚠️ Some signals have reconstruction issues - check individual results");
    }
    
    println!("\nResults saved to comparison_results/rust_results.json");
    
    Ok(())
}
