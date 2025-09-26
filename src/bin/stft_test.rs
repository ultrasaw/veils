use spectrust::StandaloneSTFT;
use num_complex::Complex;
use serde_json;
use std::fs;

#[derive(serde::Deserialize)]
struct TestData {
    signal: Vec<f64>,
    window: Vec<f64>,
    hop_length: usize,
    fs: f64,
    stft_real: Vec<Vec<f64>>,
    stft_imag: Vec<Vec<f64>>,
    reconstructed: Vec<f64>,
    stft_properties: StftProperties,
}

#[derive(serde::Deserialize)]
struct StftProperties {
    m_num: usize,
    m_num_mid: usize,
    f_pts: usize,
    p_min: i32,
    p_max: i32,
    mfft: usize,
}

fn generate_random_walk(n_samples: usize, seed: u64) -> Vec<f64> {
    // Simple linear congruential generator for reproducible results
    let mut rng_state = seed;
    let mut signal = vec![0.0; n_samples];
    let mut cumsum = 0.0;
    
    for i in 0..n_samples {
        // LCG: next = (a * current + c) % m
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let uniform = (rng_state as f64) / (u64::MAX as f64);
        // Box-Muller transform for normal distribution
        let normal = if i % 2 == 0 {
            let u1 = uniform;
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u2 = (rng_state as f64) / (u64::MAX as f64);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        } else {
            let u1 = uniform;
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u2 = (rng_state as f64) / (u64::MAX as f64);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin()
        };
        
        cumsum += normal;
        signal[i] = cumsum;
    }
    
    signal
}

fn create_hann_window(length: usize) -> Vec<f64> {
    (0..length)
        .map(|n| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / (length - 1) as f64).cos()))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running Rust STFT test...");
    
    // Test parameters (matching Python)
    let n_samples = 1000;
    let window_length = 256;
    let hop_length = 64;
    let fs = 1000.0;
    
    // Load Python test data for exact comparison
    let test_data_str = fs::read_to_string("test_data.json")
        .expect("Could not read test_data.json - run Python test first");
    let test_data: TestData = serde_json::from_str(&test_data_str)?;
    
    // Use the exact same data as Python
    let signal = test_data.signal.clone();
    let window = test_data.window.clone();
    
    println!("Signal length: {}", signal.len());
    println!("Window length: {}", window.len());
    println!("Hop length: {}", hop_length);
    println!("Sampling rate: {}", fs);
    
    // Create STFT object
    let stft = StandaloneSTFT::new(
        window,
        hop_length,
        fs,
        Some("onesided"),
        None,
        None,
        None,
    )?;
    
    println!("STFT properties:");
    println!("  m_num: {}", stft.m_num());
    println!("  m_num_mid: {}", stft.m_num_mid());
    println!("  f_pts: {}", stft.f_pts());
    println!("  p_min: {}", stft.p_min());
    println!("  p_max: {}", stft.p_max(n_samples));
    
    // Perform STFT
    println!("\nPerforming STFT...");
    let stft_result = stft.stft(&signal, None, None, None)?;
    println!("STFT shape: {} x {}", stft_result[0].len(), stft_result.len());
    
    // Perform ISTFT
    println!("Performing ISTFT...");
    let mut stft_mut = stft;
    
    // Convert Python STFT format (freq, time) to Rust format (time, freq) for comparison
    let mut python_stft_transposed = Vec::new();
    for t in 0..test_data.stft_real[0].len() {
        let mut time_slice = Vec::new();
        for f in 0..test_data.stft_real.len() {
            time_slice.push(Complex::new(test_data.stft_real[f][t], test_data.stft_imag[f][t]));
        }
        python_stft_transposed.push(time_slice);
    }
    
    let reconstructed = stft_mut.istft(&stft_result, None, None)?;
    let python_reconstructed = stft_mut.istft(&python_stft_transposed, None, None)?;
    println!("Reconstructed signal length: {}", reconstructed.len());
    
    // Calculate reconstruction error (compare with original signal)
    let min_len = signal.len().min(reconstructed.len());
    let mut error_sum = 0.0;
    let mut signal_sum = 0.0;
    
    for i in 0..min_len {
        error_sum += (signal[i] - reconstructed[i]).abs();
        signal_sum += signal[i].abs();
    }
    
    let error = error_sum / min_len as f64;
    let relative_error = error / (signal_sum / min_len as f64);
    
    println!("\nRust STFT->ISTFT reconstruction error:");
    println!("  Absolute error: {:.2e}", error);
    println!("  Relative error: {:.2e}", relative_error);
    
    // Also check Python reconstruction error for comparison
    let min_py_len = signal.len().min(test_data.reconstructed.len());
    let mut py_error_sum = 0.0;
    for i in 0..min_py_len {
        py_error_sum += (signal[i] - test_data.reconstructed[i]).abs();
    }
    let py_error = py_error_sum / min_py_len as f64;
    let py_relative_error = py_error / (signal_sum / min_py_len as f64);
    
    println!("Python STFT->ISTFT reconstruction error:");
    println!("  Absolute error: {:.2e}", py_error);
    println!("  Relative error: {:.2e}", py_relative_error);
    
    // Compare with Python results (data already loaded)
    println!("\nComparing with Python results...");
        
        // Compare properties
        println!("Property comparison:");
        println!("  m_num: Rust={}, Python={}", stft_mut.m_num(), test_data.stft_properties.m_num);
        println!("  f_pts: Rust={}, Python={}", stft_mut.f_pts(), test_data.stft_properties.f_pts);
        println!("  p_min: Rust={}, Python={}", stft_mut.p_min(), test_data.stft_properties.p_min);
        println!("  p_max: Rust={}, Python={}", stft_mut.p_max(n_samples), test_data.stft_properties.p_max);
        
        // Compare STFT results (first few values)
        // Note: Python STFT shape is (freq, time), Rust is (time, freq)
        if !stft_result.is_empty() && !test_data.stft_real.is_empty() {
            println!("\nSTFT comparison (first 3 frequency bins, first time slice):");
            for i in 0..3.min(stft_result[0].len()).min(test_data.stft_real.len()) {
                let rust_val = stft_result[0][i]; // First time slice, i-th frequency bin
                let python_val = Complex::new(test_data.stft_real[i][0], test_data.stft_imag[i][0]); // i-th frequency bin, first time slice
                let diff = (rust_val - python_val).norm();
                println!("  Bin {}: Rust={:.6}, Python={:.6}, Diff={:.2e}", 
                         i, rust_val, python_val, diff);
            }
        }
        
        // Compare reconstructed signals
        let min_recon_len = python_reconstructed.len().min(test_data.reconstructed.len());
        let mut recon_diff_sum = 0.0;
        for i in 0..min_recon_len {
            recon_diff_sum += (python_reconstructed[i] - test_data.reconstructed[i]).abs();
        }
        let recon_diff = recon_diff_sum / min_recon_len as f64;
        println!("\nReconstruction comparison (using Python STFT data):");
        println!("  Average difference: {:.2e}", recon_diff);
        
        // Also compare Rust STFT -> ISTFT vs Python STFT -> ISTFT
        let min_rust_len = reconstructed.len().min(python_reconstructed.len());
        let mut rust_diff_sum = 0.0;
        for i in 0..min_rust_len {
            rust_diff_sum += (reconstructed[i] - python_reconstructed[i]).abs();
        }
        let rust_diff = rust_diff_sum / min_rust_len as f64;
        println!("Rust STFT->ISTFT vs Python STFT->ISTFT difference: {:.2e}", rust_diff);
        
        if recon_diff < 1e-10 {
            println!("✅ Rust and Python implementations match within numerical precision!");
        } else if recon_diff < 1e-6 {
            println!("⚠️  Small differences detected, but within acceptable range");
        } else {
            println!("❌ Significant differences detected - implementation may need review");
        }
    
    println!("\nRust STFT test completed!");
    Ok(())
}
