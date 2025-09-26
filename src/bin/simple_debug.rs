use num_complex::Complex;
use serde_json;
use spectrust::StandaloneSTFT;
use std::fs;

#[derive(serde::Deserialize)]
struct SimpleDebugData {
    signal: Vec<f64>,
    window: Vec<f64>,
    hop: usize,
    fs: f64,
    stft_real: Vec<Vec<f64>>,
    stft_imag: Vec<Vec<f64>>,
    reconstructed: Vec<f64>,
    properties: Properties,
    analysis: Analysis,
}

#[derive(serde::Deserialize)]
struct Properties {
    m_num: usize,
    f_pts: usize,
    p_min: i32,
    p_max: i32,
    mfft: usize,
}

#[derive(serde::Deserialize)]
struct Analysis {
    original_impulse_idx: usize,
    reconstructed_peak_idx: usize,
    reconstructed_peak_val: f64,
    reconstruction_error: f64,
    nonzero_time_slices: Vec<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Simple Rust ISTFT Debug Test");
    println!("============================");

    // Load debug data
    let debug_data_str = fs::read_to_string("simple_debug_case.json")
        .expect("Could not read simple_debug_case.json - run Python debug script first");

    let debug_data: SimpleDebugData = serde_json::from_str(&debug_data_str)?;

    println!("Loaded debug data:");
    println!("  Signal length: {}", debug_data.signal.len());
    println!("  Window length: {}", debug_data.window.len());
    println!("  Hop: {}", debug_data.hop);
    println!(
        "  Original impulse at index: {}",
        debug_data.analysis.original_impulse_idx
    );
    println!(
        "  Python reconstruction error: {:.2e}",
        debug_data.analysis.reconstruction_error
    );

    // Create Rust STFT
    let stft = StandaloneSTFT::new(
        debug_data.window.clone(),
        debug_data.hop,
        debug_data.fs,
        Some("onesided"),
        None,
        None,
        None,
    )?;

    println!("\nRust STFT properties:");
    println!(
        "  m_num: {} (Python: {})",
        stft.m_num(),
        debug_data.properties.m_num
    );
    println!(
        "  f_pts: {} (Python: {})",
        stft.f_pts(),
        debug_data.properties.f_pts
    );
    println!(
        "  p_min: {} (Python: {})",
        stft.p_min(),
        debug_data.properties.p_min
    );
    println!(
        "  p_max: {} (Python: {})",
        stft.p_max(debug_data.signal.len()),
        debug_data.properties.p_max
    );

    // Test Rust STFT forward
    let rust_stft = stft.stft(&debug_data.signal, None, None, None)?;
    println!(
        "\nRust STFT shape: {} x {}",
        rust_stft[0].len(),
        rust_stft.len()
    );
    println!(
        "Python STFT shape: {} x {}",
        debug_data.stft_real.len(),
        debug_data.stft_real[0].len()
    );

    // Compare STFT results
    let mut max_stft_diff = 0.0f64;
    for t in 0..rust_stft.len().min(debug_data.stft_real[0].len()) {
        for f in 0..rust_stft[0].len().min(debug_data.stft_real.len()) {
            let rust_val = rust_stft[t][f];
            let python_val = Complex::new(debug_data.stft_real[f][t], debug_data.stft_imag[f][t]);
            let diff = (rust_val - python_val).norm();
            max_stft_diff = max_stft_diff.max(diff);
        }
    }
    println!("Max STFT difference: {:.2e}", max_stft_diff);

    // Test Rust ISTFT with Python data
    println!("\n=== Testing Rust ISTFT with Python STFT data ===");

    // Convert Python STFT to Rust format
    let mut python_stft_native = Vec::new();
    for f in 0..debug_data.stft_real.len() {
        let mut freq_slice = Vec::new();
        for t in 0..debug_data.stft_real[0].len() {
            freq_slice.push(Complex::new(
                debug_data.stft_real[f][t],
                debug_data.stft_imag[f][t],
            ));
        }
        python_stft_native.push(freq_slice);
    }

    let mut stft_mut = stft;

    // Debug dual window
    let dual_win = stft_mut.dual_win()?;
    println!("Rust dual window (first 5 values): {:?}", &dual_win[..5]);
    println!(
        "Rust dual window sum: {:.6}",
        dual_win.iter().map(|x| x.re).sum::<f64>()
    );
    println!(
        "Rust window sum: {:.6}",
        debug_data.window.iter().sum::<f64>()
    );
    println!(
        "Rust dual window max: {:.6}",
        dual_win.iter().map(|x| x.re).fold(0.0f64, |a, b| a.max(b))
    );
    println!(
        "Rust dual window min: {:.6}",
        dual_win
            .iter()
            .map(|x| x.re)
            .fold(f64::INFINITY, |a, b| a.min(b))
    );

    let rust_reconstructed_from_python = stft_mut.istft(&python_stft_native, None, None)?;

    println!(
        "Rust reconstruction length: {}",
        rust_reconstructed_from_python.len()
    );
    println!(
        "Python reconstruction length: {}",
        debug_data.reconstructed.len()
    );

    // Compare reconstructions
    let min_len = rust_reconstructed_from_python
        .len()
        .min(debug_data.reconstructed.len());
    let mut max_recon_diff = 0.0f64;
    let mut error_sum = 0.0;

    for i in 0..min_len {
        let diff = (rust_reconstructed_from_python[i] - debug_data.reconstructed[i]).abs();
        max_recon_diff = max_recon_diff.max(diff);
        error_sum += diff;
    }

    let mean_recon_diff = error_sum / min_len as f64;

    println!("Reconstruction comparison (Rust ISTFT with Python STFT):");
    println!("  Max difference: {:.2e}", max_recon_diff);
    println!("  Mean difference: {:.2e}", mean_recon_diff);

    // Find peak in Rust reconstruction
    let rust_peak_idx = rust_reconstructed_from_python
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let rust_peak_val = rust_reconstructed_from_python[rust_peak_idx];

    println!("\nPeak analysis:");
    println!(
        "  Original impulse: index {}, value 1.0",
        debug_data.analysis.original_impulse_idx
    );
    println!(
        "  Python peak: index {}, value {:.6}",
        debug_data.analysis.reconstructed_peak_idx, debug_data.analysis.reconstructed_peak_val
    );
    println!(
        "  Rust peak: index {}, value {:.6}",
        rust_peak_idx, rust_peak_val
    );

    // Test Rust STFT -> ISTFT pipeline
    println!("\n=== Testing Full Rust STFT->ISTFT Pipeline ===");

    // Convert Rust STFT to Python format for ISTFT
    let mut rust_stft_python_format =
        vec![vec![Complex::new(0.0, 0.0); rust_stft.len()]; rust_stft[0].len()];
    for t in 0..rust_stft.len() {
        for f in 0..rust_stft[0].len() {
            rust_stft_python_format[f][t] = rust_stft[t][f];
        }
    }

    let rust_full_pipeline = stft_mut.istft(&rust_stft_python_format, None, None)?;

    // Compare with original signal
    let pipeline_min_len = debug_data.signal.len().min(rust_full_pipeline.len());
    let mut pipeline_error_sum = 0.0;

    for i in 0..pipeline_min_len {
        pipeline_error_sum += (debug_data.signal[i] - rust_full_pipeline[i]).abs();
    }

    let pipeline_error = pipeline_error_sum / pipeline_min_len as f64;

    println!("Full pipeline reconstruction error: {:.2e}", pipeline_error);

    // Assessment
    println!("\n=== ASSESSMENT ===");

    if max_stft_diff < 1e-10 {
        println!("✅ STFT Forward: Perfect match with Python");
    } else {
        println!(
            "❌ STFT Forward: Differences detected ({:.2e})",
            max_stft_diff
        );
    }

    if mean_recon_diff < 1e-10 {
        println!("✅ ISTFT Algorithm: Perfect match with Python");
    } else {
        println!(
            "❌ ISTFT Algorithm: Differences detected ({:.2e})",
            mean_recon_diff
        );
    }

    if pipeline_error < 1e-10 {
        println!("✅ Full Pipeline: Perfect reconstruction");
    } else {
        println!(
            "❌ Full Pipeline: Reconstruction error ({:.2e})",
            pipeline_error
        );
    }

    Ok(())
}
