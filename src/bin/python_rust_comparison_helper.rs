// '''WARNING 100% AI generated file'''
//! Helper binary for Python-Rust STFT comparison
//!
//! This binary reads test parameters from a JSON file, performs STFT/ISTFT
//! using the Rust implementation, and outputs results in JSON format
//! for comparison with the Python implementation.

use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use veils::StandaloneSTFT;

#[derive(Deserialize)]
struct TestInput {
    signal: Vec<f64>,
    window: Vec<f64>,
    hop_length: usize,
    fs: f64,
    fft_mode: String,
}

#[derive(Serialize)]
struct ComplexValue {
    real: f64,
    imag: f64,
}

impl From<Complex<f64>> for ComplexValue {
    fn from(c: Complex<f64>) -> Self {
        ComplexValue {
            real: c.re,
            imag: c.im,
        }
    }
}

#[derive(Serialize)]
struct TestOutput {
    stft: Vec<Vec<ComplexValue>>,
    istft: Vec<f64>,
    properties: StftProperties,
}

#[derive(Serialize)]
struct StftProperties {
    m_num: usize,
    f_pts: usize,
    p_min: i32,
    p_max: i32,
    mfft: usize,
    hop: usize,
    fs: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get input file path from command line
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <input_json_file>", args[0]);
        std::process::exit(1);
    }

    let input_file = &args[1];

    // Read and parse input
    let input_data = fs::read_to_string(input_file)?;
    let test_input: TestInput = serde_json::from_str(&input_data)?;

    // Create STFT instance
    let mut stft = StandaloneSTFT::new(
        test_input.window,
        test_input.hop_length,
        test_input.fs,
        Some(&test_input.fft_mode),
        None, // mfft (defaults to window length)
        None, // dual_win (computed automatically)
        None, // phase_shift
    )?;

    // Perform STFT
    let stft_result = stft.stft(&test_input.signal, None, None, None)?;

    // Convert STFT result to serializable format
    let stft_serializable: Vec<Vec<ComplexValue>> = stft_result
        .iter()
        .map(|freq_bin| freq_bin.iter().map(|&val| val.into()).collect())
        .collect();

    // Perform ISTFT
    let istft_result = stft.istft(&stft_result, None, None)?;

    // Collect properties
    let properties = StftProperties {
        m_num: stft.m_num(),
        f_pts: stft.f_pts(),
        p_min: stft.p_min(),
        p_max: stft.p_max(test_input.signal.len()),
        mfft: stft.mfft(),
        hop: stft.hop(),
        fs: stft.fs(),
    };

    // Create output
    let output = TestOutput {
        stft: stft_serializable,
        istft: istft_result,
        properties,
    };

    // Output JSON to stdout
    println!("{}", serde_json::to_string(&output)?);

    Ok(())
}
