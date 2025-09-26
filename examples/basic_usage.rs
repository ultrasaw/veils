use num_complex::Complex;
use spectrust_stft::StandaloneSTFT;

/// Create a Hann window of given length
fn hann_window(length: usize) -> Vec<f64> {
    (0..length)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (length - 1) as f64).cos()))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SpectRust Basic Usage Example");
    println!("============================");

    // 1. Create a window function (Hann window)
    let window = hann_window(16);
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
            "STFT shape: {} time slices × {} frequency bins",
            stft_result.len(),
            stft_result[0].len()
        );

        // Transpose STFT result for ISTFT (convert [time][freq] to [freq][time])
        let time_slices = stft_result.len();
        let freq_bins = stft_result[0].len();
        let mut stft_transposed = vec![vec![Complex::new(0.0, 0.0); time_slices]; freq_bins];

        for t in 0..time_slices {
            for f in 0..freq_bins {
                stft_transposed[f][t] = stft_result[t][f];
            }
        }

        // Inverse STFT (perfect reconstruction)
        let reconstructed = stft.istft(&stft_transposed, None, None)?;
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

    println!("\n🎉 SpectRust demonstration complete!");
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
