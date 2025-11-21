import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ================== CONFIGURATION ==================
SAMPLE_RATE = 125  # Hz
CARDIAC_CYCLE_DURATION = 1.0  # seconds (60 bpm baseline)
TOTAL_DURATION = 120.0  # seconds
OUTPUT_SIGNAL_FILE = 'abp_signal.npy'
OUTPUT_LABELS_FILE = 'abp_labels.npy'

# ================== ABP WAVEFORM GENERATION ==================
def create_single_abp_cycle(duration=1.0, fs=125, 
                           systolic_peak=120, diastolic_trough=80, 
                           notch_depth=0.15):
    """
    Create a single realistic ABP (Arterial Blood Pressure) waveform cycle
    
    ABP waveform features:
    - Sharp systolic upstroke (rapid rise)
    - Systolic peak
    - Dicrotic notch (aortic valve closure)
    - Diastolic decay to trough
    
    Args:
        duration: Duration of one cardiac cycle in seconds
        fs: Sampling rate in Hz
        systolic_peak: Systolic pressure value
        diastolic_trough: Diastolic pressure value
        notch_depth: Depth of dicrotic notch (fraction of pulse pressure)
    
    Returns:
        time array, signal array, systolic_idx, notch_idx, trough_idx
    """
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)
    
    pulse_pressure = systolic_peak - diastolic_trough
    
    # Define key points in the cardiac cycle
    # Percentages of the cycle duration
    key_times = [
        0.0,      # Start (end of previous diastole)
        0.075,    # Mid way between peak and diastole
        0.15,     # Systolic peak (rapid upstroke)
        0.40,     # Dicrotic notch
        0.45,     # After notch (small bump)
        0.7,      # Midway between after notch bump and diastole
        1.0       # End of diastole (trough)
    ]
    
    key_values = [
        diastolic_trough,                                    # Start at trough
        0.5 * (diastolic_trough + systolic_peak),           # Midway upstroke
        systolic_peak,                                       # Systolic peak
        systolic_peak - notch_depth * pulse_pressure,       # Dicrotic notch (deep point)
        systolic_peak - notch_depth * pulse_pressure * 0.80, # Small bump after notch
        0.5 * (systolic_peak - notch_depth * pulse_pressure * 0.80 + diastolic_trough), # Midway to diastole
        diastolic_trough                                     # Diastolic trough
    ]
    
    # Create interpolation function
    interpolator = interp1d(key_times, key_values, kind='cubic')
    
    # Generate smooth waveform
    normalized_t = t / duration
    signal = interpolator(normalized_t)
    
    # Find indices of key features by searching in appropriate regions for actual peaks
    # Systolic peak: search for maximum in region around 0.15
    search_start = int(0.10 * n_samples)
    search_end = int(0.25 * n_samples)
    systolic_idx = search_start + np.argmax(signal[search_start:search_end])
    
    # Dicrotic notch: search for minimum in region around 0.40
    search_start = int(0.35 * n_samples)
    search_end = int(0.50 * n_samples)
    notch_idx = search_start + np.argmin(signal[search_start:search_end])
    
    # Diastolic trough: search for minimum in last 20% of cycle
    search_start = int(0.80 * n_samples)
    trough_idx = search_start + np.argmin(signal[search_start:])
    
    return t, signal, systolic_idx, notch_idx, trough_idx


def generate_abp_signal(total_duration=120.0, fs=125, 
                       cycle_duration=1.0,
                       add_noise=False, noise_std=1.5,
                       vary_parameters=True):
    """
    Generate a long ABP signal by repeating cardiac cycles
    
    Args:
        total_duration: Total duration in seconds
        fs: Sampling rate
        cycle_duration: Duration of one cardiac cycle
        add_noise: Whether to add physiological noise
        vary_parameters: Whether to vary BP and HR slightly over time
    
    Returns:
        signal, labels, feature_locations dict
    """
    total_samples = int(total_duration * fs)
    signal = np.zeros(total_samples)
    labels = np.zeros(total_samples, dtype=np.int64)
    
    # Track all feature locations
    systolic_peaks = []
    dicrotic_notches = []
    diastolic_troughs = []
    
    current_idx = 0
    cycle_count = 0
    
    while current_idx < total_samples:
        # Vary parameters slightly for realism
        if vary_parameters:
            # Simulate heart rate variability (±10%)
            current_cycle_duration = cycle_duration * np.random.uniform(0.90, 1.10)
            # Simulate BP variability
            systolic = np.random.uniform(115, 125)
            diastolic = np.random.uniform(75, 85)
        else:
            current_cycle_duration = cycle_duration
            systolic = 120
            diastolic = 80
        
        # Generate single cycle
        t_cycle, cycle_signal, sys_idx, notch_idx, trough_idx = create_single_abp_cycle(
            duration=current_cycle_duration,
            fs=fs,
            systolic_peak=systolic,
            diastolic_trough=diastolic,
            notch_depth=0.15
        )
        
        # Determine how many samples to copy
        cycle_samples = len(cycle_signal)
        samples_to_copy = min(cycle_samples, total_samples - current_idx)
        
        # Copy cycle to signal
        signal[current_idx:current_idx + samples_to_copy] = cycle_signal[:samples_to_copy]
        
        # Store feature locations (global indices)
        if current_idx + sys_idx < total_samples:
            systolic_peaks.append(current_idx + sys_idx)
        if current_idx + notch_idx < total_samples:
            dicrotic_notches.append(current_idx + notch_idx)
        if current_idx + trough_idx < total_samples:
            diastolic_troughs.append(current_idx + trough_idx)
        
        # Label landmarks (±3 samples window)
        for landmark_idx in [sys_idx, notch_idx, trough_idx]:
            global_idx = current_idx + landmark_idx
            if global_idx < total_samples:
                start = max(0, global_idx - 3)
                end = min(total_samples, global_idx + 4)
                labels[start:end] = 1
        
        current_idx += cycle_samples
        cycle_count += 1
    
    # Add physiological noise
    if add_noise:
        noise = np.random.normal(0, noise_std, total_samples)
        signal = signal + noise
    
    feature_locations = {
        'systolic_peaks': systolic_peaks,
        'dicrotic_notches': dicrotic_notches,
        'diastolic_troughs': diastolic_troughs
    }
    
    return signal.astype(np.float32), labels, feature_locations


# ================== VISUALIZATION ==================
def visualize_abp_waveform(signal, labels, feature_locations, fs=125, 
                          num_seconds=10, output_path='abp_waveform_inspection.png'):
    """
    Visualize the generated ABP signal with labeled features
    """
    num_samples = int(num_seconds * fs)
    signal_segment = signal[:num_samples]
    labels_segment = labels[:num_samples]
    
    t = np.arange(num_samples) / fs
    
    # Get features in this segment
    sys_peaks = [p for p in feature_locations['systolic_peaks'] if p < num_samples]
    notches = [n for n in feature_locations['dicrotic_notches'] if n < num_samples]
    troughs = [t_idx for t_idx in feature_locations['diastolic_troughs'] if t_idx < num_samples]
    
    # Find labeled regions
    landmark_idx = np.where(labels_segment == 1)[0]
    
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    
    # Plot 1: Single cycle close-up (first 2 seconds)
    cycle_samples = int(2 * fs)
    cycle_signal = signal[:cycle_samples]
    cycle_labels = labels[:cycle_samples]
    t_cycle = np.arange(cycle_samples) / fs
    
    sys_peaks_cycle = [p for p in feature_locations['systolic_peaks'] if p < cycle_samples]
    notches_cycle = [n for n in feature_locations['dicrotic_notches'] if n < cycle_samples]
    troughs_cycle = [t_idx for t_idx in feature_locations['diastolic_troughs'] if t_idx < cycle_samples]
    
    axes[0].plot(t_cycle, cycle_signal, 'b-', linewidth=2, label='ABP Signal')
    if sys_peaks_cycle:
        axes[0].scatter(np.array(sys_peaks_cycle) / fs, cycle_signal[sys_peaks_cycle], 
                       c='red', marker='^', s=300, label='Systolic Peaks', 
                       zorder=5, edgecolors='darkred', linewidth=2)
    if notches_cycle:
        axes[0].scatter(np.array(notches_cycle) / fs, cycle_signal[notches_cycle], 
                       c='orange', marker='s', s=300, label='Dicrotic Notches', 
                       zorder=5, edgecolors='darkorange', linewidth=2)
    if troughs_cycle:
        axes[0].scatter(np.array(troughs_cycle) / fs, cycle_signal[troughs_cycle], 
                       c='green', marker='v', s=300, label='Diastolic Troughs', 
                       zorder=5, edgecolors='darkgreen', linewidth=2)
    
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Pressure (mmHg)', fontsize=12)
    axes[0].set_title('ABP Waveform - Close-up View (First 2 Cardiac Cycles)', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Longer view with all features
    axes[1].plot(t, signal_segment, 'b-', linewidth=1.5, alpha=0.8, label='ABP Signal')
    if sys_peaks:
        axes[1].scatter(np.array(sys_peaks) / fs, signal_segment[sys_peaks], 
                       c='red', marker='^', s=150, label='Systolic Peaks', 
                       zorder=5, edgecolors='darkred', linewidth=1.5)
    if notches:
        axes[1].scatter(np.array(notches) / fs, signal_segment[notches], 
                       c='orange', marker='s', s=150, label='Dicrotic Notches', 
                       zorder=5, edgecolors='darkorange', linewidth=1.5)
    if troughs:
        axes[1].scatter(np.array(troughs) / fs, signal_segment[troughs], 
                       c='green', marker='v', s=150, label='Diastolic Troughs', 
                       zorder=5, edgecolors='darkgreen', linewidth=1.5)
    
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Pressure (mmHg)', fontsize=12)
    axes[1].set_title(f'ABP Signal with Detected Landmarks (First {num_seconds}s)', 
                     fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Signal with labeled regions
    axes[2].plot(t, signal_segment, 'b-', linewidth=1.5, alpha=0.7, label='ABP Signal')
    if len(landmark_idx) > 0:
        axes[2].fill_between(t, signal_segment.min() - 5, signal_segment.max() + 5, 
                            where=(labels_segment == 1), 
                            color='yellow', alpha=0.3, label='Labeled Landmark Regions')
        axes[2].scatter(landmark_idx / fs, signal_segment[landmark_idx], 
                       c='purple', marker='o', s=30, label='Landmark Labels', 
                       zorder=5, alpha=0.5)
    
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('Pressure (mmHg)', fontsize=12)
    axes[2].set_title('Signal with Labeled Landmark Regions (±3 sample windows)', 
                     fontsize=14, fontweight='bold')
    axes[2].legend(loc='upper right', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Full signal overview
    t_full = np.arange(len(signal)) / fs
    axes[3].plot(t_full, signal, 'b-', linewidth=0.5, alpha=0.8, label='ABP Signal')
    # Plot landmarks (downsample for visibility if too many)
    downsample = 5 if len(feature_locations['systolic_peaks']) > 200 else 1
    marker_size = 20 if downsample > 1 else 30
    
    # Systolic peaks
    axes[3].scatter(np.array(feature_locations['systolic_peaks'][::downsample]) / fs, 
                   signal[feature_locations['systolic_peaks'][::downsample]], 
                   c='red', marker='^', s=marker_size, label='Systolic Peaks', zorder=5, alpha=0.7)
    
    # Dicrotic notches
    axes[3].scatter(np.array(feature_locations['dicrotic_notches'][::downsample]) / fs, 
                   signal[feature_locations['dicrotic_notches'][::downsample]], 
                   c='orange', marker='s', s=marker_size, label='Dicrotic Notches', zorder=5, alpha=0.7)
    
    # Diastolic troughs
    axes[3].scatter(np.array(feature_locations['diastolic_troughs'][::downsample]) / fs, 
                   signal[feature_locations['diastolic_troughs'][::downsample]], 
                   c='green', marker='v', s=marker_size, label='Diastolic Troughs', zorder=5, alpha=0.7)
    
    axes[3].set_xlabel('Time (s)', fontsize=12)
    axes[3].set_ylabel('Pressure (mmHg)', fontsize=12)
    axes[3].set_title(f'Full ABP Signal Overview ({len(signal)/fs:.0f} seconds)', 
                     fontsize=14, fontweight='bold')
    axes[3].legend(loc='upper right', fontsize=10)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as '{output_path}'")
    plt.close()


# ================== MAIN ==================
def main():
    print("\n" + "="*70)
    print("ABP WAVEFORM DATA GENERATION")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  Sample Rate: {SAMPLE_RATE} Hz")
    print(f"  Cardiac Cycle Duration: {CARDIAC_CYCLE_DURATION} s (60 bpm)")
    print(f"  Total Duration: {TOTAL_DURATION} s")
    
    # Generate ABP signal
    print("\nGenerating ABP signal...")
    signal, labels, feature_locations = generate_abp_signal(
        total_duration=TOTAL_DURATION,
        fs=SAMPLE_RATE,
        cycle_duration=CARDIAC_CYCLE_DURATION,
        add_noise=False,
        vary_parameters=True
    )
    
    print(f"\nSignal generated successfully!")
    print(f"  Total samples: {len(signal)}")
    print(f"  Systolic peaks: {len(feature_locations['systolic_peaks'])}")
    print(f"  Dicrotic notches: {len(feature_locations['dicrotic_notches'])}")
    print(f"  Diastolic troughs: {len(feature_locations['diastolic_troughs'])}")
    print(f"  Total landmarks: {len(feature_locations['systolic_peaks']) + len(feature_locations['dicrotic_notches']) + len(feature_locations['diastolic_troughs'])}")
    print(f"  Landmark samples labeled: {np.sum(labels == 1)}")
    print(f"  Background samples: {np.sum(labels == 0)}")
    print(f"  Landmark ratio: {np.sum(labels == 1) / len(labels) * 100:.2f}%")
    
    # Save data
    print(f"\nSaving data...")
    np.save(OUTPUT_SIGNAL_FILE, signal)
    np.save(OUTPUT_LABELS_FILE, labels)
    print(f"  Signal saved to: {OUTPUT_SIGNAL_FILE}")
    print(f"  Labels saved to: {OUTPUT_LABELS_FILE}")
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_abp_waveform(signal, labels, feature_locations, SAMPLE_RATE, num_seconds=10)
    
    # Print statistics about cycles
    avg_cycle_duration = TOTAL_DURATION / len(feature_locations['systolic_peaks'])
    print(f"\nStatistics:")
    print(f"  Expected cycles: {int(TOTAL_DURATION / CARDIAC_CYCLE_DURATION)}")
    print(f"  Actual cycles: {len(feature_locations['systolic_peaks'])}")
    print(f"  Average cycle duration: {avg_cycle_duration:.3f} s")
    print(f"  Average heart rate: {60 / avg_cycle_duration:.1f} bpm")
    
    print("\n" + "="*70)
    print("Data generation complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
