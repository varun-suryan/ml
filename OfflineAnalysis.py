# Offline ABP peak, valley (trough), and dicrotic notch detection
# - loads remote waveform via PhysioNet (MIMIC-III WDB)
# - performs offline detection using scipy.signal.find_peaks
# - detects systolic peaks, diastolic troughs, and dicrotic notches
#
# pip install wfdb matplotlib scipy

import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

# ================== USER CONFIG ==================
REC_NAME = '3000003_0014'
PN_DIR = 'mimic3wdb/1.0/30/3000003'

# Segment of the record to analyze (absolute times)
ANALYZE_START_S = 420.0
ANALYZE_LEN_S = 20.0

# Plot window
PLOT_START_S = 0.0
PLOT_LEN_S = 20.0

# Detection params
MIN_PEAK_DISTANCE_MS = 400  # Minimum distance between peaks (ms)
MIN_PEAK_PROMINENCE = 5  # Minimum prominence for peak detection (mmHg)
MIN_TROUGH_PROMINENCE = 5  # Minimum prominence for trough detection (mmHg)
NOTCH_SEARCH_START_MS = 180  # Start searching for notch after peak (ms)
NOTCH_SEARCH_END_MS = 350  # Stop searching for notch after peak (ms)

# Bandpass filter params
LOWCUT = 0.5  # Low cutoff frequency (Hz)
HIGHCUT = 15.0  # High cutoff frequency (Hz)
FILTER_ORDER = 3  # Butterworth filter order

# Output
FIGSIZE = (14, 6)
OUTPUT_PNG = 'abp_offline_detections.png'
OUTPUT_CSV = 'abp_offline_landmarks.csv'


# =================================================

def find_abp_index(names, units):
    candidates = {'ABP'}
    up = [n.strip().upper() for n in names]
    for i, n in enumerate(up):
        if n in candidates:
            return i
    for i, u in enumerate(units):
        if (u or '').lower().startswith('mmhg'):
            return i
    return 0


def bandpass_filter(signal, lowcut, highcut, fs, order=3):
    """Apply Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


def detect_abp_features_offline(abp, fs):
    """
    Offline detection of ABP features: systolic peaks, diastolic troughs, and dicrotic notches.
    
    Args:
        abp: ABP signal array
        fs: Sampling frequency
        
    Returns:
        peaks, troughs, notches: Arrays of indices for each feature
    """
    # Apply bandpass filter
    abp_filtered = bandpass_filter(abp, LOWCUT, HIGHCUT, fs, FILTER_ORDER)
    
    # Convert time-based params to samples
    min_peak_distance = int(MIN_PEAK_DISTANCE_MS / 1000.0 * fs)
    notch_start = int(NOTCH_SEARCH_START_MS / 1000.0 * fs)
    notch_end = int(NOTCH_SEARCH_END_MS / 1000.0 * fs)
    
    # 1. Find systolic peaks (local maxima)
    peaks, peak_props = find_peaks(abp_filtered, 
                                     distance=min_peak_distance,
                                     prominence=MIN_PEAK_PROMINENCE)
    
    # 2. Find diastolic troughs (local minima) - invert signal
    troughs, trough_props = find_peaks(-abp_filtered,
                                        distance=min_peak_distance,
                                        prominence=MIN_TROUGH_PROMINENCE)
    
    # 3. Find dicrotic notches - search between each peak and next trough
    notches = []
    
    for i, peak_idx in enumerate(peaks):
        # Find the next trough after this peak
        next_troughs = troughs[troughs > peak_idx]
        if len(next_troughs) == 0:
            continue
        next_trough = next_troughs[0]
        
        # Search window for notch
        search_start = peak_idx + notch_start
        search_end = min(peak_idx + notch_end, next_trough)
        
        if search_start >= search_end or search_end >= len(abp_filtered):
            continue
        
        # Find the steepest negative slope (derivative) in search window
        # This corresponds to the notch location
        search_region = abp_filtered[search_start:search_end]
        if len(search_region) < 3:
            continue
        
        # Compute derivative (negative slope indicates descent)
        derivative = np.diff(search_region)
        
        # Find the point with steepest negative slope
        min_derivative_idx = np.argmin(derivative)
        notch_idx = search_start + min_derivative_idx
        
        # Validate: notch should be lower than peak and between peak and trough
        if (abp_filtered[notch_idx] < abp_filtered[peak_idx] and 
            peak_idx < notch_idx < next_trough):
            notches.append(notch_idx)
    
    notches = np.array(notches, dtype=int)
    
    return peaks, troughs, notches



# -------------------- Load ABP --------------------
rec = wfdb.rdrecord(REC_NAME, pn_dir=PN_DIR)
fs = float(rec.fs)
names = rec.sig_name
units = getattr(rec, 'units', [''] * len(names))
abp_idx = find_abp_index(names, units)
abp_all = rec.p_signal[:, abp_idx].astype(float)
unit = units[abp_idx] if abp_idx < len(units) and units[abp_idx] else 'mmHg'
label = names[abp_idx]
print(f"Loaded {REC_NAME}: {label} ({unit}), fs={fs:.1f} Hz")

# -------------------- Select analysis segment --------------------
start_idx = int(max(0.0, ANALYZE_START_S) * fs)
end_idx = len(abp_all) if ANALYZE_LEN_S is None else min(len(abp_all), start_idx + int(ANALYZE_LEN_S * fs))
abp = abp_all[start_idx:end_idx]

# -------------------- Offline detection --------------------
print("Running offline detection...")
peaks, troughs, notches = detect_abp_features_offline(abp, fs)

print(f"Detected {len(peaks)} systolic peaks, {len(notches)} dicrotic notches, {len(troughs)} diastolic troughs")

# -------------------- Static Plot --------------------
plot_start_rel = int(max(0.0, PLOT_START_S) * fs)
plot_end_rel = min(len(abp), plot_start_rel + int(PLOT_LEN_S * fs))

# time axis in absolute seconds
t_abs = (np.arange(plot_start_rel, plot_end_rel) + start_idx) / fs
y = abp[plot_start_rel:plot_end_rel]

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(t_abs, y, 'b-', lw=1.4, label='ABP')

# Convert global event indices to this plot window's relative indices
def _sel_within(win_start, win_end, arr):
    return arr[(arr >= win_start) & (arr < win_end)]

peaks_local = _sel_within(plot_start_rel, plot_end_rel, peaks) - plot_start_rel
notches_local = _sel_within(plot_start_rel, plot_end_rel, notches) - plot_start_rel
troughs_local = _sel_within(plot_start_rel, plot_end_rel, troughs) - plot_start_rel

if len(peaks_local) > 0:
    ax.plot(t_abs[peaks_local], y[peaks_local], 'ro', ms=8, label='Systolic Peaks')
if len(notches_local) > 0:
    ax.plot(t_abs[notches_local], y[notches_local], 'gv', ms=8, label='Dicrotic Notches')
if len(troughs_local) > 0:
    ax.plot(t_abs[troughs_local], y[troughs_local], 'ms', ms=8, label='Diastolic Troughs')

ax.set_xlabel('Time (s)')
ax.set_ylabel(unit)
ax.set_title(f'{REC_NAME} – {label} (detections)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
print(f"Saved plot to {OUTPUT_PNG}")
plt.close()

# -------------------- Save to CSV --------------------
import csv

# Combine all landmarks with their labels
all_landmarks = []
for idx in peaks:
    t_sec = (idx + start_idx) / fs
    all_landmarks.append({
        'Index': int(idx),
        'Timestamp_s': float(t_sec),
        'Label': 'systolic_peak',
        'Value_mmHg': float(abp[idx])
    })

for idx in notches:
    t_sec = (idx + start_idx) / fs
    all_landmarks.append({
        'Index': int(idx),
        'Timestamp_s': float(t_sec),
        'Label': 'dicrotic_notch',
        'Value_mmHg': float(abp[idx])
    })

for idx in troughs:
    t_sec = (idx + start_idx) / fs
    all_landmarks.append({
        'Index': int(idx),
        'Timestamp_s': float(t_sec),
        'Label': 'diastolic_trough',
        'Value_mmHg': float(abp[idx])
    })

# Sort by timestamp
all_landmarks.sort(key=lambda x: x['Timestamp_s'])

# Write to CSV
with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    fieldnames = ['Index', 'Timestamp_s', 'Label', 'Value_mmHg']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for landmark in all_landmarks:
        writer.writerow(landmark)

print(f"Saved landmarks to {OUTPUT_CSV}")

# -------------------- Animation code removed --------------------