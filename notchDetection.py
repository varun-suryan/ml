import numpy as np
import wfdb
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt issues
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import csv

# ---------- CONFIG ----------
REC_NAME = '3000063_0010'
PN_DIR = 'mimic3wdb/1.0/30/3000063'
ZOOM_START_S = 0
ZOOM_LEN_S = 65
# ---------------------------

def bandpass(x, fs, lo=0.5, hi=12.0, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return filtfilt(b, a, x)


def find_abp_index(names, units):
    candidates = {'ABP'}
    up = [n.strip().upper() for n in names]
    for i, u in enumerate(up):
        if u in candidates:
            return i
    for i, u in enumerate(units):
        if (u or '').lower().startswith('mmhg'):
            return i
    return 0


def detect_abp_landmarks_robust(
    abp, fs,
    min_peak_distance_s=0.30,
    peak_prom_scale=0.8,
    # notch must lie shortly after systole, but before the trough
    notch_window_s=(0.05, 0.35),
    # how far after a notch to look for the dicrotic crest
    rebound_search_s=0.25,
    # guard: notch must not be too close to trough
    trough_guard_s=0.03,
    # amplitude sanity (relative to pulse pressure)
    rebound_pp_low=0.02,  # >= 2% of PP
    rebound_pp_high=0.45,  # <= 45% of PP
):
    """
    Returns:
        sys_peaks, dicrotic_notches, dias_troughs (all np.int32 arrays; -1 if missing notch)
    Strategy:
        - band-pass filter for stability
        - systolic peaks by prominence + min distance
        - trough = deepest minimum between consecutive systolic peaks
        - candidate notches = ALL minima between peak and trough, within a timing window
        - score candidates by rebound crest amplitude (bigger rebound is better)
        - choose best that satisfies timing + amplitude constraints; otherwise next-best
    """
    N = len(abp)
    abp_f = bandpass(abp, fs)

    # --- Systolic peaks ---
    min_dist = int(min_peak_distance_s * fs)
    prom = np.std(abp_f) * peak_prom_scale
    sys_peaks, _ = find_peaks(abp_f, distance=min_dist, prominence=prom)

    # Physiologic amplitude screen on raw ABP
    keep = (abp[sys_peaks] > 60) & (abp[sys_peaks] < 250)
    sys_peaks = sys_peaks[keep]
    if len(sys_peaks) < 2:
        return (np.asarray(sys_peaks, dtype=np.int32),
                np.asarray([], dtype=np.int32),
                np.asarray([], dtype=np.int32))

    # --- Troughs: deepest min between consecutive peaks (use filtered for stability) ---
    inv = -abp_f
    dias_troughs = []
    for i in range(len(sys_peaks) - 1):
        p1, p2 = sys_peaks[i], sys_peaks[i + 1]
        if p2 - p1 < 3:
            dias_troughs.append(p1 + 1)
            continue
        mins_rel, _ = find_peaks(inv[p1:p2])  # minima of abp_f
        if mins_rel.size:
            vals = abp_f[p1:p2][mins_rel]
            trough_rel = mins_rel[np.argmin(vals)]
            dias_troughs.append(p1 + trough_rel)
        else:
            dias_troughs.append(p1 + int(np.argmin(abp[p1:p2])))
    dias_troughs = np.asarray(dias_troughs, dtype=np.int32)

    # --- Notches: pick the best candidate minimum between peak and trough ---
    dicrotic_notches = []
    for i in range(len(dias_troughs)):
        p1 = sys_peaks[i]
        d = dias_troughs[i]
        if d <= p1 + 1:
            dicrotic_notches.append(-1)
            continue

        # Candidate window (respect both the fixed window and the actual trough)
        start = p1 + int(max(0, notch_window_s[0]) * fs)
        end = min(d - int(trough_guard_s * fs), p1 + int(notch_window_s[1] * fs))
        if end <= start:
            dicrotic_notches.append(-1)
            continue

        # Find ALL local minima in abp_f within [start, end]
        cand_rel, cand_props = find_peaks(inv[start:end])
        if cand_rel.size == 0:
            dicrotic_notches.append(-1)
            continue

        cand_idx = start + cand_rel
        # Pulse pressure for this beat (raw ABP)
        PP = float(max(1e-6, abp[p1] - abp[d]))

        # For each candidate, find the rebound crest (local maximum) shortly after
        scores = []
        for c in cand_idx:
            # rebound search window: from candidate to min(d, c + rebound_search_s)
            r_end = min(d - 1, c + int(rebound_search_s * fs))
            if r_end <= c + 1:
                scores.append((-np.inf, 0.0, 0.0, c))  # invalid
                continue

            # crest as local max on filtered signal (or raw)
            crest_rel, _ = find_peaks(abp_f[c:r_end])
            if crest_rel.size:
                crest_idx = c + crest_rel[np.argmax(abp_f[c:r_end][crest_rel])]
            else:
                crest_idx = c + int(np.argmax(abp_f[c:r_end]))

            rebound_amp = float(abp[crest_idx] - abp[c])  # use raw for amplitude
            rel_rebound = rebound_amp / PP

            # Discard candidates whose notch is basically the trough
            above_trough = float(abp[c] - abp[d])  # >0 if notch above trough
            if above_trough <= 1.0:  # ~1 mmHg above trough; adjust if needed
                rel_rebound = -np.inf

            # Score: prioritize rebound amplitude relative to PP; add small bonus for prominence
            prom_here = float(cand_props['prominences'][np.where(cand_rel == (c - start))[0][0]]) if 'prominences' in cand_props else 0.0
            score = 0.8 * rel_rebound + 0.2 * (prom_here / (np.std(abp_f) + 1e-6))
            scores.append((score, rel_rebound, rebound_amp, c))

        # Rank by score (descending)
        scores.sort(reverse=True, key=lambda z: z[0])

        # Choose the best that satisfies rebound bounds; else try next (second-best, etc.)
        chosen = -1
        for sc, rel_r, abs_r, c in scores:
            if not np.isfinite(sc):
                continue
            if (rel_r >= rebound_pp_low) and (rel_r <= rebound_pp_high):
                chosen = int(c)
                break
        if chosen == -1:
            # Fallback to the highest-score candidate even if outside bounds
            chosen = int(scores[0][3]) if len(scores) else -1

        dicrotic_notches.append(chosen)

    dicrotic_notches = np.asarray(dicrotic_notches, dtype=np.int32)

    # Align lengths to the number of complete beats (one notch/trough per interval)
    sys_for_intervals = np.asarray(sys_peaks[:-1], dtype=np.int32)
    return sys_for_intervals, dicrotic_notches, dias_troughs



# ---------- Load record ----------
rec = wfdb.rdrecord(REC_NAME, pn_dir=PN_DIR)
fs = float(rec.fs)
names = np.array(rec.sig_name)
units = getattr(rec, 'units', [''] * len(names))
abp_idx = find_abp_index(names, units)
abp_full = rec.p_signal[:, abp_idx]
unit = units[abp_idx] if abp_idx < len(units) and units[abp_idx] else 'mmHg'
print(f"Using channel: {names[abp_idx]}, fs={fs} Hz")

# ---------- Handle NaN values ----------
nan_mask = np.isnan(abp_full)
if np.any(nan_mask):
    nan_count = np.sum(nan_mask)
    print(f"Warning: Found {nan_count} NaN values, interpolating...")
    
    # Interpolate NaN values with neighboring means
    nan_indices = np.where(nan_mask)[0]
    for idx in nan_indices:
        # Find valid neighbors
        left_idx = idx - 1
        right_idx = idx + 1
        
        # Search for valid left neighbor
        while left_idx >= 0 and np.isnan(abp_full[left_idx]):
            left_idx -= 1
        
        # Search for valid right neighbor
        while right_idx < len(abp_full) and np.isnan(abp_full[right_idx]):
            right_idx += 1
        
        # Calculate mean of valid neighbors
        neighbors = []
        if left_idx >= 0:
            neighbors.append(abp_full[left_idx])
        if right_idx < len(abp_full):
            neighbors.append(abp_full[right_idx])
        
        if neighbors:
            abp_full[idx] = np.mean(neighbors)
    
    print(f"NaN values replaced with neighboring means")


# ---------- Define analysis window ----------
start = int(ZOOM_START_S * fs)
end = min(len(abp_full), start + int(ZOOM_LEN_S * fs))
abp = abp_full[start:end]
print(f"Analyzing window: {ZOOM_START_S:.2f}s to {end/fs:.2f}s ({len(abp)} samples)")

# ---------- Detect on the window ----------
sys_peaks, notches, troughs = detect_abp_landmarks_robust(abp, fs)

# Adjust indices to full signal coordinates for plotting
sys_peaks = sys_peaks + start
notches = notches + start
troughs = troughs + start

# ---------- Plot zoom with markers ----------
t = np.arange(start, end) / fs

plt.figure(figsize=(12, 5))
plt.plot(t, abp, label='ABP')
sp = sys_peaks
dn = notches[notches >= 0]  # Filter out -1 (missing notches)
tr = troughs
if sp.size:
    plt.plot(sp / fs, abp_full[sp], 'o', label='Systolic peak')
if dn.size:
    plt.plot(dn / fs, abp_full[dn], 'v', label='Dicrotic notch')
if tr.size:
    plt.plot(tr / fs, abp_full[tr], 's', label='Diastolic trough')
plt.xlabel('Time (s)')
plt.ylabel(unit)
plt.title(f'{REC_NAME} – ABP landmarks (window: {ZOOM_START_S}s - {end/fs:.1f}s)')
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('abp_landmarks.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved as 'abp_landmarks.png'")
print(f"Detected: {sp.size} peaks, {dn.size} notches, {tr.size} troughs")
plt.close()

# ---------- Save labels to CSV (sorted by timestamp) ----------
csv_filename = 'abp_landmarks.csv'

# Collect all entries in a list
entries = []

# Add systolic peaks
for idx in sp:
    entries.append((idx, idx/fs, 'systolic_peak', abp_full[idx]))

# Add dicrotic notches (exclude -1 values)
for idx in dn:
    if idx >= 0:
        entries.append((idx, idx/fs, 'dicrotic_notch', abp_full[idx]))

# Add diastolic troughs
for idx in tr:
    entries.append((idx, idx/fs, 'diastolic_trough', abp_full[idx]))

# Sort by timestamp (second element in tuple)
entries.sort(key=lambda x: x[1])

# Write to CSV
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Index', 'Timestamp_s', 'Label', 'Value_mmHg'])
    
    for idx, timestamp, label, value in entries:
        writer.writerow([idx, f'{timestamp:.6f}', label, f'{value:.2f}'])

print(f"Labels saved to '{csv_filename}' (sorted by timestamp)")
print(f"Total entries: {len(entries)}")

