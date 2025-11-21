# Animate ABP streaming with online detections (save to GIF)
# - loads remote waveform via PhysioNet (MIMIC-III WDB)
# - precomputes detections using the no-filter slope-based detector
# - renders a time-progressing animation and saves as .gif
#
# pip install wfdb matplotlib pillow

import numpy as np
import wfdb
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from collections import deque

# ================== USER CONFIG ==================
REC_NAME = '3000003_0014'
PN_DIR = 'mimic3wdb/1.0/30/3000003'

# Segment of the record to analyze (absolute times)
ANALYZE_START_S = 420.0
ANALYZE_LEN_S = 20.0  # shorter windows = smaller GIFs

# Plot window (subset of analyzed segment, relative to ANALYZE_START_S)
PLOT_START_S = 0.0
PLOT_LEN_S = 20.0

# Detector params (no filtering)
PREV_SEC = 0.3
CURR_SEC = 0.3
REFRACT_MS = 220
NOTCH_DELAY_MS = 160
NOTCH_MAX_MS = 300
TROUGH_DELAY_MS = 180
PEAK_AFTER_TROUGH_DELAY_MS = 180
PEAK_SLOPE_EPS = 10.25
NOTCH_SLOPE_EPS = 10.20
TROUGH_SLOPE_EPS = 10.20

# Animation / export
FPS = 20  # GIF framerate
POINT_STEP = 2  # draw every Nth sample to keep GIF light
OUT_GIF = 'abp_detection_3.gif'
FIGSIZE = (12, 5)


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



class AsymTwoWindowDetectorNoFilter:
    """Two-window LS-slope detector (no filter), with notch timeout & trough backfill."""

    def __init__(self, fs,
                 prev_sec=0.08, curr_sec=0.06,
                 refractory_ms=220, notch_delay_ms=60, notch_max_ms=300,
                 trough_delay_ms=80, peak_after_trough_delay_ms=80,
                 peak_slope_eps=0.25, notch_slope_eps=0.20, trough_slope_eps=0.20):
        self.fs = float(fs)
        self.L_prev = max(2, int(round(prev_sec * self.fs)))
        self.L_curr = max(2, int(round(curr_sec * self.fs)))
        self.buf = deque(maxlen=self.L_prev + self.L_curr)

        # timing (samples)
        self.refrac = int(round(refractory_ms / 1000.0 * self.fs))
        self.n_delay = int(round(notch_delay_ms / 1000.0 * self.fs))
        self.n_max = int(round(notch_max_ms / 1000.0 * self.fs))
        self.t_delay = int(round(trough_delay_ms / 1000.0 * self.fs))
        self.p_after_tr_delay = int(round(peak_after_trough_delay_ms / 1000.0 * self.fs))

        # thresholds
        self.peak_eps = float(peak_slope_eps)
        self.notch_eps = float(notch_slope_eps)
        self.trough_eps = float(trough_slope_eps)

        # LS precompute
        def pre_ls(L):
            x = np.arange(L, dtype=float)
            N = float(L)
            sx = float(x.sum())
            sx2 = float((x ** 2).sum())
            den = (N * sx2 - sx ** 2) or 1.0
            return N, sx, sx2, den

        self.Np, self.sxp, self.sx2p, self.den_p = pre_ls(self.L_prev)
        self.Nc, self.sxc, self.sx2c, self.den_c = pre_ls(self.L_curr)

        # state
        self.state = 'WAIT_TROUGH'
        self.last_peak_i = -10 ** 9
        self.last_notch_i = -10 ** 9
        self.last_trough_i = -10 ** 9
        self.i = 0
        self.events = []

        # running minima
        self.runmin_y_notch = None
        self.runmin_i_notch = None
        self.runmin_y_tr = None
        self.runmin_i_tr = None

    @staticmethod
    def _ls_slope(y, L, N, sx, den, fs):
        y = np.asarray(y, float)
        sy = float(y.sum())
        sxy = float(np.dot(np.arange(L, dtype=float), y))
        m_per_sample = (N * sxy - sx * sy) / den
        return m_per_sample * fs  # per-second slope

    @staticmethod
    def _sign(x, eps):
        if x > eps:
            return 1
        if x < -eps:
            return -1
        return 0

    def step(self, raw):
        y = float(raw)
        self.buf.append(y)

        if len(self.buf) < (self.L_prev + self.L_curr):
            self.i += 1
            return

        buf_list = list(self.buf)
        
        # Smooth the buffer with a moving average
        smooth_win = 5  # Increased smoothing window
        buf_smoothed = np.convolve(buf_list, np.ones(smooth_win) / smooth_win, mode='same')
        
        prev_win = buf_smoothed[:self.L_prev]
        curr_win = buf_smoothed[self.L_prev:]
        m_prev = self._ls_slope(prev_win, self.L_prev, self.Np, self.sxp, self.den_p, self.fs)
        m_now = self._ls_slope(curr_win, self.L_curr, self.Nc, self.sxc, self.den_c, self.fs)

        # signs under per-event thresholds
        s_prev_p = self._sign(m_prev, self.peak_eps)
        s_now_p = self._sign(m_now, self.peak_eps)
        s_prev_n = self._sign(m_prev, self.notch_eps)
        s_now_n = self._sign(m_now, self.notch_eps)
        s_prev_t = self._sign(m_prev, self.trough_eps)
        s_now_t = self._sign(m_now, self.trough_eps)

        peak_pos_to_neg = (s_prev_p > 0 and s_now_p <= 0)
        notch_neg_to_pos = (s_prev_n < 0 and s_now_n >= 0)
        trough_neg_to_pos = (s_prev_t < 0 and s_now_t >= 0)

        emit_i = self.i - (self.L_curr - 1)

        if self.state == 'WAIT_PEAK':
            ok_refrac = (emit_i - self.last_peak_i > self.refrac)
            ok_after_tr = (emit_i - self.last_trough_i >= self.p_after_tr_delay)
            if peak_pos_to_neg and ok_refrac and ok_after_tr:
                self.last_peak_i = emit_i
                self.events.append({'type': 'peak', 'i': int(emit_i)})
                self.state = 'WAIT_NOTCH'
                self.runmin_y_notch = None
                self.runmin_i_notch = None
                self.runmin_y_tr = None
                self.runmin_i_tr = None

        elif self.state == 'WAIT_NOTCH':
            age = emit_i - self.last_peak_i
            if age >= self.n_delay:
                if (self.runmin_y_notch is None) or (y < self.runmin_y_notch):
                    self.runmin_y_notch, self.runmin_i_notch = y, emit_i

            if (age >= self.n_delay) and notch_neg_to_pos:
                notch_i = int(self.runmin_i_notch) if self.runmin_i_notch is not None else int(emit_i)
                self.last_notch_i = notch_i
                self.events.append({'type': 'notch', 'i': notch_i})
                self.state = 'WAIT_TROUGH'
                self.runmin_y_tr = None
                self.runmin_i_tr = None

            elif age > self.n_max:
                notch_i = int(self.runmin_i_notch) if self.runmin_i_notch is not None else int(emit_i)
                self.last_notch_i = notch_i
                self.events.append({'type': 'notch', 'i': notch_i})
                self.state = 'WAIT_TROUGH'
                self.runmin_y_tr = None
                self.runmin_i_tr = None

        elif self.state == 'WAIT_TROUGH':
            age = emit_i - self.last_notch_i
            if age >= self.t_delay:
                if (self.runmin_y_tr is None) or (y < self.runmin_y_tr):
                    self.runmin_y_tr, self.runmin_i_tr = y, emit_i

            if (age >= self.t_delay) and trough_neg_to_pos:
                tr_i = int(self.runmin_i_tr) if self.runmin_i_tr is not None else int(emit_i)
                self.last_trough_i = tr_i
                self.events.append({'type': 'trough', 'i': tr_i})
                self.state = 'WAIT_PEAK'

            elif peak_pos_to_neg and (emit_i - self.last_peak_i > self.refrac):
                tr_i = int(self.runmin_i_tr) if self.runmin_i_tr is not None else int(emit_i)
                self.last_trough_i = tr_i
                self.events.append({'type': 'trough', 'i': tr_i})
                self.state = 'WAIT_PEAK'
                self.runmin_y_notch = None
                self.runmin_i_notch = None
                self.runmin_y_tr = None
                self.runmin_i_tr = None

        self.i += 1



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

# -------------------- PRECOMPUTE detections --------------------
det = AsymTwoWindowDetectorNoFilter(
    fs,
    prev_sec=PREV_SEC, curr_sec=CURR_SEC,
    refractory_ms=REFRACT_MS,
    notch_delay_ms=NOTCH_DELAY_MS, notch_max_ms=NOTCH_MAX_MS,
    trough_delay_ms=TROUGH_DELAY_MS, peak_after_trough_delay_ms=PEAK_AFTER_TROUGH_DELAY_MS,
    peak_slope_eps=PEAK_SLOPE_EPS, notch_slope_eps=NOTCH_SLOPE_EPS, trough_slope_eps=TROUGH_SLOPE_EPS
)
for v in abp:
    det.step(v)

peaks = np.array([e['i'] for e in det.events if e['type'] == 'peak'], dtype=int)
notches = np.array([e['i'] for e in det.events if e['type'] == 'notch'], dtype=int)
troughs = np.array([e['i'] for e in det.events if e['type'] == 'trough'], dtype=int)

print(f"Detected {len(peaks)} peaks, {len(notches)} notches, {len(troughs)} troughs")

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
plt.savefig('abp_detections_static.png', dpi=150)
print("Saved plot to abp_detections_static.png")
plt.close()

# -------------------- Build animation (commented out) --------------------
"""
plot_start_rel = int(max(0.0, PLOT_START_S) * fs)
plot_end_rel = min(len(abp), plot_start_rel + int(PLOT_LEN_S * fs))

# time axis in absolute seconds
t_abs = (np.arange(plot_start_rel, plot_end_rel) + start_idx) / fs
y = abp[plot_start_rel:plot_end_rel]

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.set_xlabel('Time (s)')
ax.set_ylabel(unit)
ax.set_title(f'{REC_NAME} – {label} (streaming detections)')
ax.grid(True, alpha=0.3)


(line,) = ax.plot([], [], lw=1.4, label='ABP')
(pts_peak,) = ax.plot([], [], 'o', ms=6, label='Peaks')
(pts_notch,) = ax.plot([], [], 'v', ms=6, label='Notches')
(pts_trough,) = ax.plot([], [], 's', ms=6, label='Troughs')
ax.legend()

ax.set_xlim(t_abs[0], t_abs[-1])
# pad y-lims a bit
ymin, ymax = float(y.min()), float(y.max())
pad = 0.05 * (ymax - ymin) if ymax > ymin else 5.0
ax.set_ylim(ymin - pad, ymax + pad)


# Convert global event indices to this plot window's relative indices
def _sel_within(win_start, win_end, arr):
    return arr[(arr >= win_start) & (arr < win_end)]



peaks_local = _sel_within(plot_start_rel, plot_end_rel, peaks) - plot_start_rel
notches_local = _sel_within(plot_start_rel, plot_end_rel, notches) - plot_start_rel
troughs_local = _sel_within(plot_start_rel, plot_end_rel, troughs) - plot_start_rel

# Frames: step through samples (skip with POINT_STEP to reduce frames)
frames = np.arange(0, len(y), POINT_STEP)


def init():
    line.set_data([], [])
    pts_peak.set_data([], [])
    pts_notch.set_data([], [])
    pts_trough.set_data([], [])
    return line, pts_peak, pts_notch, pts_trough



def update(k):
    # draw line up to k
    line.set_data(t_abs[:k + 1], y[:k + 1])

    # events with index <= k appear
    pk = peaks_local[peaks_local <= k] if peaks_local.size else np.array([])
    nc = notches_local[notches_local <= k] if notches_local.size else np.array([])
    tr = troughs_local[troughs_local <= k] if troughs_local.size else np.array([])

    if pk.size:
        pts_peak.set_data(t_abs[pk], y[pk])
    if nc.size:
        pts_notch.set_data(t_abs[nc], y[nc])
    if tr.size:
        pts_trough.set_data(t_abs[tr], y[tr])

    return line, pts_peak, pts_notch, pts_trough



anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=1000 / FPS)

print(f"Saving GIF to {OUT_GIF} ...")
anim.save(OUT_GIF, writer=PillowWriter(fps=FPS))
print("Done.")
"""

# -------------------- Save to CSV --------------------
import csv
OUTPUT_CSV = 'abp_adaptive_online_landmarks.csv'

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
