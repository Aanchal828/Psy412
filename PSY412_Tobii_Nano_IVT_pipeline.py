"""
PSY412 Tobii Pro Nano I-VT candidate saccade pipeline
-----------------------------------------------------
IMPORTANT:
- The PsychoPy experiment script specifies SCREEN_WIDTH_CM=34.5 and VIEW_DIST_CM=71.7.
- The experimenter confirmed the actual viewing distance was 71.7 cm.
- Confirm the physical active-display width with a ruler before treating degrees as final.
- Set SCREEN_WIDTH_CM to the measured visible screen width and VIEW_DIST_CM to the
  actual participant eye-to-screen distance used in the session.
- Gaze coordinates in the exported files are normalized display coordinates (0..1),
  origin at top-left, as written by the experiment's Tobii Pro SDK callback.

This produces candidate events only. Inspect traces and validate thresholds before
making inferential claims. At 60 Hz, event timing is quantized to ~16.7 ms.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/raw")  # directory containing extracted ZIP files
OUT_DIR = Path("ivt_output")
OUT_DIR.mkdir(exist_ok=True)

SCREEN_WIDTH_CM = 34.5   # PROVISIONAL: confirm visible active width with ruler
SCREEN_RES_X = 1366
SCREEN_RES_Y = 768
VIEW_DIST_CM = 71.7      # confirmed: actual experiment viewing distance
THRESHOLDS_DEG_S = [30.0, 40.0, 60.0]
MIN_CONSECUTIVE_INTERVALS = 2  # ~33 ms at 60 Hz; candidate rule, validate manually
MAX_GAP_MS = 50.0        # do not calculate velocity across larger missing-data gaps

def norm_to_deg(xn, yn):
    """Exact angular eccentricity from screen centre, using physical screen geometry."""
    screen_h_cm = SCREEN_WIDTH_CM * SCREEN_RES_Y / SCREEN_RES_X
    x_cm = (xn - 0.5) * SCREEN_WIDTH_CM
    y_cm = (0.5 - yn) * screen_h_cm
    return np.degrees(np.arctan2(x_cm, VIEW_DIST_CM)), \
           np.degrees(np.arctan2(y_cm, VIEW_DIST_CM))

def load_gaze(path):
    g = pd.read_csv(path)
    # Use each eye only when its Tobii validity flag is 1 and coordinates are finite.
    left_ok = (g.left_validity == 1) & np.isfinite(g.left_x_norm) & np.isfinite(g.left_y_norm)
    right_ok = (g.right_validity == 1) & np.isfinite(g.right_x_norm) & np.isfinite(g.right_y_norm)
    x = np.where(left_ok & right_ok, (g.left_x_norm + g.right_x_norm)/2,
        np.where(left_ok, g.left_x_norm, np.where(right_ok, g.right_x_norm, np.nan)))
    y = np.where(left_ok & right_ok, (g.left_y_norm + g.right_y_norm)/2,
        np.where(left_ok, g.left_y_norm, np.where(right_ok, g.right_y_norm, np.nan)))
    gx, gy = norm_to_deg(x, y)
    return pd.DataFrame({
        "system_time_us": g.system_time_us,
        "time_s": g.system_time_us / 1e6,
        "gaze_x_deg": gx, "gaze_y_deg": gy,
        "valid_gaze": np.isfinite(gx) & np.isfinite(gy)
    })

def load_events(path):
    e = pd.read_csv(path)
    return e

def detect_candidates(g, threshold):
    """I-VT candidate events; invalid samples/gaps break runs."""
    q = g.copy().reset_index(drop=True)
    dt = q.time_s.diff().to_numpy()
    dx = q.gaze_x_deg.diff().to_numpy()
    dy = q.gaze_y_deg.diff().to_numpy()
    vel = np.sqrt(dx**2 + dy**2) / dt
    good = (q.valid_gaze.to_numpy() &
            np.r_[False, q.valid_gaze.to_numpy()[:-1]] &
            np.isfinite(vel) & (dt * 1000 <= MAX_GAP_MS))
    above = good & (vel >= threshold)
    q["velocity_deg_s"] = vel
    q["above_threshold"] = above

    events = []
    i, n = 0, len(q)
    while i < n:
        if not above[i]:
            i += 1
            continue
        start = i
        while i < n and above[i]:
            i += 1
        end = i - 1
        if end - start + 1 < MIN_CONSECUTIVE_INTERVALS:
            continue
        # Velocity is associated with the interval ending at sample index.
        start_idx = max(0, start - 1)
        end_idx = end
        x0, y0 = q.loc[start_idx, ["gaze_x_deg","gaze_y_deg"]]
        x1, y1 = q.loc[end_idx, ["gaze_x_deg","gaze_y_deg"]]
        events.append({
            "threshold_deg_s": threshold,
            "onset_time_s": q.loc[start_idx, "time_s"],
            "offset_time_s": q.loc[end_idx, "time_s"],
            "duration_ms": (q.loc[end_idx,"time_s"]-q.loc[start_idx,"time_s"])*1000,
            "peak_velocity_deg_s": np.nanmax(vel[start:end+1]),
            "amplitude_deg": float(np.hypot(x1-x0,y1-y0)),
            "direction": "right" if x1>x0 else ("left" if x1<x0 else "none")
        })
    return pd.DataFrame(events), q

def main():
    all_events = []
    for event_path in DATA_DIR.glob("*_events.csv"):
        stem = event_path.name.replace("_events.csv","")
        gaze_path = DATA_DIR / f"{stem}_gaze.csv"
        if not gaze_path.exists():
            print(f"SKIP {stem}: standard gaze file missing")
            continue
        events = load_events(event_path)
        gaze = load_gaze(gaze_path)
        for threshold in THRESHOLDS_DEG_S:
            detected, _ = detect_candidates(gaze, threshold)
            if detected.empty:
                continue
            detected["participant"] = int(events.participant.dropna().iloc[0])
            detected["condition"] = events.condition.dropna().iloc[0]
            detected["threshold_deg_s"] = threshold
            all_events.append(detected)
    if all_events:
        pd.concat(all_events, ignore_index=True).to_csv(
            OUT_DIR / "candidate_saccade_events.csv", index=False)
    print("Pipeline complete. Inspect candidate_saccade_events.csv and validate against plots.")

if __name__ == "__main__":
    main()
