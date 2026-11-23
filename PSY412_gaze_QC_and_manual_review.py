"""
PSY412 gaze quality + manual review preparation
Run this script in the same folder as the participant CSV files.

It creates:
  1. trial_gaze_quality.csv
  2. gaze_gap_report.csv
  3. manual_saccade_review.csv
  4. gaze_review_plots/ (one plot per selected trial)

This script DOES NOT automatically classify capture errors.
Confirm physical monitor width and viewing distance before interpreting degrees.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- USER SETTINGS ----------
PREFIX = "001_included_pro_anti_20261115_105145_"
GAZE_FILE = PREFIX + "gaze.csv"
EVENTS_FILE = PREFIX + "events.csv"
TRIALS_FILE = PREFIX + "trials.csv"

# These are the values currently in your PsychoPy script.
# Physically measure/confirm them before using degree values.
SCREEN_WIDTH_CM = 34.5
VIEWING_DISTANCE_CM = 71.7
SCREEN_WIDTH_PX = 1366
SCREEN_HEIGHT_PX = 768

# Plot this many trials, selected across task/stage and accuracy.
MAX_PLOTS = 20
PLOT_PRE_CUE_MS = 500
PLOT_POST_CUE_MS = 1000
# -----------------------------------

base = Path(".")
gaze = pd.read_csv(base / GAZE_FILE)
events = pd.read_csv(base / EVENTS_FILE)
trials = pd.read_csv(base / TRIALS_FILE)

# Timestamps in seconds, for easier calculations
gaze["time_s"] = gaze["system_time_us"] / 1e6
events["time_s"] = events["system_time_us"] / 1e6

# A sample is usable if at least one eye is valid AND has finite coordinates.
left_ok = (
    (gaze["left_validity"] == 1)
    & np.isfinite(gaze["left_x_norm"])
    & np.isfinite(gaze["left_y_norm"])
)
right_ok = (
    (gaze["right_validity"] == 1)
    & np.isfinite(gaze["right_x_norm"])
    & np.isfinite(gaze["right_y_norm"])
)
gaze["usable"] = left_ok | right_ok

# Combine the two eyes where available; use the one valid eye if only one is valid.
gaze["x_norm"] = np.where(
    left_ok & right_ok,
    (gaze["left_x_norm"] + gaze["right_x_norm"]) / 2,
    np.where(left_ok, gaze["left_x_norm"], gaze["right_x_norm"])
)
gaze["y_norm"] = np.where(
    left_ok & right_ok,
    (gaze["left_y_norm"] + gaze["right_y_norm"]) / 2,
    np.where(left_ok, gaze["left_y_norm"], gaze["right_y_norm"])
)
gaze.loc[~gaze["usable"], ["x_norm", "y_norm"]] = np.nan
gaze = gaze.sort_values("time_s").reset_index(drop=True)

# ---------- A. Find long gaps ----------
gaze["interval_s"] = gaze["time_s"].diff()
gap_rows = []
starts = events[events["event"] == "trial_start"]
ends = events[events["event"] == "trial_end"]

trial_intervals = []
for (_, s), (_, e) in zip(starts.iterrows(), ends.iterrows()):
    trial_intervals.append({
        "stage": s["stage"],
        "task": s["task"],
        "trial_number": s["trial_number"],
        "practice": s["practice"],
        "start_s": s["time_s"],
        "end_s": e["time_s"],
    })

for idx, row in gaze[gaze["interval_s"] > 0.5].iterrows():
    gap_start = row["time_s"] - row["interval_s"]
    gap_end = row["time_s"]
    overlapping = [
        t for t in trial_intervals
        if t["start_s"] < gap_end and t["end_s"] > gap_start
    ]
    gap_rows.append({
        "gap_start_s": gap_start,
        "gap_end_s": gap_end,
        "gap_duration_s": row["interval_s"],
        "overlapping_trial_count": len(overlapping),
        "overlapping_trials": "; ".join(
            f'{t["stage"]}_{t["task"]}_{int(t["trial_number"])}'
            for t in overlapping
        )
    })

pd.DataFrame(gap_rows).to_csv("gaze_gap_report.csv", index=False)

# ---------- B. Trial-level gaze validity ----------
quality_rows = []
for t in trial_intervals:
    segment = gaze[
        (gaze["time_s"] >= t["start_s"])
        & (gaze["time_s"] <= t["end_s"])
    ]
    n = len(segment)
    valid_n = int(segment["usable"].sum())
    quality_rows.append({
        "stage": t["stage"],
        "task": t["task"],
        "trial_number": t["trial_number"],
        "practice": t["practice"],
        "n_gaze_samples": n,
        "n_usable_samples": valid_n,
        "valid_gaze_proportion": valid_n / n if n else np.nan,
        "flag_below_70_percent": (
            valid_n / n < 0.70 if n else True
        )
    })

quality = pd.DataFrame(quality_rows)
quality.to_csv("trial_gaze_quality.csv", index=False)

# Join trial task details to the quality table.
review = trials.merge(
    quality,
    on=["stage", "task", "trial_number", "practice"],
    how="left"
)

# ---------- C. Create manual review sheet ----------
manual = review.copy()
manual["manual_first_saccade_direction"] = ""
manual["manual_first_saccade_latency_ms"] = ""
manual["manual_first_saccade_amplitude_deg"] = ""
manual["manual_capture_error_yes_no"] = ""
manual["manual_notes"] = ""
manual.to_csv("manual_saccade_review.csv", index=False)

# ---------- D. Convert gaze coordinates to approximate degrees ----------
# Assumes square pixels and a 16:9 display, with screen centre as (0,0).
screen_height_cm = SCREEN_WIDTH_CM * SCREEN_HEIGHT_PX / SCREEN_WIDTH_PX
screen_height_px = SCREEN_HEIGHT_PX

def norm_to_deg(x, y):
    x_cm = (x - 0.5) * SCREEN_WIDTH_CM
    y_cm = (0.5 - y) * screen_height_cm
    x_deg = np.degrees(np.arctan2(x_cm, VIEWING_DISTANCE_CM))
    y_deg = np.degrees(np.arctan2(y_cm, VIEWING_DISTANCE_CM))
    return x_deg, y_deg

gaze["x_deg"], gaze["y_deg"] = norm_to_deg(
    gaze["x_norm"].to_numpy(),
    gaze["y_norm"].to_numpy()
)

# ---------- E. Select and plot trials for manual inspection ----------
# Spread plots across pre/post and pro/anti; include incorrect peripheral responses.
plot_candidates = review.copy()
plot_candidates["peripheral_correct"] = pd.to_numeric(
    plot_candidates["peripheral_correct"], errors="coerce"
)
selected = []
for (stage, task), group in plot_candidates.groupby(["stage", "task"]):
    selected.extend(group.head(3).to_dict("records"))
    incorrect = group[group["peripheral_correct"] == 0]
    selected.extend(incorrect.head(2).to_dict("records"))

# De-duplicate and cap number of plots.
seen = set()
chosen = []
for r in selected:
    key = (r["stage"], r["task"], r["trial_number"], r["practice"])
    if key not in seen:
        seen.add(key)
        chosen.append(r)
chosen = chosen[:MAX_PLOTS]

plot_dir = base / "gaze_review_plots"
plot_dir.mkdir(exist_ok=True)

for r in chosen:
    cue_s = r["cue_onset_system_us"] / 1e6
    t0 = cue_s - PLOT_PRE_CUE_MS / 1000
    t1 = cue_s + PLOT_POST_CUE_MS / 1000

    segment = gaze[(gaze["time_s"] >= t0) & (gaze["time_s"] <= t1)]
    if segment.empty:
        continue

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(segment["time_s"] - cue_s, segment["x_deg"], label="Horizontal gaze (°)")
    ax.axvline(0, linestyle="--", label="Cue onset")
    ax.axvline(
        (r["cue_offset_system_us"] - r["cue_onset_system_us"]) / 1e6,
        linestyle=":",
        label="Cue offset"
    )
    ax.axvline(
        (r["peripheral_target_onset_system_us"] - r["cue_onset_system_us"]) / 1e6,
        linestyle="-.",
        label="Peripheral target onset"
    )
    ax.set_xlabel("Time relative to cue onset (seconds)")
    ax.set_ylabel("Horizontal gaze position (degrees)")
    ax.set_title(
        f'{r["stage"]} {r["task"]} trial {int(r["trial_number"])} | '
        f'cue={r["cue_side"]} | target={r["target_side"]} | '
        f'peripheral correct={r["peripheral_correct"]}'
    )
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    filename = (
        f'{r["stage"]}_{r["task"]}_trial_{int(r["trial_number"])}'
        f'_practice_{int(r["practice"])}.png'
    )
    fig.savefig(plot_dir / filename, dpi=150)
    plt.close(fig)

print("\nFinished.")
print("Created: trial_gaze_quality.csv")
print("Created: gaze_gap_report.csv")
print("Created: manual_saccade_review.csv")
print(f"Created {len(list(plot_dir.glob('*.png')))} plots in {plot_dir}/")
print("\nReminder: these plots prepare manual review; they do NOT classify saccades.")
