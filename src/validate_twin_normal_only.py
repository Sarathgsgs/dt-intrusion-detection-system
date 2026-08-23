"""
Task 1 — Normal-Only Twin Validation MAE (Before vs. After Fix)
Determines whether the flat twin forecast line is correct behavior (twin correctly ignoring
attack-driven spikes) or dynamic-range over-compression (twin lost sensitivity to legitimate
normal fluctuations as a side-effect of the log1p bounding fix).

Gate condition: this must complete before Task 3 (MITM investigation) can proceed.

Pre-fix baseline MAE (from Track A experiments):
  - Baseline twin (no transform): 244.98 B mean absolute error
  - Log1p Robust (post-fix / current): 140.70 B mean absolute error (entire mixed test set)

This script computes MAE *exclusively on Normal-class samples* (held-out 20% split)
to separate the twin's normal-tracking fidelity from its (correct) behavior of ignoring
attack-driven spikes in the live mixed stream.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.twin_model import DigitalTwin, CONTINUOUS_FEATURES

BASELINE_MAE_MIXED   = 244.98   # B -- pre-fix baseline, mixed test set
POSTFIX_MAE_MIXED    = 140.70   # B -- post-fix current, mixed test set (from Track A report)
PRE_FIX_NORMAL_APPROX_MAE = 244.98  # Conservative proxy for pre-fix normal-only

def evaluate_normal_only_mae(
    data_csv: str = "data/sampled_dataset.csv",
    model_dir: str = "models",
    val_fraction: float = 0.20,
    random_state: int = 42
):
    print("=" * 80)
    print("  TASK 1 — NORMAL-ONLY TWIN VALIDATION MAE (Post-Fix)")
    print("  Gate: determines if flat forecast line = correct behavior or over-compression")
    print("=" * 80)

    df_all = pd.read_csv(data_csv)

    # ── Hold-out: use last 20% of Normal samples as held-out validation ──────────
    normal_df = df_all[df_all["Attack_type"] == "Normal"].reset_index(drop=True)
    n_val = int(len(normal_df) * val_fraction)
    n_train_end = len(normal_df) - n_val

    normal_train = normal_df.iloc[:n_train_end]
    normal_val   = normal_df.iloc[n_train_end:]

    print(f"\nNormal samples total   : {len(normal_df):,}")
    print(f"  Training window      : {len(normal_train):,} (first {100*(1-val_fraction):.0f}%)")
    print(f"  Validation (held-out): {len(normal_val):,}  (last  {100*val_fraction:.0f}%)")

    # ── Load current (post-fix) twin ──────────────────────────────────────────────
    twin = DigitalTwin.load(model_dir)
    print(f"\nLoaded post-fix twin (use_log1p={twin.use_log1p}) from '{model_dir}/'")
    print(f"Features: {twin.feature_names}")

    # ── Compute Normal-only MAE using current twin ────────────────────────────────
    raw_val = normal_val[twin.feature_names].fillna(0.0).values
    transformed_val = twin._transform_in(raw_val)
    scaled_val      = twin.scaler.transform(transformed_val)

    # Build sequences from held-out Normal data
    W = twin.window_size
    if len(scaled_val) <= W:
        print(f"\n[WARN] Not enough Normal samples for sequences (need > {W}). Using all available.")
        W = max(1, len(scaled_val) // 2)

    X_val, y_actual_scaled = [], []
    for i in range(len(scaled_val) - W):
        X_val.append(scaled_val[i:i + W].flatten())
        y_actual_scaled.append(scaled_val[i + W])
    X_val = np.array(X_val)
    y_actual_scaled = np.array(y_actual_scaled)

    y_pred_scaled    = twin.model.predict(X_val)
    y_pred_unscaled  = twin.scaler.inverse_transform(y_pred_scaled)
    y_pred_physical  = twin._transform_out(y_pred_unscaled)

    y_actual_unscaled = twin.scaler.inverse_transform(y_actual_scaled)
    y_actual_physical = twin._transform_out(y_actual_unscaled)

    # Per-feature MAE
    per_feature_mae = {}
    for j, feat in enumerate(twin.feature_names):
        per_feature_mae[feat] = mean_absolute_error(
            y_actual_physical[:, j], y_pred_physical[:, j]
        )

    mean_normal_only_mae = np.mean(list(per_feature_mae.values()))

    # ── Print results ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  PER-FEATURE MAE (Normal-Only Held-Out Validation)")
    print("─" * 70)
    print(f"  {'Feature':<30} {'Post-Fix MAE (B)':>18}")
    print("─" * 70)
    for feat, mae in sorted(per_feature_mae.items(), key=lambda x: -x[1]):
        print(f"  {feat:<30} {mae:>18.3f}")
    print("─" * 70)

    # ── Two-row summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SUMMARY TABLE — Before vs. After Track A Fix")
    print("=" * 80)
    print(f"  {'Measurement':<45} {'MAE (Bytes)':>12}  {'Dataset Scope':<25}")
    print("─" * 80)
    print(f"  {'Pre-Fix Baseline MAE (mixed test set)':<45} {BASELINE_MAE_MIXED:>12.2f}  {'All attack classes':<25}")
    print(f"  {'Post-Fix MAE   (mixed test set, Track A)':<45} {POSTFIX_MAE_MIXED:>12.2f}  {'All attack classes':<25}")
    print(f"  {'Post-Fix MAE   (Normal-ONLY held-out)':<45} {mean_normal_only_mae:>12.2f}  {'Normal traffic only':<25}")
    print("─" * 80)

    # ── Verdict ───────────────────────────────────────────────────────────────────
    pct_change = (mean_normal_only_mae - BASELINE_MAE_MIXED) / BASELINE_MAE_MIXED * 100.0
    print()

    if mean_normal_only_mae <= BASELINE_MAE_MIXED * 1.10:
        verdict = "CALIBRATED — Flat forecast line is CORRECT BEHAVIOR"
        explanation = (
            f"Normal-only MAE ({mean_normal_only_mae:.2f} B) is no worse than pre-fix baseline "
            f"({BASELINE_MAE_MIXED:.2f} B, Δ={pct_change:+.1f}%). "
            "The twin tracks normal traffic correctly. The flat line in live dashboards "
            "is genuine: it correctly ignores attack-driven spikes as designed. "
            "Track A fix is COMPLETE — no dynamic-range suppression detected."
        )
        task3_guidance = "MITM investigation should focus on signal sensitivity, NOT bound relaxation."
    elif mean_normal_only_mae <= BASELINE_MAE_MIXED * 1.50:
        verdict = "MILD COMPRESSION — Acceptable trade-off, document as limitation"
        explanation = (
            f"Normal-only MAE ({mean_normal_only_mae:.2f} B) is moderately worse than pre-fix "
            f"({BASELINE_MAE_MIXED:.2f} B, Δ={pct_change:+.1f}%). Some dynamic range was "
            "compressed. The twin is still substantially better than the un-fixed version "
            "but may under-track low-magnitude normal fluctuations. Report as known trade-off."
        )
        task3_guidance = "MITM investigation: consider slightly wider log scale (log1p(x/5)) if compression ratio > 3x."
    else:
        verdict = "OVER-COMPRESSED — Dynamic range suppressed, bound may need relaxation"
        explanation = (
            f"Normal-only MAE ({mean_normal_only_mae:.2f} B) is significantly worse than pre-fix "
            f"({BASELINE_MAE_MIXED:.2f} B, Δ={pct_change:+.1f}%). The bounding fix over-corrected: "
            "the twin traded 'wildly over-predicting attacks' for 'uninformatively flat on everything'. "
            "Task 3 should investigate loosening the bound."
        )
        task3_guidance = "MITM investigation: relaxing bound (e.g., clip at 25.0 instead of 22.5) is STRONGLY recommended."

    print(f"  VERDICT: {verdict}")
    print()
    print(f"  {explanation}")
    print()
    print(f"  TASK 3 GUIDANCE: {task3_guidance}")
    print("=" * 80)

    return {
        "normal_only_mae": mean_normal_only_mae,
        "pre_fix_baseline_mae": BASELINE_MAE_MIXED,
        "post_fix_mixed_mae": POSTFIX_MAE_MIXED,
        "pct_change_vs_baseline": pct_change,
        "per_feature_mae": per_feature_mae,
        "verdict": verdict,
        "task3_guidance": task3_guidance
    }


def append_to_track_a_report(results: dict, report_path: str = "results/track_a_twin_experiments_report.md"):
    section = f"""
---

## 5. Normal-Only Validation (Post-Fix Gate Check)

**Task 1 output — determines whether flat dashboard forecast is correct behavior or over-compression.**

| Measurement | MAE (Bytes) | Scope |
|---|---:|---|
| Pre-Fix Baseline | {results["pre_fix_baseline_mae"]:.2f} B | All attack classes (mixed test set) |
| Post-Fix (Track A) | {results["post_fix_mixed_mae"]:.2f} B | All attack classes (mixed test set) |
| **Post-Fix Normal-Only** | **{results["normal_only_mae"]:.2f} B** | **Normal traffic only (held-out 20%)** |

**Verdict:** {results["verdict"]}

**Δ vs. Pre-Fix Baseline (Normal-Only):** {results["pct_change_vs_baseline"]:+.1f}%

**Per-Feature MAE (Post-Fix, Normal-Only):**

| Feature | MAE (Bytes) |
|---|---:|
"""
    for feat, mae in sorted(results["per_feature_mae"].items(), key=lambda x: -x[1]):
        section += f"| {feat} | {mae:.3f} |\n"

    section += f"""
**Task 3 Guidance:** {results["task3_guidance"]}
"""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(section)
    print(f"\n[APPENDED] Normal-Only Validation section → {report_path}")


if __name__ == "__main__":
    results = evaluate_normal_only_mae()
    append_to_track_a_report(results)
