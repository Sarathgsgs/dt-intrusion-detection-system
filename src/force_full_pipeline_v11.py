"""
Master Execution Script for Implementation Plan v11:
1. Retrains the Scope-Restricted Digital Twin with Delta Sequences on normal traffic.
2. Saves models/twin_model.pkl and updates dev_features.pkl and fused_features.pkl.
3. Force-regenerates data/deviation_dataset.csv.
4. Retrains all 6 IDS models and saves results/ids_metrics.csv.
5. Runs 15-class per-attack evaluation and saves results/per_attack_comparison.csv.
6. Prints strict timestamp audit and before/after metric comparison.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.twin_model import DigitalTwin, CONTINUOUS_FEATURES
from src.deviation_engine import process_and_save_deviation_dataset
from src.ids_model import train_and_evaluate_ids_suite
from src.per_attack_analysis import run_per_attack_analysis

def main():
    print("=" * 80)
    print("  IMPLEMENTATION PLAN V11: FULL DOWNSTREAM PIPELINE FORCE-REGENERATION")
    print("  Timestamp: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    
    # ── Step 1: Retrain Digital Twin with Delta Sequences ─────────────────────────
    print("\n>>> STEP 1: Retraining Digital Twin on Normal Baseline Telemetry...")
    df_raw = pd.read_csv("data/sampled_dataset.csv")
    normal_df = df_raw[df_raw["Attack_type"] == "Normal"].copy()
    
    twin = DigitalTwin(window_size=5, hidden_layer_sizes=(64, 32), alpha=0.05, max_iter=250, random_state=42)
    twin.fit(normal_df)
    
    os.makedirs("models", exist_ok=True)
    twin.save("models")
    print(f"[SUCCESS] Saved freshly trained twin to models/twin_model.pkl")
    
    # ── Step 2: Force-Regenerate Deviation Dataset ────────────────────────────────
    print("\n>>> STEP 2: Force-Regenerating data/deviation_dataset.csv...")
    if os.path.exists("data/deviation_dataset.csv"):
        os.remove("data/deviation_dataset.csv")
        print("  Deleted stale data/deviation_dataset.csv")
        
    process_and_save_deviation_dataset(
        sampled_csv="data/sampled_dataset.csv",
        output_dev_csv="data/deviation_dataset.csv",
        output_plot_path="results/deviation_separation.png",
        model_dir="models"
    )
    
    # ── Step 3: Retrain All 6 IDS Models ──────────────────────────────────────────
    print("\n>>> STEP 3: Retraining All 6 IDS Classifiers on Fresh Deviation Dataset...")
    train_and_evaluate_ids_suite(
        sampled_csv="data/sampled_dataset.csv",
        dev_csv="data/deviation_dataset.csv",
        output_metrics_csv="results/ids_metrics.csv",
        output_plot_path="results/ids_comparison.png",
        model_dir="models"
    )
    
    # ── Step 4: Run Per-Attack 15-Class Evaluation ────────────────────────────────
    print("\n>>> STEP 4: Running 15-Class Per-Attack Comparison...")
    run_per_attack_analysis(
        sampled_csv="data/sampled_dataset.csv",
        dev_csv="data/deviation_dataset.csv",
        output_csv="results/per_attack_comparison.csv",
        output_f1_csv="results/per_attack_f1.csv",
        output_plot="results/per_attack_comparison.png",
        output_summary_md="results/per_attack_analysis_summary.md",
        model_dir="models"
    )
    
    # ── Step 5: Verification & Timestamps ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  VERIFICATION & FILE TIMESTAMPS AUDIT")
    print("=" * 80)
    
    files_to_check = [
        "models/twin_model.pkl",
        "data/deviation_dataset.csv",
        "models/xgb_fused.pkl",
        "models/rf_fused.pkl",
        "results/ids_metrics.csv",
        "results/per_attack_comparison.csv"
    ]
    
    for fpath in files_to_check:
        if os.path.exists(fpath):
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M:%S")
            sz = os.path.getsize(fpath)
            print(f"  {fpath:<38} | Modified: {mtime} | Size: {sz:>10,} bytes")
        else:
            print(f"  {fpath:<38} | [ERROR] File not found!")
            
    print("\n>>> Freshly Generated IDS Metrics Table:")
    metrics_df = pd.read_csv("results/ids_metrics.csv")
    print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    main()
