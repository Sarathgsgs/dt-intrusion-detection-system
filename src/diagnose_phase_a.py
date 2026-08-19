"""
Phase A: Root-Cause Diagnosis Script
Systematic diagnosis of the Digital Twin feature scope, per-feature forecasting error decomposition,
feature scaling audit, and per-class classification breakdown.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.twin_model import DigitalTwin

def run_phase_a_diagnosis(
    sampled_csv: str = "data/sampled_dataset.csv",
    dev_csv: str = "data/deviation_dataset.csv",
    output_json: str = "results/diagnostic_phase_a.json",
    output_plot: str = "results/twin_per_feature_error.png"
):
    print("=" * 70)
    print("  PHASE A: ROOT-CAUSE DIAGNOSIS & FEATURE PARTITIONING AUDIT")
    print("=" * 70)
    
    df = pd.read_csv(sampled_csv)
    raw_feature_cols = joblib.load("models/raw_features.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    
    # ---------------------------------------------------------
    # 1. Feature Partitioning: Continuous/Physical vs Categorical/Discrete
    # ---------------------------------------------------------
    print("\n[Step 1] Partitioning 34 Features by Domain Semantics & Temporal Continuity...")
    
    categorical_keywords = [
        'port', 'flags', 'opcode', 'hw.size', 'conn', 'rst', 'syn', 'fin', 
        'ack_raw', 'trans_id', 'unit_id', 'qu', 'retransmission', 'ver', 'proto'
    ]
    
    continuous_features = []
    categorical_features = []
    
    for col in raw_feature_cols:
        nunique = df[col].nunique()
        is_discrete = any(kw in col.lower() for kw in categorical_keywords) or (nunique <= 10)
        
        # Specific semantic adjustments for industrial network telemetry
        if col in ['tcp.dstport', 'tcp.srcport', 'udp.port']:
            is_discrete = True # Port numbers are discrete identifiers, not smooth physical states
        elif col in ['tcp.flags', 'tcp.flags.ack', 'tcp.connection.syn', 'tcp.connection.fin', 'tcp.connection.rst', 'tcp.connection.synack']:
            is_discrete = True # Protocol state flags
        elif col in ['arp.opcode', 'arp.hw.size']:
            is_discrete = True # Discrete protocol codes
            
        if is_discrete:
            categorical_features.append(col)
        else:
            continuous_features.append(col)
            
    print(f"\n--> Continuous / Physical Features ({len(continuous_features)}):")
    for f in continuous_features:
        print(f"    - {f:<25} (unique values: {df[f].nunique():<6}, min: {df[f].min():<10.2f}, max: {df[f].max():<12.2f})")
        
    print(f"\n--> Categorical / Discrete Features ({len(categorical_features)}):")
    for f in categorical_features:
        print(f"    - {f:<25} (unique values: {df[f].nunique():<6}, min: {df[f].min():<10.2f}, max: {df[f].max():<12.2f})")
        
    # ---------------------------------------------------------
    # 2. Per-Feature Error Decomposition of the Digital Twin
    # ---------------------------------------------------------
    print("\n[Step 2] Decomposing Digital Twin Forecasting Errors Per Feature...")
    twin = DigitalTwin.load("models")
    normal_df = df[df["Attack_type"] == "Normal"].copy().reset_index(drop=True)
    
    # Validation split on normal data
    val_size = int(len(normal_df) * 0.2)
    val_df = normal_df.iloc[-val_size:].reset_index(drop=True)
    
    raw_val = val_df[raw_feature_cols].values
    scaled_val = twin.scaler.transform(raw_val)
    
    # Sliding sequence predictions
    W = twin.window_size
    scaled_preds = []
    scaled_targets = []
    
    for i in range(len(scaled_val) - W):
        window = scaled_val[i:i+W].flatten().reshape(1, -1)
        pred = twin.model.predict(window)
        scaled_preds.append(pred[0])
        scaled_targets.append(scaled_val[i+W])
        
    scaled_preds = np.array(scaled_preds)
    scaled_targets = np.array(scaled_targets)
    
    # Unscale predictions
    unscaled_preds = twin.scaler.inverse_transform(scaled_preds)
    unscaled_targets = twin.scaler.inverse_transform(scaled_targets)
    
    per_feature_errors = []
    for idx, col in enumerate(raw_feature_cols):
        feat_scaled_mse = float(np.mean((scaled_targets[:, idx] - scaled_preds[:, idx]) ** 2))
        feat_scaled_mae = float(np.mean(np.abs(scaled_targets[:, idx] - scaled_preds[:, idx])))
        feat_raw_mae = float(np.mean(np.abs(unscaled_targets[:, idx] - unscaled_preds[:, idx])))
        feat_type = "Continuous" if col in continuous_features else "Discrete/Categorical"
        
        per_feature_errors.append({
            "feature": col,
            "type": feat_type,
            "scaled_mse": feat_scaled_mse,
            "scaled_mae": feat_scaled_mae,
            "raw_mae": feat_raw_mae,
            "raw_min": float(df[col].min()),
            "raw_max": float(df[col].max())
        })
        
    per_feature_df = pd.DataFrame(per_feature_errors).sort_values("scaled_mse", ascending=False)
    print("\n--- Per-Feature Error Ranking (Top 10 Highest Error Features in Twin) ---")
    print(per_feature_df.head(10)[["feature", "type", "scaled_mse", "scaled_mae", "raw_mae"]].to_string(index=False))
    
    print("\n--- Summary Error by Feature Group ---")
    group_summary = per_feature_df.groupby("type")[["scaled_mse", "scaled_mae"]].mean()
    print(group_summary)
    
    # ---------------------------------------------------------
    # 3. Scaling & Magnitude Dominance Audit
    # ---------------------------------------------------------
    print("\n[Step 3] Scaling & Magnitude Dominance Audit...")
    print(f"Scaler Mean Vector Shape: {twin.scaler.mean_.shape}")
    print(f"Scaler Scale Vector Range: Min Scale={twin.scaler.scale_.min():.4f}, Max Scale={twin.scaler.scale_.max():.4f}")
    
    # High-magnitude raw features check
    dominant_features = [f for f in per_feature_errors if f["raw_max"] > 10000]
    print(f"Features with huge unscaled raw magnitudes (> 10,000): {[d['feature'] for d in dominant_features]}")
    
    # ---------------------------------------------------------
    # 4. Per-Class Breakdown: XGB-Raw vs RF-Twin-Augmented
    # ---------------------------------------------------------
    print("\n[Step 4] Computing Per-Class Performance Breakdown...")
    dev_df = pd.read_csv(dev_csv)
    
    y_true = label_encoder.transform(df["Attack_type"].astype(str))
    dev_features = joblib.load("models/dev_features.pkl")
    fused_features = joblib.load("models/fused_features.pkl")
    
    X_raw = df[raw_feature_cols].values
    X_fused = np.hstack([X_raw, dev_df[dev_features].values])
    
    _, test_idx = train_test_split(np.arange(len(y_true)), test_size=0.2, random_state=42, stratify=y_true)
    
    xgb_raw = joblib.load("models/xgb_raw.pkl")
    rf_raw = joblib.load("models/rf_raw.pkl")
    rf_fused = joblib.load("models/rf_fused.pkl")
    xgb_fused = joblib.load("models/xgb_fused.pkl")
    
    y_test = y_true[test_idx]
    X_raw_test = X_raw[test_idx]
    X_fused_test = X_fused[test_idx]
    
    y_pred_xgb_raw = xgb_raw.predict(X_raw_test)
    y_pred_rf_fused = rf_fused.predict(X_fused_test)
    y_pred_xgb_fused = xgb_fused.predict(X_fused_test)
    
    class_names = list(label_encoder.classes_)
    
    report_xgb_raw = classification_report(y_test, y_pred_xgb_raw, target_names=class_names, output_dict=True, zero_division=0)
    report_rf_fused = classification_report(y_test, y_pred_rf_fused, target_names=class_names, output_dict=True, zero_division=0)
    report_xgb_fused = classification_report(y_test, y_pred_xgb_fused, target_names=class_names, output_dict=True, zero_division=0)
    
    per_class_comparison = []
    for cls in class_names:
        f1_xgb_raw = report_xgb_raw[cls]["f1-score"]
        f1_rf_fused = report_rf_fused[cls]["f1-score"]
        f1_xgb_fused = report_xgb_fused[cls]["f1-score"]
        support = int(report_xgb_raw[cls]["support"])
        
        diff_rf = f1_rf_fused - f1_xgb_raw
        diff_xgb = f1_xgb_fused - f1_xgb_raw
        
        per_class_comparison.append({
            "Attack Class": cls,
            "Support (Test)": support,
            "XGB-Raw F1": round(f1_xgb_raw, 4),
            "RF-Twin-Augmented F1": round(f1_rf_fused, 4),
            "XGB-Twin-Augmented F1": round(f1_xgb_fused, 4),
            "RF vs Raw Diff": round(diff_rf, 4),
            "XGB vs Raw Diff": round(diff_xgb, 4),
            "Status": "Twin Advantage" if (diff_xgb > 0 or diff_rf > 0) else "Raw Advantage"
        })
        
    per_class_df = pd.DataFrame(per_class_comparison)
    print("\n--- Per-Class Performance Breakdown (Side-by-Side) ---")
    print(per_class_df.to_string(index=False))
    
    # ---------------------------------------------------------
    # 5. Plotting Per-Feature Error Decomposition
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 6))
    
    # Sort for bar plot
    plot_df = per_feature_df.sort_values("scaled_mse", ascending=True)
    colors = ['#38bdf8' if t == 'Continuous' else '#f59e0b' for t in plot_df["type"]]
    
    plt.barh(plot_df["feature"], plot_df["scaled_mse"], color=colors)
    plt.xlabel('Validation Scaled Mean Squared Error (MSE)', fontweight='bold', fontsize=11)
    plt.ylabel('Feature Name', fontweight='bold', fontsize=11)
    plt.title('Digital Twin Per-Feature Error Decomposition (Blue: Continuous, Amber: Discrete/Categorical)', fontweight='bold', fontsize=12)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    
    # Legend proxy
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#38bdf8', label='Continuous / Physical (Forecastable)'),
        Patch(facecolor='#f59e0b', label='Discrete / Categorical (Non-Smooth Dynamics)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot, dpi=300)
    plt.close()
    print(f"\n[SUCCESS] Saved per-feature error decomposition chart to: {output_plot}")
    
    # ---------------------------------------------------------
    # 6. Save Diagnostic Summary JSON
    # ---------------------------------------------------------
    diagnostic_results = {
        "continuous_features": continuous_features,
        "categorical_features": categorical_features,
        "n_continuous": len(continuous_features),
        "n_categorical": len(categorical_features),
        "group_error_summary": {
            "continuous_mean_scaled_mse": float(group_summary.loc["Continuous", "scaled_mse"]) if "Continuous" in group_summary.index else 0,
            "categorical_mean_scaled_mse": float(group_summary.loc["Discrete/Categorical", "scaled_mse"]) if "Discrete/Categorical" in group_summary.index else 0
        },
        "per_feature_ranking": per_feature_df.to_dict(orient="records"),
        "per_class_comparison": per_class_comparison
    }
    
    with open(output_json, "w") as f:
        json.dump(diagnostic_results, f, indent=2)
    print(f"[SUCCESS] Saved Phase A diagnostic report to: {output_json}")
    
    return diagnostic_results

if __name__ == "__main__":
    run_phase_a_diagnosis()
