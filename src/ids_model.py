"""
Phase B / Milestone 5: IDS Classifier Suite (Baseline vs. Twin-Augmented-v2)
Trains and compares Random Forest and XGBoost classifiers across Raw, Pure Deviation,
and Targeted Twin-Augmented-v2 feature spaces.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def train_and_evaluate_ids_suite(
    sampled_csv: str = "data/sampled_dataset.csv",
    dev_csv: str = "data/deviation_dataset.csv",
    output_metrics_csv: str = "results/ids_metrics.csv",
    output_plot_path: str = "results/ids_comparison.png",
    model_dir: str = "models"
):
    print("=" * 70)
    print("  IDS CLASSIFIER SUITE: RAW BASELINE VS. TWIN-AUGMENTED-V2 (PHASE B)")
    print("=" * 70)
    
    df_raw = pd.read_csv(sampled_csv)
    df_dev = pd.read_csv(dev_csv)
    
    raw_feature_cols = joblib.load(os.path.join(model_dir, "raw_features.pkl"))
    dev_feature_cols = joblib.load(os.path.join(model_dir, "dev_features.pkl"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    
    print(f"Features: Raw={len(raw_feature_cols)}, Targeted Continuous Dev={len(dev_feature_cols)}")
    
    y = label_encoder.transform(df_raw["Attack_type"].astype(str))
    
    # Feature matrices
    X_raw = df_raw[raw_feature_cols].values
    X_dev = df_dev[dev_feature_cols].values
    X_fused_v2 = np.hstack([X_raw, X_dev])
    
    print(f"Twin-Augmented-v2 Feature Space Shape: {X_fused_v2.shape}")
    
    # Stratified 80/20 train/test split
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    
    y_train, y_test = y[train_idx], y[test_idx]
    
    models_to_train = [
        {
            "name": "RF-Raw (Baseline)",
            "space": "Raw Telemetry (34 features)",
            "model": RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
            "X_tr": X_raw[train_idx],
            "X_te": X_raw[test_idx],
            "save_name": "rf_raw.pkl"
        },
        {
            "name": "XGB-Raw (Baseline)",
            "space": "Raw Telemetry (34 features)",
            "model": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=42),
            "X_tr": X_raw[train_idx],
            "X_te": X_raw[test_idx],
            "save_name": "xgb_raw.pkl"
        },
        {
            "name": "RF-Deviation (Pure Cont Residuals)",
            "space": "Continuous Residuals (9 features)",
            "model": RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
            "X_tr": X_dev[train_idx],
            "X_te": X_dev[test_idx],
            "save_name": "rf_dev.pkl"
        },
        {
            "name": "XGB-Deviation (Pure Cont Residuals)",
            "space": "Continuous Residuals (9 features)",
            "model": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=42),
            "X_tr": X_dev[train_idx],
            "X_te": X_dev[test_idx],
            "save_name": "xgb_dev.pkl"
        },
        {
            "name": "RF-Twin-Augmented-v2 (Targeted)",
            "space": "Raw + Continuous Residuals (43 features)",
            "model": RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
            "X_tr": X_fused_v2[train_idx],
            "X_te": X_fused_v2[test_idx],
            "save_name": "rf_fused.pkl"
        },
        {
            "name": "XGB-Twin-Augmented-v2 (Targeted)",
            "space": "Raw + Continuous Residuals (43 features)",
            "model": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=42),
            "X_tr": X_fused_v2[train_idx],
            "X_te": X_fused_v2[test_idx],
            "save_name": "xgb_fused.pkl"
        }
    ]
    
    results = []
    
    for item in models_to_train:
        print(f"\nTraining [{item['name']}] on {item['space']} ...")
        t0 = time.time()
        clf = item["model"]
        clf.fit(item["X_tr"], y_train)
        train_time = time.time() - t0
        
        t0 = time.time()
        y_pred = clf.predict(item["X_te"])
        infer_time = (time.time() - t0) / len(y_test) * 1000.0 # ms per sample
        
        acc = accuracy_score(y_test, y_pred) * 100.0
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        
        print(f"--> Acc: {acc:.2f}% | Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f} | Latency: {infer_time:.4f} ms/sample")
        
        joblib.dump(clf, os.path.join(model_dir, item["save_name"]))
        
        results.append({
            "Model Architecture": item["name"],
            "Feature Space": item["space"],
            "Accuracy (%)": round(acc, 2),
            "Macro-F1": round(macro_f1, 4),
            "Weighted-F1": round(weighted_f1, 4),
            "Macro-Precision": round(macro_prec, 4),
            "Macro-Recall": round(macro_rec, 4),
            "Train Time (s)": round(train_time, 2),
            "Inference Latency (ms/sample)": round(infer_time, 4)
        })
        
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_metrics_csv), exist_ok=True)
    results_df.to_csv(output_metrics_csv, index=False)
    print(f"\n[SUCCESS] Saved comprehensive metrics to: {output_metrics_csv}")
    print("\n" + results_df.to_string(index=False))
    
    # Generate Comparison Plot
    print("\nGenerating IDS Model Comparison Plot...")
    plt.figure(figsize=(13, 6))
    
    x = np.arange(len(results_df))
    width = 0.25
    
    plt.bar(x - width, results_df["Accuracy (%)"] / 100.0, width, label='Accuracy', color='#38bdf8')
    plt.bar(x, results_df["Macro-F1"], width, label='Macro-F1 Score', color='#10b981')
    plt.bar(x + width, results_df["Weighted-F1"], width, label='Weighted-F1 Score', color='#6366f1')
    
    plt.ylabel('Score (0.0 - 1.0)', fontweight='bold')
    plt.title('Multi-Class IDS Performance Across Baseline vs. Twin-Augmented-v2 Feature Spaces', fontweight='bold', fontsize=12)
    plt.xticks(x, results_df["Model Architecture"], rotation=20, ha='right', fontsize=9)
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved IDS comparison plot to: {output_plot_path}")
    
    return results_df

if __name__ == "__main__":
    train_and_evaluate_ids_suite()
