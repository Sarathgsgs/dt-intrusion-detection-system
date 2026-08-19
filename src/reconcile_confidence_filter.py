"""
Phase D: Confidence Filter Multi-Split Metric Reconciliation Script
Empirically benchmarks the operational alert suppression rate across 5 distinct test splits
to establish a single, verifiable, and reproducible statistic.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.confidence_filter import OperationalConfidenceFilter
from src.xai_module import XAIExplainer

def reconcile_confidence_filter_metrics(
    sampled_csv: str = "data/sampled_dataset.csv",
    dev_csv: str = "data/deviation_dataset.csv",
    output_json: str = "results/confidence_filter_reconciliation.json",
    model_dir: str = "models",
    n_splits: int = 5,
    samples_per_split: int = 1000
):
    print("=" * 70)
    print("  PHASE D: CONFIDENCE-FILTER MULTI-SPLIT METRIC RECONCILIATION")
    print("=" * 70)
    
    df_raw = pd.read_csv(sampled_csv)
    df_dev = pd.read_csv(dev_csv)
    
    raw_feature_cols = joblib.load(os.path.join(model_dir, "raw_features.pkl"))
    dev_feature_cols = joblib.load(os.path.join(model_dir, "dev_features.pkl"))
    fused_features = list(raw_feature_cols) + list(dev_feature_cols)
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    
    y = label_encoder.transform(df_raw["Attack_type"].astype(str))
    X_fused = np.hstack([df_raw[raw_feature_cols].values, df_dev[dev_feature_cols].values])
    
    explainer = XAIExplainer(model_path=os.path.join(model_dir, "xgb_fused.pkl"), model_dir=model_dir)
    
    split_results = []
    
    print(f"\nBenchmarking Alert Suppression across {n_splits} distinct random splits (Threshold: gamma >= 0.65)...")
    
    for seed in range(1, n_splits + 1):
        filter_engine = OperationalConfidenceFilter(min_confidence=0.65, min_signature_overlap=1)
        
        # Draw sample
        _, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42 + seed, stratify=y)
        eval_indices = np.random.RandomState(42 + seed).choice(test_idx, size=min(samples_per_split, len(test_idx)), replace=False)
        
        for idx in eval_indices:
            feat_vec = X_fused[idx]
            explanation = explainer.explain_sample(feat_vec, top_k=5)
            pred_dict = {
                "predicted_class": explanation["predicted_class"],
                "confidence": explanation["confidence"]
            }
            filter_engine.evaluate(pred_dict, explanation)
            
        total_inspected = filter_engine.stats["total_inspected"]
        normal_count = filter_engine.stats["normal_traffic"]
        total_alerts = total_inspected - normal_count
        passed_alerts = filter_engine.stats["passed_alerts"]
        suppressed_alerts = filter_engine.stats["suppressed_alerts"]
        
        suppression_rate = (suppressed_alerts / total_alerts * 100.0) if total_alerts > 0 else 0.0
        
        print(f"  [Split {seed}] Total Alerts: {total_alerts:<4} | Passed: {passed_alerts:<4} | Suppressed: {suppressed_alerts:<4} | Suppression Rate: {suppression_rate:.2f}%")
        
        split_results.append({
            "split_id": seed,
            "total_alerts": int(total_alerts),
            "passed_alerts": int(passed_alerts),
            "suppressed_alerts": int(suppressed_alerts),
            "suppression_rate_pct": round(suppression_rate, 2)
        })
        
    rates = [r["suppression_rate_pct"] for r in split_results]
    mean_rate = float(np.mean(rates))
    min_rate = float(np.min(rates))
    max_rate = float(np.max(rates))
    std_rate = float(np.std(rates))
    
    reconciled_metrics = {
        "confidence_threshold": 0.65,
        "n_splits": n_splits,
        "mean_suppression_rate_pct": round(mean_rate, 2),
        "min_suppression_rate_pct": round(min_rate, 2),
        "max_suppression_rate_pct": round(max_rate, 2),
        "std_dev_pct": round(std_rate, 2),
        "canonical_documented_string": f"{mean_rate:.1f}% (range: {min_rate:.1f}% - {max_rate:.1f}%)",
        "split_breakdown": split_results
    }
    
    print("\n" + "=" * 70)
    print("  RECONCILIATION SUMMARY")
    print("=" * 70)
    print(f"  Mean Alert Suppression Rate: {mean_rate:.2f}%")
    print(f"  Empirical Range:             {min_rate:.2f}% - {max_rate:.2f}% (Std: ±{std_rate:.2f}%)")
    print(f"  Canonical Documented Metric: \"{reconciled_metrics['canonical_documented_string']}\"")
    print("=" * 70)
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(reconciled_metrics, f, indent=2)
    print(f"\n[SUCCESS] Saved reconciled confidence filter metrics to: {output_json}")
    
    return reconciled_metrics

if __name__ == "__main__":
    reconcile_confidence_filter_metrics()
