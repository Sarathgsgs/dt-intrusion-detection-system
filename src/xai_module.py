"""
Phase B / Milestone 6: SHAP Local & Global Explainability Module
Calculates exact Shapley additive feature attributions for Twin-Augmented-v2 model.
"""

import os
import sys
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class XAIExplainer:
    def __init__(self, model_path: str = "models/xgb_fused.pkl", model_dir: str = "models"):
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        self.fused_features = joblib.load(os.path.join(model_dir, "fused_features.pkl"))
        self.explainer = shap.TreeExplainer(self.model)
        
    def explain_sample(self, feature_vector: np.ndarray, top_k: int = 5) -> dict:
        """
        Computes local SHAP attributions for a single telemetry sample.
        """
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
            
        prob_dist = self.model.predict_proba(feature_vector)[0]
        pred_class_idx = int(np.argmax(prob_dist))
        pred_class_name = self.label_encoder.inverse_transform([pred_class_idx])[0]
        confidence = float(prob_dist[pred_class_idx])
        
        shap_values = self.explainer.shap_values(feature_vector)
        
        if isinstance(shap_values, list):
            class_shap = shap_values[pred_class_idx][0]
        elif shap_values.ndim == 3:
            class_shap = shap_values[0, :, pred_class_idx]
        else:
            class_shap = shap_values[0]
            
        sorted_indices = np.argsort(np.abs(class_shap))[::-1][:top_k]
        
        top_attributions = []
        for idx in sorted_indices:
            feat_name = self.fused_features[idx]
            val = float(feature_vector[0, idx])
            sv = float(class_shap[idx])
            top_attributions.append({
                "feature": feat_name,
                "feature_value": round(val, 4),
                "shap_value": round(sv, 4),
                "contribution": "Increases Risk" if sv > 0 else "Decreases Risk"
            })
            
        return {
            "predicted_class": pred_class_name,
            "confidence": round(confidence, 4),
            "top_features": top_attributions
        }

def generate_global_shap_summary(
    dev_csv: str = "data/deviation_dataset.csv",
    output_plot_path: str = "results/shap_summary.png",
    model_dir: str = "models",
    n_samples: int = 500
):
    print("=" * 70)
    print("  SHAP GLOBAL ATTRIBUTION SUMMARY (PHASE B / TWIN-AUGMENTED-V2)")
    print("=" * 70)
    
    df = pd.read_csv(dev_csv)
    raw_cols = joblib.load(os.path.join(model_dir, "raw_features.pkl"))
    dev_cols = joblib.load(os.path.join(model_dir, "dev_features.pkl"))
    fused_cols = list(raw_cols) + list(dev_cols)
    
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42)
    X_sample = sample_df[fused_cols].values
    
    model = joblib.load(os.path.join(model_dir, "xgb_fused.pkl"))
    explainer = shap.TreeExplainer(model)
    
    print(f"Computing TreeExplainer SHAP values for {X_sample.shape[0]} samples across {X_sample.shape[1]} features...")
    shap_vals = explainer.shap_values(X_sample)
    
    if isinstance(shap_vals, list):
        mean_abs_shap = np.mean([np.abs(sv) for sv in shap_vals], axis=0).mean(axis=0)
    elif shap_vals.ndim == 3:
        mean_abs_shap = np.abs(shap_vals).mean(axis=(0, 2))
    else:
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        
    top_indices = np.argsort(mean_abs_shap)[::-1][:12]
    top_features = [fused_cols[i] for i in top_indices]
    top_scores = mean_abs_shap[top_indices]
    
    plt.figure(figsize=(12, 6))
    colors = ['#38bdf8' if f.startswith('dev_') else '#6366f1' for f in reversed(top_features)]
    
    plt.barh(list(reversed(top_features)), list(reversed(top_scores)), color=colors)
    plt.xlabel('Mean |SHAP Value| (Average Impact on Threat Classification)', fontweight='bold')
    plt.title('Global Feature Importance Ranking: Raw Features vs. Continuous Deviation Signals', fontweight='bold', fontsize=12)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#6366f1', label='Raw Telemetry Feature'),
        Patch(facecolor='#38bdf8', label='Continuous Physical Deviation (Twin Residual)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved global SHAP summary plot to: {output_plot_path}")

if __name__ == "__main__":
    generate_global_shap_summary()
