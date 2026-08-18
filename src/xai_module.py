"""
Milestone 6: SHAP Explainability Module (XAI)
Provides local feature attribution and global explainability for IDS detections using TreeExplainer.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class ExplainabilityModule:
    def __init__(self, model_path: str = "models/xgb_raw.pkl", feature_names_path: str = "models/raw_features.pkl", label_encoder_path: str = "models/label_encoder.pkl"):
        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(feature_names_path)
        self.label_encoder = joblib.load(label_encoder_path)
        print("Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)
        
    def explain_sample(self, feature_vector: np.ndarray, top_k: int = 5) -> dict:
        """
        Computes local SHAP attribution for a single sample.
        """
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
            
        # Prediction
        probs = self.model.predict_proba(feature_vector)[0]
        pred_class_idx = int(np.argmax(probs))
        pred_class_name = self.label_encoder.inverse_transform([pred_class_idx])[0]
        confidence = float(probs[pred_class_idx])
        
        # SHAP values
        shap_values = self.explainer.shap_values(feature_vector)
        
        # Handle multiclass vs binary shap output format
        if isinstance(shap_values, list):
            # List of arrays per class
            class_shap = shap_values[pred_class_idx][0]
        elif shap_values.ndim == 3:
            # Array of shape (n_samples, n_features, n_classes)
            class_shap = shap_values[0, :, pred_class_idx]
        else:
            class_shap = shap_values[0]
            
        # Sort top features by absolute magnitude
        top_indices = np.argsort(np.abs(class_shap))[::-1][:top_k]
        
        top_features = []
        for idx in top_indices:
            top_features.append({
                "feature": self.feature_names[idx],
                "feature_value": float(feature_vector[0, idx]),
                "shap_value": float(class_shap[idx]),
                "contribution": "Increases Risk" if class_shap[idx] > 0 else "Decreases Risk"
            })
            
        explanation = {
            "predicted_class": pred_class_name,
            "predicted_class_idx": pred_class_idx,
            "confidence": confidence,
            "top_features": top_features
        }
        return explanation
        
    def generate_global_summary(self, X_sample: np.ndarray, output_path: str = "results/shap_summary.png"):
        print(f"Generating global SHAP summary plot on {len(X_sample)} samples...")
        shap_values = self.explainer.shap_values(X_sample)
        
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            # Multiclass bar summary
            shap.summary_plot(
                shap_values, 
                X_sample, 
                feature_names=self.feature_names, 
                class_names=list(self.label_encoder.classes_),
                show=False, 
                max_display=12
            )
        elif shap_values.ndim == 3:
            # For 3D array, average across classes or plot primary
            shap.summary_plot(
                shap_values[:, :, 0], 
                X_sample, 
                feature_names=self.feature_names, 
                show=False, 
                max_display=12
            )
        else:
            shap.summary_plot(
                shap_values, 
                X_sample, 
                feature_names=self.feature_names, 
                show=False, 
                max_display=12
            )
            
        plt.title('Global SHAP Feature Importance across IIoT Attack Types', fontweight='bold', fontsize=13)
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SUCCESS] Saved SHAP summary plot to: {output_path}")

def run_xai_demo():
    xai = ExplainabilityModule("models/xgb_raw.pkl", "models/raw_features.pkl", "models/label_encoder.pkl")
    
    # Load sample test data
    df = pd.read_csv("data/sampled_dataset.csv")
    feature_cols = joblib.load("models/raw_features.pkl")
    X = df[feature_cols].values
    
    # Single sample explanation
    sample_idx = 42
    print(f"\nExplaining sample index {sample_idx} (Ground truth: {df.iloc[sample_idx]['Attack_type']}):")
    exp = xai.explain_sample(X[sample_idx])
    print(f"Predicted Class: {exp['predicted_class']} (Confidence: {exp['confidence']*100:.2f}%)")
    print("Top Influential Features:")
    for f in exp["top_features"]:
        print(f"  - {f['feature']:<22} = {f['feature_value']:<10.2f} (SHAP: {f['shap_value']:+.4f}) -> {f['contribution']}")
        
    # Global summary
    xai.generate_global_summary(X[:300], "results/shap_summary.png")

if __name__ == "__main__":
    run_xai_demo()
