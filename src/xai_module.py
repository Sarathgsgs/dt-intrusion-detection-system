"""
Phase B / Milestone 6: SHAP Local & Global Explainability Module
Calculates exact Shapley additive feature attributions for Twin-Augmented and Baseline models.
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
    def __init__(
        self, 
        model_path: str = "models/xgb_fused.pkl", 
        features_path_or_dir: str = "models",
        label_encoder_path: str = None,
        model_dir: str = "models"
    ):
        self.model = joblib.load(model_path)
        
        # Resolve label encoder path
        if label_encoder_path and os.path.exists(label_encoder_path):
            self.label_encoder = joblib.load(label_encoder_path)
        else:
            self.label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
            
        # Resolve feature names
        if features_path_or_dir.endswith(".pkl") and os.path.exists(features_path_or_dir):
            self.fused_features = joblib.load(features_path_or_dir)
        elif os.path.exists(os.path.join(model_dir, "fused_features.pkl")):
            self.fused_features = joblib.load(os.path.join(model_dir, "fused_features.pkl"))
        else:
            self.fused_features = joblib.load(os.path.join(model_dir, "raw_features.pkl"))
            
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
            feat_name = self.fused_features[idx] if idx < len(self.fused_features) else f"feature_{idx}"
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

# Alias for backward compatibility
ExplainabilityModule = XAIExplainer

if __name__ == "__main__":
    print("Testing XAIExplainer / ExplainabilityModule initialization...")
    xai = ExplainabilityModule("models/xgb_raw.pkl", "models/raw_features.pkl", "models/label_encoder.pkl")
    print("[SUCCESS] XAIExplainer loaded cleanly!")
