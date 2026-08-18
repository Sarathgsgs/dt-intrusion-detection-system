"""
Milestone 5: IDS Classifiers (Baseline vs. Twin-Deviation vs. Twin-Augmented)
Comprehensive multi-class evaluation comparing Raw Telemetry, Pure Deviation, and Twin-Augmented feature spaces.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class IDSClassifierSuite:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.metrics = {}
        
    def prepare_data(self, raw_csv: str = "data/sampled_dataset.csv", dev_csv: str = "data/deviation_dataset.csv"):
        print("Loading raw and deviation datasets...")
        raw_df = pd.read_csv(raw_csv)
        dev_df = pd.read_csv(dev_csv)
        
        target_col = "Attack_type"
        y = self.label_encoder.fit_transform(raw_df[target_col].astype(str))
        
        raw_feature_cols = [c for c in raw_df.columns if c not in ["Attack_type", "Attack_label"]]
        dev_feature_cols = [c for c in dev_df.columns if c.startswith("dev_")]
        
        X_raw = raw_df[raw_feature_cols].values
        X_dev = dev_df[dev_feature_cols].values
        X_fused = np.hstack([X_raw, X_dev])
        fused_feature_cols = raw_feature_cols + dev_feature_cols
        
        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        data_splits = {
            "raw_train": (X_raw[train_idx], y[train_idx]),
            "raw_test": (X_raw[test_idx], y[test_idx]),
            "dev_train": (X_dev[train_idx], y[train_idx]),
            "dev_test": (X_dev[test_idx], y[test_idx]),
            "fused_train": (X_fused[train_idx], y[train_idx]),
            "fused_test": (X_fused[test_idx], y[test_idx]),
            "raw_features": raw_feature_cols,
            "dev_features": dev_feature_cols,
            "fused_features": fused_feature_cols,
            "class_names": list(self.label_encoder.classes_)
        }
        return data_splits
        
    def train_and_evaluate(self, splits: dict):
        class_names = splits["class_names"]
        
        # 1. Random Forest on Raw Data
        print("\n[1/6] Training Baseline Random Forest on Raw Telemetry...")
        rf_raw = RandomForestClassifier(
            n_estimators=100, max_depth=16, class_weight="balanced", n_jobs=-1, random_state=self.random_state
        )
        rf_raw.fit(splits["raw_train"][0], splits["raw_train"][1])
        y_pred_rf_raw = rf_raw.predict(splits["raw_test"][0])
        self._record_metrics("RF-Raw (Baseline)", splits["raw_test"][1], y_pred_rf_raw, class_names)
        self.models["rf_raw"] = rf_raw
        
        # 2. XGBoost on Raw Data
        print("\n[2/6] Training Baseline XGBoost on Raw Telemetry...")
        xgb_raw = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=self.random_state, eval_metric="mlogloss"
        )
        xgb_raw.fit(splits["raw_train"][0], splits["raw_train"][1])
        y_pred_xgb_raw = xgb_raw.predict(splits["raw_test"][0])
        self._record_metrics("XGB-Raw (Baseline)", splits["raw_test"][1], y_pred_xgb_raw, class_names)
        self.models["xgb_raw"] = xgb_raw
        
        # 3. Random Forest on Deviation Features
        print("\n[3/6] Training Random Forest on Pure Twin Deviation...")
        rf_dev = RandomForestClassifier(
            n_estimators=100, max_depth=16, class_weight="balanced", n_jobs=-1, random_state=self.random_state
        )
        rf_dev.fit(splits["dev_train"][0], splits["dev_train"][1])
        y_pred_rf_dev = rf_dev.predict(splits["dev_test"][0])
        self._record_metrics("RF-Deviation (Pure)", splits["dev_test"][1], y_pred_rf_dev, class_names)
        self.models["rf_dev"] = rf_dev
        
        # 4. XGBoost on Deviation Features
        print("\n[4/6] Training XGBoost on Pure Twin Deviation...")
        xgb_dev = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=self.random_state, eval_metric="mlogloss"
        )
        xgb_dev.fit(splits["dev_train"][0], splits["dev_train"][1])
        y_pred_xgb_dev = xgb_dev.predict(splits["dev_test"][0])
        self._record_metrics("XGB-Deviation (Pure)", splits["dev_test"][1], y_pred_xgb_dev, class_names)
        self.models["xgb_dev"] = xgb_dev

        # 5. Random Forest on Twin-Augmented (Raw + Deviation)
        print("\n[5/6] Training Random Forest on Twin-Augmented (Raw + Deviation)...")
        rf_fused = RandomForestClassifier(
            n_estimators=100, max_depth=16, class_weight="balanced", n_jobs=-1, random_state=self.random_state
        )
        rf_fused.fit(splits["fused_train"][0], splits["fused_train"][1])
        y_pred_rf_fused = rf_fused.predict(splits["fused_test"][0])
        self._record_metrics("RF-Twin-Augmented", splits["fused_test"][1], y_pred_rf_fused, class_names)
        self.models["rf_fused"] = rf_fused
        
        # 6. XGBoost on Twin-Augmented (Raw + Deviation)
        print("\n[6/6] Training XGBoost on Twin-Augmented (Raw + Deviation)...")
        xgb_fused = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=self.random_state, eval_metric="mlogloss"
        )
        xgb_fused.fit(splits["fused_train"][0], splits["fused_train"][1])
        y_pred_xgb_fused = xgb_fused.predict(splits["fused_test"][0])
        self._record_metrics("XGB-Twin-Augmented", splits["fused_test"][1], y_pred_xgb_fused, class_names)
        self.models["xgb_fused"] = xgb_fused
        
    def _record_metrics(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, class_names: list):
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        
        self.metrics[model_name] = {
            "Accuracy": float(acc),
            "Macro-F1": float(macro_f1),
            "Weighted-F1": float(weighted_f1),
            "Macro-Precision": float(macro_prec),
            "Macro-Recall": float(macro_rec)
        }
        print(f"--> {model_name:<28} | Accuracy: {acc*100:.2f}% | Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f}")
        
    def save_artifacts(self, model_dir: str = "models", results_dir: str = "results", splits: dict = None):
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(model_dir, f"{name}.pkl"))
        joblib.dump(self.label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
        
        if splits:
            joblib.dump(splits["raw_features"], os.path.join(model_dir, "raw_features.pkl"))
            joblib.dump(splits["dev_features"], os.path.join(model_dir, "dev_features.pkl"))
            joblib.dump(splits["fused_features"], os.path.join(model_dir, "fused_features.pkl"))
            
        summary_data = []
        for model_name, m in self.metrics.items():
            summary_data.append({
                "Model Architecture": model_name,
                "Accuracy (%)": round(m["Accuracy"] * 100, 2),
                "Macro-F1": round(m["Macro-F1"], 4),
                "Weighted-F1": round(m["Weighted-F1"], 4),
                "Macro-Precision": round(m["Macro-Precision"], 4),
                "Macro-Recall": round(m["Macro-Recall"], 4)
            })
        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(results_dir, "ids_metrics.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n[SUCCESS] Saved comparative metrics table to: {summary_csv}")
        
        # Plot Comparison Chart
        plt.figure(figsize=(12, 5))
        x = np.arange(len(summary_df))
        width = 0.25
        
        plt.bar(x - width, summary_df["Accuracy (%)"], width, label='Accuracy (%)', color='#3b82f6')
        plt.bar(x, summary_df["Macro-F1"] * 100, width, label='Macro-F1 (x100)', color='#10b981')
        plt.bar(x + width, summary_df["Weighted-F1"] * 100, width, label='Weighted-F1 (x100)', color='#f59e0b')
        
        plt.xlabel('IDS Model Architecture & Feature Space', fontweight='bold', fontsize=11)
        plt.ylabel('Performance Score (%)', fontweight='bold', fontsize=11)
        plt.title('Baseline (Raw) vs. Pure Deviation vs. Twin-Augmented IDS Comparison', fontweight='bold', fontsize=12)
        plt.xticks(x, summary_df["Model Architecture"], rotation=15, ha='right', fontsize=9, fontweight='bold')
        plt.ylim(30, 105)
        plt.legend(loc='lower right')
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        plot_path = os.path.join(results_dir, "ids_comparison.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[SUCCESS] Saved IDS comparison chart to: {plot_path}")
        
        return summary_df

def run_ids_pipeline():
    suite = IDSClassifierSuite(random_state=42)
    splits = suite.prepare_data("data/sampled_dataset.csv", "data/deviation_dataset.csv")
    suite.train_and_evaluate(splits)
    summary_df = suite.save_artifacts("models", "results", splits)
    print("\n--- Master IDS Model Evaluation Results ---")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    run_ids_pipeline()
