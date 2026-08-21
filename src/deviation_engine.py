"""
Phase B / Milestone 4: Targeted Deviation Residual Engine
Computes absolute residual vectors strictly between incoming continuous telemetry
and the Scope-Restricted Digital Twin forecast: e_t = |y_t - y_hat_t|.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.twin_model import DigitalTwin, CONTINUOUS_FEATURES

class DeviationEngine:
    def __init__(self, twin = None, model_dir: str = "models"):
        if isinstance(twin, str):
            model_dir = twin
            self.twin = DigitalTwin.load(model_dir)
        elif twin is not None:
            self.twin = twin
        else:
            self.twin = DigitalTwin.load(model_dir)
            
        if hasattr(self.twin, "feature_names") and self.twin.feature_names is not None:
            self.continuous_features = list(self.twin.feature_names)
        elif os.path.exists(os.path.join(model_dir, "dev_features.pkl")):
            dev_feats = joblib.load(os.path.join(model_dir, "dev_features.pkl"))
            self.continuous_features = [f.replace("dev_", "") for f in dev_feats]
        else:
            self.continuous_features = list(CONTINUOUS_FEATURES)
            
        self.dev_feature_names = [f"dev_{col}" for col in self.continuous_features]
        
    def compute_deviations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes continuous deviation residuals across the dataframe.
        """
        print(f"Computing Digital Twin forecasts for {len(df)} samples across {len(self.continuous_features)} continuous signals...")
        actual_continuous = df[self.continuous_features].values
        predicted_continuous = self.twin.compute_dataset_predictions(df)
        
        # Absolute residuals
        residuals = np.abs(actual_continuous - predicted_continuous)
        
        dev_df = pd.DataFrame(residuals, columns=self.dev_feature_names, index=df.index)
        dev_df["mean_deviation"] = residuals.mean(axis=1)
        dev_df["max_deviation"] = residuals.max(axis=1)
        
        return dev_df
        
    def compute_single_deviation(self, window_df: pd.DataFrame, current_record: dict) -> dict:
        """
        Real-time single-step deviation calculation for live streaming.
        """
        window_raw = window_df[self.continuous_features].values
        pred_continuous = self.twin.predict_next_state(window_raw)
        
        actual_continuous = np.array([float(current_record[f]) for f in self.continuous_features])
        residuals = np.abs(actual_continuous - pred_continuous)
        
        dev_dict = {f"dev_{f}": float(r) for f, r in zip(self.continuous_features, residuals)}
        dev_dict["mean_deviation"] = float(residuals.mean())
        dev_dict["max_deviation"] = float(residuals.max())
        
        predicted_state_dict = {f: float(p) for f, p in zip(self.continuous_features, pred_continuous)}
        return dev_dict, predicted_state_dict

def process_and_save_deviation_dataset(
    sampled_csv: str = "data/sampled_dataset.csv",
    output_dev_csv: str = "data/deviation_dataset.csv",
    output_plot_path: str = "results/deviation_separation.png",
    model_dir: str = "models"
):
    print("=" * 70)
    print("  TARGETED DEVIATION RESIDUAL GENERATION (PHASE B / TWIN-AUGMENTED-V2)")
    print("=" * 70)
    
    df = pd.read_csv(sampled_csv)
    engine = DeviationEngine(model_dir=model_dir)
    
    dev_df = engine.compute_deviations(df)
    
    # Save dev feature names and fused feature names
    raw_features = joblib.load(os.path.join(model_dir, "raw_features.pkl"))
    dev_features = engine.dev_feature_names
    fused_features = list(raw_features) + list(dev_features)
    
    joblib.dump(dev_features, os.path.join(model_dir, "dev_features.pkl"))
    joblib.dump(fused_features, os.path.join(model_dir, "fused_features.pkl"))
    
    # Save complete targeted deviation dataset
    output_df = pd.concat([df, dev_df], axis=1)
    os.makedirs(os.path.dirname(output_dev_csv), exist_ok=True)
    output_df.to_csv(output_dev_csv, index=False)
    print(f"[SUCCESS] Exported Targeted Deviation Dataset: {output_dev_csv} ({output_df.shape[0]} rows, {output_df.shape[1]} cols)")
    
    # Statistical Separation Plot
    print("Generating targeted deviation separation analysis plot...")
    plt.figure(figsize=(14, 5))
    
    normal_devs = output_df[output_df["Attack_label"] == 0]["mean_deviation"].values
    attack_devs = output_df[output_df["Attack_label"] == 1]["mean_deviation"].values
    
    # Subplot 1: Boxplot comparison
    plt.subplot(1, 2, 1)
    box_data = [np.log1p(normal_devs), np.log1p(attack_devs)]
    plt.boxplot(box_data, tick_labels=['Normal Baseline', 'Attack Telemetry'], patch_artist=True,
                boxprops=dict(facecolor='#38bdf8', color='#0284c7'),
                medianprops=dict(color='#dc2626', lw=2))
    plt.ylabel('Log(1 + Mean Physical Residual)', fontweight='bold')
    plt.title('Targeted Residual Separation (Log Scale)', fontweight='bold', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Subplot 2: Per-attack category mean residual bar chart
    plt.subplot(1, 2, 2)
    attack_means = output_df.groupby('Attack_type')['mean_deviation'].mean().sort_values(ascending=True)
    colors = ['#10b981' if k == 'Normal' else '#f59e0b' for k in attack_means.index]
    plt.barh(attack_means.index, attack_means.values, color=colors)
    plt.xlabel('Mean Continuous Physical Deviation', fontweight='bold')
    plt.title('Physical Signal Deviation Across Attack Categories', fontweight='bold', fontsize=11)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved targeted deviation separation plot to: {output_plot_path}")
    
    return output_df

if __name__ == "__main__":
    process_and_save_deviation_dataset()
