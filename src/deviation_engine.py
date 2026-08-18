"""
Milestone 4: Deviation Engine Module
Computes residual deviation vectors |actual - predicted| between real IoT telemetry
and Digital Twin baseline forecasts.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.twin_model import DigitalTwin
except ImportError:
    from twin_model import DigitalTwin

class DeviationEngine:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.twin = DigitalTwin.load(model_dir)
        self.feature_names = self.twin.feature_names
        
    def compute_deviations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates deviation vector = |actual - predicted| for each feature.
        """
        raw_features = df[self.feature_names].values
        predicted_features = self.twin.compute_dataset_predictions(df)
        
        # Absolute residuals
        deviations = np.abs(raw_features - predicted_features)
        
        # Create deviation DataFrame
        dev_cols = [f"dev_{col}" for col in self.feature_names]
        dev_df = pd.DataFrame(deviations, columns=dev_cols, index=df.index)
        
        # Carry over labels if present
        if "Attack_type" in df.columns:
            dev_df["Attack_type"] = df["Attack_type"]
        if "Attack_label" in df.columns:
            dev_df["Attack_label"] = df["Attack_label"]
            
        return dev_df

    def compute_single_deviation(self, sequence_window: np.ndarray, current_reading: np.ndarray) -> np.ndarray:
        """
        Real-time single-step deviation calculation for live stream / API inference.
        """
        predicted = self.twin.predict_next_state(sequence_window)
        deviation = np.abs(current_reading - predicted)
        return deviation


def run_deviation_pipeline(
    input_csv: str = "data/sampled_dataset.csv",
    output_csv: str = "data/deviation_dataset.csv",
    output_plot: str = "results/deviation_separation.png"
):
    print(f"Loading dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print("Initializing Deviation Engine with trained Digital Twin...")
    engine = DeviationEngine("models")
    
    print("Computing feature-wise deviations across full dataset...")
    dev_df = engine.compute_deviations(df)
    
    # Save deviation dataset
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    dev_df.to_csv(output_csv, index=False)
    file_size_mb = os.path.getsize(output_csv) / (1024 * 1024)
    print(f"[SUCCESS] Saved deviation dataset to: {output_csv} ({file_size_mb:.2f} MB, {len(dev_df)} rows)")
    
    # Statistical analysis & Separation plot
    print("Analyzing normal vs attack deviation separation...")
    dev_cols = [c for c in dev_df.columns if c.startswith("dev_")]
    dev_df["mean_deviation_magnitude"] = dev_df[dev_cols].mean(axis=1)
    
    normal_mag = dev_df[dev_df["Attack_type"] == "Normal"]["mean_deviation_magnitude"]
    attack_mag = dev_df[dev_df["Attack_type"] != "Normal"]["mean_deviation_magnitude"]
    
    print(f"Normal Mean Deviation Magnitude: {normal_mag.mean():.4f} (std: {normal_mag.std():.4f})")
    print(f"Attack Mean Deviation Magnitude: {attack_mag.mean():.4f} (std: {attack_mag.std():.4f})")
    
    # Plot separation
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    try:
        plt.boxplot([normal_mag, attack_mag], tick_labels=['Normal Traffic', 'Attack Traffic'], patch_artist=True,
                    boxprops=dict(facecolor='#93c5fd', color='#1e40af'),
                    medianprops=dict(color='#dc2626', lw=2))
    except TypeError:
        plt.boxplot([normal_mag, attack_mag], labels=['Normal Traffic', 'Attack Traffic'], patch_artist=True,
                    boxprops=dict(facecolor='#93c5fd', color='#1e40af'),
                    medianprops=dict(color='#dc2626', lw=2))
    plt.yscale('log')
    plt.title('Log Deviation Magnitude Distribution', fontweight='bold')
    plt.ylabel('Mean Feature Deviation |y - y_hat| (log scale)')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.hist(np.log1p(normal_mag), bins=40, alpha=0.6, label='Normal (Healthy)', color='#10b981', density=True)
    plt.hist(np.log1p(attack_mag), bins=40, alpha=0.6, label='Attacks (Intrusions)', color='#ef4444', density=True)
    plt.title('Separation Density: Normal vs Intrusions', fontweight='bold')
    plt.xlabel('log(1 + Deviation Magnitude)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved separation plot to: {output_plot}")
    
    return dev_df

if __name__ == "__main__":
    run_deviation_pipeline()
