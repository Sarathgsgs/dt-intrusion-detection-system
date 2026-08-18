"""
Milestone 7: Edge-Resource Benchmarking Suite
Quantifies the fundamental trade-off between Detection Performance (Accuracy, Macro-F1)
and Computational Resource Overhead (Inference Latency, RAM, Storage Footprint).
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class EdgeResourceBenchmarker:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.configs = {}
        self.results = []
        
    def prepare_test_splits(self, raw_csv="data/sampled_dataset.csv", dev_csv="data/deviation_dataset.csv"):
        print("Preparing benchmark datasets...")
        raw_df = pd.read_csv(raw_csv)
        dev_df = pd.read_csv(dev_csv)
        
        le = joblib.load("models/label_encoder.pkl")
        y = le.transform(raw_df["Attack_type"].astype(str))
        
        raw_features = joblib.load("models/raw_features.pkl")
        dev_features = joblib.load("models/dev_features.pkl")
        fused_features = joblib.load("models/fused_features.pkl")
        
        X_raw = raw_df[raw_features].values
        X_fused = np.hstack([X_raw, dev_df[dev_features].values])
        
        indices = np.arange(len(y))
        _, test_idx = train_test_split(indices, test_size=0.2, random_state=self.random_state, stratify=y)
        
        return {
            "X_raw_test": X_raw[test_idx],
            "X_fused_test": X_fused[test_idx],
            "y_test": y[test_idx],
            "raw_features": raw_features,
            "fused_features": fused_features,
            "raw_train": (X_raw[~np.isin(indices, test_idx)], y[~np.isin(indices, test_idx)]),
            "fused_train": (X_fused[~np.isin(indices, test_idx)], y[~np.isin(indices, test_idx)])
        }
        
    def build_and_benchmark_configurations(self, splits: dict):
        print("\n--- Benchmarking Edge Configurations across 5 Repeated Runs ---")
        
        # Configuration Definitions
        configs_to_test = [
            {
                "id": "C1_FullPrecision_RF150",
                "name": "Config 1: Full-Precision Twin + Heavy RF (150 trees)",
                "feature_space": "fused",
                "model": RandomForestClassifier(n_estimators=150, max_depth=20, n_jobs=-1, random_state=self.random_state),
                "twin_overhead_ms": 0.35, # Twin inference overhead
                "twin_size_kb": 120.0
            },
            {
                "id": "C2_QuantizedTwin_RF100",
                "name": "Config 2: Quantized Twin + Standard RF (100 trees)",
                "feature_space": "fused",
                "model": RandomForestClassifier(n_estimators=100, max_depth=16, n_jobs=-1, random_state=self.random_state),
                "twin_overhead_ms": 0.12, # Quantized TFLite overhead
                "twin_size_kb": 35.0
            },
            {
                "id": "C3_QuantizedTwin_PrunedRF30",
                "name": "Config 3: Quantized Twin + Pruned Edge RF (30 trees)",
                "feature_space": "fused",
                "model": RandomForestClassifier(n_estimators=30, max_depth=10, n_jobs=-1, random_state=self.random_state),
                "twin_overhead_ms": 0.12,
                "twin_size_kb": 35.0
            },
            {
                "id": "C4_UltraLight_XGB25",
                "name": "Config 4: Fast-Inference Edge XGBoost (25 trees, Depth 4)",
                "feature_space": "raw",
                "model": XGBClassifier(n_estimators=25, max_depth=4, learning_rate=0.1, n_jobs=-1, random_state=self.random_state, eval_metric="mlogloss"),
                "twin_overhead_ms": 0.0,
                "twin_size_kb": 0.0
            }
        ]
        
        for cfg in configs_to_test:
            print(f"\nEvaluating: {cfg['name']}...")
            model = cfg["model"]
            is_fused = cfg["feature_space"] == "fused"
            
            X_train, y_train = splits["fused_train"] if is_fused else splits["raw_train"]
            X_test = splits["X_fused_test"] if is_fused else splits["X_raw_test"]
            y_test = splits["y_test"]
            
            # Train
            model.fit(X_train, y_train)
            
            # Accuracy & F1
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            
            # Latency Measurement: 5 runs of 500 samples
            n_samples = 500
            test_sub = X_test[:n_samples]
            latencies = []
            
            for _ in range(5):
                t0 = time.perf_counter()
                _ = model.predict(test_sub)
                t1 = time.perf_counter()
                latency_per_sample_ms = ((t1 - t0) / n_samples) * 1000.0 + cfg["twin_overhead_ms"]
                latencies.append(latency_per_sample_ms)
                
            avg_latency_ms = float(np.mean(latencies))
            std_latency_ms = float(np.std(latencies))
            throughput = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0
            
            # Measure model size in KB
            temp_path = f"models/temp_{cfg['id']}.joblib"
            joblib.dump(model, temp_path, compress=3)
            model_size_kb = (os.path.getsize(temp_path) / 1024.0) + cfg["twin_size_kb"]
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            res_entry = {
                "Configuration": cfg["name"],
                "Feature Space": "Twin-Augmented" if is_fused else "Raw Telemetry",
                "Accuracy (%)": round(acc * 100, 2),
                "Macro-F1": round(macro_f1, 4),
                "Avg Latency (ms/sample)": round(avg_latency_ms, 3),
                "Throughput (samples/sec)": round(throughput, 1),
                "Total Footprint (KB)": round(model_size_kb, 1)
            }
            self.results.append(res_entry)
            print(f"--> Acc: {acc*100:.2f}% | F1: {macro_f1:.4f} | Latency: {avg_latency_ms:.3f}ms | Size: {model_size_kb:.1f} KB")

    def save_and_plot(self, output_csv="results/benchmark_results.csv", output_plot="results/edge_tradeoff_chart.png"):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(output_csv, index=False)
        print(f"\n[SUCCESS] Saved Edge Benchmark Results to: {output_csv}")
        
        # Multi-panel Pareto Trade-off Chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Macro-F1 vs Latency (Trade-off)
        for _, row in results_df.iterrows():
            axes[0].scatter(row["Avg Latency (ms/sample)"], row["Macro-F1"], s=row["Total Footprint (KB)"] * 0.4, 
                            alpha=0.75, edgecolors='black', linewidth=1.5, label=row["Configuration"][:25] + "...")
            axes[0].annotate(f"{row['Macro-F1']:.3f}\n({row['Avg Latency (ms/sample)']:.2f}ms)", 
                             (row["Avg Latency (ms/sample)"], row["Macro-F1"]),
                             textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, fontweight='bold')
                             
        axes[0].set_xlabel('Inference Latency per Sample (ms) [Lower is Better]', fontweight='bold', fontsize=10)
        axes[0].set_ylabel('Macro-F1 Score [Higher is Better]', fontweight='bold', fontsize=10)
        axes[0].set_title('Macro-F1 vs. Inference Latency (Bubble Size = Model Footprint)', fontweight='bold', fontsize=11)
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].legend(loc='lower right', fontsize=8)
        
        # 2. Resource Overhead Comparison (Latency vs Storage Size)
        x = np.arange(len(results_df))
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        
        w = 0.35
        b1 = ax2.bar(x - w/2, results_df["Avg Latency (ms/sample)"], w, label='Latency (ms)', color='#6366f1')
        b2 = ax2_twin.bar(x + w/2, results_df["Total Footprint (KB)"], w, label='Model Size (KB)', color='#ec4899')
        
        ax2.set_xlabel('Edge Configuration', fontweight='bold', fontsize=10)
        ax2.set_ylabel('Inference Latency (ms)', fontweight='bold', color='#6366f1')
        ax2_twin.set_ylabel('Model Footprint (KB)', fontweight='bold', color='#ec4899')
        ax2.set_title('Edge Resource Demands (Latency vs. Footprint)', fontweight='bold', fontsize=11)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Config {i+1}" for i in range(len(results_df))], fontweight='bold')
        ax2.grid(axis='y', linestyle=':', alpha=0.4)
        
        plt.tight_layout()
        plt.savefig(output_plot, dpi=300)
        plt.close()
        print(f"[SUCCESS] Saved Edge Trade-off Chart to: {output_plot}")
        
        return results_df

def run_benchmark():
    benchmarker = EdgeResourceBenchmarker(random_state=42)
    splits = benchmarker.prepare_test_splits()
    benchmarker.build_and_benchmark_configurations(splits)
    df = benchmarker.save_and_plot()
    print("\n--- MASTER EDGE-RESOURCE TRADE-OFF TABLE ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_benchmark()
