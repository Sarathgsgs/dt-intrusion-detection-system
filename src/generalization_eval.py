"""
Milestone 9: Cross-Dataset Generalization Evaluation
Evaluates zero-shot transferability of the trained model on the unseen TON_IoT dataset (train_test_network.csv).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def evaluate_generalization(
    ton_iot_path: str = "data/train_test_network.csv",
    output_csv: str = "results/generalization_results.csv",
    output_plot: str = "results/generalization_transfer.png"
):
    print(f"Loading unseen TON_IoT dataset from: {ton_iot_path}")
    if not os.path.exists(ton_iot_path):
        print(f"Warning: {ton_iot_path} not found. Skipping generalization test.")
        return None
        
    ton_df = pd.read_csv(ton_iot_path, nrows=50000, low_memory=False)
    print(f"Loaded {len(ton_df)} rows from TON_IoT.")
    print("Columns in TON_IoT:", ton_df.columns.tolist()[:15])
    
    # Identify target column in TON_IoT (usually 'type' or 'label')
    target_col = 'type' if 'type' in ton_df.columns else ('label' if 'label' in ton_df.columns else None)
    binary_label_col = 'label' if 'label' in ton_df.columns else None
    
    print(f"TON_IoT Target column: {target_col} | Binary: {binary_label_col}")
    print("TON_IoT Class Distribution:")
    print(ton_df[target_col].value_counts())
    
    # Load trained model and features
    model = joblib.load("models/xgb_raw.pkl")
    feature_names = joblib.load("models/raw_features.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    
    # Map common network features
    # Standardize column names lowercase
    ton_df.columns = [c.lower() for c in ton_df.columns]
    
    mapped_features = np.zeros((len(ton_df), len(feature_names)))
    
    feature_mapping = {
        'tcp.dstport': ['dst_port', 'dstport', 'dport'],
        'tcp.srcport': ['src_port', 'srcport', 'sport'],
        'tcp.len': ['src_bytes', 'dst_bytes', 'bytes'],
        'tcp.flags': ['conn_state', 'flags'],
        'udp.port': ['dst_port', 'src_port']
    }
    
    for i, feat in enumerate(feature_names):
        matched = False
        if feat.lower() in ton_df.columns:
            mapped_features[:, i] = pd.to_numeric(ton_df[feat.lower()], errors='coerce').fillna(0).values
            matched = True
        elif feat in feature_mapping:
            for cand in feature_mapping[feat]:
                if cand in ton_df.columns:
                    mapped_features[:, i] = pd.to_numeric(ton_df[cand], errors='coerce').fillna(0).values
                    matched = True
                    break
        if not matched:
            mapped_features[:, i] = 0.0
            
    print(f"\nEvaluating Edge-IIoTset trained model on TON_IoT...")
    probs = model.predict_proba(mapped_features)
    pred_indices = np.argmax(probs, axis=1)
    pred_classes = label_encoder.inverse_transform(pred_indices)
    
    # Binary anomaly detection evaluation (Normal vs Attack)
    # In TON_IoT: label 0 = Normal, 1 = Attack
    if binary_label_col and binary_label_col.lower() in ton_df.columns:
        y_true_binary = pd.to_numeric(ton_df[binary_label_col.lower()], errors='coerce').fillna(0).astype(int).values
        y_pred_binary = (pred_classes != "Normal").astype(int)
        
        acc = accuracy_score(y_true_binary, y_pred_binary)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        
        print("\n--- Zero-Shot Cross-Dataset Anomaly Detection Performance ---")
        print(f"Accuracy:  {acc*100:.2f}%")
        print(f"F1-Score:  {f1:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        
        res_df = pd.DataFrame([{
            "Source Dataset (Trained)": "Edge-IIoTset",
            "Target Dataset (Unseen)": "TON_IoT",
            "Sample Count": len(ton_df),
            "Transfer Accuracy (%)": round(acc * 100, 2),
            "Transfer F1-Score": round(f1, 4),
            "Transfer Precision": round(prec, 4),
            "Transfer Recall": round(rec, 4)
        }])
        
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        res_df.to_csv(output_csv, index=False)
        print(f"[SUCCESS] Saved generalization results to: {output_csv}")
        
        # Plot Generalization
        plt.figure(figsize=(8, 5))
        metrics = ['Accuracy (%)', 'F1-Score (x100)', 'Precision (x100)', 'Recall (x100)']
        scores = [acc * 100, f1 * 100, prec * 100, rec * 100]
        bars = plt.bar(metrics, scores, color=['#3b82f6', '#10b981', '#f59e0b', '#ec4899'], width=0.5)
        plt.ylabel('Score (%)', fontweight='bold')
        plt.title('Cross-Dataset Zero-Shot Generalization: Edge-IIoTset -> TON_IoT', fontweight='bold', fontsize=11)
        plt.ylim(0, 105)
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)
            
        plt.tight_layout()
        plt.savefig(output_plot, dpi=300)
        plt.close()
        print(f"[SUCCESS] Saved generalization plot to: {output_plot}")
        
        return res_df

if __name__ == "__main__":
    evaluate_generalization()
