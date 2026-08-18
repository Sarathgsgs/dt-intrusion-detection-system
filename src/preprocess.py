"""
Milestone 1: Data Preprocessing & Stratified Sampling
Inspects, cleans, encodes, and downsamples the Edge-IIoTset dataset.
"""

import os
import json
import pandas as pd
import numpy as np

def inspect_and_preprocess(
    raw_path: str,
    output_sampled_path: str = "data/sampled_dataset.csv",
    output_dict_path: str = "data/data_dictionary.json",
    target_sample_size: int = 70000,
    random_state: int = 42
):
    print(f"Loading raw dataset from: {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"Initial shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Target columns
    target_col = 'Attack_type' if 'Attack_type' in df.columns else ('attack_type' if 'attack_type' in df.columns else None)
    binary_label_col = 'Attack_label' if 'Attack_label' in df.columns else ('attack_label' if 'attack_label' in df.columns else None)
    
    if target_col is None:
        raise ValueError("Could not locate attack type column in dataset.")
        
    print(f"\nTarget multi-class column: {target_col}")
    print(f"Target binary column: {binary_label_col}")
    
    print("\nInitial Class Distribution:")
    initial_dist = df[target_col].value_counts()
    print(initial_dist)
    
    # Columns to drop (identifiers, timestamps, ip addresses that cause data leakage / overfitting)
    drop_candidates = [
        'frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 'arp.dst.proto_ipv4',
        'http.file_data', 'http.request.full_uri', 'icmp.transmit_timestamp',
        'tcp.options', 'tcp.payload', 'udp.payload', 'dns.qry.name'
    ]
    
    actual_drops = [c for c in drop_candidates if c in df.columns]
    print(f"\nDropping non-generalizable / metadata columns: {actual_drops}")
    df = df.drop(columns=actual_drops)
    
    # Replace inf and -inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    feature_cols = [c for c in df.columns if c not in [target_col, binary_label_col]]
    print(f"\nProcessing {len(feature_cols)} feature columns...")
    
    cleaned_features = {}
    for col in feature_cols:
        series = df[col]
        # Attempt to convert to numeric
        converted = pd.to_numeric(series, errors='coerce')
        # If >80% are numbers, treat as numeric; otherwise treat as categorical
        if converted.notna().sum() > 0.5 * len(series):
            # Numeric column: fill NaN with median
            median_val = converted.median()
            if pd.isna(median_val):
                median_val = 0.0
            cleaned_features[col] = converted.fillna(median_val)
        else:
            # String / categorical: fill NaN with mode and encode
            cat_series = series.fillna('Missing').astype(str)
            cleaned_features[col] = cat_series.astype('category').cat.codes.astype(float)
            
    cleaned_df = pd.DataFrame(cleaned_features, index=df.index)
    cleaned_df[target_col] = df[target_col].astype(str)
    if binary_label_col and binary_label_col in df.columns:
        cleaned_df[binary_label_col] = pd.to_numeric(df[binary_label_col], errors='coerce').fillna(0).astype(int)
        
    # Drop constant / zero variance features
    feature_cols = list(cleaned_features.keys())
    zero_var_cols = [c for c in feature_cols if cleaned_df[c].nunique() <= 1]
    if zero_var_cols:
        print(f"Dropping zero-variance constant columns ({len(zero_var_cols)}): {zero_var_cols}")
        cleaned_df = cleaned_df.drop(columns=zero_var_cols)
        feature_cols = [c for c in feature_cols if c not in zero_var_cols]
        
    print(f"Cleaned dataset shape: {cleaned_df.shape} with {len(feature_cols)} features.")
    
    # Stratified Sampling
    print(f"\nPerforming stratified sampling to target ~{target_sample_size} rows...")
    if len(cleaned_df) > target_sample_size:
        sampled_dfs = []
        class_counts = cleaned_df[target_col].value_counts()
        total_rows = len(cleaned_df)
        
        for cls_name, count in class_counts.items():
            cls_df = cleaned_df[cleaned_df[target_col] == cls_name]
            # Stratified proportion with minimum 100 samples floor
            n_samples = max(int(target_sample_size * (count / total_rows)), min(count, 100))
            if n_samples > count:
                n_samples = count
            sampled_dfs.append(cls_df.sample(n=n_samples, random_state=random_state))
            
        sampled_df = pd.concat(sampled_dfs, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    else:
        sampled_df = cleaned_df.copy()
        
    print(f"\nSampled dataset shape: {sampled_df.shape}")
    print("\nSampled Class Distribution:")
    sampled_dist = sampled_df[target_col].value_counts()
    print(sampled_dist)
    
    # Save sampled dataset
    os.makedirs(os.path.dirname(output_sampled_path), exist_ok=True)
    sampled_df.to_csv(output_sampled_path, index=False)
    file_size_mb = os.path.getsize(output_sampled_path) / (1024 * 1024)
    print(f"\n[SUCCESS] Saved sampled dataset to: {output_sampled_path} ({file_size_mb:.2f} MB)")
    
    # Generate Data Dictionary
    data_dict = {
        "dataset_name": "Edge-IIoTset Sampled",
        "total_rows": int(len(sampled_df)),
        "total_features": int(len(feature_cols)),
        "target_multiclass": target_col,
        "target_binary": binary_label_col,
        "classes": {k: int(v) for k, v in sampled_dist.items()},
        "feature_columns": feature_cols,
        "feature_summary": {}
    }
    
    for c in feature_cols:
        data_dict["feature_summary"][c] = {
            "dtype": str(sampled_df[c].dtype),
            "min": float(sampled_df[c].min()),
            "max": float(sampled_df[c].max()),
            "mean": float(sampled_df[c].mean()),
            "std": float(sampled_df[c].std())
        }
        
    with open(output_dict_path, "w") as f:
        json.dump(data_dict, f, indent=2)
    print(f"[SUCCESS] Saved data dictionary to: {output_dict_path}")
    
    return sampled_df, data_dict

if __name__ == "__main__":
    raw_ml_path = "data/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv"
    if not os.path.exists(raw_ml_path):
        raw_ml_path = "data/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
    inspect_and_preprocess(raw_ml_path)
