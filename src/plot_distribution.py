"""
Generate Class Distribution Plot for Milestone 1
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_class_distribution(csv_path="data/sampled_dataset.csv", output_path="results/class_distribution.png"):
    df = pd.read_csv(csv_path)
    counts = df['Attack_type'].value_counts()
    
    plt.figure(figsize=(12, 6))
    colors = ['#22c55e' if cls == 'Normal' else '#ef4444' for cls in counts.index]
    bars = plt.barh(counts.index, counts.values, color=colors)
    plt.xlabel('Number of Samples', fontsize=12, fontweight='bold')
    plt.ylabel('Traffic / Attack Category', fontsize=12, fontweight='bold')
    plt.title('Stratified Sampled Edge-IIoTset Class Distribution (~70k rows)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 100, bar.get_y() + bar.get_height()/2, f"{int(width):,}", 
                 va='center', ha='left', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved class distribution plot to: {output_path}")

if __name__ == "__main__":
    plot_class_distribution()
