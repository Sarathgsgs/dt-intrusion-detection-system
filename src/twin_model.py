"""
Milestone 3: Digital Twin Forecasting Model
Learns normal industrial IoT telemetry dynamics using historical sequence windows
to forecast expected next-state sensor & network metrics.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

class DigitalTwin:
    def __init__(self, window_size: int = 5, hidden_layer_sizes=(64, 32), max_iter=200, random_state=42):
        self.window_size = window_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=self.random_state
        )
        self.feature_names = []
        self.is_fitted = False
        
    def _create_sequences(self, data: np.ndarray):
        """
        Builds (N - W, W * D) -> (N - W, D) training sequences.
        """
        X, y = [], []
        for i in range(len(data) - self.window_size):
            window = data[i:i + self.window_size].flatten()
            target = data[i + self.window_size]
            X.append(window)
            y.append(target)
        return np.array(X), np.array(y)
        
    def fit(self, normal_df: pd.DataFrame, feature_cols: list):
        self.feature_names = feature_cols
        raw_values = normal_df[feature_cols].values
        
        # Fit scaler on normal data only
        scaled_values = self.scaler.fit_transform(raw_values)
        
        X, y = self._create_sequences(scaled_values)
        print(f"Twin Training Data: {X.shape[0]} sequences (Window={self.window_size}, Inputs={X.shape[1]}, Target Outputs={y.shape[1]})")
        
        # Split train/validation
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]
        
        print("Training Digital Twin Neural Forecaster...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluate on validation
        val_preds = self.model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_preds)
        val_mae = mean_absolute_error(y_val, val_preds)
        print(f"Digital Twin Validation MSE (scaled): {val_mse:.6f} | MAE: {val_mae:.6f}")
        
        return {
            "val_mse": float(val_mse),
            "val_mae": float(val_mae),
            "n_iter": int(self.model.n_iter_),
            "loss": float(self.model.loss_) if hasattr(self.model, "loss_") and self.model.loss_ is not None else float(val_mse)
        }
        
    def predict_next_state(self, sequence_window: np.ndarray) -> np.ndarray:
        """
        Given a raw sequence of shape (W, D), scales it and predicts the unscaled next state (D,).
        """
        if not self.is_fitted:
            raise RuntimeError("Digital Twin model is not fitted yet.")
        scaled_window = self.scaler.transform(sequence_window)
        flat_input = scaled_window.flatten().reshape(1, -1)
        scaled_pred = self.model.predict(flat_input)
        unscaled_pred = self.scaler.inverse_transform(scaled_pred)
        return unscaled_pred[0]
        
    def compute_dataset_predictions(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized computation across full dataset for deviation engine.
        For initial rows < window_size, uses self-prediction / moving average.
        """
        raw_values = df[self.feature_names].values
        scaled_values = self.scaler.transform(raw_values)
        n_samples, n_features = raw_values.shape
        
        predicted_scaled = np.zeros_like(scaled_values)
        
        # For the initial warm-up rows
        for i in range(min(self.window_size, n_samples)):
            predicted_scaled[i] = scaled_values[i]
            
        # For rows >= window_size
        if n_samples > self.window_size:
            X_all = []
            for i in range(n_samples - self.window_size):
                X_all.append(scaled_values[i:i + self.window_size].flatten())
            X_all = np.array(X_all)
            preds = self.model.predict(X_all)
            predicted_scaled[self.window_size:] = preds
            
        unscaled_predictions = self.scaler.inverse_transform(predicted_scaled)
        return unscaled_predictions
        
    def save(self, model_dir: str = "models"):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, "twin_model.pkl"))
        joblib.dump(self.scaler, os.path.join(model_dir, "twin_scaler.pkl"))
        metadata = {
            "window_size": self.window_size,
            "feature_names": self.feature_names,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "is_fitted": self.is_fitted
        }
        joblib.dump(metadata, os.path.join(model_dir, "twin_metadata.pkl"))
        print(f"[SUCCESS] Digital Twin artifacts saved to {model_dir}/")
        
    @classmethod
    def load(cls, model_dir: str = "models"):
        metadata = joblib.load(os.path.join(model_dir, "twin_metadata.pkl"))
        instance = cls(
            window_size=metadata["window_size"],
            hidden_layer_sizes=metadata["hidden_layer_sizes"]
        )
        instance.model = joblib.load(os.path.join(model_dir, "twin_model.pkl"))
        instance.scaler = joblib.load(os.path.join(model_dir, "twin_scaler.pkl"))
        instance.feature_names = metadata["feature_names"]
        instance.is_fitted = metadata["is_fitted"]
        return instance

def train_and_evaluate_twin(
    csv_path: str = "data/sampled_dataset.csv",
    output_plot_path: str = "results/twin_validation.png"
):
    print(f"Loading sampled dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter to Normal-only data for training the Digital Twin
    normal_df = df[df["Attack_type"] == "Normal"].copy().reset_index(drop=True)
    print(f"Extracted {len(normal_df)} Normal telemetry samples for Digital Twin training.")
    
    feature_cols = [c for c in df.columns if c not in ["Attack_type", "Attack_label"]]
    
    twin = DigitalTwin(window_size=5, hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    metrics = twin.fit(normal_df, feature_cols)
    twin.save("models")
    
    # Validation Plot: Plot Actual vs Predicted on unseen normal segment
    print("Generating Digital Twin tracking validation plot...")
    raw_values = normal_df[feature_cols].values
    preds = twin.compute_dataset_predictions(normal_df)
    
    # Pick 2 representative features to plot (e.g. first two features)
    feat1, feat2 = feature_cols[0], feature_cols[1]
    idx1, idx2 = 0, 1
    
    time_steps = range(100, 250) # Sample window
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(time_steps, raw_values[time_steps, idx1], label='Actual Sensor Telemetry', color='#2563eb', lw=1.8)
    plt.plot(time_steps, preds[time_steps, idx1], label='Digital Twin Forecast', color='#f59e0b', linestyle='--', lw=1.8)
    plt.title(f'Twin Prediction Tracking: {feat1}', fontweight='bold')
    plt.xlabel('Time Step (Sample Index)')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(time_steps, raw_values[time_steps, idx2], label='Actual Sensor Telemetry', color='#059669', lw=1.8)
    plt.plot(time_steps, preds[time_steps, idx2], label='Digital Twin Forecast', color='#dc2626', linestyle='--', lw=1.8)
    plt.title(f'Twin Prediction Tracking: {feat2}', fontweight='bold')
    plt.xlabel('Time Step (Sample Index)')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved validation plot to: {output_plot_path}")
    
    return twin, metrics

if __name__ == "__main__":
    train_and_evaluate_twin()
