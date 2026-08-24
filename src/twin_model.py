"""
Scope-Restricted Physically Bounded Digital Twin with Log1p Normalization & Robust Regularization
Models baseline normal telemetry exclusively across continuous physical features.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Continuous physical features suitable for sequence regression
CONTINUOUS_FEATURES = [
    "icmp.checksum",
    "icmp.seq_le",
    "http.content_length",
    "tcp.ack",
    "tcp.checksum",
    "tcp.len",
    "tcp.seq",
    "udp.stream",
    "udp.time_delta"
]

# Physical protocol boundaries for industrial cyber-physical networks
PHYSICAL_BOUNDS = {
    "icmp.checksum": (0.0, 65535.0),
    "icmp.seq_le": (0.0, 65535.0),
    "http.content_length": (0.0, 10000000.0),
    "tcp.ack": (0.0, 4294967295.0),
    "tcp.checksum": (0.0, 65535.0),
    "tcp.len": (0.0, 65535.0),
    "tcp.seq": (0.0, 4294967295.0),
    "udp.stream": (0.0, 1000000.0),
    "udp.time_delta": (0.0, 3600.0)
}

# Per-feature maximum log-space bounds corresponding strictly to valid sub-saturation physical ceilings:
FEATURE_LOG_CLIPS = {
    "tcp.seq": 22.18,
    "tcp.ack": 22.18,
    "tcp.len": 11.08,
    "icmp.checksum": 11.08,
    "icmp.seq_le": 11.08,
    "tcp.checksum": 11.08,
    "http.content_length": 16.11,
    "udp.stream": 13.81,
    "udp.time_delta": 8.18
}

class DigitalTwin:
    def __init__(
        self,
        window_size: int = 5,
        hidden_layer_sizes: tuple = (64, 32),
        alpha: float = 0.05,
        max_iter: int = 250,
        random_state: int = 42,
        use_log1p: bool = True
    ):
        self.window_size = window_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.use_log1p = use_log1p
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            max_iter=self.max_iter,
            early_stopping=True,
            random_state=self.random_state
        )
        self.feature_names = CONTINUOUS_FEATURES.copy()
        self.is_fitted = False
        
    def _transform_in(self, arr: np.ndarray) -> np.ndarray:
        if self.use_log1p:
            return np.log1p(np.maximum(0, arr))
        return arr
        
    def _transform_out(self, arr: np.ndarray) -> np.ndarray:
        if self.use_log1p:
            is_1d = (arr.ndim == 1)
            arr_2d = arr.reshape(1, -1) if is_1d else arr
            clipped_log = np.zeros_like(arr_2d)
            for j, feat in enumerate(self.feature_names):
                ceil_val = FEATURE_LOG_CLIPS.get(feat, 25.0)
                clipped_log[:, j] = np.clip(arr_2d[:, j], 0.0, ceil_val)
            res = np.expm1(clipped_log)
            return res.ravel() if is_1d else res
        return arr
        
    def _create_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(len(data) - self.window_size):
            window = data[i:i + self.window_size].flatten()
            target = data[i + self.window_size]
            X.append(window)
            y.append(target)
        return np.array(X), np.array(y)
        
    def fit(self, normal_df: pd.DataFrame, feature_cols: list = None):
        if feature_cols is not None:
            self.feature_names = [f for f in feature_cols if f in CONTINUOUS_FEATURES]
        else:
            self.feature_names = CONTINUOUS_FEATURES.copy()
            
        print(f"Digital Twin configured for {len(self.feature_names)} Scope-Restricted Continuous Features:")
        print(f"  {self.feature_names}")
        
        raw_values = normal_df[self.feature_names].values
        transformed_values = self._transform_in(raw_values)
        scaled_values = self.scaler.fit_transform(transformed_values)
        
        X, y = self._create_sequences(scaled_values)
        print(f"Twin Training Data: {X.shape[0]} sequences (Window={self.window_size}, Inputs={X.shape[1]}, Outputs={y.shape[1]})")
        
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]
        
        print("Training Scope-Restricted Log1p Robust Digital Twin...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        val_preds_scaled = self.model.predict(X_val)
        val_preds_unscaled = self.scaler.inverse_transform(val_preds_scaled)
        val_preds_physical = self._transform_out(val_preds_unscaled)
        
        val_actual_physical = raw_values[split_idx + self.window_size:]
        val_mse = mean_squared_error(val_actual_physical, val_preds_physical)
        val_mae = mean_absolute_error(val_actual_physical, val_preds_physical)
        print(f"Scope-Restricted Twin Validation MSE: {val_mse:.4f} | MAE: {val_mae:.4f}")
        
        return {
            "val_mse": float(val_mse),
            "val_mae": float(val_mae),
            "n_iter": int(self.model.n_iter_),
            "loss": float(self.model.loss_) if hasattr(self.model, "loss_") and self.model.loss_ is not None else float(val_mse)
        }
        
    def predict_next_state(self, sequence_window: np.ndarray) -> np.ndarray:
        """
        Given a raw continuous sequence window (W, K), predicts unscaled, physically bounded next state (K,).
        """
        if not self.is_fitted:
            raise RuntimeError("Digital Twin model is not fitted yet.")
        transformed_window = self._transform_in(sequence_window)
        scaled_window = self.scaler.transform(transformed_window)
        flat_input = scaled_window.flatten().reshape(1, -1)
        scaled_pred = self.model.predict(flat_input)
        unscaled_pred = self.scaler.inverse_transform(scaled_pred)[0]
        physical_pred = self._transform_out(unscaled_pred)
        
        # Enforce physical network bounds
        bounded_pred = np.zeros_like(physical_pred)
        for idx, feat in enumerate(self.feature_names):
            low, high = PHYSICAL_BOUNDS.get(feat, (0.0, 1e9))
            bounded_pred[idx] = np.clip(physical_pred[idx], low, high)
            
        return bounded_pred
        
    def compute_dataset_predictions(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized computation across full dataset with physical bounding enforcement.
        """
        raw_values = df[self.feature_names].values
        transformed_values = self._transform_in(raw_values)
        scaled_values = self.scaler.transform(transformed_values)
        n_samples, n_features = raw_values.shape
        
        predicted_scaled = np.zeros_like(scaled_values)
        
        for i in range(min(self.window_size, n_samples)):
            predicted_scaled[i] = scaled_values[i]
            
        if n_samples > self.window_size:
            X_all = []
            for i in range(n_samples - self.window_size):
                X_all.append(scaled_values[i:i + self.window_size].flatten())
            X_all = np.array(X_all)
            preds = self.model.predict(X_all)
            predicted_scaled[self.window_size:] = preds
            
        unscaled_predictions = self.scaler.inverse_transform(predicted_scaled)
        physical_predictions = self._transform_out(unscaled_predictions)
        
        # Apply physical bounds across full prediction matrix
        for idx, feat in enumerate(self.feature_names):
            low, high = PHYSICAL_BOUNDS.get(feat, (0.0, 1e9))
            physical_predictions[:, idx] = np.clip(physical_predictions[:, idx], low, high)
            
        return physical_predictions
        
    def save(self, model_dir: str = "models"):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, "twin_model.pkl"))
        joblib.dump(self.scaler, os.path.join(model_dir, "twin_scaler.pkl"))
        joblib.dump(self.feature_names, os.path.join(model_dir, "continuous_features.pkl"))
        metadata = {
            "window_size": self.window_size,
            "feature_names": self.feature_names,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "alpha": self.alpha,
            "use_log1p": self.use_log1p,
            "is_fitted": self.is_fitted
        }
        joblib.dump(metadata, os.path.join(model_dir, "twin_metadata.pkl"))
        print(f"[SUCCESS] Scope-Restricted Digital Twin saved to {model_dir}/")
        
    @classmethod
    def load(cls, model_dir: str = "models"):
        metadata = joblib.load(os.path.join(model_dir, "twin_metadata.pkl"))
        instance = cls(
            window_size=metadata.get("window_size", 5),
            hidden_layer_sizes=metadata.get("hidden_layer_sizes", (64, 32)),
            alpha=metadata.get("alpha", 0.05),
            use_log1p=metadata.get("use_log1p", True)
        )
        instance.model = joblib.load(os.path.join(model_dir, "twin_model.pkl"))
        instance.scaler = joblib.load(os.path.join(model_dir, "twin_scaler.pkl"))
        instance.feature_names = metadata["feature_names"]
        instance.is_fitted = metadata["is_fitted"]
        return instance

def train_and_evaluate_scope_restricted_twin(
    csv_path: str = "data/sampled_dataset.csv",
    model_dir: str = "models",
    output_plot: str = "results/twin_forecast_validation.png"
):
    print("=" * 70)
    print("  PHASE 1: TRAINING & VALIDATING LOG1P PHYSICALLY BOUNDED DIGITAL TWIN")
    print("=" * 70)
    
    df = pd.read_csv(csv_path)
    normal_df = df[df["Attack_type"] == "Normal"].reset_index(drop=True)
    print(f"Normal Baseline Data: {len(normal_df)} rows")
    
    twin = DigitalTwin(window_size=5, hidden_layer_sizes=(64, 32), alpha=0.05, max_iter=250, random_state=42, use_log1p=True)
    metrics = twin.fit(normal_df)
    twin.save(model_dir)
    
    # Validation plotting
    print("\nGenerating physically bounded forecast validation plot...")
    test_slice = normal_df.iloc[:100]
    preds = twin.compute_dataset_predictions(test_slice)
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(twin.feature_names):
        ax = axes[i]
        actual = test_slice[col].values
        predicted = preds[:, i]
        ax.plot(actual, label="Actual Telemetry", color="#38bdf8", lw=2)
        ax.plot(predicted, label="Log1p Twin Forecast", color="#f59e0b", linestyle="--", lw=1.8)
        ax.set_title(f"{col} (Bounds: {PHYSICAL_BOUNDS[col][0]} - {PHYSICAL_BOUNDS[col][1]})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Sample Index", fontsize=8)
        ax.set_ylabel("Physical Value", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
            
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved twin forecast validation plot to: {output_plot}")
    
    return twin, metrics

if __name__ == "__main__":
    train_and_evaluate_scope_restricted_twin()
