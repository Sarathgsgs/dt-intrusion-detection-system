"""
Milestone 2: Telemetry Simulator Module
Replays dataset rows as sequential telemetry arriving live from IIoT sensors and network edge nodes.
"""

import time
import json
import os
from typing import Generator, Dict, Any, Optional
import pandas as pd

class TelemetrySimulator:
    """
    Simulates real-time IoT/IIoT telemetry ingestion from sampled datasets.
    Supports both offline fast batch replay and real-time live streaming.
    """
    def __init__(
        self,
        dataset_path: str = "data/sampled_dataset.csv",
        delay_ms: float = 0.0,
        loop: bool = False
    ):
        self.dataset_path = dataset_path
        self.delay_ms = delay_ms
        self.loop = loop
        self._df: Optional[pd.DataFrame] = None
        self._feature_cols = []
        self._target_col = "Attack_type"
        self._binary_col = "Attack_label"
        self._load_dataset()
        
    def _load_dataset(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        self._df = pd.read_csv(self.dataset_path)
        self._feature_cols = [
            c for c in self._df.columns if c not in [self._target_col, self._binary_col]
        ]
        
    @property
    def total_rows(self) -> int:
        return len(self._df) if self._df is not None else 0
        
    @property
    def feature_names(self) -> list:
        return self._feature_cols.copy()
        
    def stream(self, max_samples: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Yields telemetry records one by one.
        
        Yields:
            Dict containing:
                - 'index': Row index
                - 'timestamp': Virtual ingestion timestamp
                - 'features': Dict of {feature_name: value}
                - 'feature_vector': List of float values
                - 'attack_type': Ground truth multiclass label
                - 'attack_label': Ground truth binary label (0 = Normal, 1 = Attack)
        """
        emitted = 0
        while True:
            for idx, row in self._df.iterrows():
                if max_samples is not None and emitted >= max_samples:
                    return
                    
                features_dict = {col: float(row[col]) for col in self._feature_cols}
                feature_vector = [float(row[col]) for col in self._feature_cols]
                
                payload = {
                    "index": int(idx),
                    "timestamp": time.time(),
                    "features": features_dict,
                    "feature_vector": feature_vector,
                    "attack_type": str(row.get(self._target_col, "Unknown")),
                    "attack_label": int(row.get(self._binary_col, 0))
                }
                
                yield payload
                emitted += 1
                
                if self.delay_ms > 0:
                    time.sleep(self.delay_ms / 1000.0)
                    
            if not self.loop:
                break
                
    def get_batch(self, n_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Fast offline batch access for training and evaluation.
        """
        if n_rows is None:
            return self._df.copy()
        return self._df.iloc[:n_rows].copy()


def test_simulator():
    print("Testing TelemetrySimulator...")
    sim = TelemetrySimulator("data/sampled_dataset.csv", delay_ms=0.0)
    print(f"Total rows loaded: {sim.total_rows}")
    print(f"Number of features: {len(sim.feature_names)}")
    
    # Test fast generator
    count = 0
    start = time.time()
    for record in sim.stream(max_samples=1000):
        count += 1
        if count == 1:
            print(f"Sample Record 0: Attack_type={record['attack_type']}, Features Count={len(record['features'])}")
    elapsed = time.time() - start
    print(f"Streamed {count} records in {elapsed:.4f}s ({count/elapsed:.0f} records/sec in fast mode)")
    assert count == 1000, "Simulator failed to stream expected number of rows"
    print("[SUCCESS] TelemetrySimulator verified working perfectly!")

if __name__ == "__main__":
    test_simulator()
