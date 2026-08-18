# Full Implementation Plan: Twin-Guided Explainable Intrusion Detection System (X-IDS)
## A Resource-Aware Trade-off Analysis for Industrial IoT

**Project Scope:** Digital Twin (Active Predictor) → Deviation Engine → IDS Classifier (Baseline vs. Twin-Deviation) → SHAP Explainability → Confidence Filter → Edge-Resource Benchmarking → FastAPI Backend + Interactive React Dashboard.

---

## 📌 Execution Responsibility Matrix: [IN-IDE] vs. [EXTERNAL / USER ACTION]

| Component / Task | Location | Handled By | Details |
|---|---|---|---|
| **Python Environment & Dependencies** | Inside IDE / System | Agent + User | Python 3.10–3.12 environment setup with ML/API/Frontend dependencies. |
| **Dataset Acquisition** | Outside IDE | **User Action** | Download Edge-IIoTset (and optional TON_IoT) from Kaggle/source to `data/`. |
| **Data Cleaning, Sampling & Dictionary** | Inside IDE | **Agent [IN-IDE]** | Automated cleaning, categorical encoding, stratified sampling (~70k rows). |
| **Telemetry Simulator (`simulator.py`)** | Inside IDE | **Agent [IN-IDE]** | Real-time generator with configurable playback delay (Fast & Live stream modes). |
| **Digital Twin Model (Local or Colab)** | Hybrid | **Agent [IN-IDE]** & *(Optional Colab GPU)* | Agent builds training script + Colab Notebook. Colab used if GPU training chosen; or CPU-optimized local training. Produces `twin_model_quantized.tflite`. |
| **Deviation Engine (`deviation_engine.py`)** | Inside IDE | **Agent [IN-IDE]** | TFLite inference, residual computation $\|actual - predicted\|$, generates `deviation_dataset.csv`. |
| **IDS Classifiers (`ids_model.py`)** | Inside IDE | **Agent [IN-IDE]** | Trains & compares 4 models (RF-raw, XGB-raw, RF-dev, XGB-dev) with Macro-F1 & class balance. |
| **XAI & Confidence Filter (`xai_module.py`, `confidence_filter.py`)** | Inside IDE | **Agent [IN-IDE]** | TreeExplainer SHAP attribution + domain attack signature rules to suppress low-fidelity false positives. |
| **Edge-Resource Benchmarking (`edge_benchmark.py`)** | Inside IDE | **Agent [IN-IDE]** | Benchmarks 3–4 configurations (Full RF vs Quantized vs Pruned) for Latency (ms), RAM/Model Size (KB), and Accuracy/Macro-F1. |
| **FastAPI Backend Server (`api_server.py`)** | Inside IDE | **Agent [IN-IDE]** | REST + SSE/WebSocket endpoints for live telemetry, real-time alerts, SHAP explanations, and benchmark data. |
| **Modern React Dashboard (`dashboard/`)** | Inside IDE | **Agent [IN-IDE]** | High-aesthetic UI: live alert feed, dynamic deviation charts, SHAP waterfall/bar charts, confidence filter toggles, and trade-off comparison curves. |
| **Documentation & Evaluation Report** | Inside IDE | **Agent [IN-IDE]** | Master result tables, visualization plots, verification scripts, and demo guides. |

---

## 🛠️ What You (The User) Need to Provide

1. **Python Environment (Python 3.10, 3.11, or 3.12 recommended):**
   - Ensure a Python 3.10–3.12 runtime is available on your machine (Python 3.14 is currently too new for compiled C-extensions like `shap`, `tensorflow`, and `tflite-runtime`).
2. **Dataset CSV File:**
   - Download the **Edge-IIoTset** dataset (e.g., `DNN-EdgeIIoT-dataset.csv` or ML dataset from Kaggle: `mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot`).
   - Place it into `e:\Projects\digital twin\data\raw_edge_iiotset.csv`.
   - *(Optional for Phase 7 generalization)*: `TON_IoT` dataset placed in `data/ton_iot.csv`.
3. **Google Colab (Only for GPU-accelerated LSTM training in Phase 3 - Optional):**
   - If you want deep LSTM training on GPU, open the generated notebook `notebooks/01_train_digital_twin_colab.ipynb` in Google Colab, click *Run All*, and copy the exported `twin_model_quantized.tflite` to `models/`.
   - *Alternative:* If you prefer to stay 100% inside this IDE, we provide a local CPU-optimized Twin trainer (Lightweight GRU/Dense or Exponential Smoothing) that runs directly on your machine.

---

## 🚀 Step-by-Step Milestone-Driven Implementation Phases

### Milestone 0: Environment & Folder Structure Setup `[IN-IDE]`
- Create project directory tree:
  ```
  twin-ids-project/
  ├── data/                  # Raw, sampled, and deviation datasets
  ├── notebooks/             # Colab training notebooks & exploratory analyses
  ├── src/                   # Core Python modules
  │   ├── __init__.py
  │   ├── simulator.py       # Telemetry generator
  │   ├── twin_model.py      # Digital Twin training & inference helper
  │   ├── deviation_engine.py# Residual vector computation
  │   ├── ids_model.py       # Baseline vs Deviation Classifiers
  │   ├── xai_module.py      # SHAP TreeExplainer module
  │   ├── confidence_filter.py # Signature & confidence rule engine
  │   ├── edge_benchmark.py  # Latency, memory, and trade-off benchmark
  │   └── api_server.py      # FastAPI backend
  ├── models/                # Saved .pkl, .tflite, and scaler artifacts
  ├── results/               # Benchmark CSVs, SHAP plots, confusion matrices
  └── dashboard/             # Modern React + Vite frontend
  ```
- Install dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `fastapi`, `uvicorn`, `tflite-runtime` or `tensorflow-cpu`.
- Initialize Git repository and sanity test imports.

---

### Milestone 1: Data Acquisition & Preprocessing `[HYBRID]`
- **User Action `[EXTERNAL]`**: Place raw Edge-IIoTset CSV into `data/`.
- **Agent `[IN-IDE]`**:
  - Load and inspect all features (network telemetry, sensor measurements, attack categories).
  - Clean missing values, infinities, and drop non-informative metadata (timestamps, IP/MAC addresses).
  - Perform stratified downsampling to ~70,000 rows, preserving multiclass attack distributions (DDoS, Ransomware, SQLi, Fingerprinting, MITM, Vulnerability Scans, Normal).
  - Export `data/sampled_dataset.csv`, `data/data_dictionary.json`, and class distribution report.

---

### Milestone 2: Telemetry Simulator `[IN-IDE]`
- Write `src/simulator.py` supporting:
  - **Generator Mode**: Yields sequential telemetry records row-by-row.
  - **Fast Batch Mode**: Zero-delay replay for training and offline benchmarking.
  - **Real-Time Live Mode**: Controlled delay (e.g. 50ms–500ms) with jitter for live dashboard streaming.
- Unit tests to guarantee exact sequence order and zero data leakage.

---

### Milestone 3: Digital Twin Forecasting Model `[HYBRID: Colab or Local CPU]`
- Filter dataset to **Normal-only** operation.
- Construct temporal sliding window sequences ($t-W \dots t-1 \to t$).
- Train predictive model (LSTM/GRU) on normal behavior dynamics.
- Export & quantize model using TFLite dynamic range quantization (`models/twin_model_quantized.tflite`).
- Output validation plots: Predicted vs. Actual telemetry on unseen normal sequences.
- **Colab Option `[EXTERNAL]`**: Ready notebook generated for 1-click Colab T4 execution if desired.

---

### Milestone 4: Deviation Engine `[IN-IDE]`
- Implement `src/deviation_engine.py`:
  - Loads `models/twin_model_quantized.tflite` via lightweight runtime.
  - Calculates feature-wise deviation vector: $e_t = |y_t - \hat{y}_t|$.
  - Generates `data/deviation_dataset.csv`.
- Generates statistical separation plots (boxplots & distribution density) comparing normal deviations vs attack deviations.

---

### Milestone 5: IDS Classifiers (Baseline vs. Twin-Deviation) `[IN-IDE]`
- Implement `src/ids_model.py`:
  - **Model A (Baseline)**: Random Forest & XGBoost trained on **Raw Telemetry**.
  - **Model B (Proposed Twin-Guided)**: Random Forest & XGBoost trained on **Deviation Features**.
- Multi-class optimization with class balancing (`class_weight="balanced"` / scale pos weight).
- Full comparative evaluation: Accuracy, Macro-F1, Precision, Recall per attack class, Confusion Matrices.
- Save trained models to `models/`.

---

### Milestone 6: SHAP Explainability & Confidence Filter `[IN-IDE]`
- Implement `src/xai_module.py`:
  - High-performance SHAP `TreeExplainer` on top-performing deviation model.
  - Local feature attribution per alert + global summary plots.
- Implement `src/confidence_filter.py`:
  - Domain-specific attack signatures (e.g., DDoS features vs. Injection features).
  - Decision logic: Suppresses borderline / low-fidelity alerts where model confidence is ambiguous and SHAP attribution diverges from known attack signatures.
  - Verification test suite demonstrating alert filtering and suppression rate.

---

### Milestone 7: Edge-Resource Benchmarking Suite `[IN-IDE]`
- Implement `src/edge_benchmark.py`:
  - Benchmarks 3–4 pipeline configurations:
    1. Full-Precision Twin + Full Random Forest (150 estimators).
    2. Quantized TFLite Twin + Full Random Forest.
    3. Quantized TFLite Twin + Pruned Random Forest (50 estimators).
    4. *(Optional)* Lightweight Fast-Inference XGBoost profile.
  - Measures: Inference Latency per sample (ms, averaged over 100+ cycles), Peak Memory Usage (MB), Model Footprint (KB), and Classification Macro-F1.
  - Generates `results/benchmark_results.csv` and interactive Pareto frontier trade-off charts.

---

### Milestone 8: FastAPI Backend & Modern Interactive Dashboard `[IN-IDE]`
- **FastAPI Server (`src/api_server.py`)**:
  - `/api/stream`: SSE / WebSocket telemetry stream feed.
  - `/api/predict`: Live inference pipeline (Simulator → Twin → Deviation → IDS → SHAP → Filter).
  - `/api/benchmarks`: Serves experimental trade-off data.
  - `/api/stats`: Real-time detection statistics and suppression rates.
- **Modern React Dashboard (`dashboard/`)**:
  - Built with Vite + React + Lucide Icons + Recharts / Canvas.
  - Features:
    - **Live Monitor**: Real-time sensor charts with anomalous deviation highlights.
    - **Threat Alert Center**: Instant alert cards with confidence score and filter status (Passed / Suppressed).
    - **Explainability Panel**: Live SHAP feature attribution bar charts explaining *why* an alert fired.
    - **Resource Benchmark Studio**: Interactive trade-off plots comparing Latency, Size, and Macro-F1 across edge configurations.
    - **Interactive Simulation Controls**: Play, Pause, Speed adjustment, and manual attack injection.

---

### Milestone 9: Final Artifacts, Reports & Demo Script `[IN-IDE]`
- Generate comprehensive project summary, benchmark tables, and architecture diagrams.
- Produce a step-by-step Viva / Defense demonstration script.
