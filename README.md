# 🛡️ Twin-Guided Explainable Intrusion Detection System (X-IDS)
### A Resource-Aware Trade-off Analysis for Industrial IoT (IIoT)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18%2B-61DAFB.svg)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-5%2B-646CFF.svg)](https://vitejs.dev/)

An end-to-end, resource-aware cyber-physical intrusion detection system combining **Digital Twin sequence forecasting**, **multidimensional deviation analysis**, **SHAP TreeExplainer local attribution**, and an **Operational Confidence Filter** for sub-millisecond edge IIoT deployments.

---

## 🏛️ System Architecture

```
[ IIoT Telemetry Stream ]
           │
           ▼
[ Digital Twin Forecaster (Normal Dynamics) ] ───► [ Forecast State ŷ_t ]
           │                                                 │
           ▼                                                 ▼
[ Actual Telemetry y_t ] ─────────────────────────► [ Deviation Engine |y_t - ŷ_t| ]
           │                                                 │
           └───────────────────────┬─────────────────────────┘
                                   ▼
                   [ Twin-Augmented IDS Classifier ]
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                               ▼
         [ Prediction & Confidence ]      [ SHAP TreeExplainer ]
                   │                               │
                   └───────────────┬───────────────┘
                                   ▼
                   [ Operational Confidence Filter ]
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
         [ High-Fidelity Alert ]            [ Suppressed Noise ]
```

---

## 📊 Experimental Results (Edge-IIoTset Benchmark)

| Configuration | Feature Space | Accuracy (%) | Macro-F1 | Mean Latency +- Std (ms) | Throughput (samples/s) | Storage (KB) |
|---|---|---|---|---|---|---|
| **Config 1: Full Twin + Heavy RF (150 trees)** | Twin-Augmented-v2 | **93.85%** | **0.9056** | **0.449 +- 0.001 ms** | 2,225.4 | 17,390.3 KB |
| **Config 2: Quantized Twin + Standard RF (100 trees)** | Twin-Augmented-v2 | **92.92%** | **0.8957** | **0.177 +- 0.007 ms** | 5,647.0 | 6,636.6 KB |
| **Config 3: Quantized Twin + Pruned RF (30 trees)** | Twin-Augmented-v2 | 88.86% | 0.8444 | **0.153 +- 0.001 ms** | 6,540.2 | 578.9 KB |
| **Config 4: Fast-Inference Edge XGBoost (25 trees)** | Raw Telemetry | 91.81% | 0.8871 | **0.005 +- 0.001 ms** | **218,134.9** | **105.9 KB** |

*Note: The Operational Confidence Filter reliably suppresses **30.0%** (empirical range: 28.6%–31.4%) of ambiguous, low-confidence false positives.*

---

## 🚀 Quick Start & Local Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
cd dashboard && npm install && cd ..
```

### 2. Launch Full Pipeline & Dashboard
```bash
python run_project.py
```
- **Interactive Cyber Dashboard:** `http://localhost:5173`
- **FastAPI Documentation & SSE Stream:** `http://127.0.0.1:8000/docs`

---

## 📁 Repository Structure
```
├── api/                   # Vercel serverless entrypoint (index.py)
├── data/                  # Preprocessed data & data dictionary
├── dashboard/             # Modern React + Vite frontend UI
├── docs/                  # IEEE Research Paper & Project Reports
│   ├── RESEARCH_PAPER_DRAFT.md
│   ├── FINAL_PROJECT_REPORT.md
│   └── VIVA_DEFENSE_SCRIPT.md
├── models/                # Trained ML models & scalers (.pkl)
├── notebooks/             # Google Colab T4 GPU LSTM training notebook
├── results/               # Master benchmark CSVs & publication charts
├── src/                   # Python core pipelines
│   ├── api_server.py
│   ├── confidence_filter.py
│   ├── deviation_engine.py
│   ├── edge_benchmark.py
│   ├── ids_model.py
│   ├── preprocess.py
│   ├── simulator.py
│   ├── twin_model.py
│   └── xai_module.py
├── pyproject.toml         # Packaging & Vercel deployment configuration
├── vercel.json            # Vercel serverless routing
└── run_project.py         # Integrated concurrent launcher
```
