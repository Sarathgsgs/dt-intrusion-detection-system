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

| Configuration | Feature Space | Accuracy (%) | Macro-F1 | Latency (ms/sample) | Throughput (samples/s) | Storage (KB) |
|---|---|---|---|---|---|---|
| **Config 1: Full Twin + Heavy RF (150 trees)** | Twin-Augmented | **92.96%** | **0.8927** | 0.430 ms | 2,326.5 | 18,134.8 KB |
| **Config 2: Quantized Twin + Standard RF (100 trees)** | Twin-Augmented | 91.64% | 0.8786 | 0.176 ms | 5,680.4 | 7,291.0 KB |
| **Config 3: Quantized Twin + Pruned RF (30 trees)** | Twin-Augmented | 84.98% | 0.8041 | 0.152 ms | 6,587.5 | 571.7 KB |
| **Config 4: Fast-Inference Edge XGBoost (25 trees)** | Raw Telemetry | 91.81% | 0.8871 | **0.005 ms** | **204,592.7** | **105.9 KB** |

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
