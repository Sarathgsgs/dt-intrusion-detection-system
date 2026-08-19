# Full Implementation & Revision Plan: Twin-Guided Explainable Intrusion Detection System (X-IDS)
## A Resource-Aware Trade-off Analysis for Industrial IoT

**Project State:** Baseline Pipeline Fully Operational (Milestones 0–9 Complete).  
**Current Phase:** Scientific Refinement, Number Reconciliation & Defense Hardening (Phases A–H).  
**Repository:** [https://github.com/Sarathgsgs/dt-intrusion-detection-system.git](https://github.com/Sarathgsgs/dt-intrusion-detection-system.git)

---

## 📌 Executive Architecture & Execution Matrix

```
[ IIoT Telemetry Stream ] (34 Raw Features)
           │
           ├──────────────────────────────┬──────────────────────────────┐
           ▼                                                             ▼
[ 14 Continuous / Physical Features ]                     [ 20 Categorical / Discrete Features ]
           │                                                (Ports, Flags, Protocols)
           ▼                                                             │
[ Digital Twin Forecaster ] (MLP/LSTM)                                  │
  (Forecasts ONLY Continuous Baseline Dynamics)                          │
           │                                                             │
           ▼ ŷ_t                                                         │
[ Deviation Engine ]                                                     │
  e_t = |y_t - ŷ_t| (14 Residuals)                                       │
           │                                                             │
           └──────────────────────────────┬──────────────────────────────┘
                                          ▼
                      [ Twin-Augmented-v2 Feature Space ]
                      (20 Raw Categorical + 14 Raw Continuous + 14 Deviation Continuous = 48 Features)
                                          │
                                          ▼
                      [ Multi-Class IDS Classifiers ]
                           (Random Forest / XGBoost)
                                          │
                          ┌───────────────┴───────────────┐
                          ▼                               ▼
                [ Prediction & Confidence ]      [ SHAP TreeExplainer ]
                          │                               │
                          └───────────────┬───────────────┘
                                          ▼
                          [ Operational Confidence Filter ]
                            (Signature Match + γ Bounds)
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
            [ High-Fidelity Alert ]                [ Suppressed Noise ]
```

---

## 🛠️ Execution Responsibility Matrix

| Phase | Description | Location | Primary Files | Handled By |
|---|---|---|---|---|
| **Phase A** | Root-Cause Diagnosis & Feature Partitioning | Inside IDE | `src/preprocess.py`, `src/twin_model.py` | **Agent [IN-IDE]** |
| **Phase B** | Scope-Restricted Digital Twin & Retraining | Inside IDE | `src/twin_model.py`, `src/deviation_engine.py`, `src/ids_model.py` | **Agent [IN-IDE]** |
| **Phase C** | Per-Attack-Type Advantage Discovery | Inside IDE | `src/ids_model.py`, `results/` | **Agent [IN-IDE]** |
| **Phase D** | Confidence Filter Metric Reconciliation | Inside IDE | `src/confidence_filter.py`, `docs/` | **Agent [IN-IDE]** |
| **Phase E** | TON_IoT Generalization Dual-Model Audit | Inside IDE | `src/generalization_eval.py`, `results/` | **Agent [IN-IDE]** |
| **Phase F** | Paper & Report Value Proposition Reframing | Inside IDE | `docs/RESEARCH_PAPER_DRAFT.md`, `docs/FINAL_PROJECT_REPORT.md` | **Agent [IN-IDE]** |
| **Phase G** | Dashboard Visual & Data-Binding Fix | Inside IDE | `dashboard/src/App.jsx`, `src/api_server.py` | **Agent [IN-IDE]** |
| **Phase H** | Final Number Reconciliation & Defense Prep | Inside IDE | `docs/VIVA_DEFENSE_SCRIPT.md`, All Docs | **Agent [IN-IDE]** |

---

## 🚀 Detailed Phase-by-Phase Revision Tasks

### 📍 Phase A — Root-Cause Diagnosis (Audit & Analysis)
**Objective:** Diagnose why the initial all-feature twin-deviation model underperformed before making code changes.

1. **Feature Partitioning:**
   - Classify all 34 features into:
     - **Continuous/Physical features** (temporal dynamics, byte counts, packet length, jitter, duration).
     - **Categorical/Discrete features** (ports, protocol numbers, TCP/IP flags, connection state codes) where sequence forecasting is mathematically non-smooth.
2. **Per-Feature Error Decomposition:**
   - Update `src/twin_model.py` to evaluate validation MSE and MAE broken down **per feature** on scaled data.
   - Identify which features the Twin forecasts with high precision vs. high variance.
3. **Scaling Audit:**
   - Verify `models/twin_scaler.pkl` normalization consistency between training sequences and live deviation engine calculation.
4. **Per-Class Baseline Audit:**
   - Generate full per-class precision, recall, and F1 reports for baseline XGB-Raw vs original RF-Twin-Augmented.

**Deliverable:** Diagnostic report (`results/diagnostic_phase_a.json` & summary note).

---

### 📍 Phase B — Fix the Twin's Feature Scope & Retrain
**Objective:** Restrict the Digital Twin to forecast only physically forecastable features, passing categorical features directly to the classifier.

1. **Scope-Restricted Twin (`src/twin_model.py`):**
   - Retrain the Digital Twin on the **Continuous Feature subset only** ($K$ continuous features).
   - Export updated `models/twin_model.pkl` and `models/twin_scaler.pkl`.
2. **Targeted Deviation Engine (`src/deviation_engine.py`):**
   - Calculate residuals $e_{t, j} = |y_{t, j} - \hat{y}_{t, j}|$ strictly for the continuous subset.
   - Construct `Twin-Augmented-v2` feature space:
     $$\mathbf{z}_t = [\mathbf{x}_{\text{categorical}} \,\|\, \mathbf{x}_{\text{continuous}} \,\|\, \mathbf{e}_{\text{continuous}}]$$
3. **Retrain IDS Models (`src/ids_model.py`):**
   - Retrain RF and XGBoost on `Twin-Augmented-v2`.
   - Update `results/ids_metrics.csv` and `results/ids_comparison.png`.

**Deliverable:** Updated `data/deviation_dataset.csv`, `models/`, and comparative metrics table.

---

### 📍 Phase C — Per-Attack-Type Advantage Discovery
**Objective:** Uncover specific attack categories where Twin-Augmentation provides a distinct detection advantage over raw telemetry.

1. **Per-Attack F1 Matrix:**
   - Map per-class F1 scores: Raw Baseline vs. Twin-Augmented-v2 across all 15 attack types.
2. **Behavioral vs. Volumetric Pattern Analysis:**
   - Test hypothesis: Twin guidance excels on stealthy behavioral anomalies / tampering / MITM where raw statistical thresholds fail, while volumetric floods are dominated by raw port/packet counters.
3. **Visualization:**
   - Generate `results/per_attack_comparison.png` and `results/per_attack_f1.csv`.

**Deliverable:** Dedicated per-attack comparison chart and definitive evidence-backed finding.

---

### 📍 Phase D — Confidence-Filter Metric Reconciliation
**Objective:** Establish a single, reproducible, and verifiable suppression rate across all documentation and dashboard sessions.

1. **Multi-Seed Filter Benchmark:**
   - Run `src/confidence_filter.py` across 5 stratified test splits.
   - Record exact mean suppression rate and empirical min-max range (e.g. 28.4% – 32.1%).
2. **Documentation Synchronization:**
   - Reconcile numbers in `docs/RESEARCH_PAPER_DRAFT.md`, `docs/FINAL_PROJECT_REPORT.md`, `src/api_server.py`, and dashboard cards.

**Deliverable:** 100% consistent suppression metrics across the codebase.

---

### 📍 Phase E — TON_IoT Generalization Dual-Model Audit
**Objective:** Evaluate both Raw and Twin-Augmented models on unseen TON_IoT data with complete precision and recall reporting.

1. **Dual-Model Zero-Shot Evaluation (`src/generalization_eval.py`):**
   - Run both XGB-Raw and XGB-Twin-Augmented-v2 against `data/train_test_network.csv`.
2. **Complete Metric Reporting:**
   - Document Accuracy, Macro-F1, Precision, and **per-class Recall** (quantifying conservative vs aggressive alerting).
3. **Export Updated Results:**
   - Save to `results/generalization_results.csv` and `results/generalization_transfer.png`.

**Deliverable:** Comparative cross-dataset table explaining precision-recall trade-offs.

---

### 📍 Phase F — Reposition the Twin's Value Proposition
**Objective:** Frame the academic contribution with scientific honesty, highlighting causal interpretability and specific behavioral advantages rather than forcing an artificial aggregate win.

1. **Paper & Report Reframing (`docs/RESEARCH_PAPER_DRAFT.md`, `docs/FINAL_PROJECT_REPORT.md`):**
   - Articulate why lightweight Config 4 (XGBoost) dominates on raw latency/size, while Config 2 (Quantized Twin) provides physical grounding, explainable deviation signals, and stealth anomaly protection.
2. **Threats to Validity:**
   - Add explicit limitations and boundary condition discussions.

**Deliverable:** Publication-grade, reviewer-proof academic drafts.

---

### 📍 Phase G — Dashboard Visual & Data-Binding Fix
**Objective:** Ensure the Live Monitor dual-trace chart exhibits dynamic, responsive motion during replay.

1. **Dynamic Feature Binding:**
   - Update `dashboard/src/App.jsx` and `src/api_server.py` to stream a high-variance continuous feature (e.g. dynamic packet size / connection rate) so the Actual vs. Twin forecast traces visibly move.
2. **Visual Polish:**
   - Add contextual tooltips and attack injection status badges.

**Deliverable:** Fully responsive, moving live visualizer on `http://localhost:5173`.

---

### 📍 Phase H — Master Consistency Check & Defense Prep
**Objective:** Validate that every number, figure, and defense script is synchronized and demo-ready.

1. **Full Metric Audit:**
   - Re-verify consistency across `results/*.csv`, `docs/`, and dashboard.
2. **Examiner Q&A Expansion (`docs/VIVA_DEFENSE_SCRIPT.md`):**
   - Add fortified answers for:
     - *"Why use a Digital Twin if raw XGBoost has higher aggregate accuracy?"*
     - *"Why choose Config 2 over Config 4 in safety-critical edge environments?"*

**Deliverable:** Comprehensive, locked-in defense suite.
