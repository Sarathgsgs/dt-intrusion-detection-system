# Twin-Guided Explainable Intrusion Detection System (X-IDS) for Industrial IoT
## Final Comprehensive Technical Project Report

**Project Title:** Twin-Guided Explainable Intrusion Detection System (X-IDS) with Operational Confidence Filtering and Edge-Resource Optimization  
**Repository:** [https://github.com/Sarathgsgs/dt-intrusion-detection-system.git](https://github.com/Sarathgsgs/dt-intrusion-detection-system.git)  
**Date:** August 21, 2026  
**Status:** Complete & Validated (Post-Revision v5)

---

## Executive Summary
Industrial IoT (IIoT) infrastructures increasingly suffer from sophisticated multi-vector cyber-attacks that standard black-box machine learning models struggle to explain. This project delivers **X-IDS**, an end-to-end, resource-aware, and explainable intrusion detection framework designed specifically for resource-constrained edge gateways and SCADA/ICS controllers. 

X-IDS unites:
1. **A Scope-Restricted Continuous Digital Twin Neural Forecaster** with domain physical bounding ($0 \le \text{tcp.len} \le 65535$) to predict expected healthy telemetry dynamics.
2. **A Targeted Residual Deviation Engine** that computes continuous discrepancy vectors ($|y_t - \hat{y}_t|$).
3. **Multi-Class IDS Classifiers** operating on a 43-feature Twin-Augmented space, achieving **94.81% accuracy** (within 0.19% of raw baseline) across 15 distinct threat categories.
4. **Fine-Grained 15-Class 4-Model Parity Validation** demonstrating statistical parity across 13 of 15 classes (including $F_1 = 1.0000$ on volumetric DDoS floods).
5. **A Local SHAP Explainer & Operational Confidence Filter** that suppresses **30.0%** (range: 28.6% - 31.4%) of ambiguous false alarms.
6. **Edge Hardware Benchmarks** proving sub-millisecond latency ($0.005\text{ ms} - 0.449\text{ ms}$) and compact memory footprint ($105.9\text{ KB} - 17.39\text{ MB}$).

---

## 1. System Architecture & Methodology

```
                                  INCOMING TELEMETRY (34 RAW FEATURES)
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
         25 Discrete / State Features                                9 Continuous Features
         (Flags, Protocols, Ports)                                   (Sizes, Lengths, Jitter)
                   │                                                           │
                   │                                                           ▼
                   │                                              ┌─────────────────────────┐
                   │                                              │ Scope-Restricted Twin   │
                   │                                              │ (Physically Bounded)    │
                   │                                              └────────────┬────────────┘
                   │                                                           │
                   │                                                           ▼
                   │                                              ┌─────────────────────────┐
                   │                                              │ Residual Deviation      │
                   │                                              │ Engine: |y_t - y_hat_t| │
                   │                                              └────────────┬────────────┘
                   │                                                           │
                   └─────────────────────────────┬─────────────────────────────┘
                                                 ▼
                                43-FEATURE TWIN-AUGMENTED SPACE
                                                 │
                                                 ▼
                                ┌─────────────────────────────────┐
                                │ XGBoost / Random Forest IDS     │
                                └────────────────┬────────────────┘
                                                 │
                                                 ▼
                                ┌─────────────────────────────────┐
                                │ SHAP Local Feature Attribution  │
                                └────────────────┬────────────────┘
                                                 │
                                                 ▼
                                ┌─────────────────────────────────┐
                                │ Operational Confidence Filter   │
                                └────────────────┬────────────────┘
                                                 │
                         ┌───────────────────────┴───────────────────────┐
                         ▼                                               ▼
                PASS (Validated Alert)                         SUPPRESS (Filtered Noise)
```

---

## 2. Experimental Results & Validated Performance Metrics

### A. Multi-Class IDS Model Suite Performance (13,999 Test Samples)

| Model Architecture | Feature Space | Accuracy (%) | Macro-F1 | Weighted-F1 | Inference Latency |
|---|---|---|---|---|---|
| **RF-Raw (Baseline)** | Raw Telemetry (34 features) | 94.77% | 0.9177 | 0.9499 | 0.0131 ms/sample |
| **XGB-Raw (Baseline)** | Raw Telemetry (34 features) | **95.00%** | **0.9200** | **0.9522** | **0.0124 ms/sample** |
| **RF-Deviation (Pure)** | Continuous Residuals (9 features) | 62.30% | 0.5972 | 0.6183 | 0.0184 ms/sample |
| **XGB-Deviation (Pure)** | Continuous Residuals (9 features) | 63.05% | 0.6086 | 0.6277 | 0.0130 ms/sample |
| **RF-Twin-Augmented-v2** | Raw + Continuous Residuals (43) | **93.89%** | **0.9062** | **0.9397** | 0.0136 ms/sample |
| **XGB-Twin-Augmented-v2** | Raw + Continuous Residuals (43) | **94.81%** | **0.9158** | **0.9488** | 0.0120 ms/sample |

### B. 4-Model 15-Class Per-Attack Breakdown Table

| Attack Class | Support | RF-Raw F1 | XGB-Raw F1 | RF-Twin-v2 F1 | XGB-Twin-v2 F1 | $\Delta F_1$ (XGB) | Outcome |
|---|---|---|---|---|---|---|---|
| **XSS** | 892 | 0.9058 | **0.9084** | 0.8839 | **0.9085** | `+0.0001` | `Statistical Parity` |
| **DDoS_TCP** | 909 | 1.0000 | **1.0000** | 1.0000 | **1.0000** | `0.0000` | `Exact Parity` |
| **DDoS_UDP** | 1286 | 1.0000 | **1.0000** | 1.0000 | **1.0000** | `0.0000` | `Exact Parity` |
| **DDoS_ICMP** | 1250 | 0.9996 | **0.9996** | 0.9992 | **0.9996** | `0.0000` | `Exact Parity` |
| **Backdoor** | 904 | 0.9837 | **0.9848** | 0.9758 | **0.9848** | `0.0000` | `Statistical Parity` |
| **Normal** | 2156 | 0.9979 | **0.9979** | 0.9977 | **0.9977** | `-0.0002` | `Exact Parity` |
| **MITM \*** | 108 | 0.5806 | **0.5806** | 0.5887 | 0.5792 | `-0.0015` | `Statistical Parity` |
| **Uploading** | 911 | 0.9167 | **0.9221** | 0.9002 | **0.9205** | `-0.0016` | `Statistical Parity` |
| **Vulnerability_scanner** | 894 | 0.9773 | **0.9759** | 0.9737 | 0.9730 | `-0.0028` | `Statistical Parity` |
| **Password** | 886 | 0.8915 | **0.8990** | 0.8675 | 0.8953 | `-0.0037` | `Statistical Parity` |
| **SQL_injection** | 915 | 0.8873 | **0.8963** | 0.8601 | 0.8901 | `-0.0062` | `Statistical Parity` |
| **DDoS_HTTP** | 937 | 0.8472 | **0.8571** | 0.8274 | 0.8507 | `-0.0065` | `Statistical Parity` |
| **Port_Scanning** | 893 | 0.9511 | **0.9511** | 0.9374 | 0.9444 | `-0.0068` | `Statistical Parity` |
| **Fingerprinting \*** | 89 | 0.8889 | **0.8889** | 0.8625 | 0.8750 | `-0.0139` | `Raw Preferred` |
| **Ransomware** | 969 | 0.9379 | **0.9385** | 0.9190 | 0.9176 | `-0.0209` | `Raw Preferred` |

*(\*) Indicates low sample support ($n < 200$).*

### C. Master Edge-Resource Trade-Off Benchmark

| Configuration | Feature Space | Accuracy (%) | Macro-F1 | Mean Latency $\pm$ Std (ms) | Throughput (samples/s) | Storage (KB) |
|---|---|---|---|---|---|---|
| **Config 1: Full Twin + Heavy RF (150 trees)** | Twin-Augmented-v2 | **93.85%** | **0.9056** | **$0.449 \pm 0.001\text{ ms}$** | 2,225.4 | 17,390.3 KB |
| **Config 2: Quantized Twin + Standard RF (100 trees)** | Twin-Augmented-v2 | **92.92%** | **0.8957** | **$0.177 \pm 0.007\text{ ms}$** | 5,647.0 | 6,636.6 KB |
| **Config 3: Quantized Twin + Pruned RF (30 trees)** | Twin-Augmented-v2 | 88.86% | 0.8444 | **$0.153 \pm 0.001\text{ ms}$** | 6,540.2 | 578.9 KB |
| **Config 4: Fast-Inference Edge XGBoost (25 trees)** | Raw Telemetry | 91.81% | 0.8871 | **$0.005 \pm 0.001\text{ ms}$** | **218,134.9** | **105.9 KB** |

---

## 3. Operational Confidence Filter Reconciliation
- **Canonical Alert Suppression Rate:** **30.0%**
- **Empirical Measured Range Across Live Streaming Runs:** **28.6% – 31.4%** ($\sigma = 0.94\%$).
- **Pass Decision Threshold:** Minimum classification confidence $\ge 0.65$ AND $\ge 1$ overlapping SHAP top-risk sensor with domain attack signature.

---

## 4. Hardware Deployment Recommendations
1. **Config 4 (Fast-Inference XGBoost, 105.9 KB, 0.005 ms):** Best suited for high-frequency distributed field sensors where maximum throughput ($>218,000\text{ samples/sec}$) is essential.
2. **Config 2 (Quantized Twin + Standard RF, 6.64 MB, 0.177 ms):** Best suited for safety-critical plant gateways and SCADA systems where physical twin verification ($|y_t - \hat{y}_t|$) is required before executing actuators or alarms.

---

## 5. Deliverables & Artifact Index
- **Source Code:** [`src/`](file:///e:/Projects/digital%20twin/src/)
- **Interactive React Dashboard:** [`dashboard/`](file:///e:/Projects/digital%20twin/dashboard/)
- **Empirical CSVs & Charts:** [`results/`](file:///e:/Projects/digital%20twin/results/)
- **Viva Defense Script:** [`docs/VIVA_DEFENSE_SCRIPT.md`](file:///e:/Projects/digital%20twin/docs/VIVA_DEFENSE_SCRIPT.md)
- **Academic Paper Draft:** [`docs/RESEARCH_PAPER_DRAFT.md`](file:///e:/Projects/digital%20twin/docs/RESEARCH_PAPER_DRAFT.md)
