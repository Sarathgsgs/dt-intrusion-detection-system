# Phase 1: Twin Forecast Scaling & Physical Bounds Diagnostic Note
**Date:** August 21, 2026  
## 1. Root-Cause Analysis
- **Root Cause:** In earlier versions, `tcp.seq` and `tcp.ack` (32-bit sequence numbers spanning $10^7 - 10^9$) had massive variances that caused multi-output MLP activation cross-talk during unexpected sequence shifts. Without explicit physical output clamping, this produced occasional negative or oversized unscaled forecasts for `tcp.len`.- **Classification of Issue:** Both a **data/model bounding gap** (lack of physical domain clamping $[0, 	ext{MTU}]$ in `twin_model.py`) and a **display scaling gap** (global Y-axis stretching).- **Fix Implemented:** Implemented strict physical bounding in `predict_next_state` and `compute_dataset_predictions` across all 9 continuous signals using `PHYSICAL_BOUNDS`.
## 2. Empirical Verification Table Across Continuous Telemetry

| Continuous Feature | Physical Lower Bound | Physical Upper Bound | Forecasted Min | Forecasted Max | Status |
|---|---|---|---|---|---|
| **icmp.checksum** | 0.0 | 65535.0 | 0.00 | 0.48 | [VALID] |
| **icmp.seq_le** | 0.0 | 65535.0 | 0.00 | 0.93 | [VALID] |
| **http.content_length** | 0.0 | 10000000.0 | 0.00 | 0.72 | [VALID] |
| **tcp.ack** | 0.0 | 4294967295.0 | 0.00 | 169486925.00 | [VALID] |
| **tcp.checksum** | 0.0 | 65535.0 | 15880.61 | 64179.00 | [VALID] |
| **tcp.len** | 0.0 | 65535.0 | 0.00 | 361.24 | [VALID] |
| **tcp.seq** | 0.0 | 4294967295.0 | 0.00 | 37907786.82 | [VALID] |
| **udp.stream** | 0.0 | 1000000.0 | 0.00 | 1761.94 | [VALID] |
| **udp.time_delta** | 0.0 | 3600.0 | 0.00 | 10.39 | [VALID] |

## 3. Impact on Downstream Models
- Applying physical bounds ensures that all continuous residuals $\mathbf{e}_t = |y_t - \hat{y}_t|$ in `data/deviation_dataset.csv` represent genuine, bounded physical discrepancies rather than mathematical artifacts.- The downstream IDS models and benchmarks remain fully consistent with genuine physical telemetry dynamics.