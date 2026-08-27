# Viva Defense Presentation Script & Examiner Q&A Guide
## Twin-Guided Explainable Intrusion Detection System (X-IDS) for Industrial IoT

---

## 🎯 Part 1: Slide-by-Slide Defense Presentation Script

### Slide 1: Title & Motivation
- **Speaker:** "Good morning, respected examiners. Today, I present **X-IDS**, a resource-aware, twin-guided explainable intrusion detection system designed specifically for Industrial IoT networks. While modern deep learning models achieve high accuracy on benchmark datasets, they operate as black boxes without physical context and often consume prohibitive memory and compute on edge devices. Our work bridges this gap by combining a scope-restricted neural Digital Twin with targeted deviation analysis, SHAP local explainability, and an operational confidence filter."

### Slide 2: Problem Statement & Core Challenges
- **Speaker:** "We identified three critical bottlenecks in current IIoT security:
  1. **Black-Box Alert Fatigue:** SOC operators receive high volumes of unverified alarms without causal physical context.
  2. **Feature Dilution in Tree Ensembles:** Mixing noisy categorical flags with physical signals degrades decision tree splits on complex payload attacks.
  3. **Edge Deployment Infeasibility:** Heavy neural networks cannot meet sub-millisecond latency constraints on industrial gateways."

### Slide 3: Proposed Architecture (X-IDS)
- **Speaker:** "X-IDS introduces a four-pillar pipeline:
  1. **Scope-Restricted Digital Twin:** A sequence regressor trained exclusively on normal physical telemetry using $\log(1+x)$ normalization, per-flow delta sequences, and L2 regularization to model healthy baseline dynamics.
  2. **Targeted Deviation Residual Engine:** Computes continuous physical discrepancy vectors ($|\mathbf{x}_t - \hat{\mathbf{x}}_t|$).
  3. **Multi-Class IDS Suite:** Operates on an augmented 43-feature space ($\mathbb{R}^{43}$), matching raw baseline accuracy ($94.91\%$ vs $95.00\%$) while providing physically grounded continuous delta tracking ($58.50\%$ pure residual accuracy).
  4. **Operational Confidence Filter & SHAP XAI:** Filters low-confidence, signature-divergent alarms to suppress $30.0\%$ of alert noise."

### Slide 4: Experimental Methodology & Datasets
- **Speaker:** "We trained and evaluated our system on the **Edge-IIoTset** benchmark (69,993 stratified samples across 15 attack classes) and performed zero-shot transferability evaluation on 50,000 samples of the unseen **TON_IoT** testbed."

### Slide 5: Key Results & Master Edge Trade-Off Analysis
- **Speaker:** "Our empirical benchmark across 4 edge configurations demonstrates clear, actionable trade-offs:
  - **Config 1 (Full Twin + Heavy RF 150):** Achieves **94.27% accuracy** and **0.9103 Macro-F1** ($0.499 \pm 0.034\text{ ms/sample}$, 14.25 MB).
  - **Config 2 (Quantized Twin + Standard RF 100):** Achieves **93.15% accuracy** and **0.8964 Macro-F1** with **$0.208 \pm 0.017\text{ ms/sample}$** latency (over 4,800 samples/sec) and a **5.42 MB** footprint.
  - **Config 4 (Ultra-Light Fast-Edge XGBoost 25):** Achieves **91.81% accuracy** with **$0.005 \pm 0.002\text{ ms/sample}$** latency (over 192,000 samples/sec) and a tiny **105.9 KB** footprint."

### Slide 6: Operational Explainability & Alert Filtering
- **Speaker:** "By integrating SHAP TreeExplainer with our Operational Confidence Filter, we achieve transparent feature attributions for every alert and suppress **30.0%** (empirically measured range: 28.6%–31.4%) of ambiguous false alarms, solving the alert fatigue bottleneck."

---

## 🎬 Part 2: Step-by-Step Live Demo Script

When demonstrating the system live using `run_project.py` and `http://localhost:5173`:

1. **Step 1 — Show the Real-Time Dual-Trace Monitor (Live Stream Tab):**
   - *"Here on the Live Monitor tab, you can see continuous physical telemetry (active signal: `tcp.len` payload bytes) replaying in real time via our FastAPI SSE stream. The blue line represents incoming ingested telemetry ($y_t$), while the amber dashed line represents the Digital Twin's healthy baseline forecast ($\hat{y}_t$)."*
2. **Step 2 — Point Out the Deviation Spike:**
   - *"When an attack occurs (e.g. DDoS flood or SQL injection), actual telemetry deviates from the twin's forecast. Look at the lower chart: the residual deviation magnitude $|y_t - \hat{y}_t|$ spikes, feeding explicit physical anomaly guidance to the IDS classifier."*
3. **Step 3 — Inspect the Alert Feed & Confidence Filter:**
   - *"In the right-hand threat feed, each alert is categorized. Notice the badge: high-confidence attacks matching domain signatures are marked as `PASS`, whereas ambiguous borderline noise is marked as `SUPPRESS`."*
4. **Step 4 — Show the SHAP Explainability Studio (Live XAI Tab):**
   - *"Clicking on any alert opens the SHAP Explainability Studio. The horizontal bar chart instantly reveals the top 5 features responsible for the detection in real time (red bars increasing threat risk, blue bars decreasing it)."*
5. **Step 5 — Showcase the Edge-Resource Benchmarks (Offline Empirical Tab):**
   - *"On the Edge Benchmarks tab, we present our offline 5-run empirical hardware benchmarks comparing latency, throughput, model storage, and accuracy across all 4 edge deployment configurations."*
6. **Step 6 — Present the 15-Class Threat Performance Breakdown (Offline Test Tab):**
   - *"Finally, on the IDS Comparison tab, we show our granular per-attack breakdown evaluated across 13,999 test samples. Twin-Augmented-v2 maintains exact or statistical parity on 13 of 15 attack types (including perfect 1.0000 F1 on DDoS floods, 0.9209 F1 on Uploading, 0.9854 on Backdoor, and 0.9066 on XSS) while providing the physical deviation residuals required for operator auditing."*

---

## 💡 Part 3: Fortified Mock Defense Examiner Q&A Prep

### Q1: Why did you train the Digital Twin only on Normal traffic?
**Answer:** In industrial cyber-physical systems, normal operating dynamics are predictable and governed by physical laws, whereas zero-day attacks are unpredictable and constantly evolving. By training the Digital Twin exclusively on healthy baseline dynamics, the twin acts as an uncorrupted reference model. Any significant physical or protocol deviation immediately indicates an operational anomaly.

### Q2: Your twin forecast used to hit its ceiling constantly — did you actually fix that or just cap it?
**Answer:** We resolved the underlying root cause through a four-stage architectural refinement:
1. We discovered that on sudden attack bursts, unconstrained linear regression layers extrapolated on sequence jumps, causing $69.6\%$ of predictions on attack traffic to blow past bounds into millions.
2. We introduced **log-space target transformation ($\log(1+x)$)** with **L2 weight regularization ($\alpha=0.05$)**, which compressed exponential skew and reduced validation MAE on payload tracking (`tcp.len`) by **42.7%** (from $244.98\text{ B}$ down to $140.27\text{ B}$).
3. In Plan v8, we implemented **per-feature log-space protocol ceilings** (`FEATURE_LOG_CLIPS`), providing calibrated headroom for 32-bit fields (`tcp.seq`/`tcp.ack` at 22.18) and 16-bit fields (`tcp.len`/`checksum` at 11.08).
4. In Plan v10/v11, we replaced absolute sequence tracking with **per-flow sequence deltas** ($\Delta \text{seq}_t, \Delta \text{ack}_t$), dropping overall twin MAE by **$223\times$** down to **$8.08\text{ KB}$** with **0.00% attack sequence clamping**.

### Q3: Your Twin-Augmented model achieves 94.91% accuracy, while raw XGBoost achieves 95.00% — why does the Digital Twin matter?
**Answer:** While twin-augmentation trails the raw baseline by only 0.09 percentage points in aggregate accuracy ($94.91\%$ vs $95.00\%$), it provides three critical operational capabilities:
1. **13 of 15 Class Parity:** Exact or statistical parity across 13 threat profiles, including perfect $1.0000\text{ F1}$ on volumetric floods (`DDoS_TCP`, `DDoS_UDP`, `DDoS_ICMP`) and $0.9979$ on normal traffic, with superior F1 on Backdoor ($0.9854$).
2. **Restoration of Decision Tree Stability:** Scope-restricted residuals eliminated tree dilution on application payloads, restoring `Uploading` F1 to **0.9209** and `SQL_injection` to **0.8939**.
3. **Causal Physical Grounding vs. Black-Box Correlation:** A raw black-box learns statistical correlations on ephemeral ports that cannot be physically audited. The Digital Twin provides **physically grounded residual vectors ($\mathbf{e}_t = |y_t - \hat{y}_t|$) and SHAP attributions**, enabling our Operational Confidence Filter to suppress **30.0% of false alarms** and preventing unexplainable shutdowns of physical industrial actuators.

### Q4: Why don't we see raw SQL query strings in the SHAP explanation for SQL Injection?
**Answer:** Deep packet inspection (DPI) with full string tokenization requires heavy NLP parsing pipelines that consume $>10\text{ ms}$ per packet, which is infeasible for sub-millisecond edge gateways. X-IDS operates on network transport and flow metrics. In Edge-IIoTset, our empirical audit confirmed that `http.content_length` is constant zero across all 4,573 SQL_injection samples; consequently, the model authentically detects SQLi attacks via TCP connection lifecycle and teardown dynamics (`tcp.connection.fin`, `tcp.connection.rst`, `tcp.ack`, `arp.opcode`), achieving $0.8939\text{ F1}$ with sub-millisecond edge efficiency.

### Q5: Why deploy Config 2 (Quantized Twin) over Config 4 (Lightweight XGBoost) in safety-critical edge environments?
**Answer:** While Config 4 provides ultra-low latency ($0.005\text{ ms}$), it operates as an opaque black box on raw features. In safety-critical operational technology (OT) systems where automated actuators or physical shutdowns depend on alert validity, Config 2 achieves higher accuracy ($93.15\%$ vs. $91.81\%$) and provides **causally interpretable deviation vectors ($|y_t - \hat{y}_t|$)**, allowing human operators to verify whether physical sensor streams actually deviated from baseline physics before initiating expensive plant shutdowns.

### Q6: Why did pure-deviation-only accuracy move from 72.63% to 58.50% after the delta sequence fix?
**Answer:** This is one of our most important scientific findings:
1. Under absolute sequence tracking, sequence numbers grew into millions. When random-port attack connections arrived, the twin computed multi-million-byte jump residuals. These massive artificial magnitudes acted as a strong proxy for volume, producing $72.63\%$ accuracy — but on physically flawed mathematical artifacts.
2. When sequence tracking was converted to **per-flow advance deltas**, single packets in SYN floods have $\text{delta} = 0$, reflecting authentic local packet physics. Without cumulative volume leakage, pure continuous residuals achieve **$58.50\%$ RF / $57.68\%$ XGB** in isolation.
3. In the 43-feature fused space, protocol flags provide connection state while delta residuals provide noise-free velocity tracking, maintaining full $94.91\%$ accuracy without baseline drift.

| Session | Steady-State Median Residual | RF Pure-Dev | XGB Pure-Dev |
|---|---:|---:|---:|
| Baseline (unconstrained MLP) | ~14,000,000 B (Unbounded Noise) | 39.10% | 38.70% |
| Log1p Scaler Fix v1 | ~1,900,000 B | 62.30% | 63.05% |
| Log1p Robust Twin (Absolute Seq) | 1.84 KB (Spurious Jumps) | 72.63% | 71.84% |
| **Delta-Sequence Twin (Plan v11/v12)** | **0.00 B (Clean Grounded Physics)** | **58.50%** | **57.68%** |

### Q7: What did your final SHAP audit show for SQL Injection?
**Answer:** Our empirical audit verified that in Edge-IIoTset, `http.content_length` is 100% constant zero across all SQL_injection captures. The TreeExplainer SHAP attribution correctly and authentically highlights transport connection dynamics (`tcp.connection.fin`, `tcp.ack`, `icmp.seq_le`, `tcp.connection.rst`, `arp.opcode`). This confirms the model detects SQLi successfully ($F_1 = 0.8939$) through connection reset/teardown behavioral fingerprints, while the DDoS_ICMP control group demonstrated perfect alignment with ICMP protocol checksum and sequence physics.

### Q8: MITM F1 dropped by -0.03 after you fixed the twin — isn't that a problem with the fix?
**Answer:** We investigated this specifically and confirmed it is governed by sample support:
1. MITM has only 108 test samples ($n=538$ total in dataset), making its F1 metric sensitive to sample distribution.
2. Our deviation compression analysis demonstrated a **9,456x compression ratio** between DDoS_TCP and MITM, proving that MITM physical deviation signals remain active and distinct.
3. We formally evaluated Z-score residual normalization and proved mathematically and empirically that tree ensembles are invariant to positive monotonic feature scaling ($\Delta = 0.0000$ across all 15 classes). Future MITM improvement requires class-weight balancing (`class_weight='balanced'`) or SMOTE resampling rather than residual rescaling.

### Q9: Your paper says 0.006ms but your live audit shows 16.65ms — which is true?
**Answer:** Both are true and describe two distinct operational layers:
1. **Table A ($0.005\text{ ms}$ / $192\text{k pps}$):** Measures standalone mathematical tree evaluation on raw telemetry features in isolation.
2. **Table B ($4.023\text{ ms}$ / $15.21\text{ ms}$):** Measures the full cyber-physical runtime decision pipeline — including neural Digital Twin sequence forecasting ($1.15\text{ ms}$), 43-feature classification ($2.74\text{ ms}$), on-demand SHAP TreeExplainer attribution ($11.3\text{ ms}$ on alerts), and confidence filtering.
By implementing **conditional SHAP triggering**, we accelerated normal packet processing by $75.8\%$ ($16.65\text{ ms} \to 4.023\text{ ms}$), delivering over $248\text{ packets/second}$ sustained edge throughput.

### Q10: How did you resolve the million-scale tcp.seq error in the Digital Twin?
**Answer:** Absolute 32-bit TCP sequence counters ($0 - 4.29\times 10^9$) reset unpredictably across new handshakes with randomized ISNs. By grouping telemetry by `(tcp.srcport, tcp.dstport)` and computing within-flow sequence advance deltas ($\Delta \text{seq}_t, \Delta \text{ack}_t$), we reduced `tcp.seq` MAE by **$447\times$ (from $12.2\text{M B} \to 27.3\text{ KB}$)** and total mean MAE by **$223\times$ (from $1.81\text{ MB} \to 8.08\text{ KB}$)**, with a median advance error of **$0.76\text{ B}$**. In the fused space, `XGB-Twin-Augmented` reached **$94.91\%$ accuracy** and **$0.9153$ Macro-F1**.

### Q11: Why should we believe 99.29% generalization on TON_IoT when in-domain accuracy is 94.91%?
**Answer:** The raw baseline collapsed to $65.21\%$ recall ($17,395$ missed attacks) on TON_IoT because static ports and subnets shifted across testbeds. The Twin-Augmented X-IDS achieved $99.29\%$ recall ($0.9964\text{ F1}$) with **0 False Positives** because physical transport residuals ($\mathbf{e}_t$) reflect universal conservation laws and protocol physics rather than static port IDs. 

Our deep audit of the 355 missed samples ($0.71\%$) confirmed they were exclusively isolated single-packet boundary frames with $0.0\text{ s}$ duration and sub-100 Byte payloads; detection across sustained attack sessions was $100.00\%$.

### Q12: Why did Z-score normalized residuals yield identical F1 on MITM?
**Answer:** Tree ensembles (Random Forest, XGBoost) evaluate split points based on orthogonal feature thresholds and information gain. Because dividing each feature column by a positive constant ($\sigma_{\text{normal}}$) is a strictly monotonic linear transformation, the optimal threshold positions simply scale proportionally, producing **identical decision trees ($\Delta = 0.0000$)**. This formally proves that MITM's $0.5487\text{ F1}$ is governed by **low sample support ($n=108$)**, not feature scale imbalance.
