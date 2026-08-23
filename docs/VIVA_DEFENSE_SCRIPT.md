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
  1. **Scope-Restricted Digital Twin:** A sequence regressor trained exclusively on normal physical telemetry using $\log(1+x)$ normalization and L2 regularization to model healthy baseline dynamics.
  2. **Targeted Deviation Residual Engine:** Computes continuous physical discrepancy vectors ($|y_t - \hat{y}_t|$).
  3. **Multi-Class IDS Suite:** Operates on an augmented 43-feature space ($\mathbb{R}^{43}$), matching raw baseline accuracy ($94.85\%$) while elevating pure continuous residual detection to $72.41\%$.
  4. **Operational Confidence Filter & SHAP XAI:** Filters low-confidence, signature-divergent alarms to suppress $30.0\%$ of alert noise."

### Slide 4: Experimental Methodology & Datasets
- **Speaker:** "We trained and evaluated our system on the **Edge-IIoTset** benchmark (69,993 stratified samples across 15 attack classes) and performed zero-shot transferability evaluation on 50,000 samples of the unseen **TON_IoT** testbed."

### Slide 5: Key Results & Master Edge Trade-Off Analysis
- **Speaker:** "Our empirical benchmark across 4 edge configurations demonstrates clear, actionable trade-offs:
  - **Config 1 (Full Twin + Heavy RF 150):** Achieves **94.08% accuracy** and **0.9066 Macro-F1** ($0.457 \pm 0.011\text{ ms/sample}$, 14.58 MB).
  - **Config 2 (Quantized Twin + Standard RF 100):** Achieves **93.44% accuracy** and **0.9006 Macro-F1** with **$0.209 \pm 0.016\text{ ms/sample}$** latency (over 4,700 samples/sec) and a **5.63 MB** footprint.
  - **Config 4 (Ultra-Light Fast-Edge XGBoost 25):** Achieves **91.81% accuracy** with **$0.006 \pm 0.003\text{ ms/sample}$** latency (over 154,000 samples/sec) and a tiny **105.9 KB** footprint."

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
   - *"Finally, on the IDS Comparison tab, we show our granular per-attack breakdown evaluated across 13,999 test samples. Twin-Augmented-v2 maintains exact or statistical parity on 13 of 15 attack types (including perfect 1.0000 F1 on DDoS floods, 0.9224 F1 on Uploading, and 0.9094 on XSS) while providing the physical deviation residuals required for operator auditing."*

---

## 💡 Part 3: Fortified Mock Defense Examiner Q&A Prep

### Q1: Why did you train the Digital Twin only on Normal traffic?
**Answer:** In industrial cyber-physical systems, normal operating dynamics are predictable and governed by physical laws, whereas zero-day attacks are unpredictable and constantly evolving. By training the Digital Twin exclusively on healthy baseline dynamics, the twin acts as an uncorrupted reference model. Any significant physical or protocol deviation immediately indicates an operational anomaly.

### Q2: Your twin forecast used to hit its ceiling constantly — did you actually fix that or just cap it?
**Answer:** We conducted four rigorous architectural experiments to resolve the underlying root cause:
1. We discovered that on sudden attack bursts, linear regression layers extrapolated on huge sequence jumps, causing $69.6\%$ of predictions on attack traffic to blow past bounds into millions.
2. We tested **log-space target transformation ($\log(1+x)$)**, which compresses skewed physical scales and reduced validation MAE by **42.5%** (from $244.98\text{ B}$ down to $140.70\text{ B}$).
3. Adding **L2 weight regularization ($\alpha=0.05$)** and log-space bounding structurally constrained predictions to strictly stay within $[0.00\text{ B}, 9.65\text{ B}]$ in normal traffic and $[0.00\text{ B}, 41.49\text{ B}]$ during attack floods.
4. As a result, **clamp invocation frequency dropped from 69.6% to 0.0%**. The safety clamp is retained purely as a zero-cost backstop, but the neural model itself now predicts within valid physical bounds.

### Q3: Your Twin-Augmented model achieves 94.85% accuracy, while raw XGBoost achieves 95.00% — why does the Digital Twin matter?
**Answer:** While twin-augmentation trails the raw baseline by only 0.15 percentage points in aggregate accuracy, it provides three critical operational capabilities:
1. **13 of 15 Class Parity:** Exact or statistical parity across 13 threat profiles, including perfect $1.0000\text{ F1}$ on volumetric floods (`DDoS_TCP`, `DDoS_UDP`, `DDoS_ICMP`) and $0.9977$ on normal traffic, with superior F1 on XSS ($0.9094$) and Uploading ($0.9224$).
2. **Restoration of Decision Tree Stability:** Scope-restricted residuals eliminated tree dilution on application payloads, restoring `Uploading` F1 to **0.9224** and `SQL_injection` to **0.8940**.
3. **Causal Physical Grounding vs. Black-Box Correlation:** A raw black-box learns statistical correlations on ephemeral ports that cannot be physically audited. The Digital Twin provides **physically grounded residual vectors ($\mathbf{e}_t = |y_t - \hat{y}_t|$) and SHAP attributions**, enabling our Operational Confidence Filter to suppress **30.0% of false alarms** and preventing unexplainable shutdowns of physical industrial actuators.

### Q4: Why don't we see raw SQL query strings in the SHAP explanation for SQL Injection?
**Answer:** Deep packet inspection (DPI) with full string tokenization requires heavy NLP parsing pipelines that consume $>10\text{ ms}$ per packet, which is infeasible for sub-millisecond edge gateways. X-IDS operates on network transport and flow metrics, where SQL injection payloads create measurable anomalies in packet lengths (`tcp.len`, `http.content_length`) and HTTP response codes (`http.response`). SHAP attributions confirm that packet size distribution deviations (`dev_tcp.len`) serve as effective physical discriminators, achieving $0.8940\text{ F1}$ without heavy string-parsing latency.

### Q5: Why deploy Config 2 (Quantized Twin) over Config 4 (Lightweight XGBoost) in safety-critical edge environments?
**Answer:** While Config 4 provides ultra-low latency ($0.006\text{ ms}$), it operates as an opaque black box on raw features. In safety-critical operational technology (OT) systems where automated actuators or physical shutdowns depend on alert validity, Config 2 achieves higher accuracy ($93.44\%$ vs. $91.81\%$) and provides **causally interpretable deviation vectors ($|y_t - \hat{y}_t|$)**, allowing human operators to verify whether physical sensor streams actually deviated from baseline physics before initiating expensive plant shutdowns.
