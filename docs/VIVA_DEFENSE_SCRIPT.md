# Guide Presentation & Viva Defense Script: Twin-Guided Explainable IDS (X-IDS)
## A Resource-Aware Trade-Off Analysis for Industrial IoT

---

## 🎯 Part 1: Slide-by-Slide Presentation Script

### Slide 1: Title & Introduction
- **Speaker:** "Good morning respected guide and members of the evaluation committee. Today, I am presenting our project: *Twin-Guided Explainable Intrusion Detection System (X-IDS): A Resource-Aware Trade-Off Analysis for Industrial IoT*."
- **Core Message:** "We solve the dual challenges of securing resource-constrained IIoT edge networks with sub-millisecond latency and eliminating operator alert fatigue through causal physical explainability."

### Slide 2: Problem Statement & Motivation
- **Speaker:** "Traditional enterprise machine learning IDSs act as uninterpretable black boxes. When deployed on edge nodes, heavy models cause multi-millisecond processing delays, exhaust RAM, and trigger high volumes of ambiguous false alarms that overwhelm SOC operators. Furthermore, black-box statistical models correlate non-causal protocol artifacts rather than verifying whether physical process variables actually deviated from baseline physics."

### Slide 3: Proposed Architecture & Novelty
- **Speaker:** "Our system introduces three major architectural innovations:
  1. **Scope-Restricted Digital Twin Forecaster:** Trained exclusively on normal continuous physical dynamics ($K=9$ signals) to predict expected baseline telemetry without noise from discrete flags.
  2. **Targeted Deviation Engine:** Computes multi-dimensional physical residual vectors $\mathbf{e}_t = |\mathbf{x}_t^{\text{cont}} - \hat{\mathbf{x}}_t^{\text{cont}}|$ to form a targeted 43-feature augmented space ($\mathbf{z}_t$).
  3. **Operational Confidence Filter + SHAP XAI:** Cross-verifies model confidence ($\gamma \ge 0.65$) against local SHAP feature attributions and domain attack signatures, suppressing 30.0% of ambiguous alerts before they reach the operator."

### Slide 4: Experimental Methodology & Datasets
- **Speaker:** "We trained and evaluated our system on the **Edge-IIoTset** benchmark (69,993 stratified samples across 15 attack classes) and performed zero-shot transferability evaluation on 50,000 samples of the unseen **TON_IoT** testbed."

### Slide 5: Key Results & Master Edge Trade-Off Analysis
- **Speaker:** "Our empirical benchmark across 4 edge configurations demonstrates clear, actionable trade-offs:
  - **Config 1 (Full Twin + Heavy RF 150):** Achieves **93.88% accuracy** and **0.9074 Macro-F1** (0.454 ms/sample, 15.3 MB).
  - **Config 2 (Quantized Twin + Standard RF 100):** Achieves **92.76% accuracy** and **0.8938 Macro-F1** with **0.199 ms/sample** latency (over 5,000 samples/sec) and a **5.99 MB** footprint.
  - **Config 4 (Ultra-Light Fast-Edge XGBoost 25):** Achieves **91.81% accuracy** with **0.054 ms/sample** latency (over 18,600 samples/sec) and a tiny **105.9 KB** footprint."

### Slide 6: Operational Explainability & Alert Filtering
- **Speaker:** "By integrating SHAP TreeExplainer with our Operational Confidence Filter, we achieve transparent feature attributions for every alert and suppress **30.0%** (empirically measured range: 28.6%–31.4%) of ambiguous false alarms, solving the alert fatigue bottleneck."

---

## 🎬 Part 2: Step-by-Step Live Demo Script

When demonstrating the system live using `run_project.py` and `http://localhost:5173`:

1. **Step 1 — Show the Real-Time Dual-Trace Monitor:**
   - *"Here on the Live Monitor tab, you can see continuous physical telemetry (active signal: `tcp.len` payload bytes) replaying in real time. The blue line represents incoming ingested telemetry ($y_t$), while the amber dashed line represents the Digital Twin's healthy baseline forecast ($\hat{y}_t$)."*
2. **Step 2 — Point Out the Deviation Spike:**
   - *"When an attack occurs (e.g. DDoS flood or SQL injection), actual telemetry deviates from the twin's forecast. Look at the lower chart: the residual deviation magnitude $|y_t - \hat{y}_t|$ spikes, feeding explicit physical anomaly guidance to the IDS classifier."*
3. **Step 3 — Inspect the Alert Feed & Confidence Filter:**
   - *"In the right-hand threat feed, each alert is categorized. Notice the badge: high-confidence attacks matching domain signatures are marked as `PASS`, whereas ambiguous borderline noise is marked as `SUPPRESS`."*
4. **Step 4 — Show the SHAP Explainability Studio:**
   - *"Clicking on any alert opens the SHAP Explainability Studio. The horizontal bar chart instantly reveals the top 5 features responsible for the detection (red bars increasing threat risk, blue bars decreasing it)."*
5. **Step 5 — Showcase the Edge-Resource Benchmarks:**
   - *"Finally, on the Edge Benchmarks tab, we present our master trade-off curves comparing latency, throughput, model storage, and accuracy across all 4 edge deployment configurations."*

---

## 💡 Part 3: Fortified Mock Defense Examiner Q&A Prep

### Q1: Why did you train the Digital Twin only on Normal traffic?
**Answer:** In industrial cyber-physical systems, normal operating dynamics are predictable and governed by physical laws, whereas zero-day attacks are unpredictable and constantly evolving. By training the Digital Twin exclusively on healthy baseline dynamics, the twin acts as an uncorrupted reference model. Any significant physical or protocol deviation immediately indicates an operational anomaly.

### Q2: Why did you restrict the Digital Twin to continuous physical features rather than all 34 features?
**Answer:** Our Phase A diagnostic audit revealed that 25 of the 34 features are discrete or categorical (binary TCP connection flags, MQTT header flags, and ephemeral port numbers). Sequence regressors (LSTMs/MLPs) cannot model non-smooth discrete states, resulting in 60% higher forecasting error on categorical flags. Restricting the Twin to continuous physical telemetry reduced validation MSE by **30%** (from 0.7246 to 0.5080) and eliminated noisy residual channels, boosting Random Forest accuracy by **+3.2%** (from 90.60% to 93.80%).

### Q3: Your Twin-Augmented model achieves 94.81% accuracy, while raw XGBoost achieves 95.00% — why should an industrial facility value the Digital Twin?
**Answer:** A pure raw-feature black box learns statistical correlations on port numbers and IP flags, but cannot verify whether physical sensor streams actually deviated from baseline physics. In safety-critical OT environments (e.g. power generation, chemical refining), automated actuators or emergency shutoffs cannot rely on an uninterpretable black box. The Digital Twin provides **physically grounded residual vectors ($\mathbf{e}_t = |y_t - \hat{y}_t|$) and SHAP causal attributions**, giving operators auditable physical proof of sensor/flow anomalies before initiating expensive plant shutdowns.

### Q4: Config 4 (Fast-Edge XGBoost) dominates on latency (0.054 ms) and model footprint (105.9 KB) — why not just deploy Config 4 everywhere?
**Answer:** Config 4 is indeed the optimal choice for high-throughput, resource-starved network interfaces and sensor microcontrollers. However, for critical edge supervisory controllers, **Config 2 (Quantized Twin + RF)** provides higher overall detection accuracy (**92.76% vs. 91.81%**) while executing well within the sub-millisecond control loop requirement (**0.199 ms/sample**, over 5,000 samples/sec). Crucially, Config 2 provides the physical residual confirmation that Config 4 cannot supply.

### Q5: How does your Operational Confidence Filter differ from simple thresholding?
**Answer:** Traditional systems apply a simple probability cutoff (e.g. $> 0.5$). Our Operational Confidence Filter performs a two-stage evaluation: first, checking probability confidence ($\gamma \ge 0.65$), and second, cross-verifying whether the top positive SHAP-attributed features match the known domain attack signature (e.g. UDP jitter and packet lengths for UDP DDoS, or HTTP payload metrics for SQL injection). If a model is confident for the wrong reasons (spurious correlation), the alert is suppressed. This reliably eliminates **30.0% of ambiguous false alarms** (range: 28.6%–31.4%).

### Q6: How did you validate that your model does not overfit to a single dataset?
**Answer:** We conducted zero-shot cross-dataset generalization testing by taking our model trained on Edge-IIoTset and evaluating it directly on 50,000 samples of the unseen TON_IoT dataset (`train_test_network.csv`). The model achieved a **100.0% Transfer Precision (0 False Positives)** with **65.21% Recall**. This confirms that the model adopts a conservative, zero-false-alarm posture when transferred to unfamiliar industrial network topologies.
