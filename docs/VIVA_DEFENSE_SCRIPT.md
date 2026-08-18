# Guide Presentation & Viva Defense Script
## Twin-Guided Explainable Intrusion Detection System (X-IDS)

---

## 🎯 Part 1: Slide-by-Slide Guide Presentation Script

### Slide 1: Title & Introduction
- **Speaker:** "Good morning respected guide and examiners. Today, I am presenting our project: *Twin-Guided Explainable Intrusion Detection System: A Resource-Aware Trade-off Analysis for Industrial IoT*."
- **Core Message:** "We address the critical challenge of securing IIoT edge networks under severe resource constraints, sub-millisecond latency requirements, and the persistent problem of SOC alert fatigue."

### Slide 2: Problem Statement & Motivation
- **Speaker:** "Traditional enterprise IDS cannot be deployed on industrial edge gateways. Heavy deep learning models introduce multi-millisecond inference delays, exhaust memory, behave as opaque black boxes, and generate thousands of false-positive alarms that overwhelm security operators."

### Slide 3: Proposed Architecture & Novelty
- **Speaker:** "Our system introduces three major innovations:
  1. **A Digital Twin Forecaster** trained exclusively on healthy baseline dynamics to predict expected telemetry.
  2. **A Deviation Engine** that computes multi-dimensional residual error vectors $\|y_t - \hat{y}_t\|$.
  3. **An Operational Confidence Filter powered by SHAP XAI** that suppresses low-confidence false alarms by checking domain attack signatures before alerts reach the operator."

### Slide 4: Experimental Methodology & Datasets
- **Speaker:** "We trained and evaluated our system on the **Edge-IIoTset** benchmark (~70,000 stratified samples across 15 attack classes) and tested zero-shot cross-dataset generalization on the **TON_IoT** dataset."

### Slide 5: Key Results & Edge Trade-Off Analysis
- **Speaker:** "Our experimental benchmark across 4 edge configurations demonstrates clear trade-offs:
  - Our Quantized Twin configuration achieves **91.64% accuracy** and **0.8786 Macro-F1** with only **0.176 ms/sample** latency and a **7.29 MB** footprint.
  - Our Ultra-Light Fast-Edge configuration achieves **91.81% accuracy** with an ultra-fast **0.005 ms/sample** latency (over 204,000 samples/sec throughput) and a tiny **105.9 KB** model footprint."

### Slide 6: Operational Explainability & Alert Filtering
- **Speaker:** "By integrating SHAP TreeExplainer with our Operational Confidence Filter, we achieve transparent feature attributions for every alert and suppress 35–45% of ambiguous false alarms, solving the alert fatigue bottleneck."

---

## 🎬 Part 2: Step-by-Step Live Demo Script

When demonstrating the system live using `run_project.py` and `http://localhost:5173`:

1. **Step 1 — Show the Real-Time Dual-Trace Monitor:**
   - *"Here on the Live Monitor tab, you can see the telemetry stream replaying in real time. The blue line represents incoming sensor readings, while the amber dashed line represents the Digital Twin's healthy forecast."*
2. **Step 2 — Point Out the Deviation Spike:**
   - *"When an attack occurs (e.g. DDoS or SQL injection), the actual telemetry diverges from the twin's forecast. Look at the lower chart: the residual deviation magnitude spikes, triggering the IDS classifier."*
3. **Step 3 — Inspect the Alert Feed & Confidence Filter:**
   - *"In the right-hand threat feed, each alert is categorized. Notice the badge: high-confidence attacks matching domain signatures are marked as `PASS`, whereas ambiguous borderline noise is marked as `SUPPRESS`."*
4. **Step 4 — Show the SHAP Explainability Studio:**
   - *"Clicking on any alert opens the SHAP Explainability Studio. The horizontal bar chart instantly reveals the top 5 sensor features responsible for the detection (red bars increasing threat probability, green bars decreasing it)."*
5. **Step 5 — Showcase the Edge-Resource Benchmarks:**
   - *"Finally, on the Edge Benchmarks tab, we present our master trade-off curves comparing latency, throughput, model storage, and accuracy across all 4 edge deployment configurations."*

---

## 💡 Part 3: Mock Defense Examiner Q&A Prep

### Q1: Why did you train the Digital Twin only on Normal traffic?
**Answer:** In industrial cyber-physical systems, normal operating state transitions are predictable and follow physical laws, whereas zero-day attacks are unpredictable and constantly evolving. By training the Digital Twin exclusively on healthy baseline dynamics, the twin acts as an uncorrupted reference model. Any significant physical or protocol deviation immediately indicates an operational anomaly.

### Q2: Why did pure deviation features achieve lower accuracy than raw or twin-augmented features?
**Answer:** Pure deviation features compute absolute residuals $\|y_t - \hat{y}_t\|$, which isolate anomaly magnitude but discard base telemetry magnitudes and directional signs. When we augment raw features with deviation residuals (Twin-Augmented space), the model retains the absolute signal levels while gaining explicit anomaly guidance, achieving 94.75% accuracy and robust multi-class separation.

### Q3: How does your Confidence Filter differ from simple thresholding?
**Answer:** Traditional systems only apply a probability threshold (e.g. $> 0.5$). Our Confidence Filter performs a two-stage evaluation: first, checking probability confidence ($\ge 0.65$), and second, cross-verifying whether the top positive SHAP-attributed features match the known domain attack signature (e.g. UDP jitter and packet lengths for UDP DDoS, or HTTP payload metrics for SQL injection). If the model is confident for the wrong reasons (spurious correlation), the alert is suppressed.

### Q4: How is this system suitable for edge deployment?
**Answer:** Our benchmarking suite proved that our Quantized Twin architecture achieves a latency of **0.176 ms/sample** with **7.29 MB** footprint, and our Fast-Edge XGBoost configuration operates at **0.005 ms/sample** with **105.9 KB** footprint. Both easily satisfy sub-millisecond industrial control loop requirements on 1–2 core edge hardware with $< 512$ MB RAM.

### Q5: How did you validate that your model doesn't overfit to a single dataset?
**Answer:** We conducted zero-shot cross-dataset generalization testing by taking our model trained on Edge-IIoTset and evaluating it directly on 50,000 samples of the unseen TON_IoT dataset (`train_test_network.csv`). The model achieved a Transfer F1-Score of **0.7878** and a Transfer Precision of **100.0%**, demonstrating zero false-positive transfers across disparate testbeds.
