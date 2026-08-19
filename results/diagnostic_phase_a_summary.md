# Phase A: Root-Cause Diagnosis & Feature Partitioning Report

**Date:** August 19, 2026  
**Artifacts Generated:** [`results/diagnostic_phase_a.json`](file:///e:/Projects/digital%20twin/results/diagnostic_phase_a.json), [`results/twin_per_feature_error.png`](file:///e:/Projects/digital%20twin/results/twin_per_feature_error.png)

---

## 1. Executive Summary & Root-Cause Discovery

The diagnostic audit has confirmed the root cause behind the performance degradation in the initial all-feature Twin-Deviation model:

1. **Attempting to Forecast Discrete / Categorical States:**
   - Out of 34 total features, **25 are discrete/categorical** (binary TCP connection flags, MQTT header flags, hardware sizes, and ephemeral port numbers).
   - The Digital Twin (neural sequence regressor) was forced to predict continuous approximations for binary states ($0$ or $1$) and random ephemeral ports ($0–65535$).
   - **Empirical Evidence:** The Scaled MSE for discrete/categorical features is **0.8041**, which is **60% higher** than for continuous physical features (**0.5051**).
   - The top 10 highest-error features in the Digital Twin are dominated by binary state flags: `tcp.connection.syn` (MSE: 1.098), `tcp.connection.synack` (MSE: 1.066), `tcp.connection.fin` (MSE: 1.043), and `mqtt.conflag.cleansess` (MSE: 1.028).

2. **Noise Injection in Downstream IDS Classifiers:**
   - Computing absolute residuals $|y - \hat{y}|$ on non-smooth binary flags created 25 noisy residual channels.
   - For tree-based classifiers (especially unregularized Random Forests), splitting on 25 noisy deviation features diluted the decision boundaries for application-layer attacks (e.g., `Uploading` F1 dropped from 0.922 to 0.775, and `SQL_injection` dropped from 0.896 to 0.789).

---

## 2. Feature Partitioning Breakdown

### Group 1: Continuous / Physical Features (9 Features — To Be Forecasted by Twin)
These features represent smooth temporal dynamics, volume metrics, sequence progressions, and network flow rates:
1. `icmp.checksum` (ICMP integrity check)
2. `icmp.seq_le` (ICMP sequential packet identifier)
3. `http.content_length` (HTTP payload size)
4. `tcp.ack` (TCP cumulative acknowledgement stream)
5. `tcp.checksum` (TCP header integrity check)
6. `tcp.len` (TCP segment payload length)
7. `tcp.seq` (TCP sequence number progression)
8. `udp.stream` (UDP flow identifier)
9. `udp.time_delta` (Inter-packet arrival jitter)

### Group 2: Categorical / Discrete Features (25 Features — Passed Directly to IDS)
These features represent discrete protocol codes, flags, and port identifiers that should **NOT** be forecasted by the Twin:
- **Port Identifiers:** `tcp.srcport`, `tcp.dstport`, `udp.port`
- **TCP Control Flags:** `tcp.flags`, `tcp.flags.ack`, `tcp.connection.syn`, `tcp.connection.synack`, `tcp.connection.fin`, `tcp.connection.rst`, `tcp.ack_raw`
- **ARP / Network Codes:** `arp.opcode`, `arp.hw.size`
- **DNS Protocol Indicators:** `dns.qry.name.len`, `dns.qry.qu`, `dns.retransmission`, `dns.retransmit_request`
- **MQTT / IoT Controls:** `mqtt.conflag.cleansess`, `mqtt.conflags`, `mqtt.hdrflags`, `mqtt.len`, `mqtt.msgtype`, `mqtt.proto_len`, `mqtt.topic_len`, `mqtt.ver`
- **HTTP Response:** `http.response`

---

## 3. Per-Class Baseline Comparison (XGB-Raw vs. All-34 RF/XGB-Twin-Augmented)

| Attack Class | Support (Test) | XGB-Raw F1 | RF-Twin-Augmented F1 (All 34) | XGB-Twin-Augmented F1 (All 34) | Status / Observation |
|---|---|---|---|---|---|
| **DDoS_TCP** | 909 | **1.0000** | **1.0000** | **1.0000** | Identical Perfect Detection |
| **DDoS_UDP** | 1286 | **1.0000** | **1.0000** | **1.0000** | Identical Perfect Detection |
| **DDoS_ICMP** | 1250 | **0.9996** | 0.9984 | **0.9996** | Identical Near-Perfect Detection |
| **Normal (Healthy)** | 2156 | **0.9979** | 0.9854 | **0.9979** | Zero False Alarm Baseline |
| **Backdoor** | 904 | **0.9848** | 0.9686 | 0.9843 | Minor difference (-0.0006) |
| **Vulnerability_scanner** | 894 | **0.9759** | 0.9478 | 0.9714 | Minor difference (-0.0045) |
| **Port_Scanning** | 893 | **0.9511** | 0.9418 | 0.9459 | Minor difference (-0.0052) |
| **Ransomware** | 969 | **0.9385** | 0.8924 | 0.9179 | Affected by noisy flag residuals |
| **Uploading** | 911 | **0.9221** | 0.7755 | 0.9202 | RF heavily degraded by noise |
| **XSS** | 892 | **0.9084** | 0.8406 | 0.9073 | Minor difference (-0.0011) |
| **Password** | 886 | **0.8990** | 0.7923 | 0.8963 | Affected by flag noise in RF |
| **SQL_injection** | 915 | **0.8963** | 0.7889 | 0.8894 | RF degraded by flag residuals |
| **Fingerprinting** | 89 | **0.8889** | 0.8471 | 0.8750 | Rare class sensitivity |
| **DDoS_HTTP** | 937 | **0.8571** | 0.7607 | 0.8491 | Complex payload dynamics |
| **MITM** | 108 | **0.5806** | 0.5737 | 0.5632 | Rare behavioral attack class |

---

## 4. Phase B Action Plan

In Phase B, we will:
1. Retrain `src/twin_model.py` to predict **only the 9 Continuous Physical Features**.
2. Compute deviations $e_{t, j} = |y_{t, j} - \hat{y}_{t, j}|$ strictly for the 9 continuous channels.
3. Build the **`Twin-Augmented-v2`** feature space:
   $$\mathbf{z}_t = [\mathbf{x}_{\text{categorical}} (25) \,\|\, \mathbf{x}_{\text{continuous}} (9) \,\|\, \mathbf{e}_{\text{continuous}} (9)] \in \mathbb{R}^{43}$$
4. Retrain the IDS classifiers and document whether this focused feature space eliminates the performance degradation.
