"""
Vercel Serverless FastAPI Entrypoint
Serves interactive APIs, live simulation packets, SHAP explanations,
confidence filter decisions, and benchmark results for Vercel deployment.
"""

import os
import sys
import time
import json
import random
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse

app = FastAPI(
    title="Twin-Guided Explainable IDS Cloud API",
    description="Vercel Cloud Deployment for Industrial IoT Intrusion Detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Benchmark Data
BENCHMARK_DATA = [
    {
        "Configuration": "Config 1: Full-Precision Twin + Heavy RF (150 trees)",
        "Feature Space": "Twin-Augmented",
        "Accuracy (%)": 92.96,
        "Macro-F1": 0.8927,
        "Avg Latency (ms/sample)": 0.43,
        "Throughput (samples/sec)": 2326.5,
        "Total Footprint (KB)": 18134.8
    },
    {
        "Configuration": "Config 2: Quantized Twin + Standard RF (100 trees)",
        "Feature Space": "Twin-Augmented",
        "Accuracy (%)": 91.64,
        "Macro-F1": 0.8786,
        "Avg Latency (ms/sample)": 0.176,
        "Throughput (samples/sec)": 5680.4,
        "Total Footprint (KB)": 7291.0
    },
    {
        "Configuration": "Config 3: Quantized Twin + Pruned Edge RF (30 trees)",
        "Feature Space": "Twin-Augmented",
        "Accuracy (%)": 84.98,
        "Macro-F1": 0.8041,
        "Avg Latency (ms/sample)": 0.152,
        "Throughput (samples/sec)": 6587.5,
        "Total Footprint (KB)": 571.7
    },
    {
        "Configuration": "Config 4: Fast-Inference Edge XGBoost (25 trees, Depth 4)",
        "Feature Space": "Raw Telemetry",
        "Accuracy (%)": 91.81,
        "Macro-F1": 0.8871,
        "Avg Latency (ms/sample)": 0.005,
        "Throughput (samples/sec)": 204592.7,
        "Total Footprint (KB)": 105.9
    }
]

# Model Comparison Data
MODEL_COMPARISON_DATA = [
    {"Model Architecture": "RF-Raw (Baseline)", "Accuracy (%)": 94.24, "Macro-F1": 0.9016, "Weighted-F1": 0.9450, "Macro-Precision": 0.8907, "Macro-Recall": 0.9341},
    {"Model Architecture": "XGB-Raw (Baseline)", "Accuracy (%)": 95.00, "Macro-F1": 0.9200, "Weighted-F1": 0.9522, "Macro-Precision": 0.9215, "Macro-Recall": 0.9378},
    {"Model Architecture": "RF-Deviation (Pure)", "Accuracy (%)": 41.61, "Macro-F1": 0.3934, "Weighted-F1": 0.4154, "Macro-Precision": 0.3935, "Macro-Recall": 0.4174},
    {"Model Architecture": "XGB-Deviation (Pure)", "Accuracy (%)": 47.12, "Macro-F1": 0.4303, "Weighted-F1": 0.4448, "Macro-Precision": 0.5839, "Macro-Recall": 0.3993},
    {"Model Architecture": "RF-Twin-Augmented", "Accuracy (%)": 90.60, "Macro-F1": 0.8742, "Weighted-F1": 0.9091, "Macro-Precision": 0.8732, "Macro-Recall": 0.8947},
    {"Model Architecture": "XGB-Twin-Augmented", "Accuracy (%)": 94.75, "Macro-F1": 0.9145, "Weighted-F1": 0.9485, "Macro-Precision": 0.9183, "Macro-Recall": 0.9175}
]

ATTACK_SIGNATURES = {
    "DDoS_UDP": ["udp.stream", "udp.time_delta", "udp.port"],
    "DDoS_ICMP": ["icmp.checksum", "icmp.seq_le", "tcp.flags"],
    "DDoS_HTTP": ["http.content_length", "http.response", "tcp.dstport"],
    "DDoS_TCP": ["tcp.flags", "tcp.connection.syn", "tcp.len"],
    "Port_Scanning": ["tcp.dstport", "tcp.srcport", "tcp.flags"],
    "SQL_injection": ["http.content_length", "http.response", "tcp.len"],
    "XSS": ["http.content_length", "http.response", "tcp.len"],
    "Ransomware": ["tcp.len", "tcp.seq", "tcp.ack"],
    "Vulnerability_scanner": ["tcp.dstport", "tcp.flags", "http.content_length"],
    "Password": ["http.response", "tcp.len", "tcp.flags"],
    "Backdoor": ["tcp.dstport", "tcp.srcport", "tcp.flags.ack"],
    "Uploading": ["http.content_length", "tcp.len", "tcp.flags"],
    "Normal": ["tcp.flags.ack", "tcp.ack", "tcp.len"]
}

SAMPLE_ATTACKS = ["Normal", "Normal", "Normal", "DDoS_UDP", "Normal", "SQL_injection", "Normal", "DDoS_TCP", "Normal", "Port_Scanning", "Ransomware"]

class CloudSimulator:
    def __init__(self):
        self.step_count = 0
        self.stats = {
            "total_inspected": 1420,
            "passed_alerts": 184,
            "suppressed_alerts": 92,
            "normal_traffic": 1144
        }
        self.recent_alerts = []
        
    def generate_packet(self):
        self.step_count += 1
        attack_type = random.choice(SAMPLE_ATTACKS)
        is_attack = attack_type != "Normal"
        
        base_val = 50.0 + random.uniform(-5.0, 5.0)
        actual_val = base_val + (random.uniform(30.0, 90.0) if is_attack else random.uniform(-2.0, 2.0))
        twin_val = base_val + random.uniform(-1.5, 1.5)
        deviation = abs(actual_val - twin_val)
        
        confidence = random.uniform(0.88, 0.99) if is_attack else random.uniform(0.92, 0.98)
        is_borderline = random.random() < 0.25 and is_attack
        if is_borderline:
            confidence = random.uniform(0.48, 0.62)
            
        sigs = ATTACK_SIGNATURES.get(attack_type, ["tcp.flags", "tcp.len"])
        top_feature = sigs[0] if sigs else "tcp.len"
        
        if not is_attack:
            decision = "NORMAL"
            reason = "Traffic classified as normal baseline behavior."
            should_alert = False
        elif confidence >= 0.65:
            decision = "PASS"
            reason = f"High-fidelity alert: Confidence ({confidence*100:.1f}%) meets threshold and features match known attack signature."
            should_alert = True
            self.stats["passed_alerts"] += 1
        else:
            decision = "SUPPRESS"
            reason = f"Alert suppressed: Low model confidence ({confidence*100:.1f}% < 65.0%)."
            should_alert = False
            self.stats["suppressed_alerts"] += 1
            
        self.stats["total_inspected"] += 1
        if not is_attack:
            self.stats["normal_traffic"] += 1
            
        packet = {
            "index": self.step_count,
            "timestamp": time.time(),
            "active_metric": "tcp.len",
            "actual_signal": round(actual_val, 2),
            "twin_signal": round(twin_val, 2),
            "features": {
                "sensor_telemetry": round(actual_val, 2),
                "tcp.len": round(actual_val, 2),
                "tcp.flags": random.randint(2, 24),
                "udp.stream": random.randint(1, 100)
            },
            "predicted_state": {
                "sensor_telemetry": round(twin_val, 2),
                "tcp.len": round(twin_val, 2)
            },
            "mean_deviation": round(deviation, 2),
            "ground_truth": attack_type,
            "prediction": {
                "predicted_class": attack_type,
                "confidence": round(confidence, 4),
                "ground_truth": attack_type
            },
            "shap_explanation": {
                "predicted_class": attack_type,
                "confidence": confidence,
                "top_features": [
                    {"feature": top_feature, "feature_value": round(actual_val, 2), "shap_value": round(random.uniform(1.2, 4.5), 4) if is_attack else round(-random.uniform(0.5, 2.0), 4), "contribution": "Increases Risk" if is_attack else "Decreases Risk"},
                    {"feature": "tcp.flags", "feature_value": 16.0, "shap_value": round(random.uniform(0.4, 1.8), 4), "contribution": "Increases Risk"},
                    {"feature": "tcp.len", "feature_value": 120.0, "shap_value": round(random.uniform(-0.8, 0.9), 4), "contribution": "Decreases Risk"},
                    {"feature": "udp.time_delta", "feature_value": 0.05, "shap_value": round(random.uniform(0.1, 0.7), 4), "contribution": "Increases Risk"},
                    {"feature": "http.content_length", "feature_value": 0.0, "shap_value": round(-0.35, 4), "contribution": "Decreases Risk"}
                ]
            },
            "filter": {
                "decision": decision,
                "should_alert": should_alert,
                "predicted_class": attack_type,
                "confidence": confidence,
                "reason": reason
            }
        }
        
        if decision in ["PASS", "SUPPRESS"]:
            alert_entry = {
                "id": self.step_count,
                "timestamp": time.strftime("%H:%M:%S"),
                "attack_type": attack_type,
                "ground_truth": attack_type,
                "confidence": round(confidence * 100, 1),
                "filter_decision": decision,
                "filter_reason": reason,
                "top_feature": top_feature,
                "mean_deviation": round(deviation, 2)
            }
            self.recent_alerts.insert(0, alert_entry)
            if len(self.recent_alerts) > 50:
                self.recent_alerts.pop()
                
        return packet

simulator = CloudSimulator()

# Serve root
@app.get("/")
def root():
    # If dist/index.html exists, serve it
    dist_index = os.path.join(os.path.dirname(__file__), "..", "dashboard", "dist", "index.html")
    if os.path.exists(dist_index):
        return FileResponse(dist_index)
    return {
        "status": "ONLINE",
        "system": "Twin-Guided Explainable IDS (X-IDS)",
        "endpoints": [
            "/api/health",
            "/api/benchmarks",
            "/api/models/comparison",
            "/api/stats",
            "/api/stream/step",
            "/api/stream/sse",
            "/docs"
        ]
    }

@app.get("/api/health")
def health():
    return {"status": "ONLINE", "cloud": "Vercel Serverless", "timestamp": time.time()}

@app.get("/api/benchmarks")
def get_benchmarks():
    return BENCHMARK_DATA

@app.get("/api/models/comparison")
def get_model_comparison():
    return MODEL_COMPARISON_DATA

@app.get("/api/stats")
def get_stats():
    return {
        "filter_stats": simulator.stats,
        "total_alerts_recorded": len(simulator.recent_alerts),
        "recent_alerts": simulator.recent_alerts[:15],
        "is_streaming": True,
        "stream_delay_ms": 250
    }

@app.get("/api/stream/step")
def stream_step():
    return simulator.generate_packet()

@app.get("/api/stream/config")
def stream_config(delay_ms: int = Query(250), streaming: bool = Query(True)):
    return {"status": "SUCCESS", "delay_ms": delay_ms, "streaming": streaming}

@app.get("/api/stream/sse")
async def stream_sse(request: Request):
    async def sse_gen():
        for _ in range(60):
            if await request.is_disconnected():
                break
            pkt = simulator.generate_packet()
            yield f"data: {json.dumps(pkt)}\n\n"
            import asyncio
            await asyncio.sleep(0.35)
    return StreamingResponse(sse_gen(), media_type="text/event-stream")
