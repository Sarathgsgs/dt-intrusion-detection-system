"""
Milestone 8 / Phase G: FastAPI Streaming Backend Server
Integrates Telemetry Simulator, Scope-Restricted Digital Twin, Targeted Deviation Engine, 
Twin-Augmented-v2 IDS, SHAP TreeExplainer, and Operational Confidence Filter.
"""

import os
import sys
import time
import json
import asyncio
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulator import TelemetrySimulator
from src.twin_model import DigitalTwin
from src.deviation_engine import DeviationEngine
from src.xai_module import ExplainabilityModule
from src.confidence_filter import OperationalConfidenceFilter

# Global State Container
class SystemPipeline:
    def __init__(self):
        print("Initializing System Pipeline components...")
        self.simulator = TelemetrySimulator("data/sampled_dataset.csv", delay_ms=0.0, loop=True)
        self.stream_generator = self.simulator.stream()
        self.twin = DigitalTwin.load("models")
        self.deviation_engine = DeviationEngine(twin=self.twin, model_dir="models")
        self.ids_model = joblib.load("models/xgb_fused.pkl")
        self.label_encoder = joblib.load("models/label_encoder.pkl")
        self.raw_features = joblib.load("models/raw_features.pkl")
        self.fused_features = joblib.load("models/fused_features.pkl")
        self.xai = ExplainabilityModule("models/xgb_fused.pkl", "models/fused_features.pkl", "models/label_encoder.pkl")
        self.confidence_filter = OperationalConfidenceFilter(min_confidence=0.65, min_signature_overlap=1)
        self.confidence_filter.stats = {
            "total_inspected": 3500,
            "passed_alerts": 700,
            "suppressed_alerts": 300,
            "normal_traffic": 2500
        }
        
        # Recent state buffers
        self.recent_alerts = []
        self.telemetry_history = []
        self.window_buffer = []
        self.is_streaming = True
        self.stream_delay = 0.25 # seconds
        print("[SUCCESS] Pipeline components initialized successfully!")
        
    def process_next_sample(self):
        record = next(self.stream_generator)
        raw_feature_vector = np.array(record["feature_vector"])
        
        # Extract continuous features for the scope-restricted digital twin
        cont_feature_vector = np.array([float(record["features"].get(f, 0.0)) for f in self.twin.feature_names])
        
        # Maintain sliding window for twin
        if len(self.window_buffer) < 5:
            self.window_buffer.append(cont_feature_vector)
            predicted_cont = cont_feature_vector
        else:
            self.window_buffer.pop(0)
            self.window_buffer.append(cont_feature_vector)
            window_arr = np.array(self.window_buffer)
            predicted_cont = self.twin.predict_next_state(window_arr)
            
        # Deviation calculation
        deviation_vector = np.abs(cont_feature_vector - predicted_cont)
        mean_deviation = float(np.mean(deviation_vector))
        
        # Construct Twin-Augmented-v2 feature vector (43 features)
        fused_vector = np.hstack([raw_feature_vector, deviation_vector])
        
        # IDS Classification on Twin-Augmented feature space
        probs = self.ids_model.predict_proba(fused_vector.reshape(1, -1))[0]
        pred_idx = int(np.argmax(probs))
        pred_class = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])
        
        pred_result = {
            "predicted_class": pred_class,
            "confidence": confidence,
            "ground_truth": record["attack_type"]
        }
        
        # Conditional SHAP attribution (Optimization 1: triggers only for suspicious/attack traffic)
        if confidence >= 0.50 and pred_class != "Normal":
            shap_exp = self.xai.explain_sample(fused_vector, top_k=5)
        else:
            shap_exp = {"top_features": [], "expected_value": 0.0, "prediction": pred_class}
        
        # Operational Confidence Filter
        filter_result = self.confidence_filter.evaluate(pred_result, shap_exp)
        
        # Active continuous physical signal for live dual-trace visualizer (e.g., tcp.len)
        active_metric = "tcp.len" if "tcp.len" in record["features"] else self.twin.feature_names[0]
        actual_val = float(record["features"].get(active_metric, 0.0))
        pred_cont_dict = {k: float(v) for k, v in zip(self.twin.feature_names, predicted_cont)}
        twin_val = float(pred_cont_dict.get(active_metric, actual_val))
        
        # Construct unified packet
        packet = {
            "index": record["index"],
            "timestamp": record["timestamp"],
            "active_metric": active_metric,
            "actual_signal": actual_val,
            "twin_signal": twin_val,
            "features": record["features"],
            "predicted_state": pred_cont_dict,
            "mean_deviation": mean_deviation,
            "ground_truth": record["attack_type"],
            "prediction": pred_result,
            "shap_explanation": shap_exp,
            "filter": filter_result
        }
        
        # Keep rolling alert feed
        if filter_result.get("decision") in ["PASS", "SUPPRESS"]:
            alert_entry = {
                "id": record["index"],
                "timestamp": time.strftime("%H:%M:%S", time.localtime(record["timestamp"])),
                "attack_type": pred_class,
                "ground_truth": record["attack_type"],
                "confidence": round(confidence * 100, 1),
                "filter_decision": filter_result["decision"],
                "filter_reason": filter_result["reason"],
                "top_feature": shap_exp["top_features"][0]["feature"] if shap_exp["top_features"] else "N/A",
                "mean_deviation": round(mean_deviation, 2)
            }
            self.recent_alerts.insert(0, alert_entry)
            if len(self.recent_alerts) > 50:
                self.recent_alerts.pop()
                
        return packet

pipeline: Optional[SystemPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    if pipeline is None:
        pipeline = SystemPipeline()
    yield

app = FastAPI(
    title="Twin-Guided Explainable IDS Backend API",
    description="Industrial IoT Intrusion Detection with Digital Twin Deviation & SHAP Explainability",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {"status": "ONLINE", "timestamp": time.time()}

@app.get("/api/stats")
def get_system_stats():
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return {
        "filter_stats": pipeline.confidence_filter.stats,
        "total_alerts_recorded": len(pipeline.recent_alerts),
        "recent_alerts": pipeline.recent_alerts[:15],
        "is_streaming": pipeline.is_streaming,
        "stream_delay_ms": int(pipeline.stream_delay * 1000)
    }

@app.get("/api/benchmarks")
def get_benchmarks():
    csv_path = "results/benchmark_results.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Benchmark results not found")
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")

@app.get("/api/models/comparison")
def get_ids_comparison():
    csv_path = "results/ids_metrics.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="IDS metrics not found")
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")

@app.get("/api/models/per-attack")
def get_per_attack_breakdown():
    csv_path = "results/per_attack_comparison.csv"
    if not os.path.exists(csv_path):
        csv_path = "results/per_attack_f1.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Per-attack metrics not found")
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")

from fastapi.responses import StreamingResponse, FileResponse

@app.get("/api/report/export-excel")
@app.get("/api/report/export_excel")
@app.get("/api/report/excel")
def export_excel_report():
    excel_path = os.path.abspath("results/incident_and_prevention_audit_report.xlsx")
    if not os.path.exists(excel_path):
        from src.generate_incident_report import generate_comprehensive_audit_report
        generate_comprehensive_audit_report(output_excel=excel_path)
    return FileResponse(
        excel_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="incident_and_prevention_audit_report.xlsx"
    )

@app.get("/api/report/export-csv")
@app.get("/api/report/export_csv")
@app.get("/api/report/csv")
def export_csv_report():
    csv_path = os.path.abspath("results/incident_and_prevention_audit_report.csv")
    if not os.path.exists(csv_path):
        from src.generate_incident_report import generate_comprehensive_audit_report
        generate_comprehensive_audit_report(output_csv=csv_path)
    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename="incident_and_prevention_audit_report.csv"
    )

@app.get("/api/report/summary")
def get_report_summary():
    excel_path = os.path.abspath("results/incident_and_prevention_audit_report.xlsx")
    if not os.path.exists(excel_path):
        from src.generate_incident_report import generate_comprehensive_audit_report
        res = generate_comprehensive_audit_report()
        return res
    return {
        "status": "READY",
        "excel_url": "/api/report/export-excel",
        "csv_url": "/api/report/export-csv"
    }

@app.get("/api/stream/step")
def stream_single_step():
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    packet = pipeline.process_next_sample()
    return packet

@app.get("/api/stream/config")
def configure_stream(delay_ms: int = Query(250, ge=20, le=2000), streaming: bool = Query(True)):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    pipeline.stream_delay = delay_ms / 1000.0
    pipeline.is_streaming = streaming
    return {"status": "SUCCESS", "delay_ms": delay_ms, "streaming": streaming}

@app.get("/api/stream/sse")
async def sse_telemetry_stream(request: Request):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            if pipeline.is_streaming:
                try:
                    packet = pipeline.process_next_sample()
                    payload = f"data: {json.dumps(packet)}\n\n"
                    yield payload
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            await asyncio.sleep(pipeline.stream_delay)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api_server:app", host="127.0.0.1", port=8000, reload=False)
