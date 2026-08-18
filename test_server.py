import urllib.request
import json

endpoints = [
    "/api/health",
    "/api/stats",
    "/api/benchmarks",
    "/api/models/comparison",
    "/api/stream/step"
]

print("--- Testing FastAPI Backend Endpoints ---")
for ep in endpoints:
    url = f"http://127.0.0.1:8000{ep}"
    try:
        req = urllib.request.urlopen(url, timeout=5)
        res = json.loads(req.read().decode())
        print(f"[OK] {ep:<25} -> Response keys/count: {len(res) if isinstance(res, (list, dict)) else 'OK'}")
    except Exception as e:
        print(f"[FAIL] {ep:<25} -> Error: {e}")
