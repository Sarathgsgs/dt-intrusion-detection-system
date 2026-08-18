"""
Project Runner & Integrated Launcher
Starts the FastAPI Backend and Vite Dashboard concurrently for live demonstration.
"""

import os
import sys
import subprocess
import time
import shutil

def main():
    print("=" * 70)
    print("  TWIN-GUIDED EXPLAINABLE INTRUSION DETECTION SYSTEM (X-IDS)")
    print("  FastAPI Backend + React Cyber Dashboard Launcher")
    print("=" * 70)
    
    python_exe = sys.executable
    
    # 1. Start FastAPI Backend
    print("\n[1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...")
    backend_proc = subprocess.Popen(
        [python_exe, "src/api_server.py"],
        cwd=os.path.abspath(".")
    )
    
    time.sleep(2)
    
    # 2. Start React Dashboard
    npm_cmd = "npm.cmd" if os.name == "nt" and shutil.which("npm.cmd") else "npm"
    print(f"\n[2/2] Starting Vite React Dashboard on http://localhost:5173 ...")
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=os.path.abspath("dashboard")
    )
    
    print("\n" + "=" * 70)
    print("  🚀 SYSTEM ONLINE & READY!")
    print("  - 🌐 React Interactive Dashboard: http://localhost:5173")
    print("  - 📡 FastAPI Backend & SSE Stream: http://127.0.0.1:8000")
    print("  - 📚 API Interactive Docs (Swagger): http://127.0.0.1:8000/docs")
    print("  Press Ctrl+C to terminate both servers.")
    print("=" * 70 + "\n")
    
    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping servers gracefully...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
