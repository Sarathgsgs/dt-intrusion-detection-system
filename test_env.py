import sys
import os

def test_environment():
    print(f"Python Version: {sys.version}")
    print(f"Executing from: {sys.executable}")
    
    modules = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-Learn"),
        ("xgboost", "XGBoost"),
        ("shap", "SHAP"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("matplotlib", "Matplotlib"),
        ("joblib", "Joblib")
    ]
    
    success = True
    print("\n--- Verifying Core Libraries ---")
    for mod_name, label in modules:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "loaded")
            print(f"[OK] {label:<15} (version: {ver})")
        except ImportError as e:
            print(f"[FAIL] {label:<15} - Error: {e}")
            success = False
            
    print("\n--- Directory Structure Check ---")
    required_dirs = ["data", "notebooks", "src", "models", "results"]
    for d in required_dirs:
        exists = os.path.isdir(d)
        print(f"[{'OK' if exists else 'MISSING'}] Folder: {d}/")
        
    if success:
        print("\nAll core dependencies and directories verified successfully!")
    else:
        print("\nSome dependencies failed to load.")
    return success

if __name__ == "__main__":
    test_environment()
