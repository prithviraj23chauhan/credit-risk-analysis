# src/model_registry.py
import joblib
from pathlib import Path
from .config import load_config

ROOT = Path(__file__).resolve().parents[1]

def load_model_bundle(model_name: str = None):
    cfg = load_config()

    if model_name is None:
        model_name = cfg["default_model"]

    model_cfg = cfg["models"][model_name]
    model_path = ROOT / model_cfg["path"]

    bundle = joblib.load(model_path)
    bundle["model_name"] = model_name
    bundle["decision_threshold"] = cfg.get("decision_threshold", 0.5)
    bundle["type"] = model_cfg.get("type", "sklearn")

    return bundle
