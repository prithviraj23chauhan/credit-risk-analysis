# src/pipeline.py
import pandas as pd
from .model_registry import load_model_bundle

_bundle = load_model_bundle()  # uses default_model from config
_model = _bundle["model"]
_feature_cols = _bundle["feature_cols"]
_feature_means = _bundle.get("feature_means", {})
_threshold = _bundle.get("decision_threshold", 0.5)

def prepare_input_df(input_data):
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        df = pd.DataFrame(input_data)

    # Fill missing columns with mean (or 0)
    for col in _feature_cols:
        if col not in df.columns:
            df[col] = _feature_means.get(col, 0.0)

    df = df[_feature_cols]
    return df

def predict_single(input_dict):
    df = prepare_input_df(input_dict)
    proba = _model.predict_proba(df)[0, 1]
    label = int(proba >= _threshold)
    return {
        "model": _bundle["model_name"],
        "threshold": _threshold,
        "label": label,
        "default_probability": float(proba),
    }

def predict_batch(input_list_or_df):
    df = prepare_input_df(input_list_or_df)
    proba = _model.predict_proba(df)[:, 1]
    labels = (proba >= _threshold).astype(int)

    result = df.copy()
    result["default_proba"] = proba
    result["predicted_label"] = labels
    result["model"] = _bundle["model_name"]
    result["threshold"] = _threshold
    return result
