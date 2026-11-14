"""Evaluation metrics and persistence helpers (the filename is retained for compatibility)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true et y_pred doivent être non vides et de même longueur.")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if not len(y_true):
        raise ValueError("Aucune paire de valeurs finies à évaluer.")
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denominator = np.maximum(np.abs(y_true), 1e-8)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "MAPE_pct": float(np.mean(np.abs((y_true-y_pred) / denominator)) * 100),
        "n": int(len(y_true))
    }

def evaluate_models(y_true, predictions):
    rows = []
    for model_name, pred in predictions.items():
        rows.append({"model": model_name, **regression_metrics(y_true, pred)})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

def save_evaluation(results, output_dir, filename="evaluation_metrics.csv"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    results.to_csv(path, index=False)
    return path

def save_predictions(y_true, predictions, output_dir, filename="predictions.csv"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"y_true": np.asarray(y_true).reshape(-1)})
    for name, values in predictions.items():
        frame[name] = np.asarray(values).reshape(-1)
    path = out / filename
    frame.to_csv(path, index=False)
    return path
