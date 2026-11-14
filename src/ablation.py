"""Ablation-study helpers: compare feature groups under the same split."""
import pandas as pd
from .test_evaluation import regression_metrics

def define_ablation_sets(features):
    features = list(features)
    return {
        "all_features": features,
        "without_target_lags": [f for f in features if not f.startswith("pm25_lag") and f not in
                                ("pm25_moy3", "pm25_moy7", "pm25_std7", "pm25_diff1")],
        "without_meteorological_lags": [f for f in features if not any(
            token in f.lower() for token in ("temp", "press", "wind", "humidity", "u_", "v_"))],
        "target_lags_only": [f for f in features if f.startswith("pm25_lag") or f in
                             ("pm25_moy3", "pm25_moy7", "pm25_std7", "pm25_diff1")],
    }

def run_ablation(feature_sets, fit_predict, y_true):
    """fit_predict(name, features) must return predictions for the same test rows."""
    rows = []
    predictions = {}
    for name, features in feature_sets.items():
        if not features:
            continue
        pred = fit_predict(name, features)
        predictions[name] = pred
        rows.append({"experiment": name, "n_features": len(features),
                     **regression_metrics(y_true, pred)})
    return pd.DataFrame(rows).sort_values("RMSE"), predictions
