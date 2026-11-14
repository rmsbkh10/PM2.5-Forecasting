"""Model interpretation helpers for tree models and permutation importance."""
import pandas as pd
import numpy as np

def permutation_importance_table(model, X, y, feature_names, scoring="neg_mean_absolute_error",
                                 n_repeats=10, random_state=42):
    from sklearn.inspection import permutation_importance
    result = permutation_importance(
        model, X, y, scoring=scoring, n_repeats=n_repeats,
        random_state=random_state, n_jobs=-1
    )
    table = pd.DataFrame({
        "feature": list(feature_names),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    })
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)

def shap_values(model, X, feature_names=None, max_samples=500):
    """Optional SHAP integration; returns explainer and values."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Installe SHAP pour cette analyse : pip install shap") from exc
    sample = X[:max_samples] if hasattr(X, "__getitem__") else X
    explainer = shap.Explainer(model)
    values = explainer(sample)
    if feature_names is not None and getattr(values, "feature_names", None) is None:
        values.feature_names = list(feature_names)
    return explainer, values

def save_importance(table, output_path):
    from pathlib import Path
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path
