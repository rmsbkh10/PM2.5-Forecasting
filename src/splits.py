import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GroupKFold, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .config import CONFIG

def make_temporal_folds(dev, date, n_splits=None, gap=None):
    """Split unique calendar dates, keeping all stations from a date together."""
    n_splits = n_splits or CONFIG.temporal_splits
    gap = CONFIG.gap_days if gap is None else gap
    dates = np.array(sorted(dev[date].dropna().unique()))
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    folds = []
    for train_date_idx, val_date_idx in splitter.split(dates):
        train_dates, val_dates = dates[train_date_idx], dates[val_date_idx]
        train_idx = np.flatnonzero(dev[date].isin(train_dates).to_numpy())
        val_idx = np.flatnonzero(dev[date].isin(val_dates).to_numpy())
        if len(train_idx) and len(val_idx):
            folds.append((train_idx, val_idx))
    if not folds:
        raise ValueError("Impossible de construire les folds temporels. Réduis n_splits ou gap.")
    return folds

def make_spatial_folds(dev, station, n_splits=None):
    n_groups = dev[station].nunique()
    n_splits = min(n_splits or CONFIG.spatial_splits, n_groups)
    if n_splits < 2:
        raise ValueError("Au moins deux stations sont nécessaires pour la validation spatiale.")
    return list(GroupKFold(n_splits=n_splits).split(dev, groups=dev[station]))

def make_random_folds(dev, n_splits=5, random_state=None):
    return list(KFold(n_splits=n_splits, shuffle=True,
                      random_state=random_state or CONFIG.random_state).split(dev))

def save_folds(dev, folds, filename, output_dir=None):
    out = Path(output_dir or CONFIG.folds_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = [
        {"fold": i, "train": dev.iloc[tr]["row_id"].astype(int).tolist(),
         "val": dev.iloc[va]["row_id"].astype(int).tolist()}
        for i, (tr, va) in enumerate(folds)
    ]
    with open(out / filename, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return out / filename

def make_preprocessing_pipeline():
    """Fit this pipeline on a training fold only, then transform validation/test."""
    return Pipeline([
        ("imputation", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaling", StandardScaler())
    ])

def prepare_fold(dev, features, train_idx, val_idx):
    pipeline = make_preprocessing_pipeline()
    x_train = pipeline.fit_transform(dev.iloc[train_idx][features])
    x_val = pipeline.transform(dev.iloc[val_idx][features])
    y_train = dev.iloc[train_idx].iloc[:, dev.columns.get_loc(features[0])] if False else dev.iloc[train_idx]
    return pipeline, x_train, x_val
