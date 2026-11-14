from pathlib import Path
import zipfile
import pandas as pd
import numpy as np
from .config import CONFIG, detect_columns, get_predictors

def locate_dataset(path=None):
    """Locate the CSV in the configured data/raw directory (or an explicit path)."""
    candidate = Path(path) if path else Path(CONFIG.data_file)
    if candidate.is_file():
        return candidate
    raw_dir = candidate.parent
    if raw_dir.exists():
        for archive_path in raw_dir.glob("*.zip"):
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(raw_dir / "extracted")
        matches = list(raw_dir.rglob("combined_gee_pm25.csv"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Dataset introuvable : {candidate}. Place combined_gee_pm25.csv dans data/raw/."
    )

def load_dataset(path=None):
    csv_path = locate_dataset(path)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Le fichier CSV est vide.")
    return df, csv_path

def initial_exploration(df):
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "head": df.head(),
        "missing": df.isna().sum().sort_values(ascending=False),
        "description": df.describe(include="all").T,
        "duplicates": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str)
    }

def clean_dataset(df, config=CONFIG):
    """Clean target, normalize dates, deduplicate and aggregate station/day."""
    df = df.copy()
    cols = detect_columns(df, config)
    date, station = cols["date"], cols["station"]
    lat, lon, target = cols["lat"], cols["lon"], cols["target"]

    if station == "station_id" and station not in df.columns:
        df[station] = (
            "ST_" + df[lat].round(4).astype("string") + "_" +
            df[lon].round(4).astype("string")
        )
    df[date] = pd.to_datetime(df[date], errors="coerce").dt.normalize()
    df[target] = pd.to_numeric(df[target], errors="coerce")

    audit = {
        "rows_before": len(df),
        "invalid_dates": int(df[date].isna().sum()),
        "target_missing": int(df[target].isna().sum()),
        "target_le_zero": int((df[target] <= config.target_min_exclusive).sum()),
        "target_ge_max": int((df[target] >= config.target_max_exclusive).sum()),
    }
    df = df.dropna(subset=[date, station, target])
    df = df[
        (df[target] > config.target_min_exclusive) &
        (df[target] < config.target_max_exclusive)
    ].copy()

    audit["duplicate_rows"] = int(df.duplicated().sum())
    key = [date, lat, lon, station]
    audit["duplicate_station_date_key"] = int(df.duplicated(key).sum())
    df = df.drop_duplicates()

    aggregation = {
        c: ("mean" if pd.api.types.is_numeric_dtype(df[c]) else "first")
        for c in df.columns if c not in (station, date)
    }
    df = df.groupby([station, date], as_index=False).agg(aggregation)
    df = df.sort_values([station, date]).reset_index(drop=True)
    audit["rows_after"] = len(df)
    audit["stations"] = int(df[station].nunique())
    audit["date_min"] = str(df[date].min().date()) if len(df) else None
    audit["date_max"] = str(df[date].max().date()) if len(df) else None

    predictors, leakage = get_predictors(df, cols)
    audit["excluded_leakage_columns"] = leakage
    audit["predictors"] = predictors
    audit["columns"] = cols
    return df, audit

def descriptive_statistics(df, target, predictors):
    variables = [target] + list(predictors)
    result = df[variables].describe().T
    result["skewness"] = df[variables].skew()
    result["kurtosis"] = df[variables].kurtosis()
    return result

def missingness_report(df, columns=None):
    columns = columns or df.columns.tolist()
    return (100 * df[columns].isna().mean()).sort_values(ascending=False).rename("pct_missing")

def outlier_report(df, variables, sample_size=20000, random_state=42):
    """Estimate IQR, z-score, Isolation Forest and LOF outlier percentages."""
    from scipy import stats
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    rows = []
    for col in variables:
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(x) < 5:
            rows.append({"variable": col, "pct_iqr": np.nan, "pct_zscore": np.nan,
                         "pct_isolation_forest": np.nan, "pct_lof": np.nan})
            continue
        q1, q3 = x.quantile([.25, .75])
        iqr = q3 - q1
        sample = x.sample(min(len(x), sample_size), random_state=random_state).to_numpy().reshape(-1, 1)
        iso_pct = 100 * (-IsolationForest(random_state=random_state).fit(sample)
                         .score_samples(sample) > .7).mean()
        n_neighbors = min(20, len(sample) - 1)
        lof_pct = (100 * (LocalOutlierFactor(n_neighbors=n_neighbors)
                          .fit_predict(sample) == -1).mean()) if n_neighbors >= 2 else np.nan
        z = np.abs(stats.zscore(x, nan_policy="omit"))
        rows.append({
            "variable": col,
            "pct_iqr": 100 * ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean(),
            "pct_zscore": 100 * (z > 3).mean(),
            "pct_isolation_forest": iso_pct,
            "pct_lof": lof_pct
        })
    return pd.DataFrame(rows).set_index("variable")

def save_preprocessed(df, audit, output_dir=None):
    out = Path(output_dir or CONFIG.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cleaned_data.csv", index=False)
    import json
    with open(out / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False, default=str)
    return out
