import numpy as np
import pandas as pd
from .config import CONFIG

def create_features(df, target, date, station, predictors, sequence_gap="D"):
    """
    Build next-day supervised features. All target-derived features use shift(1)
    or older values, so the target at day t is never used as an input at day t.
    Missing calendar days are explicitly inserted per station.
    """
    df = df.copy()
    df[date] = pd.to_datetime(df[date])
    df = df.sort_values([station, date])
    grids = []
    for station_id, group in df.groupby(station):
        group = group.set_index(date).sort_index()
        full_index = pd.date_range(group.index.min(), group.index.max(), freq=sequence_gap)
        group = group.reindex(full_index)
        group[station] = station_id
        grids.append(group.rename_axis(date))
    if not grids:
        raise ValueError("Aucune station disponible après nettoyage.")
    data = pd.concat(grids).reset_index()
    grouped = data.groupby(station, sort=False)

    for lag in (1, 2, 3):
        data[f"pm25_lag{lag}"] = grouped[target].shift(lag)
    for col in predictors:
        data[f"{col}_lag1"] = grouped[col].shift(1)
    data["pm25_moy3"] = grouped[target].transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
    data["pm25_moy7"] = grouped[target].transform(lambda s: s.shift(1).rolling(7, min_periods=7).mean())
    data["pm25_std7"] = grouped[target].transform(lambda s: s.shift(1).rolling(7, min_periods=7).std())
    data["pm25_diff1"] = data["pm25_lag1"] - data["pm25_lag2"]

    day_of_year = data[date].dt.dayofyear
    day_of_week = data[date].dt.dayofweek
    data["jour_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    data["jour_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    data["semaine_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    data["semaine_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    feature_names = (
        [f"pm25_lag{k}" for k in (1, 2, 3)] +
        [f"{col}_lag1" for col in predictors] +
        ["pm25_moy3", "pm25_moy7", "pm25_std7", "pm25_diff1",
         "jour_sin", "jour_cos", "semaine_sin", "semaine_cos"]
    )
    data = data[data[target].notna()].copy().reset_index(drop=True)
    data["row_id"] = np.arange(len(data), dtype=int)
    if target in feature_names:
        raise AssertionError("Fuite de données : la cible est présente dans FEATURES.")
    return data, feature_names

def check_no_target_leakage(data, target, features, threshold=.999):
    corr = data[features].corrwith(data[target]).abs()
    max_corr = float(corr.max()) if len(corr.dropna()) else np.nan
    if target in features:
        raise AssertionError("La cible est directement incluse dans les variables explicatives.")
    suspicious = corr[corr >= threshold].to_dict()
    if suspicious:
        raise AssertionError(f"Features suspectes de fuite (corrélation >= {threshold}) : {suspicious}")
    return {"max_abs_correlation": max_corr, "correlations": corr.sort_values(ascending=False)}

def split_development_test(data, date, test_start=None):
    cutoff = pd.Timestamp(test_start or CONFIG.test_start)
    development = data[data[date] < cutoff].reset_index(drop=True)
    future_test = data[data[date] >= cutoff].reset_index(drop=True)
    if development.empty:
        raise ValueError("Aucune donnée avant TEST_START : vérifie la date de coupure.")
    if future_test.empty:
        raise ValueError("Aucune donnée dans le test futur : vérifie TEST_START.")
    return development, future_test

def temporal_diagnostics(df, target, date, station):
    from scipy import stats
    from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox

    rows = []
    for station_id, group in df.groupby(station):
        series = (group.set_index(date)[target].sort_index()
                  .asfreq("D"))
        observed = series.dropna()
        if len(observed) < 60:
            continue
        filled_for_acf = series.interpolate(limit_direction="both")
        max_lag = min(10, len(filled_for_acf) // 2 - 1)
        if max_lag < 2:
            continue
        try:
            acf_values = acf(filled_for_acf, nlags=max_lag, missing="drop")
            pacf_values = pacf(observed, nlags=min(2, len(observed)//2 - 1), method="ywm")
            adf_p = adfuller(observed)[1]
            kpss_p = kpss(observed, nlags="auto")[1]
            lb_p = acorr_ljungbox(observed, lags=[min(7, len(observed)//5)],
                                  return_df=True)["lb_pvalue"].iloc[0]
            x = (observed.index - observed.index[0]).days
            slope = stats.linregress(x, observed.values)
            half = len(observed) // 2
            drift_p = stats.ks_2samp(observed.iloc[:half], observed.iloc[half:]).pvalue
            monthly = observed.groupby(observed.index.month).mean()
            rows.append({
                "node_id": station_id, "n_records": len(observed),
                "t_min": observed.index.min().date(), "t_max": observed.index.max().date(),
                "pct_missing_intervals": 100 * series.isna().mean(),
                "acf_lag1": acf_values[1], "acf_lag7": acf_values[7] if len(acf_values)>7 else np.nan,
                "pacf_lag1": pacf_values[1], "ljung_box_p": lb_p,
                "adf_p": adf_p, "kpss_p": kpss_p,
                "trend_slope_per_day": slope.slope, "trend_pval": slope.pvalue,
                "seasonal_amplitude": monthly.max() - monthly.min(),
                "target_drift_pval": drift_p,
                "n_days_above_25": int((observed > 25).sum()),
                "n_days_above_50": int((observed > 50).sum())
            })
        except (ValueError, np.linalg.LinAlgError):
            continue
    return pd.DataFrame(rows)
