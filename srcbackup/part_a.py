# ==========================================
# PM2.5 FORECASTING - PARTIE A (exploration + prétraitement)
# Modèle : ARIMA-WOA-LSTM (G22) — à coller dans UNE cellule Colab
# Le fichier combined_gee_pm25.csv doit être dans le panneau Files (à gauche)
# ==========================================



import pandas as pd
import numpy as np
import os
import zipfile
import glob


# ==========================================
# ETAPE 1-2 : Trouver le fichier (déjà déposé dans Colab)
# ==========================================
matches = glob.glob("/content/**/combined_gee_pm25.csv", recursive=True)
if not matches:
    for z in glob.glob("/content/*.zip"):
        with zipfile.ZipFile(z, "r") as archive:
            archive.extractall("/content/pm25_data")
    matches = glob.glob("/content/**/combined_gee_pm25.csv", recursive=True)
if not matches:
    raise FileNotFoundError("combined_gee_pm25.csv introuvable : glisse-le dans le panneau Files à gauche.")
csv_path = matches[0]
print("Fichier trouvé :", csv_path)

# ==========================================
# ETAPE 3 : Charger les données
# ==========================================
df = pd.read_csv(csv_path)
print("Lignes, colonnes :", df.shape)
print("Colonnes :", df.columns.tolist())

# ==========================================
# ETAPE 4 à 8 : Aperçu, infos, manquants, statistiques, doublons
# ==========================================
display(df.head())
df.info()
display(df.isnull().sum())
display(df.describe(include="all"))
print("Nombre de doublons :", df.duplicated().sum())
print("\nEXPLORATION TERMINEE !")

# ==========================================
# ETAPE 9 : Configuration et détection des colonnes
# ==========================================
import json, warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
warnings.filterwarnings("ignore")
np.random.seed(42)

GROUPE = "BXX-YY"                      # <-- ton numéro de binôme
OUT = f"results/{GROUPE}"
os.makedirs(OUT, exist_ok=True)

def trouver(mots):
    """Renvoie la première colonne dont le nom contient un des mots."""
    for c in df.columns:
        if any(m in c.lower() for m in mots):
            return c
    return None

# Si une détection est fausse, écris le nom exact à la main (ex. TARGET = "PM2.5")
DATE    = trouver(["date", "time"])
STATION = trouver(["station", "site", "gee_id", "location", "city", "name"])
LAT     = trouver(["lat"])
LON     = trouver(["lon", "lng"])
TARGET  = trouver(["pm2.5", "pm25", "pm2_5"])
print("Date:", DATE, "| Station:", STATION, "| Lat:", LAT, "| Lon:", LON, "| Cible:", TARGET)

# Pas de colonne station ? -> chaque couple (latitude, longitude) = une station
if STATION is None:
    df["station_id"] = "ST_" + df[LAT].round(4).astype(str) + "_" + df[LON].round(4).astype(str)
    STATION = "station_id"
    print("Pas de colonne station : identifiant créé à partir des coordonnées ->", df[STATION].nunique(), "stations")

df[DATE] = pd.to_datetime(df[DATE], errors="coerce").dt.normalize()

# ==========================================
# ETAPE 10 : Audit du dataset (§3.1)
# ==========================================
print("Période :", df[DATE].min().date(), "->", df[DATE].max().date())
print("Nombre de stations :", df[STATION].nunique())
print("Latitude :", df[LAT].min(), "->", df[LAT].max(), "| Longitude :", df[LON].min(), "->", df[LON].max())
print("Dates illisibles :", df[DATE].isna().sum())
print("Stations avec plusieurs coordonnées :", (df.groupby(STATION)[[LAT, LON]].nunique() > 1).any(axis=1).sum())
print("Mesures par station (min / médiane / max) :", df.groupby(STATION).size().agg(["min", "median", "max"]).tolist())

# Variables dérivées de la cible = fuite -> on les exclut (PM10 contient la PM2.5, AQI calculé avec)
FUITE = [c for c in df.columns if c != TARGET and any(m in c.lower() for m in ["pm10", "aqi", "pm2", "pm25"])]
print("Variables exclues (fuite) :", FUITE)

# Prédicteurs = colonnes numériques sauf identifiants, coordonnées, cible et fuite
EXCLURE = [DATE, STATION, LAT, LON, TARGET] + FUITE
PRED = [c for c in df.select_dtypes("number").columns if c not in EXCLURE and not c.lower().endswith(("_id", "id"))]
print("Prédicteurs :", PRED)

# ==========================================
# ETAPE 11 : Nettoyage de la cible et doublons (§5.1)
# ==========================================
n0 = len(df)
n_manquant = df[TARGET].isna().sum()
n_bas = (df[TARGET] <= 0).sum()
n_haut = (df[TARGET] >= 500).sum()
df = df[df[TARGET].notna() & (df[TARGET] > 0) & (df[TARGET] < 500)]
print(f"Supprimées : {n_manquant} PM2.5 manquantes, {n_bas} <= 0, {n_haut} >= 500  -> {n0} -> {len(df)} lignes")

n_dup_total = df.duplicated().sum()
n_dup_cle = df.duplicated([DATE, LAT, LON, STATION]).sum()
df = df.drop_duplicates()
df = df.groupby([STATION, DATE], as_index=False).agg({c: "mean" if pd.api.types.is_numeric_dtype(df[c]) else "first"
                                                      for c in df.columns if c not in [STATION, DATE]})
print(f"Doublons : {n_dup_total} (toutes colonnes), {n_dup_cle} (date, lat, lon, station) -> 1 ligne par station et par jour")

# ==========================================
# ETAPE 12 : Statistiques descriptives (§5.1)
# ==========================================
VARS = [TARGET] + PRED
stats_df = df[VARS].describe().T
stats_df["skewness"] = df[VARS].skew()
stats_df["kurtosis"] = df[VARS].kurt()
display(stats_df.round(3))

# ==========================================
# ETAPE 13 : Graphiques des variables clés
# ==========================================
CLES = [TARGET] + [c for c in PRED if any(m in c.lower() for m in
        ["aod", "no2", "co", "o3", "temp", "press", "wind", "u_", "v_"])][:8]
fig, ax = plt.subplots(2, len(CLES), figsize=(3 * len(CLES), 6))
for i, c in enumerate(CLES):
    sns.histplot(df[c], bins=50, ax=ax[0, i]); ax[0, i].set_title(c, fontsize=9)
    sns.boxplot(y=df[c], ax=ax[1, i])
plt.tight_layout(); plt.show()

df["mois"] = df[DATE].dt.month
df["saison"] = df["mois"].map({12: "Hiver", 1: "Hiver", 2: "Hiver", 3: "Printemps", 4: "Printemps", 5: "Printemps",
                               6: "Eté", 7: "Eté", 8: "Eté", 9: "Automne", 10: "Automne", 11: "Automne"})
sns.boxplot(data=df, x="mois", y=TARGET); plt.title("PM2.5 par mois"); plt.show()

# ==========================================
# ETAPE 14 : Valeurs manquantes et outliers
# ==========================================
manq = (100 * df[VARS].isna().mean()).sort_values(ascending=False)
print("% de valeurs manquantes :"); display(manq.round(2))

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
lignes = []
for c in VARS:
    x = df[c].dropna()
    q1, q3 = x.quantile([.25, .75]); iqr = q3 - q1
    ech = x.sample(min(len(x), 20000), random_state=42).values.reshape(-1, 1)
    lignes.append({"variable": c,
                   "%_IQR": 100 * ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean(),
                   "%_zscore": 100 * (np.abs(stats.zscore(x)) > 3).mean(),
                   # score d'anomalie > 0.7 = anomalie nette (Liu et al., 2008)
                   "%_IsolationForest": 100 * (-IsolationForest(random_state=42).fit(ech).score_samples(ech) > 0.7).mean(),
                   "%_LOF": 100 * (LocalOutlierFactor(20).fit_predict(ech) == -1).mean()})
outliers = pd.DataFrame(lignes).set_index("variable")
display(outliers.round(2))
# On NE supprime PAS ces valeurs : un pic de PM2.5 est souvent un vrai épisode de pollution.

# ==========================================
# ETAPE 15 : Corrélations, colinéarité et VIF
# ==========================================
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(df[VARS].corr("pearson"), cmap="coolwarm", center=0, ax=ax[0]); ax[0].set_title("Pearson")
sns.heatmap(df[VARS].corr("spearman"), cmap="coolwarm", center=0, ax=ax[1]); ax[1].set_title("Spearman")
plt.tight_layout(); plt.show()

corr = df[VARS].corr().abs()
paires = [(a, b, round(corr.loc[a, b], 2)) for i, a in enumerate(VARS) for b in VARS[i + 1:] if corr.loc[a, b] > 0.9]
print("Paires très corrélées |r| > 0.9 :", paires if paires else "aucune")

from statsmodels.stats.outliers_influence import variance_inflation_factor
X = df[PRED].fillna(df[PRED].median())
X = (X - X.mean()) / X.std()
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=PRED)
print("VIF (> 10 = colinéarité forte) :"); display(vif.round(2))

# ==========================================
# ETAPE 16 : PM2.5 vs prédicteurs, saisons, extrêmes
# ==========================================
print("Corrélation de Spearman PM2.5 / prédicteurs par saison :")
display(df.groupby("saison").apply(lambda g: g[PRED].corrwith(g[TARGET], method="spearman")).T.round(2))

y = df[TARGET]
print(f"Jours > 25 µg/m³ : {100 * (y > 25).mean():.1f} %  |  Jours > 50 µg/m³ : {100 * (y > 50).mean():.1f} %")
print(f"Skewness PM2.5 : {y.skew():.2f}  -> après log1p : {np.log1p(y).skew():.2f}")

# ==========================================
# ETAPE 17 : Fichier Exploratory_Data.csv
# ==========================================
rows = []
for c in VARS:
    x = df[c]
    r = {"model_id": "arima_woa_lstm_pm25", "model_group": "G22", "variable": c,
         "type": "numeric", "n_total": len(x), "n_missing": x.isna().sum(), "pct_missing": 100 * x.isna().mean(),
         "n_unique": x.nunique(), "n_duplicates_key": n_dup_cle,
         "min": x.min(), "q1": x.quantile(.25), "median": x.median(), "q3": x.quantile(.75), "max": x.max(),
         "mean": x.mean(), "std": x.std(), "skewness": x.skew(), "kurtosis": x.kurt(),
         "iqr": x.quantile(.75) - x.quantile(.25),
         "pct_out_iqr": outliers.loc[c, "%_IQR"], "pct_out_z": outliers.loc[c, "%_zscore"],
         "pct_out_if": outliers.loc[c, "%_IsolationForest"], "pct_out_lof": outliers.loc[c, "%_LOF"],
         "max_abs_corr": corr[c].drop(c).max(), "corr_partner": corr[c].drop(c).idxmax(),
         "vif": vif.get(c, np.nan), "is_leakage_suspect": False}
    if c == TARGET:
        r.update(family="Target", unit="µg/m³", n_sentinel_low=n_bas, n_sentinel_high=n_haut,
                 n_removed_total=n0 - len(df) + n_dup_total)
    else:
        ok = df[[c, TARGET]].dropna()
        rho, p = stats.spearmanr(ok[c], ok[TARGET])
        r.update(target_assoc=f"spearman={rho:.3f}", target_pval=p)
    rows.append(r)
pd.DataFrame(rows).to_csv(f"{OUT}/Exploratory_Data.csv", index=False)
print("Sauvegardé :", f"{OUT}/Exploratory_Data.csv")

# ==========================================
# ETAPE 18 : Exploration temporelle (§5.2)
# ==========================================
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.simplefilter("ignore", InterpolationWarning)
from statsmodels.stats.diagnostic import acorr_ljungbox

# Série médiane de toutes les stations (un point par jour)
serie = df.groupby(DATE)[TARGET].median().asfreq("D")
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(serie.dropna(), lags=40, ax=ax[0]); plot_pacf(serie.dropna(), lags=20, ax=ax[1], method="ywm")
plt.show()

tempo = []
for st, g in df.groupby(STATION):
    s = g.set_index(DATE)[TARGET].sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))   # grille journalière régulière
    x = s.dropna()
    if len(x) < 60:
        continue
    a = acf(s.values, nlags=10, missing="conservative")
    pente = stats.linregress((x.index - x.index[0]).days, x.values)
    tempo.append({"model_id": "arima_woa_lstm_pm25", "model_group": "G22", "variable": TARGET, "node_id": st,
                  "n_records": len(x), "t_min": x.index.min().date(), "t_max": x.index.max().date(),
                  "sampling_interval_median_sec": x.index.to_series().diff().dt.total_seconds().median(),
                  "pct_missing_intervals": 100 * s.isna().mean(),
                  "acf_lag1": a[1], "acf_lag7": a[7], "pacf_lag1": pacf(x, nlags=2)[1],
                  "ljung_box_p": acorr_ljungbox(x, lags=[7])["lb_pvalue"].iloc[0],
                  "adf_p": adfuller(x)[1], "kpss_p": kpss(x, nlags="auto")[1],
                  "trend_slope_per_day": pente.slope, "trend_pval": pente.pvalue,
                  "seasonal_amplitude": x.groupby(x.index.month).mean().pipe(lambda m: m.max() - m.min()),
                  "target_drift_pval": stats.ks_2samp(x[:len(x) // 2], x[len(x) // 2:]).pvalue,
                  "n_days_above_25": (x > 25).sum(), "n_days_above_50": (x > 50).sum(),
                  "pct_days_above_25": 100 * (x > 25).mean(), "pct_days_above_50": 100 * (x > 50).mean(),
                  "cv_strategy_recommended": "temporal", "gap_between_train_test_sec": 86400})
tempo = pd.DataFrame(tempo)
tempo.to_csv(f"{OUT}/Exploratory_temporal.csv", index=False)
display(tempo[["node_id", "n_records", "pct_missing_intervals", "acf_lag1", "acf_lag7", "ljung_box_p", "adf_p", "kpss_p"]].round(3).head())

# Dérive : médiane, IQR et % de jours pollués par année
derive = df.groupby(df[DATE].dt.year)[TARGET].agg(
    mediane="median", iqr=lambda s: s.quantile(.75) - s.quantile(.25),
    pct_sup_25=lambda s: 100 * (s > 25).mean(), pct_sup_50=lambda s: 100 * (s > 50).mean())
display(derive.round(2))

# Lead-lag : corrélation entre PM2.5 du jour t et prédicteur du jour t-k
ll = {c: [df.groupby(STATION).apply(lambda g: g[TARGET].corr(g[c].shift(k))).median() for k in range(0, 4)] for c in CLES[1:]}
print("corr(PM2.5_t, X_{t-k}) pour k = 0, 1, 2, 3 :"); display(pd.DataFrame(ll, index=["k=0", "k=1", "k=2", "k=3"]).T.round(2))

# ==========================================
# ETAPE 19 : Création des features SANS FUITE (§5.3.4)
# ==========================================
# But : prévoir la PM2.5 de DEMAIN -> chaque ligne n'utilise que le passé (shift >= 1)
df = df.sort_values([STATION, DATE])
grille = []
for st, g in df.groupby(STATION):          # grille journalière : un jour manquant = NaN (pas de lag à travers un trou)
    g = g.set_index(DATE).reindex(pd.date_range(g[DATE].min(), g[DATE].max(), freq="D"))
    g[STATION] = st
    grille.append(g)
data = pd.concat(grille).rename_axis(DATE).reset_index()
grp = data.groupby(STATION)

for k in [1, 2, 3]:
    data[f"pm25_lag{k}"] = grp[TARGET].shift(k)                       # PM2.5 des jours précédents
for c in PRED:
    data[f"{c}_lag1"] = grp[c].shift(1)                               # prédicteurs de la veille
data["pm25_moy3"] = grp[TARGET].transform(lambda s: s.shift(1).rolling(3).mean())
data["pm25_moy7"] = grp[TARGET].transform(lambda s: s.shift(1).rolling(7).mean())
data["pm25_std7"] = grp[TARGET].transform(lambda s: s.shift(1).rolling(7).std())
data["pm25_diff1"] = data["pm25_lag1"] - data["pm25_lag2"]
jour = data[DATE].dt.dayofyear
data["jour_sin"], data["jour_cos"] = np.sin(2 * np.pi * jour / 365.25), np.cos(2 * np.pi * jour / 365.25)
data["semaine_sin"] = np.sin(2 * np.pi * data[DATE].dt.dayofweek / 7)
data["semaine_cos"] = np.cos(2 * np.pi * data[DATE].dt.dayofweek / 7)

FEATURES = ([f"pm25_lag{k}" for k in [1, 2, 3]] + [f"{c}_lag1" for c in PRED] +
            ["pm25_moy3", "pm25_moy7", "pm25_std7", "pm25_diff1", "jour_sin", "jour_cos", "semaine_sin", "semaine_cos"])
data = data[data[TARGET].notna()].reset_index(drop=True)                # on garde les jours où la cible existe
data["row_id"] = np.arange(len(data))

# Vérification anti-fuite : aucune feature ne doit être (quasi) égale à la PM2.5 du jour
corr_cible = data[FEATURES].corrwith(data[TARGET]).abs()
assert TARGET not in FEATURES and corr_cible.max() < 0.999, "FUITE !"
print("Test anti-fuite OK (corrélation max feature/cible :", round(corr_cible.max(), 2), ")")
print(f"{len(FEATURES)} features créées :", FEATURES)

# ==========================================
# ETAPE 20 : Séparation développement / test futur
# ==========================================
TEST_START = pd.Timestamp("2025-01-01")
dev = data[data[DATE] < TEST_START].reset_index(drop=True)
test = data[data[DATE] >= TEST_START].reset_index(drop=True)
print(f"Développement : {len(dev)} lignes | Test futur (jamais touché avant la fin) : {len(test)} lignes")

# ==========================================
# ETAPE 21 : Folds de validation (§5.4)
# ==========================================
from sklearn.model_selection import TimeSeriesSplit, GroupKFold, KFold

def sauver(nom, folds):
    json.dump([{"fold": i, "train": dev.loc[tr, "row_id"].tolist(), "val": dev.loc[va, "row_id"].tolist()}
               for i, (tr, va) in enumerate(folds)], open(f"{OUT}/{nom}", "w"))

# 1) Temporel : on coupe sur les dates (même date de coupure pour toutes les stations), 1 jour d'écart
dates = np.sort(dev[DATE].unique())
f_temp = [(dev.index[dev[DATE].isin(dates[a])], dev.index[dev[DATE].isin(dates[b])])
          for a, b in TimeSeriesSplit(n_splits=5, gap=1).split(dates)]
sauver("folds_temporal.json", f_temp)
# 2) Spatial : des stations jamais vues en validation
f_spat = list(GroupKFold(n_splits=5).split(dev, groups=dev[STATION]))
sauver("folds_spatial.json", f_spat)
# 3) Aléatoire : diagnostic seulement
f_rand = list(KFold(n_splits=5, shuffle=True, random_state=42).split(dev))
sauver("folds_standard.json", f_rand)

for i, (tr, va) in enumerate(f_temp):
    print(f"Fold temporel {i} : train {dev.loc[tr, DATE].min().date()} -> {dev.loc[tr, DATE].max().date()} | "
          f"val {dev.loc[va, DATE].min().date()} -> {dev.loc[va, DATE].max().date()}")

# ==========================================
# ETAPE 22 : Pipeline de prétraitement (fit sur le TRAIN uniquement)
# ==========================================
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([("imputation", SimpleImputer(strategy="median", add_indicator=True)),  # médiane + indicateur de manquant
                     ("normalisation", StandardScaler())])
tr, va = f_temp[0]
X_train = pipeline.fit_transform(dev.loc[tr, FEATURES])    # fit sur le train
X_val = pipeline.transform(dev.loc[va, FEATURES])          # transform seulement sur la validation
print("Fold 0 :", X_train.shape, X_val.shape, "| NaN restants :", np.isnan(X_val).sum())

# ==========================================
# ETAPE 23 : Diagnostics pour ARIMA (préparation de la partie B)
# ==========================================
p_adf, p_kpss = adfuller(serie.dropna())[1], kpss(serie.dropna(), nlags="auto")[1]
d = 0 if (p_adf < 0.05 and p_kpss > 0.05) else 1
print(f"ADF p = {p_adf:.3f} | KPSS p = {p_kpss:.3f} -> ordre de différenciation d = {d}")
print(f"ACF lag 7 médiane des stations = {tempo['acf_lag7'].median():.2f} -> saisonnalité hebdomadaire (s=7) si nettement > 0")

# Sauvegarde finale
data.to_csv(f"{OUT}/data_features.csv", index=False)
json.dump({"features": FEATURES, "target": TARGET, "test_start": "2025-01-01", "d_arima": d},
          open(f"{OUT}/feature_config.json", "w"), indent=1)
print("\nPARTIE A TERMINEE ! Fichiers dans", OUT)
y = df[TARGET]
print(f"> 25 : {100 * (y > 25).mean():.1f} %")
print(f"> 50 : {100 * (y > 50).mean():.1f} %")