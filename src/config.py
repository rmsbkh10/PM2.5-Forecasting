from dataclasses import dataclass, field
from pathlib import Path
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

@dataclass
class ProjectConfig:
    group: str = "G22"
    data_file: Path = ROOT / "data" / "raw" / "combined_gee_pm25.csv"
    output_dir: Path = ROOT / "results" / "G22"
    figures_dir: Path = ROOT / "figures"
    folds_dir: Path = ROOT / "folds"
    test_start: str = "2025-01-01"
    random_state: int = 42
    target_min_exclusive: float = 0.0
    target_max_exclusive: float = 500.0
    temporal_splits: int = 5
    spatial_splits: int = 5
    gap_days: int = 1
    sequence_length: int = 7
    forecast_horizon: int = 1
    date_column: str | None = None
    station_column: str | None = None
    latitude_column: str | None = None
    longitude_column: str | None = None
    target_column: str | None = None
    predictors: list[str] = field(default_factory=list)

    def prepare_directories(self):
        for path in (self.output_dir, self.figures_dir, self.folds_dir):
            Path(path).mkdir(parents=True, exist_ok=True)
        return self

CONFIG = ProjectConfig()
np.random.seed(CONFIG.random_state)

DATE_HINTS = ("date", "time")
STATION_HINTS = ("station", "site", "gee_id", "location", "city", "name")
LAT_HINTS = ("lat",)
LON_HINTS = ("lon", "lng")
TARGET_HINTS = ("pm2.5", "pm25", "pm2_5")
LEAKAGE_HINTS = ("pm10", "aqi", "pm2", "pm25")

def find_column(columns, hints):
    """Return the first matching column name, case-insensitively."""
    for col in columns:
        low = str(col).lower()
        if any(hint in low for hint in hints):
            return col
    return None

def detect_columns(df, config=CONFIG):
    """Detect dataset columns and validate the minimum schema."""
    date = config.date_column or find_column(df.columns, DATE_HINTS)
    station = config.station_column or find_column(df.columns, STATION_HINTS)
    lat = config.latitude_column or find_column(df.columns, LAT_HINTS)
    lon = config.longitude_column or find_column(df.columns, LON_HINTS)
    target = config.target_column or find_column(df.columns, TARGET_HINTS)

    missing = [name for name, value in {
        "date/time": date, "latitude": lat, "longitude": lon, "PM2.5 target": target
    }.items() if value is None]
    if missing:
        raise ValueError(
            "Colonnes indispensables introuvables : " + ", ".join(missing) +
            ". Renseigne leurs noms exacts dans ProjectConfig."
        )
    if station is None:
        if lat not in df or lon not in df:
            raise ValueError("Impossible de créer station_id sans latitude et longitude.")
        station = "station_id"
    config.date_column, config.station_column = date, station
    config.latitude_column, config.longitude_column = lat, lon
    config.target_column = target
    return {"date": date, "station": station, "lat": lat, "lon": lon, "target": target}

def get_predictors(df, columns):
    excluded = {
        columns["date"], columns["station"], columns["lat"],
        columns["lon"], columns["target"]
    }
    leakage = [
        c for c in df.columns
        if c != columns["target"] and any(h in str(c).lower() for h in LEAKAGE_HINTS)
    ]
    excluded.update(leakage)
    predictors = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in excluded and not str(c).lower().endswith(("_id", "id"))
    ]
    return predictors, leakage
