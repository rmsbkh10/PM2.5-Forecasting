"""End-to-end implementation of Part A: exploration, cleaning, features and folds."""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .config import CONFIG
from .preprocessing import (load_dataset, initial_exploration, clean_dataset,
                            descriptive_statistics, missingness_report,
                            outlier_report, save_preprocessed)
from .features import (create_features, check_no_target_leakage,
                       split_development_test, temporal_diagnostics)
from .splits import (make_temporal_folds, make_spatial_folds, make_random_folds,
                     save_folds, make_preprocessing_pipeline)

def run_part_a(csv_path=None, config=CONFIG, make_plots=True):
    config.prepare_directories()
    raw, source = load_dataset(csv_path)
    exploration = initial_exploration(raw)
    df, audit = clean_dataset(raw, config)
    cols = audit["columns"]
    target, date, station = cols["target"], cols["date"], cols["station"]
    predictors = audit["predictors"]
    variables = [target] + predictors

    stats = descriptive_statistics(df, target, predictors)
    stats.to_csv(config.output_dir / "descriptive_statistics.csv")
    missingness_report(df, variables).to_csv(config.output_dir / "missingness.csv")
    outliers = outlier_report(df, variables)
    outliers.to_csv(config.output_dir / "outlier_report.csv")

    if make_plots:
        selected = [target] + [c for c in predictors if any(
            token in c.lower() for token in ("aod", "no2", "co", "o3", "temp", "press", "wind", "u_", "v_")
        )][:8]
        if selected:
            fig, axes = plt.subplots(2, len(selected), figsize=(3*len(selected), 6), squeeze=False)
            for i, col in enumerate(selected):
                sns.histplot(df[col].dropna(), bins=50, ax=axes[0, i])
                axes[0, i].set_title(col)
                sns.boxplot(y=df[col].dropna(), ax=axes[1, i])
            fig.tight_layout()
            fig.savefig(config.figures_dir / "key_variables_distribution.png", dpi=160)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df.assign(month=df[date].dt.month), x="month", y=target, ax=ax)
        ax.set_title("PM2.5 par mois")
        fig.tight_layout()
        fig.savefig(config.figures_dir / "pm25_by_month.png", dpi=160)
        plt.close(fig)

        corr = df[variables].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        fig.tight_layout()
        fig.savefig(config.figures_dir / "correlation_heatmap.png", dpi=160)
        plt.close(fig)

    temporal = temporal_diagnostics(df, target, date, station)
    temporal.to_csv(config.output_dir / "Exploratory_temporal.csv", index=False)
    data, feature_names = create_features(df, target, date, station, predictors)
    leakage_check = check_no_target_leakage(data, target, feature_names)
    dev, future_test = split_development_test(data, date, config.test_start)

    temporal_folds = make_temporal_folds(dev, date)
    spatial_folds = make_spatial_folds(dev, station)
    random_folds = make_random_folds(dev)
    save_folds(dev, temporal_folds, "folds_temporal.json", config.folds_dir)
    save_folds(dev, spatial_folds, "folds_spatial.json", config.folds_dir)
    save_folds(dev, random_folds, "folds_standard.json", config.folds_dir)

    # Demonstrate train-only fitting on the first temporal fold.
    train_idx, val_idx = temporal_folds[0]
    pipeline = make_preprocessing_pipeline()
    x_train = pipeline.fit_transform(dev.iloc[train_idx][feature_names])
    x_val = pipeline.transform(dev.iloc[val_idx][feature_names])

    data.to_csv(config.output_dir / "data_features.csv", index=False)
    dev.to_csv(config.output_dir / "development_data.csv", index=False)
    future_test.to_csv(config.output_dir / "future_test_data.csv", index=False)
    with open(config.output_dir / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump({"features": feature_names, "target": target,
                   "date": date, "station": station,
                   "predictors": predictors, "test_start": config.test_start,
                   "group": config.group}, f, indent=2, ensure_ascii=False)
    save_preprocessed(df, audit, config.output_dir)

    summary = {
        "source": str(source), "raw_shape": exploration["shape"],
        "clean_shape": df.shape, "development_rows": len(dev),
        "future_test_rows": len(future_test), "n_features": len(feature_names),
        "n_stations": int(df[station].nunique()),
        "date_min": str(df[date].min().date()), "date_max": str(df[date].max().date()),
        "first_fold_train_shape": list(x_train.shape),
        "first_fold_validation_shape": list(x_val.shape),
        "max_feature_target_abs_corr": leakage_check["max_abs_correlation"]
    }
    with open(config.output_dir / "part_a_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return {"cleaned": df, "features": data, "development": dev,
            "future_test": future_test, "feature_names": feature_names,
            "audit": audit, "temporal_diagnostics": temporal,
            "summary": summary}
