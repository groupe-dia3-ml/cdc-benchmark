"""Full experiment, including a written freeze before accessing test labels."""

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from .data_loader import download_dataset, verify_dataset
from .evaluate import (
    bootstrap_intervals,
    choose_threshold,
    evaluate_probabilistic_predictions,
    get_calibration_data,
    subgroup_metrics,
)
from .models import candidate_models
from .preprocessing import ALL_PREDICTORS, LIFESTYLE_AND_DEMO, ROOT, split_data


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def selected_features(model, names):
    selector = model.named_steps.get("select")
    return list(np.asarray(names)[selector.get_support()]) if selector is not None else list(names)


def plot_eda(x, y, out):
    x.describe().T.to_csv(out / "development_describe.csv")
    x.corr().to_csv(out / "development_correlations.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    y.value_counts().sort_index().plot.bar(ax=axes[0], color=["#40678b", "#b95e40"], rot=0)
    axes[0].set(title="Development labels", xlabel="Survey status", ylabel="Respondents")
    for label, color in [(0, "#40678b"), (1, "#b95e40")]:
        axes[1].hist(
            x.loc[y == label, "BMI"], bins=35, density=True, alpha=0.5, label=str(label), color=color
        )
    axes[1].set(title="BMI by observed status", xlabel="BMI", ylabel="Density")
    axes[1].legend()
    age_rates = y.groupby(x["Age"]).mean()
    axes[2].plot(age_rates.index, age_rates.values, "o-", color="#40678b")
    axes[2].set(title="Status by age band", xlabel="Age band (1–13)", ylabel="Positive fraction")
    fig.tight_layout()
    fig.savefig(out / "eda.png", dpi=150)
    plt.close(fig)


def run(output_dir: str | Path | None = None, bootstrap_repeats: int = 200):
    with threadpool_limits(limits=2):
        return _run(Path(output_dir) if output_dir else ROOT / "results", bootstrap_repeats)


def _run(out: Path, bootstrap_repeats: int):
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    path = download_dataset()
    df, checksum = verify_dataset(path)
    x_dev, x_test, y_dev, y_test = split_data(df)
    plot_eda(x_dev, y_dev, out)
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "src").glob("*.py"))
    }
    write_json(
        out / "data_audit.json",
        {
            "source": json.loads(path.with_suffix(".provenance.json").read_text())
            if path.with_suffix(".provenance.json").exists()
            else {"sha256": checksum, "retrieval_date": "unknown (pre-existing verified file)"},
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": int(df.isna().sum().sum()),
            "target_counts": {str(k): int(v) for k, v in df["Diabetes_binary"].value_counts().items()},
            "development_rows": len(x_dev),
            "test_rows": len(x_test),
            "repeated_development_predictor_profiles": int(x_dev.duplicated().sum()),
            "duplicate_policy": "Retained: identical responses do not identify repeated participants",
            "development_index_sha256": hashlib.sha256(
                x_dev.index.to_numpy(dtype="int64").tobytes()
            ).hexdigest(),
            "test_index_sha256": hashlib.sha256(x_test.index.to_numpy(dtype="int64").tobytes()).hexdigest(),
        },
    )
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x_dev, y_dev))
    configurations, oof, summaries, fold_rows = {}, {}, [], []
    for group, name, template in candidate_models():
        key = f"{group}/{name}"
        names = LIFESTYLE_AND_DEMO if group == "lifestyle" else ALL_PREDICTORS
        x = x_dev[names]
        probabilities = np.zeros(len(y_dev))
        scores, counts, seconds = [], [], []
        for fold, (train, valid) in enumerate(folds, 1):
            model = clone(template)
            tick = perf_counter()
            model.fit(x.iloc[train], y_dev.iloc[train])
            elapsed = perf_counter() - tick
            probability = model.predict_proba(x.iloc[valid])[:, 1]
            probabilities[valid] = probability
            metrics = evaluate_probabilistic_predictions(y_dev.iloc[valid], probability)
            selected = selected_features(model, names)
            scores.append(metrics["average_precision"])
            counts.append(len(selected))
            seconds.append(elapsed)
            fold_rows.append(
                {
                    "candidate": key,
                    "fold": fold,
                    **metrics,
                    "fit_seconds": elapsed,
                    "feature_count": len(selected),
                    "features": ",".join(selected),
                }
            )
        summary = {
            "candidate": key,
            "group": group,
            "model": name,
            "cv_ap_mean": float(np.mean(scores)),
            "cv_ap_sd": float(np.std(scores, ddof=1)),
            "feature_count_mean": float(np.mean(counts)),
            "cv_fit_seconds": sum(seconds),
        }
        summaries.append(summary)
        configurations[key] = (template, names)
        oof[key] = probabilities
        print(f"{key}: CV AP {summary['cv_ap_mean']:.4f} ± {summary['cv_ap_sd']:.4f}", flush=True)
        pd.DataFrame(summaries).to_csv(out / "cv_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "cv_folds.csv", index=False)
    cv = pd.DataFrame(summaries).sort_values("cv_ap_mean", ascending=False, kind="stable")
    best = str(cv.iloc[0]["candidate"])
    finalists = {best, "enriched/majority", "enriched/logistic_C1", "lifestyle/logistic_C1"}
    for group in ("lifestyle", "enriched"):
        finalists.add(str(cv[cv["group"] == group].iloc[0]["candidate"]))
    for prefix in ("mi_", "rfe_", "l1_"):
        finalists.add(str(cv[cv["model"].str.startswith(prefix)].iloc[0]["candidate"]))
    frozen = {}
    threshold_rows = []
    for key in sorted(finalists):
        threshold = 0.5 if key.endswith("majority") else choose_threshold(y_dev, oof[key])
        frozen[key] = {
            "threshold": threshold,
            "features": configurations[key][1],
            "parameters": configurations[key][0].get_params(deep=True),
        }
        for t in sorted(set([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, threshold])):
            threshold_rows.append(
                {"candidate": key, **evaluate_probabilistic_predictions(y_dev, oof[key], t)}
            )
    pd.DataFrame(threshold_rows).to_csv(out / "development_thresholds.csv", index=False)
    # The test labels have not been used for model fitting, selection, EDA or thresholds.
    write_json(
        out / "frozen_protocol.json",
        {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": 42,
            "primary_selection_metric": "mean five-fold development average precision",
            "primary_candidate": best,
            "threshold_rule": "maximize development OOF F1; majority uses 0.5",
            "candidate_count": len(configurations),
            "finalists": frozen,
            "source_sha256": source_hashes,
            "cv_caveat": "Selection uses the same CV scores; they are optimistic, not nested estimates.",
        },
    )
    test_rows, feature_rows, coefficient_rows = [], [], []
    probabilities_by_key = {}
    for key in sorted(finalists):
        template, names = configurations[key]
        fitted = clone(template).fit(x_dev[names], y_dev)
        probability = fitted.predict_proba(x_test[names])[:, 1]
        probabilities_by_key[key] = probability
        threshold = frozen[key]["threshold"]
        for label, t in [("default", 0.5), ("development_f1", threshold)]:
            test_rows.append(
                {
                    "candidate": key,
                    "operating_point": label,
                    **evaluate_probabilistic_predictions(y_test, probability, t),
                }
            )
        selected = selected_features(fitted, names)
        feature_rows.append(
            {"candidate": key, "feature_count": len(selected), "features": ",".join(selected)}
        )
        if key == "enriched/logistic_C1":
            coef = fitted.named_steps["clf"].coef_[0]
            scales = fitted.named_steps["scale"].scale_
            coefficient_rows = [
                {
                    "feature": name,
                    "coefficient_per_sd": float(b),
                    "odds_ratio_per_sd": float(np.exp(b)),
                    "coefficient_raw_unit": float(b / s),
                }
                for name, b, s in zip(names, coef, scales, strict=True)
            ]
    metrics_table = pd.DataFrame(test_rows)
    metrics_table.to_csv(out / "test_metrics.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(out / "selected_features.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(out / "logistic_coefficients.csv", index=False)
    primary_probability = probabilities_by_key[best]
    threshold = frozen[best]["threshold"]
    intervals = bootstrap_intervals(y_test, primary_probability, threshold, repeats=bootstrap_repeats)
    write_json(
        out / "test_intervals.json",
        {
            "candidate": best,
            "repeats": bootstrap_repeats,
            "method": "95% stratified percentile bootstrap; fixed fit and observed prevalence",
            "intervals": intervals,
        },
    )
    subgroup_metrics(x_test, y_test, primary_probability, threshold).to_csv(
        out / "subgroup_metrics.csv", index=False
    )
    train_hash = pd.util.hash_pandas_object(x_dev, index=False)
    test_hash = pd.util.hash_pandas_object(x_test, index=False)
    write_json(
        out / "profile_overlap.json",
        {
            "test_profiles_also_in_development": int(test_hash.isin(train_hash).sum()),
            "test_rows": len(x_test),
            "interpretation": "Predictor-profile hash overlap is not proof of participant overlap; retained by design.",
        },
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for key in ["lifestyle/logistic_C1", "enriched/logistic_C1", best]:
        probability = probabilities_by_key[key]
        PrecisionRecallDisplay.from_predictions(y_test, probability, ax=axes[0], name=key)
        RocCurveDisplay.from_predictions(y_test, probability, ax=axes[1], name=key)
        observed, predicted = get_calibration_data(y_test, probability)
        axes[2].plot(predicted, observed, "o-", label=key)
    axes[0].axhline(y_test.mean(), color="gray", ls=":")
    axes[2].plot([0, 1], [0, 1], "--", color="gray")
    axes[2].set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        title="Quantile-bin calibration",
    )
    for ax in axes:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "test_curves.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 8))
    ordered = cv.sort_values("cv_ap_mean")
    ax.barh(ordered["candidate"], ordered["cv_ap_mean"], xerr=ordered["cv_ap_sd"], color="#40678b")
    ax.set(xlabel="Development average precision (mean ± fold SD)", title="All predeclared candidates")
    fig.tight_layout()
    fig.savefig(out / "cv_comparison.png", dpi=150)
    plt.close(fig)
    manifest = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": perf_counter() - started,
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "matplotlib"]
        },
        "source_sha256": source_hashes,
        "data_sha256": checksum,
        "seed": 42,
        "primary_candidate": best,
        "test_threshold": threshold,
    }
    write_json(out / "run_manifest.json", manifest)
    print(f"Finished. Primary model: {best}; outputs: {out}", flush=True)
    return manifest


if __name__ == "__main__":
    run()
