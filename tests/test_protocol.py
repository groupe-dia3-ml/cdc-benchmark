import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import average_precision_score

from src.data_loader import validate_frame, verify_dataset
from src.evaluate import bootstrap_intervals, choose_threshold, evaluate_probabilistic_predictions
from src.models import get_simple_models, selection_models
from src.preprocessing import ALL_PREDICTORS, split_data


def survey(n=200):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({name: rng.integers(0, 2, n) for name in ALL_PREDICTORS})
    for name in ["Age", "Education", "Income", "GenHlth"]:
        df[name] += 1
    df["BMI"] = rng.integers(18, 40, n)
    df["Diabetes_binary"] = np.tile([0, 0, 0, 1], n // 4)
    df["ID"] = np.arange(n)
    return df


def test_split_is_disjoint_reproducible_and_excludes_metadata():
    df = survey()
    first = split_data(df)
    second = split_data(df)
    x_train, x_test, y_train, y_test = first
    assert set(x_train.index).isdisjoint(x_test.index)
    assert len(x_train) + len(x_test) == len(df)
    assert set(x_train) == set(ALL_PREDICTORS)
    assert y_train.mean() == y_test.mean() == 0.25
    for a, b in zip(first, second, strict=True):
        assert a.equals(b)


@pytest.mark.parametrize(
    "column,value", [("Age", 14), ("BMI", np.nan), ("MentHlth", 31), ("Diabetes_binary", 2)]
)
def test_schema_rejects_bad_values(column, value):
    df = survey().astype(float)
    df.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_frame(df, exact_export=False)


def test_checksum_is_enforced(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("ID,Diabetes_binary\n0,1\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_dataset(path)


def test_repeated_profiles_are_retained():
    df = survey()
    df.loc[1, ALL_PREDICTORS] = df.loc[0, ALL_PREDICTORS]
    validate_frame(df, exact_export=False)
    parts = split_data(df)
    assert len(parts[0]) + len(parts[1]) == len(df)


def test_pipeline_scaler_sees_only_training_rows():
    x_train, x_test, y_train, _ = split_data(survey())
    model = clone(get_simple_models()["logistic_C1"]).fit(x_train, y_train)
    np.testing.assert_allclose(model.named_steps["scale"].mean_, x_train.mean())
    x_test.loc[:, "BMI"] = 90
    model.predict_proba(x_test)
    np.testing.assert_allclose(model.named_steps["scale"].mean_, x_train.mean())
    assert all("select" in pipeline.named_steps for pipeline in selection_models().values())


def test_threshold_and_ap_semantics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.1, 0.2, 0.3])
    threshold = choose_threshold(y, p)
    assert threshold == 0.2
    metrics = evaluate_probabilistic_predictions(y, p, threshold)
    assert metrics["f1"] == 1
    assert metrics["average_precision"] == average_precision_score(y, p)
    assert metrics["tn"] == metrics["tp"] == 2
    intervals = bootstrap_intervals(y, p, threshold, repeats=10)
    assert intervals["roc_auc"] == [1, 1]


def test_single_class_metrics_and_invalid_probabilities():
    metrics = evaluate_probabilistic_predictions([0, 0], np.array([0.1, 0.2]))
    assert metrics["roc_auc"] is None
    assert metrics["recall"] is None
    with pytest.raises(ValueError):
        evaluate_probabilistic_predictions([0, 1], [0.1, np.inf])
