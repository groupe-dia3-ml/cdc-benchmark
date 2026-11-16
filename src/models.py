"""Small, predeclared search budget with every learned transform in a pipeline."""

from functools import partial

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def logistic(c=1.0, **kwargs):
    return LogisticRegression(C=c, max_iter=2000, random_state=42, **kwargs)


def get_simple_models():
    return {
        "majority": Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
        **{
            f"logistic_C{c:g}": Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("clf", logistic(c)),
                ]
            )
            for c in (0.1, 1.0, 10.0)
        },
        "logistic_balanced": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", logistic(class_weight="balanced")),
            ]
        ),
        "tree_depth3": Pipeline([("clf", DecisionTreeClassifier(max_depth=3, random_state=42))]),
        "tree_depth5": Pipeline([("clf", DecisionTreeClassifier(max_depth=5, random_state=42))]),
        "gaussian_nb": Pipeline([("scale", StandardScaler()), ("clf", GaussianNB())]),
        "random_forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=12,
                        min_samples_leaf=10,
                        n_jobs=2,
                        random_state=42,
                    ),
                )
            ]
        ),
        **{
            f"gradient_boosting_leaf{leaves}": Pipeline(
                [
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_iter=150,
                            max_leaf_nodes=leaves,
                            learning_rate=0.08,
                            l2_regularization=1.0,
                            early_stopping=False,
                            random_state=42,
                        ),
                    )
                ]
            )
            for leaves in (15, 31)
        },
    }


def selection_models():
    # Every CDC predictor is integer-coded; treat it as discrete for MI.
    mi = partial(mutual_info_classif, discrete_features=True, random_state=42)
    models = {}
    for k in (5, 10):
        models[f"mi_k{k}"] = Pipeline(
            [
                ("select", SelectKBest(mi, k=k)),
                ("scale", StandardScaler()),
                ("clf", logistic()),
            ]
        )
        models[f"rfe_k{k}"] = Pipeline(
            [
                ("scale", StandardScaler()),
                ("select", RFE(logistic(), n_features_to_select=k, step=4)),
                ("clf", logistic()),
            ]
        )
    for c in (0.001, 0.01):
        models[f"l1_C{c:g}"] = Pipeline(
            [
                ("scale", StandardScaler()),
                ("select", SelectFromModel(logistic(c, solver="liblinear", l1_ratio=1))),
                ("clf", logistic()),
            ]
        )
    return models


def candidate_models():
    candidates = [
        (group, name, model)
        for group in ("lifestyle", "enriched")
        for name, model in get_simple_models().items()
    ]
    return candidates + [("enriched", name, model) for name, model in selection_models().items()]
