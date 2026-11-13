import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_simple_models():
    """
    Returns a dictionary of simple, interpretable candidate models.
    All models support predict_proba() to estimate P(Y=1 | X).
    """
    models = {
        "Logistic_Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Logistic_Regression_Balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
        ]),
        "Decision_Tree_d3": DecisionTreeClassifier(max_depth=3, random_state=42),
        "Decision_Tree_d5": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Gaussian_Naive_Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB())
        ]),
        "KNN_k15": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15))
        ])
    }
    return models
