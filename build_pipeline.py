"""
Run once from the loan-campaign-api/ directory to build and save the sklearn Pipeline.

    python build_pipeline.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path("../project2-loan-campaign/loan-campaign/data/Loan_Modelling.csv")
OUT_PATH = Path("models/pipeline.pkl")

edu_map = {1: "Undergrad", 2: "Graduate", 3: "Advanced_Professional"}
numeric_cols = [
    "Age", "Income", "Family", "CCAvg", "Mortgage",
    "Securities_Account", "CD_Account", "Online", "CreditCard",
]


def load_and_prepare(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df = df.drop(columns=["ID", "Experience"])
    df["Education"] = df["Education"].map(edu_map)
    df["ZIPCode"] = df["ZIPCode"].astype(str).str[:2]
    X = df.drop(columns=["Personal_Loan"])
    y = df["Personal_Loan"]
    return X, y


def find_best_alpha(preprocessor, X_train, y_train, X_test, y_test) -> float:
    X_tr = preprocessor.fit_transform(X_train)
    X_te = preprocessor.transform(X_test)

    path = DecisionTreeClassifier(random_state=1).cost_complexity_pruning_path(X_tr, y_train)
    alphas = abs(path.ccp_alphas[:-1])

    best_alpha, best_f1 = 0.0, 0.0
    for alpha in alphas:
        clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=1)
        clf.fit(X_tr, y_train)
        score = f1_score(y_test, clf.predict(X_te))
        if score > best_f1:
            best_f1, best_alpha = score, alpha

    print(f"Best alpha: {best_alpha:.8f}  |  Test F1 during search: {best_f1:.4f}")
    return best_alpha


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("zip", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), ["ZIPCode"]),
        ("edu", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), ["Education"]),
        ("num", "passthrough", numeric_cols),
    ])


def main():
    print("Loading data...")
    X, y = load_and_prepare(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=1
    )

    print("Finding best ccp_alpha...")
    preprocessor = build_preprocessor()
    best_alpha = find_best_alpha(preprocessor, X_train, y_train, X_test, y_test)

    print("Fitting final pipeline on full dataset...")
    final_preprocessor = build_preprocessor()
    pipeline = Pipeline([
        ("preprocessor", final_preprocessor),
        ("classifier", DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=1)),
    ])
    pipeline.fit(X, y)

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
