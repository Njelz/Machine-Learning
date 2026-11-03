import csv
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import TrainQuantileClipper


NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def add_derived_features(frame):
    result = frame.copy()
    result["capital_net"] = (
        result["capital-gain"] - result["capital-loss"]
    )
    result["hours_per_age"] = (
        result["hours-per-week"] / (result["age"] - 16)
    )
    return result


def make_preprocessor(numeric_columns):
    numeric_pipeline = Pipeline([
        ("clip", TrainQuantileClipper()),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])


def main():
    random_state = 42
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    adult = fetch_openml(name="adult", version=2, as_frame=True)
    X = adult.data.copy()
    y = adult.target.astype(str).str.strip()

    # Split before cleaning; keep the test rows intact.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=random_state,
    )

    # Remove exact duplicates from the training split only.
    train_frame = X_train.copy()
    train_frame["income"] = y_train.to_numpy()
    before = len(train_frame)
    train_frame = train_frame.drop_duplicates()
    removed = before - len(train_frame)

    y_train = (train_frame.pop("income") == ">50K").astype(int)
    X_train = train_frame

    # Convert the target to 0/1. The positive class is income >50K.
    y_test = (y_test == ">50K").astype(int)

    print(f"Training rows before duplicate removal: {before}")
    print(f"Training duplicates removed: {removed}")
    print(f"Training rows after duplicate removal: {len(X_train)}")
    print(f"Test rows kept untouched: {len(X_test)}")

    results = []
    fitted_engineered_preprocessor = None

    for use_derived in (False, True):
        if use_derived:
            train_features = add_derived_features(X_train)
            test_features = add_derived_features(X_test)
            numeric_columns = NUMERIC_COLUMNS + [
                "capital_net",
                "hours_per_age",
            ]
            model_name = "Logistic Regression with derived features"
        else:
            train_features = X_train.copy()
            test_features = X_test.copy()
            numeric_columns = NUMERIC_COLUMNS
            model_name = "Logistic Regression without derived features"

        preprocessor = make_preprocessor(numeric_columns)

        # Fit preprocessing on training only; transform test with those fitted rules.
        train_ready = preprocessor.fit_transform(train_features)
        test_ready = preprocessor.transform(test_features)

        model = LogisticRegression(max_iter=1000, random_state=random_state)
        model.fit(train_ready, y_train)
        predictions = model.predict(test_ready)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        print(f"\n{model_name}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 score: {f1:.4f}")

        results.append([model_name, f"{accuracy:.6f}", f"{f1:.6f}"])

        if use_derived:
            fitted_engineered_preprocessor = preprocessor

    with (results_dir / "baseline_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["model", "accuracy", "f1"])
        writer.writerows(results)

    joblib.dump(
        fitted_engineered_preprocessor,
        results_dir / "preprocessor.joblib",
    )

    print(f"\nSaved comparison to: {results_dir / 'baseline_comparison.csv'}")
    print(f"Saved fitted preprocessor to: {results_dir / 'preprocessor.joblib'}")


if __name__ == "__main__":
    main()
