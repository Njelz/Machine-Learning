import csv
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

random.seed(42)
np.random.seed(42)

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

models = {
    "Logistic Regression (standardized)": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    ),
}

results = []
results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    results.append((name, accuracy))

    model_file = results_dir / f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
    joblib.dump(model, model_file)

    size_bytes = model_file.stat().st_size
    print(f"{name}: test accuracy = {accuracy:.4f}")
    print(f"{name}: saved model size = {size_bytes} bytes ({size_bytes / 1024:.2f} KB)")

output_file = results_dir / "baseline_accuracy.csv"

with output_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["model", "test_accuracy"])
    for name, accuracy in results:
        writer.writerow([name, f"{accuracy:.4f}"])

print(f"Saved accuracy results to: {output_file}")
