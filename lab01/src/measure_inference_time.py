import csv
import statistics
import time
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

sample = X_test[0:1]
results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    # One untimed prediction warms up the model.
    model.predict(sample)

    # Time 100 predictions of the same single example.
    times_seconds = []
    for _ in range(100):
        start_time = time.perf_counter()
        model.predict(sample)
        times_seconds.append(time.perf_counter() - start_time)

    median_ms = statistics.median(times_seconds) * 1000
    results.append((name, median_ms))
    print(f"{name}: median inference = {median_ms:.6f} ms")

output_file = Path(__file__).parent.parent / "results" / "inference_time.csv"

with output_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["model", "median_inference_time_ms"])
    writer.writerows(results)

print(f"Saved results to: {output_file}")
