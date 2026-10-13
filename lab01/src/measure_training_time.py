import csv
import random
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

random.seed(42)
np.random.seed(42)

# Use the same dataset and 70/30 split as the baseline experiment.
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

# A factory creates a fresh model for every fit.
def make_models():
    return {
        "Logistic Regression (standardized)": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42),
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }


results = []

for name, make_model in make_models().items():
    # Warm-up run: do not include it in the timing results.
    make_model().fit(X_train, y_train)

    # Measure five fresh model fits.
    times_seconds = []
    for run_number in range(1, 6):
        model = make_model()

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed_seconds = time.perf_counter() - start_time

        times_seconds.append(elapsed_seconds)
        print(f"{name}, run {run_number}: {elapsed_seconds:.6f} seconds")

    median_seconds = statistics.median(times_seconds)
    results.append((name, median_seconds))
    print(f"{name}, median: {median_seconds:.6f} seconds\n")

# Save the results in lab01/results/training_time.csv.
output_file = Path(__file__).parent.parent / "results" / "training_time.csv"
with output_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["model", "median_training_time_seconds"])
    for name, median_seconds in results:
        writer.writerow([name, f"{median_seconds:.6f}"])

print(f"Saved results to: {output_file}")
