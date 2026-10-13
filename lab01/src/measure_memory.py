import gc
from pathlib import Path

import psutil
from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MIB = 1024 * 1024


def make_model(name):
    if name == "Logistic Regression (standardized)":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42),
        )

    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )

    raise ValueError(f"Unknown model: {name}")


def fit_model_and_get_starting_memory(name, X_train, y_train):
    """Called by memory_profiler; report RSS just before fitting."""
    starting_rss_mib = psutil.Process().memory_info().rss / MIB
    model = make_model(name)
    model.fit(X_train, y_train)
    return starting_rss_mib


def train_and_predict_100_times(model, sample):
    """Called by memory_profiler to measure inference memory."""
    starting_rss_mib = psutil.Process().memory_info().rss / MIB

    for _ in range(100):
        model.predict(sample)

    return starting_rss_mib


def main():
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )
    sample = X_test[0:1]

    model_names = [
        "Logistic Regression (standardized)",
        "Random Forest",
    ]
    results = []

    for name in model_names:
        gc.collect()

        # Measure peak process memory while fitting.
        peak_fit_mib, starting_fit_mib = memory_usage(
            (
                fit_model_and_get_starting_memory,
                (name, X_train, y_train),
                {},
            ),
            interval=0.01,
            max_usage=True,
            retval=True,
            backend="psutil",
        )

        # Fit a model in this script so we can measure prediction memory separately.
        model = make_model(name)
        model.fit(X_train, y_train)

        # Warm-up prediction, not included in the memory measurement.
        model.predict(sample)
        gc.collect()

        # Measure peak process memory during 100 predictions.
        peak_inference_mib, starting_inference_mib = memory_usage(
            (
                train_and_predict_100_times,
                (model, sample),
                {},
            ),
            interval=0.01,
            max_usage=True,
            retval=True,
            backend="psutil",
        )

        fit_increase_mib = max(0, peak_fit_mib - starting_fit_mib)
        inference_increase_mib = max(
            0, peak_inference_mib - starting_inference_mib
        )

        print(f"\n{name}")
        print(
            f"  Training: peak RSS {peak_fit_mib:.2f} MiB; "
            f"increase over start {fit_increase_mib:.2f} MiB"
        )
        print(
            f"  Inference: peak RSS {peak_inference_mib:.2f} MiB; "
            f"increase over start {inference_increase_mib:.2f} MiB"
        )

        results.append(
            (
                name,
                peak_fit_mib,
                fit_increase_mib,
                peak_inference_mib,
                inference_increase_mib,
            )
        )

    output_file = (
        Path(__file__).parent.parent / "results" / "memory_usage.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as file:
        import csv

        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "training_peak_rss_mib",
                "training_increase_mib",
                "inference_peak_rss_mib",
                "inference_increase_mib",
            ]
        )
        writer.writerows(results)

    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()
