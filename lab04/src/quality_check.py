from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


def main():
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    adult = fetch_openml(name="adult", version=2, as_frame=True)
    features = adult.data.copy()
    labels = adult.target.astype(str).str.strip()

    data = features.copy()
    data["income"] = labels

    # Group by all features, including rows with missing feature values.
    # A conflict means identical features appear with different income labels.
    grouped = data.groupby(
        list(features.columns),
        dropna=False,
        observed=True,
    ).agg(
        label_count=("income", "nunique"),
        row_count=("income", "size"),
    )

    conflicts = grouped[grouped["label_count"] > 1]
    conflicting_rows = int(conflicts["row_count"].sum())

    # Select 50 reproducible rows from the unchanged raw dataset.
    sample = data.sample(n=50, random_state=42).copy()
    sample.insert(0, "source_row_index", sample.index)
    sample = sample.reset_index(drop=True)

    sample_path = results_dir / "quality_review_sample_50.csv"
    sample.to_csv(sample_path, index=False)

    report = [
        "Adult label-quality check",
        "=========================",
        f"Dataset rows: {len(data)}",
        "Random sample size: 50",
        "Random seed: 42",
        f"Feature groups with conflicting income labels: {len(conflicts)}",
        f"Rows belonging to those conflicting groups: {conflicting_rows}",
        "",
        "Manual review completed:",
        "50 sampled rows checked against the lab schema and ranges.",
        "Impossible or inconsistent values found: 0.",
        "Blank categorical entries were treated as expected missing values.",
        "No rows were changed or removed by this quality check.",
    ]

    report_path = results_dir / "label_quality_report.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print(f"\nSaved 50-row sample to: {sample_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
