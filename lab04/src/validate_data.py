from pathlib import Path

import pandas as pd
from pandas.api.types import is_integer_dtype
from sklearn.datasets import fetch_openml


CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "income",
]

INTEGER_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}


def main():
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    raw_data = adult.data.copy()
    raw_data["income"] = adult.target

    issues = []
    report = []

    report.append(f"Rows: {len(raw_data)}")
    report.append(f"Columns: {len(raw_data.columns)}")
    report.append("\nColumn checks:")

    expected_columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    print("Actual columns: ", list(raw_data.columns))
    print("Expected columns: ", expected_columns)

    if list(raw_data.columns) != expected_columns:
        issues.append("Column names or order do not match the expected schema.")
        report.append("  Column order: FAIL")
    else:
        report.append("  Column order: PASS")

    report.append("\nData types:")
    for column in INTEGER_COLUMNS:
        passed = column in raw_data and is_integer_dtype(raw_data[column].dtype)
        report.append(
            f"  {column}: {raw_data[column].dtype} "
            f"({'PASS' if passed else 'FAIL; expected integer'})"
        )
        if not passed:
            issues.append(f"{column} should have an integer type.")

    for column in CATEGORICAL_COLUMNS:
        passed = (
            column in raw_data
            and isinstance(raw_data[column].dtype, pd.CategoricalDtype)
        )
        report.append(
            f"  {column}: {raw_data[column].dtype} "
            f"({'PASS' if passed else 'FAIL; expected category'})"
        )
        if not passed:
            issues.append(f"{column} should have category type.")

    report.append("\nMissing values and '?' markers:")
    for column in raw_data.columns:
        null_count = int(raw_data[column].isna().sum())
        question_count = int(
            raw_data[column].astype("string").eq("?").sum()
        )
        report.append(
            f"  {column}: nulls={null_count}, '?' markers={question_count}"
        )

    report.append("\nRange checks:")
    for column, (minimum, maximum) in RANGES.items():
        values = raw_data[column]
        violations = int(((values < minimum) | (values > maximum)).sum())
        passed = violations == 0
        report.append(
            f"  {column} [{minimum}, {maximum}]: "
            f"{violations} violations ({'PASS' if passed else 'FAIL'})"
        )
        if not passed:
            issues.append(f"{column} has {violations} range violations.")

    exact_duplicates = int(raw_data.duplicated().sum())
    report.append(f"\nExact duplicate rows: {exact_duplicates}")
    report.append("\nIncome label counts:")
    report.append(raw_data["income"].value_counts(dropna=False).to_string())

    report.append("\nValidation summary:")
    if issues:
        report.append(f"Found {len(issues)} issue(s):")
        report.extend(f"  - {issue}" for issue in issues)
    else:
        report.append("No schema or range violations found.")

    output = Path(__file__).parent.parent / "results" / "validation_report.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print(f"\nSaved validation report to: {output}")
    print("\nNo rows were removed or changed.")


if __name__ == "__main__":
    main()
