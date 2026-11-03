from sklearn.datasets import fetch_openml


def main():
    # Fetch the dataset as a pandas DataFrame.
    adult = fetch_openml(name="adult", version=2, as_frame=True)

    # Preserve the raw feature table; add the target as the 15th column.
    raw_data = adult.data.copy()
    raw_data["income"] = adult.target

    print(f"Rows: {len(raw_data)}")
    print(f"Columns: {len(raw_data.columns)}")
    print(f"Column names: {list(raw_data.columns)}")
    print("\nData types:")
    print(raw_data.dtypes)
    print("\nMissing-value markers ('?') by column:")
    print((raw_data.astype("string") == "?").sum())


if __name__ == "__main__":
    main()
