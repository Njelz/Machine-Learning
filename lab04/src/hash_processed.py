import hashlib
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from train_baselines import (
    NUMERIC_COLUMNS,
    add_derived_features,
    make_preprocessor,
)


def hash_array(hasher, name, values, dtype):
    array = np.asarray(values, dtype=dtype)
    hasher.update(name.encode("utf-8") + b"\0")
    hasher.update(np.asarray(array.shape, dtype="<i8").tobytes())
    hasher.update(array.tobytes(order="C"))


def hash_matrix(hasher, name, matrix):
    hasher.update(name.encode("utf-8") + b"\0")

    csr = sparse.csr_matrix(matrix, dtype=np.float64)
    csr.sum_duplicates()
    csr.sort_indices()

    hash_array(hasher, name + ":shape", csr.shape, "<i8")
    hash_array(hasher, name + ":indptr", csr.indptr, "<i8")
    hash_array(hasher, name + ":indices", csr.indices, "<i8")
    hash_array(hasher, name + ":data", csr.data, "<f8")


def main():
    print("Loading Adult dataset...")
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    X = adult.data.copy()
    y = adult.target.astype(str).str.strip()

    print("Creating reproducible stratified split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    print("Removing exact duplicates from training split only...")
    train_frame = X_train.copy()
    train_frame["income"] = y_train.to_numpy()
    train_frame = train_frame.drop_duplicates()

    y_train = (train_frame.pop("income") == ">50K").astype(np.uint8)
    X_train = train_frame
    y_test = (y_test == ">50K").astype(np.uint8)

    print("Fitting preprocessing on training split...")
    X_train = add_derived_features(X_train)
    X_test = add_derived_features(X_test)
    numeric_columns = NUMERIC_COLUMNS + ["capital_net", "hours_per_age"]

    preprocessor = make_preprocessor(numeric_columns)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("Hashing processed train and test data...")
    hasher = hashlib.sha256()
    hasher.update(b"lab04-processed-adult-v1\0")

    hash_matrix(hasher, "X_train", X_train_processed)
    hash_array(hasher, "y_train", y_train.to_numpy(), "u1")
    hash_matrix(hasher, "X_test", X_test_processed)
    hash_array(hasher, "y_test", y_test.to_numpy(), "u1")

    digest = hasher.hexdigest()
    output = Path(__file__).parent.parent / "results" / "processed_sha256.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "SHA256 of transformed train/test features and labels\n"
        "Preprocessing fit on training split only; training duplicates removed only.\n"
        f"SHA256: {digest}\n",
        encoding="utf-8",
    )

    print(f"Processed train shape: {X_train_processed.shape}")
    print(f"Processed test shape:  {X_test_processed.shape}")
    print(f"SHA256: {digest}")
    print(f"Saved hash to: {output}")


if __name__ == "__main__":
    main()
