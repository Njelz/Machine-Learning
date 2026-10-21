import csv
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def forward(X, p):
    z1 = X @ p["W1"] + p["b1"]
    a1 = np.maximum(0, z1)
    z2 = a1 @ p["W2"] + p["b2"]
    a2 = np.maximum(0, z2)
    logits = a2 @ p["W3"] + p["b3"]

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)

    cache = (X, z1, a1, z2, a2)
    return probabilities, cache


def backward(Y, p, cache, probabilities):
    X, z1, a1, z2, a2 = cache
    d3 = (probabilities - Y) / len(X)

    dW3 = a2.T @ d3
    db3 = d3.sum(axis=0)

    d2 = (d3 @ p["W3"].T) * (z2 > 0)
    dW2 = a1.T @ d2
    db2 = d2.sum(axis=0)

    d1 = (d2 @ p["W2"].T) * (z1 > 0)
    dW1 = X.T @ d1
    db1 = d1.sum(axis=0)

    return dW1, db1, dW2, db2, dW3, db3


X, y = load_digits(return_X_y=True)
X = X / 16.0
Y = np.eye(10)[y]

X_train, _, Y_train, _, = train_test_split(
    X, Y, test_size=0.30, stratify=y, random_state=42
)

batch_X = X_train[:32]
batch_Y = Y_train[:32]
results = []

for width in (16, 64):
    rng = np.random.default_rng(42)
    p = {
        "W1": rng.normal(0, 0.01, (64, width)),
        "b1": np.zeros(width),
        "W2": rng.normal(0, 0.01, (width, width)),
        "b2": np.zeros(width),
        "W3": rng.normal(0, 0.01, (width, 10)),
        "b3": np.zeros(10),
    }

    # Warm-up
    probabilities, cache = forward(batch_X, p)
    backward(batch_Y, p, cache, probabilities)

    forward_times = []
    for _ in range(100):
        start = time.perf_counter()
        probabilities, cache = forward(batch_X, p)
        forward_times.append(time.perf_counter() - start)

    # Use a fresh, untimed forward result for backward timing.
    probabilities, cache = forward(batch_X, p)
    backward_times = []
    for _ in range(100):
        start = time.perf_counter()
        backward(batch_Y, p, cache, probabilities)
        backward_times.append(time.perf_counter() - start)

    forward_ms = statistics.median(forward_times) * 1000
    backward_ms = statistics.median(backward_times) * 1000

    print(
        f"Width {width}: forward median={forward_ms:.6f} ms; "
        f"backward median={backward_ms:.6f} ms"
    )
    results.append([width, forward_ms, backward_ms])

output = Path(__file__).parent.parent / "results" / "pass_times.csv"
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["width", "forward_median_ms", "backward_median_ms"])
    writer.writerows(results)

print(f"Saved results to: {output}")
