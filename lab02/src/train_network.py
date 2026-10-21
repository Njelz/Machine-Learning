import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def forward(X, params):
    z1 = X @ params["W1"] + params["b1"]
    a1 = np.maximum(0, z1)

    z2 = a1 @ params["W2"] + params["b2"]
    a2 = np.maximum(0, z2)

    probabilities = softmax(a2 @ params["W3"] + params["b3"])
    cache = (X, z1, a1, z2, a2)
    return probabilities, cache


def cross_entropy(probabilities, Y):
    safe = np.clip(probabilities, 1e-12, 1.0)
    return -np.mean(np.sum(Y * np.log(safe), axis=1))


def train_batch(X, Y, params, learning_rate):
    probabilities, cache = forward(X, params)
    inputs, z1, a1, z2, a2 = cache
    n = len(X)

    d3 = (probabilities - Y) / n
    gradients = {}
    gradients["W3"] = a2.T @ d3
    gradients["b3"] = d3.sum(axis=0)

    d2 = (d3 @ params["W3"].T) * (z2 > 0)
    gradients["W2"] = a1.T @ d2
    gradients["b2"] = d2.sum(axis=0)

    d1 = (d2 @ params["W2"].T) * (z1 > 0)
    gradients["W1"] = inputs.T @ d1
    gradients["b1"] = d1.sum(axis=0)

    for name in params:
        params[name] -= learning_rate * gradients[name]


def make_params(width, rng):
    return {
        "W1": rng.normal(0, 0.01, size=(64, width)),
        "b1": np.zeros(width),
        "W2": rng.normal(0, 0.01, size=(width, width)),
        "b2": np.zeros(width),
        "W3": rng.normal(0, 0.01, size=(width, 10)),
        "b3": np.zeros(10),
    }


def main():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Y = np.eye(10)[y]

    X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(
        X, Y, y, test_size=0.30, stratify=y, random_state=42
    )

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    log_rows = []
    summary_rows = []
    curves = {}

    for width in (16, 64):
        for learning_rate in (0.001, 0.01, 0.1):
            # Reset seed so each configuration is reproducible.
            rng = np.random.default_rng(42)
            params = make_params(width, rng)
            curve = []

            for epoch in range(1, 51):
                order = rng.permutation(len(X_train))

                for start in range(0, len(order), 32):
                    batch_indices = order[start : start + 32]
                    train_batch(
                        X_train[batch_indices],
                        Y_train[batch_indices],
                        params,
                        learning_rate,
                    )

                train_probs, _ = forward(X_train, params)
                epoch_loss = cross_entropy(train_probs, Y_train)
                curve.append(epoch_loss)

                log_rows.append(
                    [width, learning_rate, epoch, epoch_loss]
                )
                print(
                    f"width={width}, lr={learning_rate}, "
                    f"epoch={epoch:02d}/50, loss={epoch_loss:.4f}"
                )

            test_probs, _ = forward(X_test, params)
            predictions = np.argmax(test_probs, axis=1)
            accuracy = np.mean(predictions == y_test)

            curves[(width, learning_rate)] = curve
            summary_rows.append([width, learning_rate, accuracy])
            print(
                f"FINAL width={width}, lr={learning_rate}: "
                f"test accuracy={accuracy:.4f}\n"
            )

    with (results_dir / "training_log.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["width", "learning_rate", "epoch", "train_loss"])
        writer.writerows(log_rows)

    with (results_dir / "test_accuracy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["width", "learning_rate", "test_accuracy"])
        writer.writerows(summary_rows)

    for (width, learning_rate), curve in curves.items():
        plt.plot(
            range(1, 51),
            curve,
            label=f"W={width}, lr={learning_rate}",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training cross-entropy loss")
    plt.title("Lab 2: Training loss by width and learning rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "loss_curves.png", dpi=150)
    plt.close()

    print(f"Saved logs, accuracy, and plot in: {results_dir}")


if __name__ == "__main__":
    main()
