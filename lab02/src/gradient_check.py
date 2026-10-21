from pathlib import Path

import mpmath as mp
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def relu(values):
    return np.maximum(0, values)


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def forward(X, params):
    z1 = X @ params["W1"] + params["b1"]
    a1 = relu(z1)

    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)

    logits = a2 @ params["W3"] + params["b3"]
    probabilities = softmax(logits)

    cache = {
        "X": X,
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "a2": a2,
    }
    return probabilities, cache


def loss_and_gradients(X, Y, params):
    probabilities, cache = forward(X, params)
    sample_count = X.shape[0]

    safe_probabilities = np.clip(probabilities, 1e-12, 1.0)
    loss = -np.mean(
        np.sum(Y * np.log(safe_probabilities), axis=1)
    )

    # Softmax plus cross-entropy gradient.
    d_logits = (probabilities - Y) / sample_count

    dW3 = cache["a2"].T @ d_logits
    db3 = np.sum(d_logits, axis=0)

    d_a2 = d_logits @ params["W3"].T
    d_z2 = d_a2 * (cache["z2"] > 0)

    dW2 = cache["a1"].T @ d_z2
    db2 = np.sum(d_z2, axis=0)

    d_a1 = d_z2 @ params["W2"].T
    d_z1 = d_a1 * (cache["z1"] > 0)

    dW1 = cache["X"].T @ d_z1
    db1 = np.sum(d_z1, axis=0)

    gradients = {
        "W1": dW1,
        "W2": dW2,
        "W3": dW3,
        "b1": db1,
        "b2": db2,
        "b3": db3,
    }
    return loss, gradients


def high_precision_loss(X, labels, params, changed_weight=None, delta=None):
    """Compute summed cross-entropy with higher precision for the numerical check."""
    with mp.workdps(40):
        weights = {}
        biases = {}

        for name in ("W1", "W2", "W3"):
            weights[name] = [
                [mp.mpf(str(float(value))) for value in row]
                for row in params[name]
            ]

        for name in ("b1", "b2", "b3"):
            biases[name] = [
                mp.mpf(str(float(value))) for value in params[name]
            ]

        if changed_weight is not None:
            name, index = changed_weight
            row, column = index
            weights[name][row][column] += delta

        total_loss = mp.mpf("0")
        zero = mp.mpf("0")

        for example, label in zip(X, labels):
            x = [mp.mpf(str(float(value))) for value in example]

            z1 = [
                sum(
                    (x[k] * weights["W1"][k][j] for k in range(64)),
                    zero,
                ) + biases["b1"][j]
                for j in range(len(biases["b1"]))
            ]
            a1 = [max(zero, value) for value in z1]

            width = len(biases["b2"])
            z2 = [
                sum(
                    (a1[k] * weights["W2"][k][j] for k in range(width)),
                    zero,
                ) + biases["b2"][j]
                for j in range(width)
            ]
            a2 = [max(zero, value) for value in z2]

            logits = [
                sum(
                    (a2[k] * weights["W3"][k][j] for k in range(width)),
                    zero,
                ) + biases["b3"][j]
                for j in range(10)
            ]

            largest_logit = max(logits)
            log_sum_exp = largest_logit + mp.log(
                sum((mp.exp(value - largest_logit) for value in logits), zero)
            )
            total_loss += log_sum_exp - logits[int(label)]

        return total_loss


def same_relu_regions(X, params, weight_name, index, epsilon):
    """Only check weights whose +/- nudges don't cross a ReLU boundary."""
    _, base_cache = forward(X, params)
    base_z1 = base_cache["z1"] > 0
    base_z2 = base_cache["z2"] > 0

    for direction in (1.0, -1.0):
        changed = {name: value.copy() for name, value in params.items()}
        changed[weight_name][index] += direction * epsilon
        _, cache = forward(X, changed)

        if not np.array_equal(cache["z1"] > 0, base_z1):
            return False
        if not np.array_equal(cache["z2"] > 0, base_z2):
            return False

    return True


def main():
    np.random.seed(42)

    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Y = np.eye(10)[y]

    X_train, _, Y_train, _, y_train, _ = train_test_split(
        X,
        Y,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    X_batch = X_train[:32]
    Y_batch = Y_train[:32]
    labels_batch = y_train[:32]

    check_rng = np.random.default_rng(42)
    report_lines = []

    for width in (16, 64):
        params = {
            "W1": np.random.normal(0, 0.01, size=(64, width)),
            "b1": np.zeros(width),
            "W2": np.random.normal(0, 0.01, size=(width, width)),
            "b2": np.zeros(width),
            "W3": np.random.normal(0, 0.01, size=(width, 10)),
            "b3": np.zeros(10),
        }

        loss, gradients = loss_and_gradients(X_batch, Y_batch, params)
        locations = [
            (name, index)
            for name in ("W1", "W2", "W3")
            for index in np.ndindex(params[name].shape)
        ]
        check_rng.shuffle(locations)

        epsilon = 1e-5
        passing = 0
        errors = []
        checked = 0

        for name, index in locations:
            # ReLU has no single derivative at its zero boundary.
            if not same_relu_regions(
                X_batch, params, name, index, epsilon
            ):
                continue

            delta = mp.mpf("0.00001")
            loss_plus = high_precision_loss(
                X_batch,
                labels_batch,
                params,
                changed_weight=(name, index),
                delta=delta,
            )
            loss_minus = high_precision_loss(
                X_batch,
                labels_batch,
                params,
                changed_weight=(name, index),
                delta=-delta,
            )

            numerical_gradient = float(
                (loss_plus - loss_minus)
                / (2 * delta * X_batch.shape[0])
            )
            analytical_gradient = gradients[name][index]

            relative_error = abs(
                analytical_gradient - numerical_gradient
            ) / max(
                1e-8,
                abs(analytical_gradient) + abs(numerical_gradient),
            )

            checked += 1
            errors.append(relative_error)
            if relative_error < 1e-6:
                passing += 1

            print(
                f"Width {width}, {name}{index}: "
                f"analytic={analytical_gradient:.3e}, "
                f"numerical={numerical_gradient:.3e}, "
                f"relative error={relative_error:.3e}"
            )

            if checked == 10:
                break

        if checked < 10:
            raise RuntimeError(
                f"Could only find {checked} differentiable weights to check."
            )

        message = (
            f"Width {width}: loss={loss:.6f}; "
            f"{passing}/10 gradients have relative error < 1e-6; "
            f"maximum relative error={max(errors):.3e}"
        )
        print(message)
        report_lines.append(message)

        assert passing >= 9, (
            f"Gradient check failed for width {width}: "
            f"only {passing}/10 passed."
        )

    output_file = (
        Path(__file__).parent.parent / "results" / "gradient_check.txt"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Gradient checks passed. Saved results to: {output_file}")


if __name__ == "__main__":
    main()
