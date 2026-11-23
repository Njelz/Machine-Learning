import random
import statistics
import time

import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from autodiff import Value, binary_cross_entropy


SEED = 42
EPOCHS = 100
LEARNING_RATE = 0.1
RUNS = 5


def linear(inputs, weights, bias):
    result = bias
    for input_value, weight in zip(inputs, weights):
        result = result + input_value * weight
    return result


def make_custom_model():
    hidden_weights = [
        [Value(random.uniform(-0.5, 0.5)) for _ in range(2)]
        for _ in range(8)
    ]
    hidden_biases = [Value(0.0) for _ in range(8)]
    output_weights = [Value(random.uniform(-0.5, 0.5)) for _ in range(8)]
    output_bias = Value(0.0)

    parameters = (
        [weight for unit in hidden_weights for weight in unit]
        + hidden_biases
        + output_weights
        + [output_bias]
    )

    model = (hidden_weights, hidden_biases, output_weights, output_bias)
    return model, parameters


def predict_custom(features, model):
    hidden_weights, hidden_biases, output_weights, output_bias = model
    inputs = [Value(float(feature)) for feature in features]

    hidden = [
        linear(inputs, weights, bias).relu()
        for weights, bias in zip(hidden_weights, hidden_biases)
    ]

    return linear(hidden, output_weights, output_bias).sigmoid()


def train_custom_one_epoch(model, parameters, X, y):
    total_loss = Value(0.0)

    for features, label in zip(X, y):
        probability = predict_custom(features, model)
        total_loss = total_loss + binary_cross_entropy(probability, label)

    mean_loss = total_loss * (1.0 / len(X))
    mean_loss.backward()

    for parameter in parameters:
        parameter.data -= LEARNING_RATE * parameter.grad

    return mean_loss.data


def evaluate_custom(model, X, y):
    correct = 0

    for features, label in zip(X, y):
        probability = predict_custom(features, model)
        predicted_class = int(probability.data >= 0.5)
        correct += int(predicted_class == int(label))

    return correct / len(y)


def run_pytorch_once(X_train, y_train, X_test, y_test, seed):
    torch.manual_seed(seed)

    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    start = time.perf_counter()

    for _ in range(EPOCHS):
        predictions = model(X_train)
        loss = criterion(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - start

    with torch.no_grad():
        predictions = model(X_test)
        predicted_classes = (predictions >= 0.5).float()
        accuracy = (predicted_classes == y_test).float().mean().item()

    return elapsed, loss.item(), accuracy


def run_custom_once(X_train, y_train, X_test, y_test, seed):
    random.seed(seed)
    model, parameters = make_custom_model()

    start = time.perf_counter()

    for _ in range(EPOCHS):
        loss = train_custom_one_epoch(model, parameters, X_train, y_train)

    elapsed = time.perf_counter() - start
    accuracy = evaluate_custom(model, X_test, y_test)

    return elapsed, loss, accuracy


def main():
    torch.set_num_threads(1)

    X, y = make_moons(
        n_samples=500,
        noise=0.2,
        random_state=SEED,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,

    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    
    print("\nTensor inspection:")
    transposed_view = X_train_t.T

    for name, tensor in [
        ("X_train_t", X_train_t),
        ("transposed_view", transposed_view),
    ]:
        print(
            f"{name}: shape={tensor.shape}, "
            f"dtype={tensor.dtype}, "
            f"contiguous={tensor.is_contiguous()}"
        )

    pytorch_times = []
    custom_times = []

    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")
    print(f"Runs: {RUNS}; epochs per run: {EPOCHS}")
    print()

    for run in range(RUNS):
        seed = SEED + run

        pytorch_time, pytorch_loss, pytorch_accuracy = run_pytorch_once(
            X_train_t, y_train_t, X_test_t, y_test_t, seed
        )

        custom_time, custom_loss, custom_accuracy = run_custom_once(
            X_train, y_train, X_test, y_test, seed
        )

        pytorch_times.append(pytorch_time)
        custom_times.append(custom_time)

        print(
            f"Run {run + 1}: "
            f"PyTorch={pytorch_time:.4f}s "
            f"(accuracy={pytorch_accuracy:.2f}), "
            f"custom={custom_time:.4f}s "
            f"(accuracy={custom_accuracy:.2f})"
        )

    print()
    print(
        "PyTorch median training time:",
        f"{statistics.median(pytorch_times):.4f} seconds",
    )
    print(
        "Custom autodiff median training time:",
        f"{statistics.median(custom_times):.4f} seconds",
    )
    print(f"Last-run PyTorch loss: {pytorch_loss:.4f}")
    print(f"Last-run custom loss: {custom_loss:.4f}")


if __name__ == "__main__":
    main()