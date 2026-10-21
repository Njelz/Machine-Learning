import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def relu(values):
    return np.maximum(0, values)


def softmax(logits):
    # Subtract the row maximum for numerical stability.
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def cross_entropy(probabilities, one_hot_labels):
    safe_probabilities = np.clip(probabilities, 1e-12, 1.0)
    return -np.mean(
        np.sum(one_hot_labels * np.log(safe_probabilities), axis=1)
    )


def forward(X, W1, b1, W2, b2, W3, b3):
    # Input -> hidden layer 1 -> ReLU
    z1 = X @ W1 + b1
    a1 = relu(z1)

    # Hidden layer 1 -> hidden layer 2 -> ReLU
    z2 = a1 @ W2 + b2
    a2 = relu(z2)

    # Hidden layer 2 -> 10 output scores -> probabilities
    logits = a2 @ W3 + b3
    probabilities = softmax(logits)

    return probabilities


# Load and scale the inputs, then use a training batch for the shape check.
X, y = load_digits(return_X_y=True)
X = X / 16.0

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

batch = X_train[:32]
rng = np.random.default_rng(42)

for width in (16, 64):
    # Weight dimensions: input-to-hidden, hidden-to-hidden, hidden-to-output.
    W1 = rng.normal(0, 0.01, size=(64, width))
    b1 = np.zeros(width)

    W2 = rng.normal(0, 0.01, size=(width, width))
    b2 = np.zeros(width)

    W3 = rng.normal(0, 0.01, size=(width, 10))
    b3 = np.zeros(10)

    probabilities = forward(batch, W1, b1, W2, b2, W3, b3)
    row_sums = probabilities.sum(axis=1)
    batch_labels = np.eye(10)[y_train[:32]]
    loss = cross_entropy(probabilities, batch_labels)


    print(f"\nHidden width: {width}")
    print(f"W1 shape: {W1.shape}")
    print(f"W2 shape: {W2.shape}")
    print(f"W3 shape: {W3.shape}")
    print(f"Output shape: {probabilities.shape}")
    print(f"First output row sums to: {row_sums[0]:.8f}")
    print(f"Untrained batch loss: {loss:.4f}")


    assert probabilities.shape == (32, 10)
    assert np.allclose(row_sums, 1.0, atol=1e-6)

print("\nForward-pass checks passed.")
