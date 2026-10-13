from collections import Counter

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Loading the built-in dataset: X contains measurements, y contains the class labels.
X, y = load_breast_cancer(return_X_y=True)

# Keep 30% aside for testing; stratify keeps the class proportions similar.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

print("All examples, features:", X.shape)
print("Training examples:", X_train.shape[0])
print("Test examples:", X_test.shape[0])
print("All labels (0=malignant, 1=benign):", dict(sorted(Counter(y).items())))
print("Training labels:", dict(sorted(Counter(y_train).items())))
print("Test labels:", dict(sorted(Counter(y_test).items())))
