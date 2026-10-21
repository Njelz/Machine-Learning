import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load the digit images as 64 pixel values and their labels, 0 through 9.
X, y = load_digits(return_X_y=True)

# Pixel values range from 0 to 16; rescale them to the range 0 to 1.
X = X / 16.0

# Convert each label to a vector of 10 zeros and ones.
Y = np.eye(10)[y]

# Split features and one-hot labels together.
X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(
    X,
    Y,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

print("All feature data:", X.shape)
print("All one-hot labels:", Y.shape)
print("Training features:", X_train.shape)
print("Test features:", X_test.shape)
print("Training one-hot labels:", Y_train.shape)
print("Test one-hot labels:", Y_test.shape)
print("First digit label:", y_train[0])
print("Its one-hot vector:", Y_train[0])