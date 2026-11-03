import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TrainQuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each numeric column to its training-set 1st–99th percentiles."""

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(values, 0.01, axis=0)
        self.upper_ = np.nanquantile(values, 0.99, axis=0)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_, self.upper_)
