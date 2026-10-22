import numpy as np


class StandardScaler:

    def __init__(self):
        pass

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        return self 

    def transform(self, X):
        X = np.asarray(X, dtype=float)

        return (X - self.mean_) / self.scale_