
import numpy as np
from itertools import combinations_with_replacement


class PolynomialFeatures:

    def __init__(self, degree=2, include_bias=False):
        if degree < 1:
            raise ValueError("degree must be at least 1.")

        self.degree = degree
        self.include_bias = include_bias

    def fit(self, X):
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        self.n_features_in_ = X.shape[1]

        return self


    def transform(self, X):
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError(
                "PolynomialFeatures must be fitted before transform."
            )

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"but got {X.shape[1]}."
            )

        features = []

        if self.include_bias:
            features.append(np.ones(X.shape[0]))

        for degree in range(1, self.degree + 1):
            for indices in combinations_with_replacement(
                range(self.n_features_in_), degree
            ):
                feature = np.prod(X[:, indices], axis=1)
                features.append(feature)

        return np.column_stack(features)


    def fit_transform(self, X):
        return self.fit(X).transform(X)
