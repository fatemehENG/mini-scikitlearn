
import numpy as np
from minislearn.base.classifier import BaseClassifier


class KNN(BaseClassifier):

    def __init__(self, n_neighbors=5):
        if not isinstance(n_neighbors, (int, np.integer)) or n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")

        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] == 0 or X.shape[0] != len(y):
            raise ValueError("X and y must contain the same non-zero number of samples.")

        if self.n_neighbors > X.shape[0]:
            raise ValueError("n_neighbors cannot exceed the number of training samples.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)

        return self

    def predict_proba(self, X):
        if not hasattr(self, "X_train"):
            raise RuntimeError("Model must be fitted before prediction.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError("X has an incorrect number of features.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        probabilities = []

        for x in X:
            distances = np.linalg.norm(
                self.X_train - x,
                axis=1
            )

            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_labels = self.y_train[nearest_indices]

            class_probabilities = [
                np.mean(nearest_labels == cls)
                for cls in self.classes_
            ]

            probabilities.append(class_probabilities)

        return np.asarray(probabilities)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
