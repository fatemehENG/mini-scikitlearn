
import numpy as np
from minislearn.base.classifier import BaseClassifier


class LogisticRegression(BaseClassifier):

    def __init__(
        self,
        learning_rate=0.01,
        n_iterations=1000,
        fit_intercept=True
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        if not isinstance(n_iterations, (int, np.integer)) or n_iterations <= 0:
            raise ValueError("n_iterations must be a positive integer.")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept

    def _sigmoid(self, z):
        # Numerically stable sigmoid
        z = np.asarray(z, dtype=float)
        result = np.empty_like(z)

        positive = z >= 0
        result[positive] = 1 / (1 + np.exp(-z[positive]))

        exp_z = np.exp(z[~positive])
        result[~positive] = exp_z / (1 + exp_z)

        return result

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain samples and features.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain only finite values.")

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only 0 and 1.")

        n_samples, n_features = X.shape

        self.n_features_in_ = n_features
        self.classes_ = np.array([0, 1])
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        self.loss_history_ = []

        for _ in range(self.n_iterations):
            z = X @ self.coef_

            if self.fit_intercept:
                z = z + self.intercept_

            y_hat = self._sigmoid(z)
            error = y_hat - y

            dw = (X.T @ error) / n_samples
            self.coef_ -= self.learning_rate * dw

            if self.fit_intercept:
                db = np.sum(error) / n_samples
                self.intercept_ -= self.learning_rate * db

            # Recalculate probabilities after the parameter update
            z = X @ self.coef_
            if self.fit_intercept:
                z = z + self.intercept_

            y_hat = self._sigmoid(z)
            loss = self._compute_loss(y, y_hat)
            self.loss_history_.append(loss)

        return self

    def predict_proba(self, X):
        if not hasattr(self, "coef_"):
            raise RuntimeError("Model must be fitted before prediction.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"but got {X.shape[1]}."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        z = X @ self.coef_

        if self.fit_intercept:
            z = z + self.intercept_

        return self._sigmoid(z)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    def _compute_loss(self, y, y_hat):
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        loss = -np.mean(
            y * np.log(y_hat)
            + (1 - y) * np.log(1 - y_hat)
        )

        return loss
