
import numpy as np

from minislearn.base.classifier import BaseClassifier
from minislearn.tree.decision_tree import DecisionTreeClassifier


class RandomForestClassifier(BaseClassifier):

    def __init__(
        self,
        n_estimators=10,
        max_depth=None,
        max_features=2,
        random_state=None,
        n_estimator=None
    ):
        # Backward compatibility with the old parameter name
        if n_estimator is not None:
            n_estimators = n_estimator

        if (
            not isinstance(n_estimators, (int, np.integer))
            or isinstance(n_estimators, (bool, np.bool_))
            or n_estimators <= 0
        ):
            raise ValueError("n_estimators must be a positive integer.")

        self.n_estimators = n_estimators
        self.n_estimator = n_estimators  #
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain samples and features.")

        if X.shape[0] != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.estimators_ = []

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y, rng)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features
            )

            tree.fit(X_sample, y_sample)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        if not hasattr(self, "estimators_"):
            raise RuntimeError("Model must be fitted before prediction.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than during fit."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        predictions = np.asarray([
            tree.predict(X)
            for tree in self.estimators_
        ])

        final_predictions = []

        for i in range(X.shape[0]):
            values, counts = np.unique(
                predictions[:, i],
                return_counts=True
            )
            final_predictions.append(values[np.argmax(counts)])

        return np.asarray(final_predictions)

    def _bootstrap_sample(self, X, y, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        n_samples = X.shape[0]
        indices = rng.choice(
            n_samples,
            size=n_samples,
            replace=True
        )

        return X[indices], y[indices]
