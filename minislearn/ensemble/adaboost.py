
import numpy as np


from minislearn.base.classifier import BaseClassifier
from minislearn.tree.decision_tree import DecisionTreeClassifier


class AdaBoostClassifier(BaseClassifier):

    def __init__(
        self,
        n_estimators=50,
        learning_rate=1.0,
        max_depth=1
    ):
        if (
            not isinstance(n_estimators, (int, np.integer))
            or isinstance(n_estimators, (bool, np.bool_))
            or n_estimators <= 0
        ):
            raise ValueError("n_estimators must be a positive integer.")

        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite.")

        if (not isinstance(max_depth, (int, np.integer)) 

            or isinstance(max_depth, (bool, np.bool_))
            or max_depth <= 0):

            raise ValueError("max_depth must be a positive integer.")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

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
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError(
                "AdaBoostClassifier supports exactly two classes."
            )

        self.n_features_in_ = X.shape[1]

        # Convert the two classes to -1 and +1
        y_signed = np.where(y == self.classes_[0], -1.0, 1.0)

        n_samples = len(y)
        sample_weights = np.full(n_samples, 1.0 / n_samples)

        self.estimators_ = []
        self.estimator_weights_ = []

        epsilon = 1e-15

        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=None
            )

            tree.fit(X, y, sample_weight=sample_weights)

            predictions = tree.predict(X)
            predictions_signed = np.where(
                predictions == self.classes_[0],
                -1.0,
                1.0
            )

            incorrect = predictions != y
            error = np.sum(sample_weights * incorrect)

            # A weak learner must perform better than random guessing
            if error >= 0.5:
                if not self.estimators_:
                    raise ValueError(
                        "The weak learner must have an error below 0.5."
                    )
                break

            # Avoid an infinite estimator weight when error is zero
            error_for_log = max(error, epsilon)

            alpha = (
                self.learning_rate
                * 0.5
                * np.log((1.0 - error_for_log) / error_for_log)
            )

            self.estimators_.append(tree)
            self.estimator_weights_.append(alpha)

            # Stop if this estimator classifies every sample correctly
            if error <= epsilon:
                break

            # Increase the relative weights of misclassified samples
            sample_weights *= np.exp(
                -alpha * y_signed * predictions_signed
            )

            sample_weights /= sample_weights.sum()

        self.estimator_weights_ = np.asarray(
            self.estimator_weights_,
            dtype=float
        )

        if not self.estimators_:
            raise ValueError("No useful estimator could be trained.")

        return self

    def predict(self, X):
        if not hasattr(self, "estimators_"):
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than during fit."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        scores = np.zeros(X.shape[0])

        for tree, alpha in zip(
            self.estimators_,
            self.estimator_weights_
        ):
            predictions = tree.predict(X)

            predictions_signed = np.where(
                predictions == self.classes_[0],
                -1.0,
                1.0
            )

            scores += alpha * predictions_signed

        return np.where(
            scores >= 0,
            self.classes_[1],
            self.classes_[0]
        )
