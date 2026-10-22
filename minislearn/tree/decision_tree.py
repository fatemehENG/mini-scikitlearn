
import numpy as np
from minislearn.base.classifier import BaseClassifier


class Node:

    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeClassifier(BaseClassifier):

    def __init__(self, max_depth=None, max_features=None):
        if max_depth is not None:
            if (
                not isinstance(max_depth, (int, np.integer))
                or isinstance(max_depth, (bool, np.bool_))
                or max_depth < 1
            ):
                raise ValueError("max_depth must be a positive integer or None.")

        if max_features is not None:
            if (
                not isinstance(max_features, (int, np.integer))
                or isinstance(max_features, (bool, np.bool_))
                or max_features < 1
            ):
                raise ValueError("max_features must be a positive integer or None.")

        self.max_depth = max_depth
        self.max_features = max_features

 

    
    def _gini_impurity(self, y, sample_weight=None):
        classes, inverse = np.unique(y, return_inverse=True)

        if sample_weight is None:
            counts = np.bincount(inverse)
        else:
            counts = np.bincount(
                inverse,
                weights=sample_weight
            )

        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities ** 2)


    
    def _find_best_split(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape
        feature_indices = self._select_features(n_features)

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=float)

        best_feature = None
        best_threshold = None
        best_gini = float("inf")
        total_weight = sample_weight.sum()

        for feature_index in feature_indices:
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if not left_mask.any() or not right_mask.any():
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                w_left = sample_weight[left_mask]
                w_right = sample_weight[right_mask]

                weighted_gini = (
                    w_left.sum() / total_weight
                    * self._gini_impurity(y_left, w_left)
                    + w_right.sum() / total_weight
                    * self._gini_impurity(y_right, w_right)
                )

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    
    def _most_common_class(self, y, sample_weight=None):
        classes, inverse = np.unique(y, return_inverse=True)

        if sample_weight is None:
            counts = np.bincount(inverse)
        else:
            counts = np.bincount(inverse, weights=sample_weight)

        return classes[np.argmax(counts)]




    def _build_tree(self, X , y, depth, sample_weight=None):
        n_samples = len(y)

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=float)

        if len(np.unique(y)) == 1:
            return Node(value=y[0])

        if self.max_depth is not None and depth >= self.max_depth:
            return Node(
                value=self._most_common_class(y, sample_weight)
            )

        feature_index, threshold = self._find_best_split(

            X, y, sample_weight
        )

        if feature_index is None:
            return Node(
                value=self._most_common_class(y, sample_weight)
            )

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        if not left_mask.any() or not right_mask.any():
            return Node(
                value=self._most_common_class(y, sample_weight)
            )

        X_left = X[left_mask]
        y_left = y[left_mask]
        weight_left = sample_weight[left_mask]

        X_right = X[right_mask]
        y_right = y[right_mask]
        weight_right = sample_weight[right_mask]

        left = self._build_tree(
            X_left,

            y_left,
            depth + 1,
            weight_left
        )

        right =self._build_tree(
            X_right,
            y_right,
            depth + 1,
            weight_right
        )

        return Node(
            feature_index= feature_index,
            threshold=threshold,
            left =left,
            right=right
        )

    

    def fit(self, X, y, sample_weight=None):
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

        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)

            if sample_weight.ndim != 1 or len(sample_weight) != len(y):
                raise ValueError(
                    "sample_weight must match the number of samples."
                )

            if (
                not np.all(np.isfinite(sample_weight))
                or np.any(sample_weight < 0)
                or sample_weight.sum() <= 0
            ):
                raise ValueError(
                    "sample_weight must be finite, non-negative, "
                    "and have positive sum."
                )

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        self.tree_ = self._build_tree(
            X,
            y,
            depth=0,
            sample_weight=sample_weight
        )

        return self

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)

        return self._predict_one(x, node.right)

    def _check_is_fitted(self):
        if not hasattr(self, "tree_"):
            raise RuntimeError(
                "This DecisionTreeClassifier is not fitted yet."
            )

    def predict(self, X):
        self._check_is_fitted()

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than during fit."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        predictions = [
            self._predict_one(x, self.tree_)
            for x in X
        ]

        return np.asarray(predictions)

    def _select_features(self, n_features):
        if self.max_features is None:
            return np.arange(n_features)

        n_selected = min(self.max_features, n_features)

        return np.random.choice(
            n_features,
            size=n_selected,
            replace=False
        )
