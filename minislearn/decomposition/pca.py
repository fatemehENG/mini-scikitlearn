import numpy as np


class PCA:

    def __init__(self, n_components=3):
        if (
            not isinstance(n_components, (int, np.integer))
            or isinstance(n_components, (bool, np.bool_))
            or n_components <= 0
        ):
            raise ValueError(
                "n_components must be a positive integer."
            )

        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[0] < 2:
            raise ValueError("PCA requires at least 2 samples.")

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        if self.n_components > X.shape[1]:
            raise ValueError(
                "n_components cannot exceed the number of features."
            )

        self.n_features_in_ = X.shape[1]
        self.mean_ = np.mean(X, axis=0)

        X_centered = X - self.mean_

        covariance_matrix = np.cov(
            X_centered,
            rowvar=False
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance_matrix
        )

        sorted_indices = np.argsort(eigenvalues)[::-1]

        # Keep the largest eigenvalues and their eigenvectors
        self.explained_variance_ = eigenvalues[
            sorted_indices[:self.n_components]
        ]



        self.components_ = eigenvectors[
            :, sorted_indices[:self.n_components]
        ]

        total_variance = np.sum(eigenvalues)

        if total_variance > 0:
            self.explained_variance_ratio_ = (
                self.explained_variance_ / total_variance
            )
        else:
            self.explained_variance_ratio_ = np.zeros(
                self.n_components
            )

        return self

    def transform(self, X):
    
        if not hasattr(self, "components_"):
            raise ValueError("PCA has not been fitted yet.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than training data."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        X_centered = X - self.mean_

        return X_centered @ self.components_

    def fit_transform(self, X):
        return self.fit(X).transform(X)