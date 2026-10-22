import numpy as np

from minislearn.base.regressor import BaseRegressor


class LinearRegression(BaseRegressor):

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        self.n_features_in_ = X.shape[1]

        if self.fit_intercept:
            X = np.column_stack((np.ones(X.shape[0]), X))

        # self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y

        # self.coef_ = np.linalg.solve(
        #         X.T @ X,
        #         X.T @ y
        #     )

        try:
            self.coef_ = np.linalg.solve(
                X.T @ X,
                X.T @ y
            )

        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.pinv(X) @ y



        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X):

        if not hasattr(self, "coef_"):
            raise RuntimeError("Model must be fitted before prediction.")

        X = np.asarray(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"but got {X.shape[1]}."
            )

        predictions = X @ self.coef_ + self.intercept_

        return predictions




