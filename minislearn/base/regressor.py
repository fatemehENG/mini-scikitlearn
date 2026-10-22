from minislearn.base.estimator import BaseEstimator


class BaseRegressor(BaseEstimator):

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError