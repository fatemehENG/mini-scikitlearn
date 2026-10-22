from minislearn.base.estimator import BaseEstimator


class BaseClassifier(BaseEstimator):


    def predict(self, X):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    