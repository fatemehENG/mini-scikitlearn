import numpy as np

from minislearn.model_selection.cross_validation import cross_val_score


class DummyClassifier:

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def test_cross_val_score_returns_scores():

    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
        [10]
    ])

    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    model = DummyClassifier()

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5
    )

    assert len(scores) == 5


def test_cross_val_score_values_are_between_zero_and_one():

    X = np.arange(10).reshape(10, 1)
    y = np.zeros(10)

    model = DummyClassifier()

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5
    )

    assert np.all(scores >= 0)
    assert np.all(scores <= 1)


def test_cross_val_score_with_perfect_model():

    X = np.arange(10).reshape(10, 1)
    y = np.zeros(10)

    model = DummyClassifier()

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5
    )

    assert np.all(scores == 1.0)