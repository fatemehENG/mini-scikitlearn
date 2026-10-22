from minislearn.linear_model.logistic_regression import LogisticRegression
from minislearn.base.classifier import BaseClassifier
import numpy as np 


def test_logistic_regression_is_classifier():
    model = LogisticRegression()

    assert isinstance(model, BaseClassifier)

def test_logistic_regression_get_params():

    model = LogisticRegression(
        learning_rate=0.1,
        n_iterations=500,
        fit_intercept=False
    )

    params = model.get_params()

    assert params == {
        "learning_rate": 0.1,
        "n_iterations": 500,
        "fit_intercept": False
    }


def test_logistic_regression_set_params():

    model = LogisticRegression()

    model.set_params(
        learning_rate=0.05,
        n_iterations=200
    )

    assert model.learning_rate == 0.05
    assert model.n_iterations == 200

def test_logistic_regression_fit_predict():

    X = [
        [1],
        [2],
        [3],
        [8],
        [9],
        [10]
    ]

    y = [0, 0, 0, 1, 1, 1]

    model = LogisticRegression(
        learning_rate=0.1,
        n_iterations=1000
    )

    model.fit(X, y)

    predictions = model.predict([
        [2],
        [9]
    ])

    assert predictions[0] == 0
    assert predictions[1] == 1

def test_sigmoid():
    model = LogisticRegression()

    assert np.isclose(model._sigmoid(0), 0.5)
    assert model._sigmoid(-10) < 0.01
    assert model._sigmoid(10) > 0.99


def test_logistic_regression_predict_proba():

    X = [
        [1],
        [2],
        [3],
        [8],
        [9],
        [10]
    ]

    y = [0, 0, 0, 1, 1, 1]

    model = LogisticRegression(
        learning_rate=0.1,
        n_iterations=1000
    )

    model.fit(X, y)

    probabilities = model.predict_proba([
        [2],
        [9]
    ])

    assert probabilities[0] < 0.5
    assert probabilities[1] > 0.5

def test_logistic_regression_loss():

    model = LogisticRegression()

    y = np.array([1, 0])

    good_predictions = np.array([0.99, 0.01])
    bad_predictions = np.array([0.01, 0.99])

    good_loss = model._compute_loss(
        y,
        good_predictions
    )

    bad_loss = model._compute_loss(
        y,
        bad_predictions
    )

    assert good_loss < bad_loss

def test_logistic_regression_loss_perfect_prediction():

    model = LogisticRegression()

    y = np.array([1, 0])

    predictions = np.array([1.0, 0.0])

    loss = model._compute_loss(y, predictions)

    assert loss < 1e-10

def test_logistic_regression_loss_history():

    X = [
        [1],
        [2],
        [3],
        [8],
        [9],
        [10]
    ]

    y = [0, 0, 0, 1, 1, 1]

    model = LogisticRegression(
        learning_rate=0.1,
        n_iterations=100
    )

    model.fit(X, y)

    assert len(model.loss_history_) == 100

