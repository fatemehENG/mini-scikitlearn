from minislearn.linear_model.linear_regression import LinearRegression
from minislearn.base.regressor import BaseRegressor
import numpy as np 
import pytest


def test_linear_regression_is_regressor():
    model = LinearRegression()

    assert isinstance(model, BaseRegressor)

def test_linear_regression_get_params():
    model = LinearRegression(fit_intercept=True)

    params = model.get_params()

    assert params == {"fit_intercept": True}

def test_linear_regression_set_params():
    model = LinearRegression(fit_intercept=True)

    model.set_params(fit_intercept=False)

    assert model.fit_intercept is False

def test_linear_regression_fit_predict():
    X = [[1], [2], [3], [4], [5]]
    y = [3, 5, 7, 9, 11]

    model = LinearRegression()

    model.fit(X, y)

    predictions = model.predict([[6]])

    assert abs(predictions[0] - 13) < 1e-6

def test_linear_regression_parameters():
    X = [[1], [2], [3], [4], [5]]
    y = [3, 5, 7, 9, 11]

    model = LinearRegression()

    model.fit(X, y)

    assert abs(model.coef_[0] - 2) < 1e-6
    assert abs(model.intercept_ - 1) < 1e-6

def test_linear_regression_without_intercept():
    X = [[1], [2], [3], [4], [5]]
    y = [2, 4, 6, 8, 10]

    model = LinearRegression(fit_intercept=False)

    model.fit(X, y)

    assert abs(model.coef_[0] - 2) < 1e-6
    assert model.intercept_ == 0.0

def test_linear_regression_multiple_features():
    X = [
        [1, 1],
        [2, 1],
        [1, 2],
        [3, 2],
        [2, 3]
    ]

    y = [6, 8, 9, 13, 14]

    model = LinearRegression()

    model.fit(X, y)

    assert abs(model.coef_[0] - 2) < 1e-6
    assert abs(model.coef_[1] - 3) < 1e-6
    assert abs(model.intercept_ - 1) < 1e-6

def test_linear_regression_predict_multiple_features():
    X = [
        [1, 1],
        [2, 1],
        [1, 2],
        [3, 2],
        [2, 3]
    ]

    y = [6, 8, 9, 13, 14]

    model = LinearRegression()

    model.fit(X, y)

    prediction = model.predict([[10, 5]])

    assert abs(prediction[0] - 36) < 1e-6

# def test_linear_regression_singular_matrix():
#     X = [
#         [1, 1],
#         [2, 2],
#         [3, 3],
#         [4, 4]
#     ]

#     y = [2, 4, 6, 8]

#     model = LinearRegression()

#     with pytest.raises(np.linalg.LinAlgError):
#         model.fit(X, y)

def test_linear_regression_singular_matrix():
    X = [
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4]
    ]

    y = [2, 4, 6, 8]

    model = LinearRegression()

    model.fit(X, y)

    predictions = model.predict([[5, 5]])

    assert np.isclose(predictions[0], 10)

def test_predict_before_fit():
    model = LinearRegression()

    with pytest.raises(RuntimeError):
        model.predict([[5]])

def test_predict_wrong_number_of_features():

    X = [
        [1, 2],
        [2, 3],
        [3, 4]
    ]

    y = [8, 12, 16]

    model = LinearRegression()

    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict([[5]])