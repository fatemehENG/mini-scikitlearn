
import numpy as np
import pytest

from minislearn.preprocessing.polynomial_features import (
    PolynomialFeatures
)


def test_polynomial_features_degree_2():
    X = np.array([[2], [3], [4]])

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    expected = np.array([
        [2, 4],
        [3, 9],
        [4, 16]
    ])

    np.testing.assert_array_equal(X_poly, expected)


def test_polynomial_features_degree_3():
    X = np.array([[2], [3]])

    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)

    expected = np.array([
        [2, 4, 8],
        [3, 9, 27]
    ])

    np.testing.assert_array_equal(X_poly, expected)


def test_transform_before_fit_raises_error():
    poly = PolynomialFeatures(degree=2)

    with pytest.raises(RuntimeError):
        poly.transform(np.array([[2]]))


def test_wrong_number_of_features_raises_error():
    poly = PolynomialFeatures(degree=2)
    poly.fit(np.array([[1, 2], [3, 4]]))

    with pytest.raises(ValueError):
        poly.transform(np.array([[1]]))


def test_invalid_degree_raises_error():
    with pytest.raises(ValueError):
        PolynomialFeatures(degree=0)



def test_interaction_features_degree_2():
    X = np.array([
        [2, 3],
        [4, 5]
    ])

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    expected = np.array([
        [2, 3, 4, 6, 9],
        [4, 5, 16, 20, 25]
    ])

    np.testing.assert_array_equal(X_poly, expected)
