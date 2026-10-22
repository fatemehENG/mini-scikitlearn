import numpy as np
import pytest

from minislearn.preprocessing.standard_scaler import StandardScaler


def test_standard_scaler_fit():
    X = np.array([
        [10, 100],
        [20, 200],
        [30, 300]
    ])

    scaler = StandardScaler()
    scaler.fit(X)

    assert np.allclose(scaler.mean_, [20, 200])
    assert np.allclose(
        scaler.scale_,
        [np.std([10, 20, 30]), np.std([100, 200, 300])]
    )


def test_standard_scaler_transform():
    X = np.array([
        [10, 100],
        [20, 200],
        [30, 300]
    ])

    scaler = StandardScaler()
    scaler.fit(X)

    X_scaled = scaler.transform(X)

    assert np.allclose(np.mean(X_scaled, axis=0), [0, 0])
    assert np.allclose(np.std(X_scaled, axis=0), [1, 1])


def test_standard_scaler_transform_new_data():
    X_train = np.array([
        [10, 100],
        [20, 200],
        [30, 300]
    ])

    X_test = np.array([
        [40, 400]
    ])

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_scaled = scaler.transform(X_test)

    assert X_scaled.shape == (1, 2)


def test_standard_scaler_fit_returns_self():
    X = np.array([
        [10, 100],
        [20, 200],
        [30, 300]
    ])

    scaler = StandardScaler()

    result = scaler.fit(X)

    assert result is scaler 

import numpy as np

from minislearn.preprocessing.train_test_split import train_test_split


def test_train_test_split_shapes():
    X = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
        [5, 50],
        [6, 60],
        [7, 70],
        [8, 80],
        [9, 90],
        [10, 100]
    ])

    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    assert X_train.shape == (8, 2)
    assert X_test.shape == (2, 2)
    assert y_train.shape == (8,)
    assert y_test.shape == (2,)


def test_train_test_split_keeps_X_and_y_together():
    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5]
    ])

    y = np.array([10, 20, 30, 40, 50])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.4,
        random_state=42
    )

    for x, target in zip(X_train, y_train):
        assert target == x[0] * 10

    for x, target in zip(X_test, y_test):
        assert target == x[0] * 10


def test_train_test_split_random_state():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    result1 = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    result2 = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    for a, b in zip(result1, result2):
        assert np.array_equal(a, b) 