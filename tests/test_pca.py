import numpy as np 
from minislearn.decomposition.pca import PCA 

def test_pca_fit():

    X = [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6]
    ]

    model = PCA(n_components=2)

    model.fit(X)

    assert model.mean_.shape == (3,)

    assert model.components_.shape == (3, 2)

def test_pca_transform():

    X = [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6]
    ]

    model = PCA(n_components=2)

    model.fit(X)

    X_reduced = model.transform(X)

    assert X_reduced.shape == (4, 2)

def test_pca_transform_uses_training_mean():

    X_train = [
        [1, 2],
        [3, 4],
        [5, 6]
    ]

    X_new = [
        [2, 3]
    ]

    model = PCA(n_components=1)

    model.fit(X_train)

    X_reduced = model.transform(X_new)

    assert X_reduced.shape == (1, 1) 

import pytest


def test_pca_transform_before_fit():

    model = PCA(n_components=2)

    with pytest.raises(ValueError):
        model.transform([
            [1, 2, 3]
        ])


def test_pca_transform_wrong_number_of_features():

    X = [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6]
    ]

    model = PCA(n_components=2)

    model.fit(X)

    with pytest.raises(ValueError):
        model.transform([
            [1, 2]
        ])

def test_pca_too_many_components():

    X = [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6]
    ]

    model = PCA(n_components=4)

    with pytest.raises(ValueError):
        model.fit(X)

def test_pca_invalid_n_components():
    X = [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]
    ]

    with pytest.raises(ValueError):
        PCA(n_components=0).fit(X)

    with pytest.raises(ValueError):
        PCA(n_components=-1).fit(X)

def test_pca_single_sample():
    X = [
        [1, 2, 3]
    ]

    model = PCA(n_components=2)

    with pytest.raises(ValueError):
        model.fit(X)

def test_pca_n_features_in():
    X = [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]
    ]

    model = PCA(n_components=2)
    model.fit(X)

    assert model.n_features_in_ == 3 

def test_pca_components_are_orthogonal():
    X = [
        [1, 2, 3],
        [2, 4, 1],
        [3, 1, 5],
        [4, 3, 2]
    ]

    model = PCA(n_components=2)
    model.fit(X)

    dot_product = np.dot(
        model.components_[:, 0],
        model.components_[:, 1]
    )

    assert np.isclose(dot_product, 0)

def test_pca_fit_transform():
    X = [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6]
    ]

    model = PCA(n_components=2)

    X_reduced = model.fit_transform(X)

    assert X_reduced.shape == (4, 2)