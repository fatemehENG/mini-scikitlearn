from minislearn.cluster.kmeans import KMeans
import numpy as np 
import pytest 




def test_kmeans_initial_centers():

    X = [
        [1, 1],
        [2, 2],
        [8, 8],
        [9, 9]
    ]

    model = KMeans(n_clusters=2)

    model.fit(X)

    assert model.cluster_centers_.shape == (2, 2)


def test_kmeans_fit():
    X= [
        [1,1],
        [2,2],
        [8,8],
        [9,9]
    ]
    model= KMeans(n_clusters=2)
    model.fit(X)
    assert model.cluster_centers_.shape == (2,2)
    assert model.labels_.shape ==(4,)

def test_kmeans_clusters_similar_points_together():

    X = [
        [1, 1],
        [2, 2],
        [8, 8],
        [9, 9]
    ]

    model = KMeans(n_clusters=2)

    model.fit(X)

    labels = model.labels_

    # دو نقطه نزدیک اول باید در یک خوشه باشند
    assert labels[0] == labels[1]

    # دو نقطه نزدیک آخر باید در یک خوشه باشند
    assert labels[2] == labels[3]

    # این دو گروه باید در خوشه‌های متفاوت باشند
    assert labels[0] != labels[2]

def test_kmeans_predict():
    X= [
        [1,1],
        [2,2],
        [8,8],
        [9,9]
    ]

    model= KMeans(n_clusters=2)
    model.fit(X)
    predictions= model.predict([
        [1.5,1.5],
        [8.5, 8.5]
    ])

    assert predictions.shape == (2,)
    assert predictions[0] != predictions[1]

def test_kmeans_predict_wrong_number_of_features():
    X = [
        [1, 1],
        [2, 2],
        [8, 8],
        [9, 9]
    ]

    model= KMeans(n_clusters=2)
    model.fit(X)

    with pytest.raises(ValueError):
        model.predict([[1,2,3]])