from minislearn.neighbors.knn import KNN
from minislearn.base.classifier import BaseClassifier


def test_get_params():
    model = KNN(n_neighbors=3)

    params = model.get_params()

    assert params == {"n_neighbors": 3}


def test_set_params():
    model = KNN(n_neighbors=3)

    model.set_params(n_neighbors=10)

    assert model.n_neighbors == 10


def test_set_invalid_param():
    model = KNN(n_neighbors=3)

    try:
        model.set_params(unknown=10)
        assert False
    except ValueError:
        assert True


def test_knn_is_classifier():
    model = KNN()

    assert isinstance(model, BaseClassifier)

# def test_knn_predict_proba():

#     X = [
#         [1],
#         [2],
#         [8],
#         [9]
#     ]

#     y = [0, 0, 1, 1]

#     model = KNN(k=3)

#     model.fit(X, y)

#     probabilities = model.predict_proba([[2]])

#     assert probabilities.shape == (1, 2)

#     assert np.isclose(probabilities[0].sum(), 1.0)

#     assert probabilities[0, 0] > probabilities[0, 1]