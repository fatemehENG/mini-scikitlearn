import numpy as np 



from minislearn.metrics.regression import (mean_absolute_error,
                                            mean_squared_error,
                                            root_mean_squared_error,
                                            r2_score)

from minislearn.metrics.classification import (accuracy_score,
                                                precision_score,
                                                 recall_score,
                                                f1_score,
                                                confusion_matrix)



def test_mean_absolute_error():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([2, 2, 5])

    result = mean_absolute_error(y_true, y_pred)

    assert result == 1


def test_mean_squared_error():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([2, 2, 5])

    result = mean_squared_error(y_true, y_pred)

    assert result == 5 / 3

def test_root_mean_squared_error():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([2, 2, 5])

    result = root_mean_squared_error(y_true, y_pred)

    assert np.isclose(result, np.sqrt(5 / 3))

def test_r2_score():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 5])

    result = r2_score(y_true, y_pred)

    assert np.isclose(result, -1.0)

def test_r2_score_perfect_prediction():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])

    result = r2_score(y_true, y_pred)

    assert result == 1


def test_accuracy_score():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    result = accuracy_score(y_true, y_pred)

    assert result == 0.8

def test_precision_score():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])

    result = precision_score(y_true, y_pred)

    assert np.isclose(result, 2 / 3)

def test_recall_score():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])

    result = recall_score(y_true, y_pred)

    assert np.isclose(result, 2 / 3)

def test_f1_score():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])

    result = f1_score(y_true, y_pred)

    assert np.isclose(result, 2 / 3)

def test_confusion_matrix():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])

    result = confusion_matrix(y_true, y_pred)

    expected = np.array([
        [1, 1],
        [1, 2]
    ])

    assert np.array_equal(result, expected)
    