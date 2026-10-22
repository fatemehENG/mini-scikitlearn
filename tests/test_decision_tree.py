from minislearn.tree.decision_tree import DecisionTreeClassifier 
from minislearn.base.classifier import BaseClassifier 
import numpy as np 
import pytest

import numpy as np
import pytest

from minislearn.tree.decision_tree import DecisionTreeClassifier





def test_decision_tree_is_classifier():
    model= DecisionTreeClassifier()
    assert isinstance(model, BaseClassifier)

def test_decision_tree_get_params():
    model= DecisionTreeClassifier(max_depth=3)
    params= model.get_params()
    assert params['max_depth']==3

def test_decision_tree_set_params():

    model = DecisionTreeClassifier(max_depth=3)
    model.set_params(max_depth=5)
    assert model.max_depth == 5

def test_decision_tree_fit_predict():
    X= [
        [1],
        [2],
        [3],
        [8],
        [9],
        [10]
    ]

    y= [0,0,0,1,1,1]

    model= DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)


    predictions = model.predict([
        [2],
        [9]
    ])

    assert predictions.tolist()==[0,1]

def _test_gini_impurity():
    model= DecisionTreeClassifier()
    y= np.array([0,0,0,1,1])
    gini= model._gini_impurity(y)
    assert np.isclose(gini, 0.48)

def test_gini_impurity_pure_node():
    model= DecisionTreeClassifier()
    y= np.array([1,1,1,1])
    gini= model._gini_impurity(y)
    assert np.isclose(gini,0.0)

def test_find_best_split():

    X = np.array([
        [1],
        [2],
        [3],
        [8],
        [9],
        [10]
    ])

    y = np.array([0, 0, 0, 1, 1, 1])

    model = DecisionTreeClassifier()

    feature, threshold = model._find_best_split(X, y)

    assert feature == 0
    assert threshold == 3

def test_decision_tree_invalid_data():
    X= [[1],[2],[3]]
    y= [0,1]
    model= DecisionTreeClassifier()
    with pytest.raises(ValueError):
        model.fit(X,y) 

def test_decision_tree_predict_before_fit():

    model = DecisionTreeClassifier()

    with pytest.raises(RuntimeError):
        model.predict([[5]])

def test_decision_tree_wrong_number_of_features():

    X = [
        [1],
        [2],
        [3],
        [8],
        [9],
        [10]
    ]

    y = [0, 0, 0, 1, 1, 1]

    model = DecisionTreeClassifier()
    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict([
            [1, 2]
        ])


def test_decision_tree_predict_single_class():
    X= [
        [1],
        [2],
        [3],
        [4]
    ]
    y=[1,1,1,1]

    model= DecisionTreeClassifier()
    model.fit(X,y)

    predictions= model.predict([[10],[20]])
    assert np.array_equal(predictions, np.array([1,1]))

def test_select_all_features():
    model= DecisionTreeClassifier()
    selected_features= model._select_features(n_features=4)

    assert len(selected_features)==4 
    assert set(selected_features)== {0,1,2,3}

def test_select_random_features():
    model= DecisionTreeClassifier(max_features=2)
    selected_features= model._select_features(n_features=4)
    assert len(selected_features)==2 
    assert len(set(selected_features))==2 
    assert all (feature in [0,1,2,3] for feature in selected_features)

    