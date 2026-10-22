from minislearn.tree.decision_tree import DecisionTreeClassifier 
from minislearn.ensemble.random_forest import RandomForestClassifier 
import numpy as np 

def test_random_forest_get_params():
    model= RandomForestClassifier(n_estimator=10, max_depth=3)
    params= model.get_params()

    assert params["n_estimator"]==10 
    assert params["max_depth"]==3 

def test_random_forest_set_params():
    model= RandomForestClassifier(n_estimator=10, max_depth=3)
    model.set_params(n_estimator=20, max_depth=5)
    params= model.get_params()
    assert params["n_estimator"]==20 
    assert params["max_depth"]== 5 


def test_random_forest_fit():

    X = [
        [1],
        [2],
        [8],
        [9]
    ]

    y = [0, 0, 1, 1]

    model = RandomForestClassifier(
        n_estimator=3,
        max_depth=2
    )

    model.fit(X, y)

    assert len(model.estimators_) == 3

def test_random_forest_predict():
    X= [
        [1],
        [2],
        [8],
        [9]
    ]

    y= [0,0,1,1]

    model = RandomForestClassifier(
    n_estimator=3,
    max_depth=2,
    random_state=42
)
    model.fit(X,y)

    predictions = model.predict([
        [2],
        [9]
    ])

    print("predictions:", predictions)

    for i, tree in enumerate(model.estimators_):
        print("tree", i, "prediction:", tree.predict([[2], [9]]))

    assert all(pred in model.classes_ for pred in predictions)

def test_bootstrap_sample():
    X= np.array([
        [1],
        [2],
        [3],
        [4],
        [5]
    ])

    y = np.array([0, 0, 0, 1, 1]) 
    model= RandomForestClassifier(n_estimator=3)
    X_sample, y_sample= model._bootstrap_sample(X,y)

    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape

def test_bootstrap_sample_with_replacement():

    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5]
    ])

    y = np.array([0, 0, 0, 1, 1])

    model = RandomForestClassifier()

    X_sample, y_sample = model._bootstrap_sample(X, y)

    unique_samples = np.unique(X_sample, axis=0)

    assert len(unique_samples) <= len(X_sample)

def test_random_forest_number_of_trees():

    X = [
        [1],
        [2],
        [8],
        [9]
    ]

    y = [0, 0, 1, 1]

    model = RandomForestClassifier(
        n_estimator=5,
        max_depth=2
    )

    model.fit(X, y)

    assert len(model.estimators_) == 5

def test_random_forest_max_features():

    X = [
        [1, 10, 100],
        [2, 20, 200],
        [8, 80, 800],
        [9, 90, 900]
    ]

    y = [0, 0, 1, 1]

    model = RandomForestClassifier(
        n_estimator=3,
        max_depth=2,
        max_features=2
    )

    model.fit(X, y)

    for tree in model.estimators_:
        assert tree.max_features == 2 

def test_random_forest_random_state():

    X = [
        [1],
        [2],
        [8],
        [9]
    ]

    y = [0, 0, 1, 1]

    model1 = RandomForestClassifier(
        n_estimator=3,
        max_depth=2,
        random_state=42
    )

    model2 = RandomForestClassifier(
        n_estimator=3,
        max_depth=2,
        random_state=42
    )

    model1.fit(X, y)
    model2.fit(X, y)

    predictions1 = model1.predict([[2], [9]])
    predictions2 = model2.predict([[2], [9]])

    assert predictions1.tolist() == predictions2.tolist()

def test_random_forest_default_random_state():

    X = [
        [1],
        [2],
        [8],
        [9]
    ]

    y = [0, 0, 1, 1]

    model = RandomForestClassifier(
        n_estimator=3,
        max_depth=2
    )

    model.fit(X, y)

    predictions = model.predict([
        [2],
        [9]
    ])

    assert predictions.shape == (2,)