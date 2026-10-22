import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):

    X = np.asarray(X)
    y = np.asarray(y)

    if random_state is not None:
        np.random.seed(random_state)

    indices = np.random.permutation(len(X))

    test_count = int(len(X) * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

