import numpy as np

from minislearn.linear_model.logistic_regression import LogisticRegression
from minislearn.preprocessing.train_test_split import train_test_split
from minislearn.metrics.classification import accuracy_score


X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
])

y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))