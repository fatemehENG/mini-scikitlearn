import numpy as np

from minislearn.linear_model.linear_regression import LinearRegression
from minislearn.preprocessing.train_test_split import train_test_split
from minislearn.metrics.regression import mean_squared_error, r2_score


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
    3,
    5,
    7,
    9,
    11,
    13,
    15,
    17
])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Predictions:", y_pred)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))