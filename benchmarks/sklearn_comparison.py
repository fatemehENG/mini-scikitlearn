import time
import numpy as np

from minislearn.linear_model.linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression


X = np.random.rand(1000, 5)
y = np.random.rand(1000)


# Mini-sklearn
start = time.perf_counter()

model = LinearRegression()
model.fit(X, y)
model.predict(X)

mini_time = time.perf_counter() - start


# Scikit-learn
start = time.perf_counter()

model = SklearnLinearRegression()
model.fit(X, y)
model.predict(X)

sklearn_time = time.perf_counter() - start


print("Mini-sklearn time:", mini_time)
print("Scikit-learn time:", sklearn_time)