
import numpy as np

from minislearn.preprocessing.polynomial_features import PolynomialFeatures
from minislearn.linear_model.linear_regression import LinearRegression


X = np.array([[1], [2], [3], [4]])
y = np.array([6, 11, 18, 27])


poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

X_new = np.array([[5]])
X_new_poly = poly.transform(X_new)

predictions = model.predict(X_new_poly)

print("Polynomial features:")
print(X_poly)

print("Prediction for x=5:")
print(predictions)
