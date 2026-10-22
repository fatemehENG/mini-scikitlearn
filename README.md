
# Mini-Sklearn

A small machine learning library built from scratch using Python and NumPy.

The goal of this project is to understand how common machine learning algorithms work internally and practice writing modular, reusable, and testable Python code.

## Algorithms

- Linear Regression
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- AdaBoost
- K-Means Clustering
- Principal Component Analysis (PCA)

## Additional Components

- Base estimator, classifier, and regressor classes
- Data preprocessing utilities
- Regression and classification metrics
- Cross-validation utilities

## Requirements

- Python 3.10+
- NumPy

Other dependencies required for development and testing will be listed in `requirements.txt`.

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Example

```python
from minislearn.linear_model.linear_regression import LinearRegression

X = [[1], [2], [3], [4]]
y = [2, 4, 6, 8]

model = LinearRegression()
model.fit(X, y)

predictions = model.predict([[5], [6]])
print(predictions)
```

## Running Tests

Run the test suite from the project root:

```bash
python -m pytest -v
```

## Project Structure

- `minislearn/`: Machine learning algorithms and utilities
- `tests/`: Automated tests for the implemented components

## Project Status

This project is a learning and portfolio project focused on implementing machine learning algorithms and understanding their internal mechanisms.

## License

See the `LICENSE` file for licensing terms.