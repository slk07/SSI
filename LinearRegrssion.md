Given the following set of data points, (1, 5.5), (3, 8.5), (6, 13), (7, 14.5). Calculate the equation of the straight line (linear regression line) that best fits these data points.
```
import numpy as np
from sklearn.linear_model import LinearRegression

# Data points
X = np.array([[1], [3], [6], [7]])   # Feature matrix, must be 2D
y = np.array([5.5, 8.5, 13, 14.5])   # Target values

# the model
model = LinearRegression()
model.fit(X, y)

# slope (coefficient) and intercept
slope = model.coef_[0]
intercept = model.intercept_

print(f"Equation of the line: y = {slope:.4f}x + {intercept:.4f}")
```
