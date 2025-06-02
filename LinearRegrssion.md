Given the following set of data points, (1, 5.5), (3, 8.5), (6, 13), (7, 14.5). Calculate the equation of the straight line (linear regression line) that best fits these data points.
```
import numpy as np
from sklearn.linear_model import LinearRegression

# Data points
X = np.array([[1], [3], [6], [7]])   # Feature matrix, must be 2D
y = np.array([5.5, 10, 13, 14.5])   # Target values

# the model
model = LinearRegression()
model.fit(X, y)

# slope (coefficient) and intercept
slope = model.coef_[0]
intercept = model.intercept_

print(f"Equation of the line: y = {slope:.4f}x + {intercept:.4f}")
```
## prediction
```
y_pred = model.predict(X)
y_pred
```
## evaluation
```
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y - y_pred) / y)) * 100
rsq = r2_score(y, y_pred)

print("mea: ", mae)
print("mse: ", mse)
print("rmea: ", rmse)
print("rsq: ", rsq)
```

