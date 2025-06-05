Given the following set of data points, (1, 5.5), (3, 8.5), (6, 13), (7, 14.5). Calculate the equation of the straight line (linear regression line) that best fits these data points.
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Data points
X = np.array([[1], [3], [6], [7]])
y = np.array([5.5, 10, 13, 14.5])
```
## model
```
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),  # ลองปรับ alpha ดูได้
    "Lasso": Lasso(alpha=0.5)   # ลองปรับ alpha ดูได้
}
```
## prediction & evaluaiton
```
preds = {}
metrics = {}
for name, model in models.items():
    model.fit(X, y)  # train model
    y_pred = model.predict(X)
    preds[name] = y_pred
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    rsq = r2_score(y, y_pred)
    metrics[name] = {
        "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": rsq,
        "slope": model.coef_[0], "intercept": model.intercept_
    }
```
## plot
```
x_line = np.linspace(0, 8, 100).reshape(-1, 1)
plt.scatter(X, y, color="black", label="Data")
colors = {"Linear": "blue", "Ridge": "green", "Lasso": "red"}
for name, model in models.items():
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, color=colors[name], label=name)

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression vs Ridge vs Lasso")
plt.show()
```
## metrics table
```
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
df = pd.DataFrame(metrics).T
print(df[["MAE", "MSE", "RMSE", "R2", "slope", "intercept"]])
```

