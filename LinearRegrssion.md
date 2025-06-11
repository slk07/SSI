Given the following set of data points, (1, 5.5), (3, 8.5), (6, 13), (7, 14.5). Calculate the equation of the straight line (linear regression line) that best fits these data points.
# 1 model: OLS
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
## evaluaiton
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
# 3 models
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
    model.fit(X, y)
    y_pred = model.predict(X)
    preds[name] = y_pred

    slope = model.coef_[0]
    intercept = model.intercept_

    # Print the equation of the fitted line
    print(f"{name} Regression Equation: y = {slope:.4f}x + {intercept:.4f}")

    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    rsq = r2_score(y, y_pred)
    metrics[name] = {
        "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": rsq,
        "slope": slope, "intercept": intercept
    }
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

# 2.
```
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import matplotlib.patches as mpatches

# Data
X = np.array([[1, 2], [3, 4], [6, 5], [7, 8]])
y = np.array([1, 4, 9, 14])

models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.5)
}
colors = {"Linear": "blue", "Ridge": "green", "Lasso": "red"}
alphas = {"Linear": 0.4, "Ridge": 0.4, "Lasso": 0.4}

metrics = {}

# Fit and print equation
for name, model in models.items():
    model.fit(X, y)
    coefs = ' + '.join([f'{model.coef_[i]:.4f}*x{i+1}' for i in range(X.shape[1])])
    print(f"{name} Regression Equation: y = {coefs} + {model.intercept_:.4f}")

    y_pred = model.predict(X)
    
    # พิมพ์ y_pred เทียบ y จริง
    print(f"\n{name} Regression Prediction vs Actual:")
    for xi, yi, ypi in zip(X, y, y_pred):
        print(f"x = {xi}, y_true = {yi:.4f}, y_pred = {ypi:.4f}")

    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    rsq = r2_score(y, y_pred)
    metrics[name] = {
        "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": rsq,
        "coef1": model.coef_[0], "coef2": model.coef_[1], "intercept": model.intercept_
    }

# Prepare meshgrid for surface
x1_range = np.linspace(X[:,0].min(), X[:,0].max(), 20)
x2_range = np.linspace(X[:,1].min(), X[:,1].max(), 20)
x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
X_grid = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]

# Plot 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Data points
ax.scatter(X[:,0], X[:,1], y, color='black', s=50, label='Data')

# Regression surfaces (label only here, one per model)
for name, model in models.items():
    y_surf = model.predict(X_grid).reshape(x1_mesh.shape)
    ax.plot_surface(x1_mesh, x2_mesh, y_surf, alpha=alphas[name], color=colors[name])

# Custom legend (no duplicates)
custom_lines = [
    mpatches.Patch(color='blue', alpha=0.4, label='Linear'),
    mpatches.Patch(color='green', alpha=0.4, label='Ridge'),
    mpatches.Patch(color='red', alpha=0.4, label='Lasso'),
]
ax.legend(
    handles=[plt.Line2D([], [], color='k', marker='o', linestyle='', label='Data')] + custom_lines
)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Linear, Ridge, Lasso Regression Surfaces')
plt.tight_layout()
plt.show()

# Show metrics table
pd.set_option('display.float_format', lambda x: '%.4f' % x)
df = pd.DataFrame(metrics).T
print(df[["MAE", "MSE", "RMSE", "R2", "coef1", "coef2", "intercept"]])
```
