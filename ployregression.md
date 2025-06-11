```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate synthetic data WITHOUT noise
X = np.linspace(1, 10, 20)
y_poly = 5 + 3 * X**2    # ไม่มี noise

# Transform data using Polynomial Features (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X.reshape(-1, 1))

# Fit Polynomial Regression Model
model_poly = LinearRegression().fit(X_poly, y_poly)

# Predict y values
y_pred_poly = model_poly.predict(X_poly)

# Plot Polynomial Regression Results
plt.figure(figsize=(6, 5))
plt.scatter(X, y_poly, label="Actual Data", color="blue")
plt.plot(X, y_pred_poly, label="Polynomial Fit (deg=2)", color="red")
plt.title("Polynomial Regression (Degree 2) [No Noise]")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```
