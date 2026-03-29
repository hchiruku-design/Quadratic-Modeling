import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('../data/weather.csv')

x = data['Day']
y = data['Temperature']

# Fit quadratic model
coeff = np.polyfit(x, y, 2)
model = np.poly1d(coeff)

# Predictions
y_pred = model(x)

# Print equation
print("Quadratic Equation:")
print(f"y = {coeff[0]:.3f}x^2 + {coeff[1]:.3f}x + {coeff[2]:.3f}")

# Plot
plt.scatter(x, y, label="Actual Data")
plt.plot(x, y_pred, color='red', label="Quadratic Fit")
plt.xlabel("Day")
plt.ylabel("Temperature")
plt.legend()

# Save graph
plt.savefig('../outputs/graph.png')

plt.show()
