import numpy as np
import matplotlib.pyplot as plt

# Sample data (Area vs Price)
X = np.array([1500, 1600, 1700, 1800, 2000])
y = np.array([300000, 320000, 340000, 360000, 400000])

# Normalize features (optional but helpful)
X = (X - X.mean()) / X.std()

# Initialize parameters
m = 0  # slope
c = 0  # intercept

# Hyperparameters
learning_rate = 0.01
epochs = 1000
n = len(X)

# Loss function history
loss_history = []

# Gradient descent loop
for i in range(epochs):
    y_pred = m * X + c
    error = y - y_pred
    
    # Mean Squared Error
    loss = (error ** 2).mean()
    loss_history.append(loss)
    
    # Gradients
    dm = (-2/n) * sum(X * error)
    dc = (-2/n) * sum(error)
    
    # Update parameters
    m = m - learning_rate * dm
    c = c - learning_rate * dc

# Final model
print(f"Final slope (m): {m:.3f}")
print(f"Final intercept (c): {c:.3f}")

# Plot
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, m * X + c, color='red', label='Fitted Line')
plt.title('Manual Linear Regression with Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()

# Plot loss
plt.plot(loss_history)
plt.title('Loss over Iterations')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)
plt.show()
