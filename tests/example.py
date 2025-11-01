from smolml.models.regression import PolynomialRegression
from smolml.core.ml_array import MLArray, randn
import smolml.utils.optimizers as optimizers
import smolml.utils.losses as losses
import smolml.utils.initializers as initializers

# Generate non-linear sample data (e.g., quadratic relationship)
# y ≈ 2x² + 3x + 1 + noise
X_data = [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
y_data = [[3.0], [0.5], [1.0], [6.0], [15.0]]

# Convert to MLArray
X = MLArray(X_data)
y = MLArray(y_data)

# Initialize polynomial regression model with degree 2
# This will create features: [x, x²]
model = PolynomialRegression(
    input_size=1,
    degree=2,
    optimizer=optimizers.SGD(learning_rate=0.05),
    loss_function=losses.mse_loss,
    initializer=initializers.XavierUniform()
)

# Print initial model summary
print(model)

# Train the model
print("\nStarting training...")
losses_history = model.fit(X, y, iterations=200, verbose=True, print_every=20)
print("Training complete.")

# Print final model summary
print(model)

# Make predictions on new data
X_new = MLArray([[3.0]])
prediction = model.predict(X_new)
print(f"\nPrediction for {X_new.to_list()}: {prediction.to_list()}")

# Example with multiple features and interactions
# Generate 2-feature data: y = 2x₁ + 3x₂ + x₁² + x₁x₂ + noise
# For 2 features with degree 2, it creates: [x₁, x₂, x₁², x₁x₂, x₂²]
import numpy as np

X_1 = randn(30, 1)
X_2 = randn(30, 1)

X_data = []
y_data = []
for i in range(30):
    x1 = X_1.data[i][0].data
    x2 = X_2.data[i][0].data
    X_data.append([x1, x2])
    y_val = 2*x1 + 3*x2 + x1*x1 + x1*x2 + np.random.randn()*0.1
    y_data.append([y_val])

X_multi = MLArray(X_data)
y_multi = MLArray(y_data)

model_multi = PolynomialRegression(
    input_size=2,
    degree=2,
    optimizer=optimizers.SGD(learning_rate=0.05),
    loss_function=losses.mse_loss,
    initializer=initializers.XavierUniform()
)

print("\n" + "="*50)
print("Training multi-feature polynomial regression...")
print(model_multi)
model_multi.fit(X_multi, y_multi, iterations=200, verbose=True, print_every=20)

X_test = MLArray([[1.5, 2.0]])
pred_multi = model_multi.predict(X_test)
print(f"\nPrediction for {X_test.to_list()}: {pred_multi.to_list()}")