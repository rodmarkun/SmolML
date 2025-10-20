# SmolML - Regression: Predicting Continuous Values

Building upon the core concepts of automatic differentiation (`Value`) and N-dimensional arrays (`MLArray`) explained in the [SmolML Core](https://github.com/rodmarkun/SmolML/tree/main/smolml/core), we can now implement various machine learning models. This section focuses on **regression models**, which are used to predict continuous numerical outputs. Think of predicting house prices, stock values, or temperature based on input features.

While deep neural networks offer immense power, simpler models like Linear Regression or its extension, Polynomial Regression, are often excellent starting points, computationally efficient, and highly interpretable. They share the same fundamental learning principle as complex networks: minimizing a loss function by adjusting parameters using gradient descent, all powered by our automatic differentiation engine using the `Value` class.

## Regression Fundamentals: Learning from Data

The goal in regression is to find a mathematical function that maps input features (like the square footage of a house) to a continuous output (like its price). This function has internal parameters (often called **weights** or coefficients, and a **bias** or intercept) that determine its exact shape.

<div align="center">
  <img src="https://github.com/user-attachments/assets/79874cec-8650-4628-af1f-ca6fdc4debe5" width="600">
</div>

 > *(I highly recommend to [check out this deep-dive into Linear Regression by MLU-Explain](https://mlu-explain.github.io/linear-regression/), it's very visual!)*

How do we find the *best* parameters?
1.  **Prediction:** We start with initial (often random) parameters and use the model to make predictions on our training data.
2.  **Loss Calculation:** We compare these predictions to the actual known values using a **loss function** (like Mean Squared Error - MSE). This function quantifies *how wrong* the model currently is. A lower loss is better.
3.  **Gradient Calculation:** Just like in the core explanation, we need to know how to adjust each parameter to reduce the loss. Our `Value` objects and the concept of **backpropagation** automatically calculate the **gradient** of the loss with respect to each parameter (weights and bias). Remember, the gradient points towards the steepest *increase* in loss.
4.  **Parameter Update:** We use an **optimizer** (like Stochastic Gradient Descent - SGD) to nudge the parameters in the *opposite* direction of their gradients, taking a small step towards lower loss.
5.  **Iteration:** We repeat steps 1-4 many times (iterations or epochs), gradually improving the model's parameters until the loss is minimized or stops decreasing significantly.

This iterative process allows the regression model to "learn" the underlying relationship between the inputs and outputs from the data.

## The `Regression` Base Class

To streamline the implementation of different regression algorithms, in SmolML we made a `Regression` base class (in `regression.py`). This class handles the common structure and the training loop logic. Specific models like `LinearRegression` inherit from it.

Here's how it works:

* **Initialization (`__init__`)**:
    * Accepts the `input_size` (number of expected input features), a `loss_function`, an `optimizer` instance, and a weight `initializer`.
    * Crucially, it initializes the model's **trainable parameters**:
        * `self.weights`: An `MLArray` holding the coefficients for each input feature. Its shape is determined by `input_size`, and values are set by the `initializer`.
        * `self.bias`: A scalar `MLArray` (initialized to 1) representing the intercept term.
    * Because `weights` and `bias` are `MLArray`s, they inherently contain `Value` objects. This ensures they are part of the computational graph and their gradients can be automatically computed during training.

* **Training (`fit`)**:
    * This method orchestrates the gradient descent loop described earlier. For a specified number of `iterations`:
        1.  **Forward Pass:** Calls `self.predict(X)` (which must be implemented by the subclass) to get predictions `y_pred`. This builds the computational graph for the prediction step.
        2.  **Loss Calculation:** Computes `loss = self.loss_function(y, y_pred)`. This `loss` is the final `MLArray` (usually containing a single `Value`) representing the overall error for this iteration.
        3.  **Backward Pass:** Invokes `loss.backward()`. This triggers the automatic differentiation process, calculating the gradients of the loss with respect to all involved `Value` objects, including those within `self.weights` and `self.bias`.
        4.  **Parameter Update:** Uses `self.optimizer.update(...)` to adjust `self.weights` and `self.bias` based on their computed gradients (`weights.grad()` and `bias.grad()`) and the optimizer's logic (e.g., learning rate).
        5.  **Gradient Reset:** Calls `self.restart(X, y)` to zero out all gradients (`.grad` attributes of the `Value` objects) in the parameters and data, preparing for the next iteration.

* **Prediction (`predict`)**:
    * Defined in the base class but raises `NotImplementedError`. Why? Because the core logic of *how* to make a prediction differs between regression types (e.g., linear vs. polynomial). Each subclass *must* provide its own `predict` method defining its specific mathematical formula using `MLArray` operations.

* **Gradient Reset (`restart`)**:
    * A helper that simply calls the `.restart()` method on the `weights`, `bias`, input `X`, and target `y` `MLArray`s. This efficiently resets the `.grad` attribute of all underlying `Value` objects to zero.

* **Representation (`__repr__`)**:
    * Provides a nicely formatted string summary of the configured model, including its type, parameter shapes, optimizer, loss function, and estimated memory usage.

## Specific Models Implemented

<div align="center">
  <img src="https://github.com/user-attachments/assets/8b282ca1-7c17-460d-a64c-61b0624627f9" width="600">
</div>

### `LinearRegression`

This is the most fundamental regression model. It assumes a direct linear relationship between the input features `X` and the output `y`. The goal is to find the best weights `w` and bias `b` such that $y \approx Xw + b$.

* **Implementation (`regression.py`)**:
    * Inherits directly from `Regression`.
    * Its primary contribution is overriding the `predict` method.
* **Prediction (`predict`)**:
    * Implements the linear equation: `return X @ self.weights + self.bias`.
    * It takes the input `X` (`MLArray`), performs matrix multiplication (`@`) with `self.weights` (`MLArray`), and adds `self.bias` (`MLArray`). Because `X`, `weights`, and `bias` are all `MLArray`s containing `Value` objects, this single line of code automatically constructs the necessary computational graph for backpropagation.
* **Training**:
    * Uses the `fit` method directly inherited from the `Regression` base class without modification. The base class handles the entire training loop using the `predict` logic provided by `LinearRegression`.

### `PolynomialRegression`

What if the relationship isn't a straight line? Polynomial Regression extends linear regression by fitting a polynomial curve (e.g., $y \approx w_2 x^2 + w_1 x + b$) to the data.

* **Implementation (`regression.py`)**:
    * Also inherits from `Regression`.
* **The Core Idea**: Instead of directly fitting `X` to `y`, it first *transforms* the input features `X` into polynomial features (e.g., adding $X^2$, $X^3$, etc.) and then applies a standard *linear regression* model to these *new, transformed* features.
* **Initialization (`__init__`)**:
    * Takes an additional `degree` argument, specifying the highest power to include in the feature transformation (e.g., `degree=2` means include $X$ and $X^2$).
    * It calls the base class `__init__`, but the `input_size` passed to the base class is effectively the number of *polynomial features*, not the original number of features. The weights will correspond to these transformed features.
* **Feature Transformation (`transform_features`)**:
    * This crucial method takes the original input `X` and generates the new polynomial features. For an input `X` and `degree=d`, it calculates $X, X^2, \dots, X^d$ using `MLArray` operations (like element-wise multiplication `*`) and concatenates them into a new `MLArray`. This ensures the transformation is also potentially part of the graph if needed (though often it's pre-calculated).
* **Prediction (`predict`)**:
    1.  It first calls `X_poly = self.transform_features(X)` to get the polynomial features.
    2.  Then, it performs a standard linear prediction using these transformed features: `return X_poly @ self.weights + self.bias`. The `self.weights` here correspond to the coefficients of the polynomial terms.
* **Training (`fit`)**:
    * It overrides the base `fit` method slightly.
    1.  Before the main loop, it transforms the entire training input `X` into `X_poly = self.transform_features(X)`.
    2.  It then calls the *base class's* `fit` method (`super().fit(...)`) but passes `X_poly` (instead of `X`) as the input data.
    * The inherited `fit` method then proceeds as usual, calculating loss based on the predictions from `X_poly`, backpropagating gradients through the linear prediction part *and* the feature transformation step, and updating the weights associated with the polynomial terms.

## Example Usage

Here's a conceptual example of how you might use `LinearRegression`:

```python
from smolml.models.regression import LinearRegression
from smolml.core.ml_array import MLArray
import smolml.utils.optimizers as optimizers
import smolml.utils.losses as losses

# Sample Data (e.g., 2 features, 3 samples)
X_data = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
# Target values (continuous)
y_data = [[3.5], [5.5], [7.5]]

# Convert to MLArray
X = MLArray(X_data)
y = MLArray(y_data)

# Initialize the model
# Expects 2 input features
model = LinearRegression(input_size=2,
                         optimizer=optimizers.SGD(learning_rate=0.01),
                         loss_function=losses.mse_loss)

# Print initial model summary
print(model)

# Train the model
print("\nStarting training...")
losses_history = model.fit(X, y, iterations=100, verbose=True, print_every=10)
print("Training complete.")

# Print final model summary (weights/bias will have changed)
print(model)

# Make predictions on new data
X_new = MLArray([[4.0, 5.0]])
prediction = model.predict(X_new)
print(f"\nPrediction for {X_new.to_list()}: {prediction.to_list()}")
```

## Regression wrap-up

These regression classes showcase how the foundational `Value` and `MLArray` we implemented can be used to design and train classic machine learning models! In just a few lines of code! Isn't that cool?
