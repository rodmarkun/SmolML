# SmolML - Loss Functions: Measuring How Wrong We Are

**What is a loss function?**

During training, we need a way to measure how "wrong" our model's predictions are compared to the actual target values (ground truth). This measure is the **loss** (or cost, or error). The goal of training is to adjust the model's parameters (weights/biases) to **minimize** this loss value.

<div align="center">
  <img src="https://github.com/user-attachments/assets/6fe8332d-904f-45f8-a2f6-9bca50ffd576" width="500">
</div>

Different loss functions are suited for different types of problems (regression vs. classification) and have different properties (e.g., sensitivity to outliers).

## Mean Squared Error (MSE)

Standard choice for **regression** problems:

$$L = \frac{1}{N} \sum_{i=1}^{N} (y_{pred, i} - y_{true, i})^2$$

```python
def mse_loss(y_pred, y_true):
    diff = y_pred - y_true
    squared_diff = diff * diff
    return squared_diff.mean()
```

Penalizes larger errors more heavily due to the squaring. Sensitive to outliers.

## Mean Absolute Error (MAE)

Another common choice for **regression**, less sensitive to outliers:

$$L = \frac{1}{N} \sum_{i=1}^{N} |y_{pred, i} - y_{true, i}|$$

```python
def mae_loss(y_pred, y_true):
    diff = (y_pred - y_true).abs()
    return diff.mean()
```

## Binary Cross-Entropy

Standard loss for **binary classification** where the model outputs a probability:

```python
def binary_cross_entropy(y_pred, y_true):
    epsilon = 1e-15  # Prevent log(0)
    y_pred = MLArray([[max(min(p, 1 - epsilon), epsilon) for p in row] for row in y_pred.data])
    return -(y_true * y_pred.log() + (1 - y_true) * (1 - y_pred).log()).mean()
```

The `epsilon` clipping prevents numerical issues when taking `log(0)`.

## Categorical Cross-Entropy

Standard loss for **multi-class classification**, comparing predicted probability distributions to true labels:

```python
def categorical_cross_entropy(y_pred, y_true):
    epsilon = 1e-15
    y_pred = MLArray([[max(p, epsilon) for p in row] for row in y_pred.data])
    return -(y_true * y_pred.log()).sum(axis=1).mean()
```

Expects `y_pred` to be a probability distribution across classes (output of softmax) and `y_true` to be one-hot encoded.

## Huber Loss

A hybrid that behaves like MSE for small errors and like MAE for large errors:

```python
def huber_loss(y_pred, y_true, delta=1.0):
    diff = y_pred - y_true
    abs_diff = diff.abs()
    quadratic = 0.5 * diff * diff
    linear = delta * abs_diff - 0.5 * delta * delta
    return MLArray([[quad if abs_d <= delta else lin
                    for quad, lin, abs_d in zip(row_quad, row_lin, row_abs)]
                    for row_quad, row_lin, row_abs in zip(quadratic.data, linear.data, abs_diff.data)]).mean()
```

The `delta` parameter controls the transition point. Useful for **regression** when you want robustness to outliers while maintaining smooth gradients near the minimum.

## Run the tests!

You can check out how these loss functions behave by running `loss_tests.py` in the `tests/` folder! It also outputs some fancy images to let you compare each of them.

[Next Section - Optimizers](https://github.com/rodmarkun/SmolML/tree/main/smolml/utils/optimizers)

## Resources & Readings

- [ML - Common Loss Functions](https://www.geeksforgeeks.org/machine-learning/ml-common-loss-functions/)
