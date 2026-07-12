from smolml.core.ml_array import MLArray

"""
//////////////////////
/// LOSS FUNCTIONS ///
//////////////////////

Loss functions for training machine learning models.
Each function computes error between predictions and true values.
"""

def mse_loss(y_pred, y_true):
    """
    Mean Squared Error Loss.
    Standard regression loss function.
    """
    diff = y_pred - y_true
    squared_diff = diff * diff
    return squared_diff.mean()

def mae_loss(y_pred, y_true):
    """
    Mean Absolute Error Loss.
    Less sensitive to outliers than MSE.
    """
    diff = (y_pred - y_true).abs()
    return diff.mean()

def binary_cross_entropy(y_pred, y_true):
    """
    Binary Cross-Entropy Loss.
    For binary classification problems.
    Expects y_pred to be in range (0,1).
    """
    epsilon = 1e-15  # Prevent log(0)
    y_pred = MLArray([[max(min(p, 1 - epsilon), epsilon) for p in row] for row in y_pred.data])
    return -(y_true * y_pred.log() + (1 - y_true) * (1 - y_pred).log()).mean()

def categorical_cross_entropy(y_pred, y_true):
    """
    Categorical Cross-Entropy Loss.
    For multi-class classification problems.
    Expects y_pred to be probability distribution.
    """
    epsilon = 1e-15
    y_pred = MLArray([[max(p, epsilon) for p in row] for row in y_pred.data])
    return -(y_true * y_pred.log()).sum(axis=1).mean()

def huber_loss(y_pred, y_true, delta=1.0):
    """
    Huber Loss.
    Combines MSE and MAE - quadratic for small errors, linear for large ones.
    More robust to outliers than MSE while maintaining smoothness.
    """
    diff = y_pred - y_true
    abs_diff = diff.abs()
    quadratic = 0.5 * diff * diff
    linear = delta * abs_diff - 0.5 * delta * delta
    return MLArray([[quad if abs_d <= delta else lin 
                    for quad, lin, abs_d in zip(row_quad, row_lin, row_abs)]
                    for row_quad, row_lin, row_abs in zip(quadratic.data, linear.data, abs_diff.data)]).mean()