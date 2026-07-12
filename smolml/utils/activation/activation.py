from smolml.core.ml_array import MLArray
import math

"""
////////////////////////////
/// ACTIVATION FUNCTIONS ///
////////////////////////////

Activation functions for neural network layers.
Each function applies non-linear transformations element-wise.
"""

def _element_wise_activation(x, activation_fn):
    """
    Helper function to apply activation function element-wise to n-dimensional MLArray
    """
    if len(x.shape) == 0:  # scalar
        return MLArray(activation_fn(x.data))
    
    def apply_recursive(data):
        if isinstance(data, list):
            return [apply_recursive(d) for d in data]
        return activation_fn(data)
    
    return MLArray(apply_recursive(x.data))

def relu(x):
    """
    Rectified Linear Unit (ReLU) activation.
    Computes max(0,x) for each element.
    Standard choice for deep networks.
    """
    return _element_wise_activation(x, lambda val: val.relu())

def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU activation.
    Returns x if x > 0, else alpha * x.
    Prevents dying ReLU problem with small negative slope.
    """
    def leaky_relu_single(val):
        if val > 0:
            return val
        return val * alpha
    
    return _element_wise_activation(x, leaky_relu_single)

def elu(x, alpha=1.0):
    """
    Exponential Linear Unit.
    Returns x if x > 0, else alpha * (e^x - 1).
    Smoother alternative to ReLU with negative values.
    """
    def elu_single(val):
        if val > 0:
            return val
        return alpha * (val.exp() - 1)
    
    return _element_wise_activation(x, elu_single)

def sigmoid(x):
    """
    Sigmoid activation.
    Maps inputs to (0,1) range using 1/(1 + e^-x).
    Used for binary classification output.
    """
    def sigmoid_single(val):
        return 1 / (1 + (-val).exp())
    
    return _element_wise_activation(x, sigmoid_single)

def softmax(x, axis=-1):
    """
    Softmax activation.
    Normalizes inputs into probability distribution.
    Used for multi-class classification output.
    """
    # Handle scalar case
    if len(x.shape) == 0:
        return MLArray(1.0)  # Softmax of a scalar is always 1
        
    # Handle negative axis
    if axis < 0:
        axis += len(x.shape)
        
    # Handle 1D case
    if len(x.shape) == 1:
        max_val = x.max()
        exp_x = (x - max_val).exp()
        sum_exp = exp_x.sum()
        return exp_x / sum_exp
    
    # Handle multi-dimensional case
    def apply_softmax_along_axis(data, curr_depth=0):
        """
        Recursively applies softmax along specified axis
        """
        if curr_depth == axis:
            if isinstance(data[0], list):
                # Convert to transpose without using zip
                transposed = []
                for i in range(len(data[0])):
                    slice_data = [row[i] for row in data]
                    # Find max for numerical stability
                    max_val = max(slice_data)
                    # Compute exp(x - max)
                    exp_vals = [(val - max_val).exp() for val in slice_data]
                    # Compute sum
                    sum_exp = sum(exp_vals)
                    # Compute softmax
                    softmax_vals = [exp_val / sum_exp for exp_val in exp_vals]
                    transposed.append(softmax_vals)
                    
                # Convert back from transpose without using zip
                result = []
                for i in range(len(data)):
                    row = [transposed[j][i] for j in range(len(transposed))]
                    result.append(row)
                return result
            else:
                # Direct computation for 1D slice
                max_val = max(data)
                exp_vals = [(val - max_val).exp() for val in data]
                sum_exp = sum(exp_vals)
                return [exp_val / sum_exp for exp_val in exp_vals]
        
        # Recursive case: not at target axis yet
        return [apply_softmax_along_axis(subarray, curr_depth + 1) 
                for subarray in data]
    
    result = apply_softmax_along_axis(x.data)
    return MLArray(result)

def tanh(x):
    """
    Hyperbolic tangent activation.
    Maps inputs to [-1,1] range.
    """
    return _element_wise_activation(x, lambda val: val.tanh())

def linear(x):
    """
    Linear/Identity activation.
    Passes input through unchanged.
    """
    return x