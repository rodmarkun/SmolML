# SmolML - Activation Functions: Adding Non-Linearity

**Why do we need them?**

Imagine building a neural network. If you just stack linear operations (like matrix multiplications and additions), the entire network, no matter how deep, behaves like a single *linear* transformation. This severely limits the network's ability to learn complex, non-linear patterns often found in real-world data (like image recognition, language translation, etc.).

**Activation functions** introduce **non-linearity** into the network, typically applied element-wise after a linear transformation in a layer. This allows the network to approximate much more complicated functions.

<div align="center">
  <img src="https://github.com/user-attachments/assets/c610f284-dbf2-4a69-8f88-5433a28276cb" width="600">
</div>

Most of our activation functions use a helper that applies the transformation element-wise to any n-dimensional MLArray:

```python
def _element_wise_activation(x, activation_fn):
    if len(x.shape) == 0:  # scalar
        return MLArray(activation_fn(x.data))

    def apply_recursive(data):
        if isinstance(data, list):
            return [apply_recursive(d) for d in data]
        return activation_fn(data)

    return MLArray(apply_recursive(x.data))
```

This recursively traverses the nested structure of the MLArray and applies the activation function to each `Value` element.

## ReLU (Rectified Linear Unit)

The most common activation for hidden layers: outputs the input if positive, otherwise zero.

$$f(x) = \max(0, x)$$

```python
def relu(x):
    return _element_wise_activation(x, lambda val: val.relu())
```

Computationally efficient and helps mitigate vanishing gradients in deep networks.

## Leaky ReLU

Like ReLU, but allows a small gradient for negative inputs to prevent "dying neurons":

$$f(x) = x \text{ if } x > 0, \text{ else } \alpha x$$

```python
def leaky_relu(x, alpha=0.01):
    def leaky_relu_single(val):
        if val > 0:
            return val
        return val * alpha

    return _element_wise_activation(x, leaky_relu_single)
```

## ELU (Exponential Linear Unit)

Similar to Leaky ReLU but uses an exponential curve for negative inputs:

$$f(x) = x \text{ if } x > 0, \text{ else } \alpha (e^x - 1)$$

```python
def elu(x, alpha=1.0):
    def elu_single(val):
        if val > 0:
            return val
        return alpha * (val.exp() - 1)

    return _element_wise_activation(x, elu_single)
```

Smoother than ReLU/Leaky ReLU and can speed up learning.

## Sigmoid

Squashes input values into the range (0, 1):

$$f(x) = \frac{1}{1 + e^{-x}}$$

```python
def sigmoid(x):
    def sigmoid_single(val):
        return 1 / (1 + (-val).exp())

    return _element_wise_activation(x, sigmoid_single)
```

Often used in the output layer for **binary classification** to interpret outputs as probabilities.

## Softmax

Transforms a vector into a probability distribution (values are non-negative and sum to 1):

```python
def softmax(x, axis=-1):
    # Handle scalar case
    if len(x.shape) == 0:
        return MLArray(1.0)

    # Handle negative axis
    if axis < 0:
        axis += len(x.shape)

    # Handle 1D case
    if len(x.shape) == 1:
        max_val = x.max()
        exp_x = (x - max_val).exp()  # Numerical stability
        sum_exp = exp_x.sum()
        return exp_x / sum_exp

    # Handle multi-dimensional case recursively...
```

Essential for the output layer in **multi-class classification**. The `axis` argument determines along which dimension the normalization occurs. Note the numerical stability trick: subtracting the max value before exponentiation prevents overflow.

## Tanh (Hyperbolic Tangent)

Squashes input values into the range (-1, 1):

$$f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

```python
def tanh(x):
    return _element_wise_activation(x, lambda val: val.tanh())
```

Similar to sigmoid but zero-centered, which can be beneficial in some cases.

## Linear

Simply returns the input unchanged:

```python
def linear(x):
    return x
```

Used when no non-linearity is needed, for example in the output layer of a regression model.

## Run the tests!

You can check out how these activation functions behave by running `activation_tests.py` in the `tests/` folder! It also outputs some fancy images to let you compare each of them.

[Next Section - Weight Initializers](https://github.com/rodmarkun/SmolML/tree/main/smolml/utils/initializers)

## Resources & Readings

- [Wikipedia - Activation function](https://en.wikipedia.org/wiki/Activation_function)
