# SmolML - The Utility Room!

Welcome to the utility components of SmolML! This directory houses the supporting building blocks required for constructing, training, and analyzing machine learning models within our framework. 

Let's go over them one by one.

## Activation Functions

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

### ReLU (Rectified Linear Unit)

The most common activation for hidden layers: outputs the input if positive, otherwise zero.

$$f(x) = \max(0, x)$$

```python
def relu(x):
    return _element_wise_activation(x, lambda val: val.relu())
```

Computationally efficient and helps mitigate vanishing gradients in deep networks.

### Leaky ReLU

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

### ELU (Exponential Linear Unit)

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

### Sigmoid

Squashes input values into the range (0, 1):

$$f(x) = \frac{1}{1 + e^{-x}}$$

```python
def sigmoid(x):
    def sigmoid_single(val):
        return 1 / (1 + (-val).exp())

    return _element_wise_activation(x, sigmoid_single)
```

Often used in the output layer for **binary classification** to interpret outputs as probabilities.

### Softmax

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

### Tanh (Hyperbolic Tangent)

Squashes input values into the range (-1, 1):

$$f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

```python
def tanh(x):
    return _element_wise_activation(x, lambda val: val.tanh())
```

Similar to sigmoid but zero-centered, which can be beneficial in some cases.

### Linear

Simply returns the input unchanged:

```python
def linear(x):
    return x
```

Used when no non-linearity is needed, for example in the output layer of a regression model.

## Weight Initializers

**Why is initialization important?**

When you create a neural network layer, its weights and biases need starting values. Choosing these initial values poorly can drastically hinder training:
* **Too small:** Gradients might become tiny as they propagate backward (vanishing gradients), making learning extremely slow or impossible.
* **Too large:** Gradients might explode, leading to unstable training (NaN values).
* **Symmetry:** If all weights start the same, neurons in the same layer will learn the same thing, defeating the purpose of having multiple neurons.

**Weight initializers** provide strategies to set these starting weights intelligently, breaking symmetry and keeping signals/gradients in a reasonable range.

All initializers inherit from a base class that provides a helper for creating arrays:

```python
class WeightInitializer:
   @staticmethod
   def _create_array(generator, dims):
       total_elements = reduce(mul, dims)
       flat_array = [generator() for _ in range(total_elements)]
       return MLArray(flat_array).reshape(*dims)
```

This creates an MLArray of the desired shape by generating random values and reshaping.

### Random Uniform

Simple uniform initialization in a specified range:

```python
class RandomUniform(WeightInitializer):
    @staticmethod
    def initialize(*dims, limit=1.0):
        dims = RandomUniform._process_dims(dims)
        return RandomUniform._create_array(lambda: random.uniform(-limit, limit), dims)
```

### Xavier/Glorot Initialization

Scales the initialization variance based on both `fan_in` (input units) and `fan_out` (output units). Works well with `sigmoid` and `tanh` activations.

**Xavier Uniform:**

$$\text{limit} = \sqrt{\frac{6}{fan_{in} + fan_{out}}}$$

```python
class XavierUniform(WeightInitializer):
   @staticmethod
   def initialize(*dims):
       dims = XavierUniform._process_dims(dims)
       fan_in = dims[0] if len(dims) > 0 else 1
       fan_out = dims[-1] if len(dims) > 1 else fan_in
       limit = math.sqrt(6. / (fan_in + fan_out))
       return XavierUniform._create_array(lambda: random.uniform(-limit, limit), dims)
```

**Xavier Normal:**

$$\sigma = \sqrt{\frac{2}{fan_{in} + fan_{out}}}$$

```python
class XavierNormal(WeightInitializer):
   @staticmethod
   def initialize(*dims):
       dims = XavierNormal._process_dims(dims)
       fan_in = dims[0] if len(dims) > 0 else 1
       fan_out = dims[-1] if len(dims) > 1 else fan_in
       std = math.sqrt(2. / (fan_in + fan_out))
       return XavierNormal._create_array(lambda: random.gauss(0, std), dims)
```

### He/Kaiming Initialization

Designed specifically for ReLU activations, accounting for the fact that ReLU zeros out half the inputs:

$$\sigma = \sqrt{\frac{2}{fan_{in}}}$$

```python
class HeInitialization(WeightInitializer):
   @staticmethod
   def initialize(*dims):
       dims = HeInitialization._process_dims(dims)
       fan_in = dims[0] if len(dims) > 0 else 1
       std = math.sqrt(2. / fan_in)
       return HeInitialization._create_array(lambda: random.gauss(0, std), dims)
```

## Loss Functions

**What is a loss function?**

During training, we need a way to measure how "wrong" our model's predictions are compared to the actual target values (ground truth). This measure is the **loss** (or cost, or error). The goal of training is to adjust the model's parameters (weights/biases) to **minimize** this loss value.

<div align="center">
  <img src="https://github.com/user-attachments/assets/6fe8332d-904f-45f8-a2f6-9bca50ffd576" width="500">
</div>

Different loss functions are suited for different types of problems (regression vs. classification) and have different properties (e.g., sensitivity to outliers).

### Mean Squared Error (MSE)

Standard choice for **regression** problems:

$$L = \frac{1}{N} \sum_{i=1}^{N} (y_{pred, i} - y_{true, i})^2$$

```python
def mse_loss(y_pred, y_true):
    diff = y_pred - y_true
    squared_diff = diff * diff
    return squared_diff.mean()
```

Penalizes larger errors more heavily due to the squaring. Sensitive to outliers.

### Mean Absolute Error (MAE)

Another common choice for **regression**, less sensitive to outliers:

$$L = \frac{1}{N} \sum_{i=1}^{N} |y_{pred, i} - y_{true, i}|$$

```python
def mae_loss(y_pred, y_true):
    diff = (y_pred - y_true).abs()
    return diff.mean()
```

### Binary Cross-Entropy

Standard loss for **binary classification** where the model outputs a probability:

```python
def binary_cross_entropy(y_pred, y_true):
    epsilon = 1e-15  # Prevent log(0)
    y_pred = MLArray([[max(min(p, 1 - epsilon), epsilon) for p in row] for row in y_pred.data])
    return -(y_true * y_pred.log() + (1 - y_true) * (1 - y_pred).log()).mean()
```

The `epsilon` clipping prevents numerical issues when taking `log(0)`.

### Categorical Cross-Entropy

Standard loss for **multi-class classification**, comparing predicted probability distributions to true labels:

```python
def categorical_cross_entropy(y_pred, y_true):
    epsilon = 1e-15
    y_pred = MLArray([[max(p, epsilon) for p in row] for row in y_pred.data])
    return -(y_true * y_pred.log()).sum(axis=1).mean()
```

Expects `y_pred` to be a probability distribution across classes (output of softmax) and `y_true` to be one-hot encoded.

### Huber Loss

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

---

## Optimizers

**What do optimizers do?**

Once we have calculated the loss, we know how wrong the model is. We also use backpropagation (handled by `MLArray`'s automatic differentiation) to calculate the **gradients** (how the loss changes with respect to each weight and bias in the model), as we already saw in [SmolML - Core](https://github.com/rodmarkun/SmolML/tree/main/smolml/core).

An **optimizer** an algorithm that uses these gradients to actually *update* the model's parameters (weights and biases) in a way that aims to decrease the loss over time.

All optimizers inherit from a base class:

```python
class Optimizer:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(self, object, object_idx, param_names):
        raise NotImplementedError
```

### SGD (Stochastic Gradient Descent)

The simplest optimizer: moves parameters directly opposite to the gradient.

$$\theta = \theta - \alpha \nabla_\theta L$$

```python
class SGD(Optimizer):
    def update(self, object, object_idx, param_names):
        new_params = tuple(
            getattr(object, name) - self.learning_rate * getattr(object, name).grad()
            for name in param_names
        )
        return new_params
```

Easy to understand, but can be slow and get stuck in local minima.

### SGD with Momentum

Adds a "momentum" term that accumulates past gradients, accelerating descent in consistent directions:

$$v = \beta v + \alpha \nabla_\theta L$$
$$\theta = \theta - v$$

```python
class SGDMomentum(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum_coefficient: float = 0.9):
        super().__init__(learning_rate)
        self.momentum_coefficient = momentum_coefficient
        self.velocities = {}

    def update(self, object, object_idx, param_names):
        # Initialize velocities for this layer if not exist
        if object_idx not in self.velocities:
            self.velocities[object_idx] = {
                name: zeros(*getattr(object, name).shape) for name in param_names
            }

        new_params = []
        for name in param_names:
            # Update velocity
            v = self.velocities[object_idx][name]
            v = self.momentum_coefficient * v + self.learning_rate * getattr(object, name).grad()
            self.velocities[object_idx][name] = v

            # Compute new parameter
            new_params.append(getattr(object, name) - v)

        return tuple(new_params)
```

The `velocities` dictionary maintains the velocity state per parameter across layers.

### AdaGrad (Adaptive Gradient)

Adapts the learning rate *per parameter*: smaller updates for frequently changing parameters, larger updates for infrequent ones.

$$\theta = \theta - \frac{\alpha}{\sqrt{G + \epsilon}} \nabla_\theta L$$

```python
class AdaGrad(Optimizer):
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)
        self.epsilon = 1e-8
        self.squared_gradients = {}

    def update(self, object, object_idx, param_names):
        if object_idx not in self.squared_gradients:
            self.squared_gradients[object_idx] = {
                name: zeros(*getattr(object, name).shape) for name in param_names
            }

        new_params = []
        for name in param_names:
            # Update squared gradients sum
            self.squared_gradients[object_idx][name] += getattr(object, name).grad()**2

            # Compute new parameter
            new_params.append(
                getattr(object, name) - (self.learning_rate /
                (self.squared_gradients[object_idx][name] + self.epsilon).sqrt()) *
                getattr(object, name).grad()
            )

        return tuple(new_params)
```

Good for sparse data (like in NLP), but learning rate monotonically decreases and can become too small.

### Adam (Adaptive Moment Estimation)

Combines Momentum (1st moment) and RMSProp/AdaGrad (2nd moment) with bias correction:

$$m = \beta_1 m + (1 - \beta_1) \nabla_\theta L$$
$$v = \beta_2 v + (1 - \beta_2) (\nabla_\theta L)^2$$
$$\hat{m} = \frac{m}{1 - \beta_1^t}, \quad \hat{v} = \frac{v}{1 - \beta_2^t}$$
$$\theta = \theta - \alpha \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

> Here's a [strongly recommended read](https://medium.com/@daga.yash/bias-correction-in-adam-the-statistical-intuition-5908daa01168) on the Adam optimizer in case you're interested! :)

```python
class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.01, exp_decay_gradients: float = 0.9, exp_decay_squared: float = 0.999):
        super().__init__(learning_rate)
        self.exp_decay_gradients = exp_decay_gradients
        self.exp_decay_squared = exp_decay_squared
        self.gradients_momentum = {}
        self.squared_gradients_momentum = {}
        self.epsilon = 1e-8
        self.timestep = 1

    def update(self, object, object_idx, param_names):
        # Initialize momentums if not exist
        if object_idx not in self.gradients_momentum:
            self.gradients_momentum[object_idx] = {
                name: zeros(*getattr(object, name).shape) for name in param_names
            }
            self.squared_gradients_momentum[object_idx] = {
                name: zeros(*getattr(object, name).shape) for name in param_names
            }

        new_params = []
        for name in param_names:
            # Update biased first moment estimate
            self.gradients_momentum[object_idx][name] = (
                self.exp_decay_gradients * self.gradients_momentum[object_idx][name] +
                (1 - self.exp_decay_gradients) * getattr(object, name).grad()
            )

            # Update biased second moment estimate
            self.squared_gradients_momentum[object_idx][name] = (
                self.exp_decay_squared * self.squared_gradients_momentum[object_idx][name] +
                (1 - self.exp_decay_squared) * getattr(object, name).grad()**2
            )

            # Compute bias-corrected moments
            m = self.gradients_momentum[object_idx][name] / (1 - self.exp_decay_gradients ** self.timestep)
            v = self.squared_gradients_momentum[object_idx][name] / (1 - self.exp_decay_squared ** self.timestep)

            # Compute new parameter
            new_params.append(
                getattr(object, name) - self.learning_rate * m / (v.sqrt() + self.epsilon)
            )

        self.timestep += 1
        return tuple(new_params)
```

Often considered a robust, effective default optimizer for many problems.

## Run the tests!

Remember that you can check out how do each of these components work by running the correspondent test in the `tests/` folder! They also output some fancy images to let you compare each of them.

## Resources & Readings

- [ML - Common Loss Functions](https://www.geeksforgeeks.org/machine-learning/ml-common-loss-functions/)
- [Bias Correction in Adam: The Statistical Intuition](https://medium.com/@daga.yash/bias-correction-in-adam-the-statistical-intuition-5908daa01168)
- [DeeplearningAI - Initializing Neural Networks (Interactive)](https://www.deeplearning.ai/ai-notes/initialization/index.html)