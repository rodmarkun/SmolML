# SmolML - Optimizers: Learning from Gradients

**What do optimizers do?**

Once we have calculated the loss, we know how wrong the model is. We also use backpropagation (handled by `MLArray`'s automatic differentiation) to calculate the **gradients** (how the loss changes with respect to each weight and bias in the model), as we already saw in [SmolML - Core](https://github.com/rodmarkun/SmolML/tree/main/smolml/core).

An **optimizer** is an algorithm that uses these gradients to actually *update* the model's parameters (weights and biases) in a way that aims to decrease the loss over time.

All optimizers inherit from a base class:

```python
class Optimizer:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(self, object, object_idx, param_names):
        raise NotImplementedError
```

## SGD (Stochastic Gradient Descent)

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

## SGD with Momentum

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

## AdaGrad (Adaptive Gradient)

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

## Adam (Adaptive Moment Estimation)

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

You can check out how these optimizers compare against each other by running `optimizer_tests.py` in the `tests/` folder! It also outputs some fancy images to let you compare each of them.

[Next Section - Memory Utilities](https://github.com/rodmarkun/SmolML/tree/main/smolml/utils/memory)

## Resources & Readings

- [Bias Correction in Adam: The Statistical Intuition](https://medium.com/@daga.yash/bias-correction-in-adam-the-statistical-intuition-5908daa01168)
