# SmolML - Weight Initializers: A Good Start Matters

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

## Random Uniform

Simple uniform initialization in a specified range:

```python
class RandomUniform(WeightInitializer):
    @staticmethod
    def initialize(*dims, limit=1.0):
        dims = RandomUniform._process_dims(dims)
        return RandomUniform._create_array(lambda: random.uniform(-limit, limit), dims)
```

## Xavier/Glorot Initialization

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

## He/Kaiming Initialization

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

[Next Section - Loss Functions](https://github.com/rodmarkun/SmolML/tree/main/smolml/utils/losses)

## Resources & Readings

- [DeeplearningAI - Initializing Neural Networks (Interactive)](https://www.deeplearning.ai/ai-notes/initialization/index.html)
