# SmolML: A Machine Learning Library from Scratch!

<div align="center">
<b>SmolML is a pure Python machine learning library built entirely from the ground up for educational purposes. Made to teach and understand how ML really works!</b>
<img src="https://github.com/user-attachments/assets/c00b89e9-58a3-44d8-b9c3-4b47052eb150" width="600" alt="SmolML Cover">
</div>

---

## What is SmolML? 🤔

Ever wondered what goes on *inside* those powerful machine learning libraries like [Scikit-Learn](https://scikit-learn.org/stable/), [PyTorch](https://pytorch.org/), or [TensorFlow](https://www.tensorflow.org/)? How does a neural network *actually* learn? How is gradient descent implemented? How do different data-handling tools work?

SmolML is a fully functional (though *simplified*) machine learning library built using **only pure Python** and basic (`collections`, `random`, and `math`) modules. No NumPy, no SciPy, no C++ extensions. Just Python, all the way down.

The goal is to provide a **transparent, understandable, and educational** implementation of core machine learning concepts.

## Walkthrough 📖

You can read these guides of the different sections of SmolML in any order, though this list presents the recommended order for learning.

- [SmolML - Core: Automatic Differentiation & N-Dimensional Arrays](https://github.com/rodmarkun/SmolML/tree/main/smolml/core)
- [SmolML - Regression: Predicting Continuous Values](https://github.com/rodmarkun/SmolML/tree/main/smolml/models/regression)
- [SmolML - Neural Networks: Backpropagation to the limit](https://github.com/rodmarkun/SmolML/tree/main/smolml/models/nn)
- [SmolML - Tree Models: Decisions, Decisions!](https://github.com/rodmarkun/SmolML/tree/main/smolml/models/tree)
- [SmolML - K-Means: Finding Groups in Your Data!](https://github.com/rodmarkun/SmolML/tree/main/smolml/models/unsupervised)
- [SmolML - Preprocessing: Make your data meaningful](https://github.com/rodmarkun/SmolML/tree/main/smolml/preprocessing)
- [SmolML - The utility room!](https://github.com/rodmarkun/SmolML/tree/main/smolml/utils)

### Approach

We believe the best way to truly understand complex topics like machine learning is often to **build them yourself**. Production libraries are fantastic tools, but their internal complexity and optimizations can sometimes obscure the fundamental principles.

SmolML strips away these layers to focus on the core ideas:
* Every major component is **built from scratch**, letting you trace the logic from basic operations to complex algorithms.
* See how concepts like automatic differentiation (autograd), optimization algorithms, and model architectures are **implemented in code**, so that you can implement them yourself.
* Relying only on Python's standard library makes the codebase **accessible and easy to explore** without external setup hurdles.
* Code is written with **understanding**, not peak performance, as the primary goal.

> In order to learn as much as possible, we recommend reading through the guides, checking the code, and then trying to implement your own versions of these components.

## Features

SmolML provides an implementation of the essential building blocks for any Machine Learning library:

* **The Foundation: Custom Arrays & Autograd Engine:**
    * **Automatic Differentiation (`Value`):** A simple autograd engine that tracks operations and computes gradients automatically. (See `smolml/core/value.py`)
    * **N-dimensional Arrays (`MLArray`):** A custom array implementation inspired by [NumPy](https://numpy.org/) (though simplified), supporting common mathematical operations needed for ML. Extremely inefficient due to being written in Python, but ideal for understanding N-Dimensional Arrays, one of the most underrated skills of a ML engineer. (See `smolml/core/ml_array.py`)

* **Essential Preprocessing:**
    * **Scalers (`StandardScaler`, `MinMaxScaler`):** Fundamental tools to prepare your data, because algorithms tend to perform better when features are on a similar scale. (See `smolml/preprocessing/scalers.py`)

* **Build Your Own Neural Networks:**
    * **Activation Functions:** Non-linearities like `relu`, `sigmoid`, `softmax`, `tanh` that allow networks to learn complex patterns. (See `smolml/utils/activation.py`)
    * **Weight Initializers:** Smart strategies (`Xavier`, `He`) to set initial network weights for stable training. (See `smolml/utils/initializers.py`)
    * **Loss Functions:** Ways to measure model error (`mse_loss`, `binary_cross_entropy`, `categorical_cross_entropy`). (See `smolml/utils/losses.py`)
    * **Optimizers:** Algorithms like `SGD`, `Adam`, and `AdaGrad` that update model weights based on gradients to minimize loss. (See `smolml/utils/optimizers.py`)

* **Classic ML Models:**
    * **Regression:** Implementations of `Linear` and `Polynomial` regression.
    * **Neural Networks:** A flexible framework for building feed-forward neural networks.
    * **Tree-Based Models:** `Decision Tree` and `Random Forest` implementations for classification and regression.
    * **K-Means:** `KMeans` unsupervised clustering algorithm for grouping similar data points together.

## Limitations

SmolML is built for **learning**, and thus it should not be used for production. Being pure Python, it's **WAAAAY** slower and uses a ton more memory than libraries using optimized C/C++/Fortran backends (like NumPy).

It's best suited for small datasets and toy problems where understanding the mechanics is more important than computation time. **Do not** use SmolML for production applications. Stick to battle-tested libraries like Scikit-learn, PyTorch, TensorFlow, JAX, etc., for real-world tasks.

## Getting Started

The best way to use SmolML is to clone this repository and explore the code and examples.

```bash
git clone https://github.com/rodmarkun/SmolML
cd SmolML
# Explore the code in the smolml/ directory!
```

You can also run the multiple tests in the `tests/` folder. Just install the `requirements.txt` (this is for comparing SmolML against another standard libraries like TensorFlow, sklearn, etc, and generate plots with matplotlib. SmolML does not use any of these libraries whatsoever).

```bash
cd tests
pip install -r requirements
```

## Contributing

Contributions are always welcome! If you're interested in contributing to SmolML, please fork the repository and create a new branch for your changes. When you're done with your changes, submit a pull request to merge your changes into the main branch.

## Supporting SmolML

If you want to support SmolML, you can:
- **Star** :star: the project in Github!
- **Donate** :coin: to my [Ko-fi](https://ko-fi.com/rodmarkun) page!
- **Share** :heart: the project with your friends!
