# SmolML - Memory Utilities: How Big is My Model?

**Why measure memory?**

SmolML is written in pure Python for clarity, and that comes at a price: every single number in an `MLArray` is a full `Value` object carrying its data, its gradient, and the bookkeeping needed for backpropagation. That's *much* heavier than a tightly packed NumPy array! These utilities let you inspect exactly how much memory your models are using, which is a great way to appreciate why production libraries rely on optimized C/C++ backends.

Models like `NeuralNetwork`, `DecisionTree` or `RandomForest` use these functions internally when printing their architecture summaries via `__repr__`.

## Formatting Sizes

A tiny helper turns raw byte counts into something human-readable:

```python
def format_size(size_bytes):
    """Helper function to format size in bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"
```

## Measuring the Building Blocks

Everything in SmolML is built on `Value` objects, so we start there, using Python's `sys.getsizeof` to account for each attribute:

```python
def calculate_value_size(value: 'Value') -> int:
    total = sys.getsizeof(value)
    total += sys.getsizeof(value.data)  # float
    total += sys.getsizeof(value.grad)  # float
    total += sys.getsizeof(value._prev)  # set
    total += sys.getsizeof(value._op)    # str
    total += sys.getsizeof(value._backward)  # function
    return total
```

An `MLArray` is just a nested structure of `Value` objects, so we traverse it recursively:

```python
def calculate_mlarray_size(arr: 'MLArray') -> int:
    def get_nested_size(data):
        if isinstance(data, Value):
            return calculate_value_size(data)
        elif isinstance(data, list):
            size = sys.getsizeof(data)
            size += sum(get_nested_size(item) for item in data)
            return size
        else:
            return sys.getsizeof(data)

    return sys.getsizeof(arr) + get_nested_size(arr.data)
```

## Measuring Whole Models

On top of these building blocks, we provide one calculator per model type:

* `calculate_regression_size`: sums the weights, bias, and any optimizer state (like Adam's momentum arrays) of a regression model.
* `calculate_neural_network_size`: reports a per-layer breakdown (weights and biases) plus the optimizer state.
* `calculate_decision_tree_size`: walks the tree recursively with `calculate_decision_node_size`, also reporting structure stats like node counts and max depth.
* `calculate_random_forest_size`: aggregates the size of every tree in the forest, along with averages like tree depth and node count.

Each returns a dictionary with a `total` plus a detailed breakdown. For example, for a neural network:

```python
def calculate_neural_network_size(model: 'NeuralNetwork') -> Dict[str, Any]:
    size_info = {
        'total': 0,
        'layers': [],
        'optimizer': {
            'size': sys.getsizeof(model.optimizer),
            'state': {}
        }
    }

    # Calculate size of each layer
    for layer in model.layers:
        layer_info = {
            'weights_size': calculate_mlarray_size(layer.weights),
            'biases_size': calculate_mlarray_size(layer.biases),
        }
        layer_info['total'] = layer_info['weights_size'] + layer_info['biases_size']
        size_info['layers'].append(layer_info)
        size_info['total'] += layer_info['total']
    ...
```

Try printing one of your models and you'll see these numbers in action — a good reminder that understanding *costs* is part of understanding machine learning systems!

[Next Section - Tree Models](https://github.com/rodmarkun/SmolML/tree/main/smolml/models/tree)

## Resources & Readings

- [Python Docs - sys.getsizeof](https://docs.python.org/3/library/sys.html#sys.getsizeof)
