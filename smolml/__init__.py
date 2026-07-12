# smolml/__init__.py

# Import core components
from .core import MLArray, Value

# Import models
from .models.nn import DenseLayer, NeuralNetwork
from .models.regression import LinearRegression, PolynomialRegression
from .models.tree import DecisionTree, RandomForest
from .models.unsupervised import KMeans

# Import preprocessing
from .preprocessing import (
    BPETokenizer,
    CharacterTokenizer,
    MinMaxScaler,
    StandardScaler,
    Tokenizer,
    WordPiece,
    WordTokenizer,
)

# Import utilities
from .utils.activation import (
    relu, leaky_relu, elu, sigmoid, softmax, tanh, linear
)
from .utils.initializers import (
    WeightInitializer, XavierUniform, XavierNormal, HeInitialization
)
from .utils.losses import (
    mse_loss, mae_loss, binary_cross_entropy, 
    categorical_cross_entropy, huber_loss
)
from .utils.optimizers import (
    Optimizer, SGD, SGDMomentum, AdaGrad, Adam
)
from .utils.memory import (format_size, calculate_value_size, calculate_mlarray_size, calculate_neural_network_size)

# Version of the smolml package
__version__ = '0.1.0'

__all__ = [
    # Core
    'MLArray',
    'Value',
    
    # Models - Neural Networks
    'DenseLayer',
    'NeuralNetwork',
    
    # Models - Regression
    'LinearRegression',
    'PolynomialRegression',
    
    # Models - Tree-based
    'DecisionTree',
    'RandomForest',

    # Models - Unsupervised
    'KMeans',
    
    # Preprocessing
    'StandardScaler',
    'MinMaxScaler',
    'Tokenizer',
    'CharacterTokenizer',
    'WordTokenizer',
    'BPETokenizer',
    'WordPiece',
    
    # Utils - Activation Functions
    'relu',
    'leaky_relu',
    'elu',
    'sigmoid',
    'softmax',
    'tanh',
    'linear',
    
    # Utils - Initializers
    'WeightInitializer',
    'XavierUniform',
    'XavierNormal',
    'HeInitialization',
    
    # Utils - Loss Functions
    'mse_loss',
    'mae_loss',
    'binary_cross_entropy',
    'categorical_cross_entropy',
    'huber_loss',

    # Utils - Optimizers
    'Optimizer',
    'SGD',
    'SGDMomentum',
    'AdaGrad',
    'Adam',

    # Utils - Memory
    'format_size',
    'calculate_value_size',
    'calculate_mlarray_size',
    'calculate_neural_network_size'
]
