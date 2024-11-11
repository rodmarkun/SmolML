# smolml/__init__.py

# Import core components
from .core import MLArray, Value

# Import models
from .models.nn import DenseLayer, NeuralNetwork
from .models.regression import LinearRegression, PolynomialRegression
from .models.tree import DecisionTree, RandomForest

# Import preprocessing
from .preprocessing import StandardScaler, MinMaxScaler

# Import utilities
from .utils.activation import (
    relu, leaky_relu, elu, sigmoid, softmax, tanh, linear
)
from .utils.initializers import (
    WeightInitializer, XavierUniform, XavierNormal, HeInitialization
)
from .utils.losses import (
    mse_loss, mae_loss, binary_cross_entropy, 
    categorical_cross_entropy, huber_loss, log_cosh_loss
)
from .utils.optimizers import (
    Optimizer, SGD, SGDMomentum, AdaGrad, Adam
)

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
    
    # Preprocessing
    'StandardScaler',
    'MinMaxScaler',
    
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
    'log_cosh_loss',
    
    # Utils - Optimizers
    'Optimizer',
    'SGD',
    'SGDMomentum',
    'AdaGrad',
    'Adam'
]