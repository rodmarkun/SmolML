from smolml.core.ml_array import MLArray
import math
import random
from functools import reduce
from operator import mul

"""
////////////////////
/// INITIALIZERS ///
////////////////////
"""

class WeightInitializer:
   """
   Base class for neural network weight initialization strategies.
   Provides common utilities for creating weight arrays.
   """
   @staticmethod
   def initialize(*dims):
       """
       Abstract method for initializing weights.
       Must be implemented by concrete initializer classes.
       """
       raise NotImplementedError("Subclasses must implement this method")

   @staticmethod
   def _create_array(generator, dims):
       """
       Creates an MLArray with given dimensions using a generator function.
       Flattens dimensions and reshapes array to desired shape.
       """
       total_elements = reduce(mul, dims)
       flat_array = [generator() for _ in range(total_elements)]
       return MLArray(flat_array).reshape(*dims)

class RandomUniform(WeightInitializer):
    """
    Simple random uniform initialization.
    Generates weights from uniform distribution in range [-limit, limit].
    """
    @staticmethod
    def initialize(*dims, limit=1.0):
        """
        Initializes weights using uniform distribution in range [-limit, limit].
        
        Args:
            *dims: Dimensions of the weight array
            limit: Upper bound for uniform distribution (default: 1.0)
        
        Examples:
            RandomUniform.initialize(3, 4)  # 3x4 matrix with values in [-1, 1]
            RandomUniform.initialize(3, 4, limit=0.5)  # 3x4 matrix with values in [-0.5, 0.5]
        """
        dims = RandomUniform._process_dims(dims)
        return RandomUniform._create_array(lambda: random.uniform(-limit, limit), dims)

    @staticmethod
    def _process_dims(dims):
        """
        Processes input dimensions to handle both tuple/list and separate arguments.
        """
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            return dims[0]
        return dims

class XavierUniform(WeightInitializer):
   """
   Xavier/Glorot uniform initialization.
   Generates weights from uniform distribution with variance based on layer dimensions.
   """
   @staticmethod
   def initialize(*dims):
       """
       Initializes weights using uniform distribution scaled by input/output dimensions.
       Good for layers with tanh or sigmoid activation.
       """
       dims = XavierUniform._process_dims(dims)
       fan_in = dims[0] if len(dims) > 0 else 1
       fan_out = dims[-1] if len(dims) > 1 else fan_in
       limit = math.sqrt(6. / (fan_in + fan_out))
       return XavierUniform._create_array(lambda: random.uniform(-limit, limit), dims)

   @staticmethod
   def _process_dims(dims):
       """
       Processes input dimensions to handle both tuple/list and separate arguments.
       """
       if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
           return dims[0]
       return dims

class XavierNormal(WeightInitializer):
   """
   Xavier/Glorot normal initialization.
   Generates weights from normal distribution with variance based on layer dimensions.
   """
   @staticmethod
   def initialize(*dims):
       """
       Initializes weights using normal distribution scaled by input/output dimensions.
       Good for layers with tanh or sigmoid activation.
       """
       dims = XavierNormal._process_dims(dims)
       fan_in = dims[0] if len(dims) > 0 else 1
       fan_out = dims[-1] if len(dims) > 1 else fan_in
       std = math.sqrt(2. / (fan_in + fan_out))
       return XavierNormal._create_array(lambda: random.gauss(0, std), dims)

   @staticmethod
   def _process_dims(dims):
       """
       Processes input dimensions to handle both tuple/list and separate arguments.
       """
       if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
           return dims[0]
       return dims

class HeInitialization(WeightInitializer):
   """
   He/Kaiming initialization.
   Generates weights from normal distribution with variance based on input dimension.
   """
   @staticmethod
   def initialize(*dims):
       """
       Initializes weights using normal distribution scaled by input dimension.
       Optimal for layers with ReLU activation.
       """
       dims = HeInitialization._process_dims(dims)
       fan_in = dims[0] if len(dims) > 0 else 1
       std = math.sqrt(2. / fan_in)
       return HeInitialization._create_array(lambda: random.gauss(0, std), dims)

   @staticmethod
   def _process_dims(dims):
       """
       Processes input dimensions to handle both tuple/list and separate arguments.
       """
       if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
           return dims[0]
       return dims