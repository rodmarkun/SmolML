from smolml.core.ml_array import MLArray
from smolml.core.value import Value

"""
///////////////
/// SCALERS ///
///////////////
"""

class StandardScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    Transforms features to have mean=0 and standard deviation=1.
    """
    def __init__(self):
        """
        Initializes scaler with empty mean and standard deviation attributes.
        """
        self.mean = None
        self.std = None
        
    def fit(self, X):
        """
        Computes mean and standard deviation of input features for later scaling.
        Stores values internally for transform step.
        """
        if not isinstance(X, MLArray):
            X = MLArray(X)
        
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        
        # Handle zero standard deviation
        if len(X.shape) <= 1:  # Single value or 1D array
            if isinstance(self.std.data, (int, float)) and self.std.data == 0:
                self.std = MLArray(1.0)
        else:
            # Replace zero standard deviations with 1
            def replace_zeros(data):
                if isinstance(data, Value):
                    return Value(1.0) if data.data == 0 else data
                return [replace_zeros(d) for d in data]
            
            self.std.data = replace_zeros(self.std.data)

    def transform(self, X):
        """
        Standardizes features using previously computed mean and std.
        Z-score normalization: z = (x - μ) / σ
        """
        if not isinstance(X, MLArray):
            X = MLArray(X)
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        """
        Convenience method that fits scaler and transforms data in one step.
        """
        self.fit(X)
        return self.transform(X)

class MinMaxScaler:
   """
   Transforms features by scaling to a fixed range, typically [0, 1].
   Preserves zero values and handles sparse matrices.
   """
   def __init__(self):
       """
       Initializes scaler with empty min and max attributes.
       """
       self.max = None
       self.min = None

   def fit(self, X):
       """
       Computes min and max values of input features for later scaling.
       Stores values internally for transform step.
       """
       if not isinstance(X, MLArray):
           X = MLArray(X)
       self.max = X.max(axis=0)
       self.min = X.min(axis=0)

   def transform(self, X):
       """
       Scales features using previously computed min and max values.
       MinMax formula: x_scaled = (x - x_min) / (x_max - x_min)
       """
       return (X - self.min) / (self.max - self.min)

   def fit_transform(self, X):
       """
       Convenience method that fits scaler and transforms data in one step.
       """
       self.fit(X)
       return self.transform(X)
