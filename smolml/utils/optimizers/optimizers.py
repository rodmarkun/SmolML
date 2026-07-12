from smolml.core.ml_array import zeros

class Optimizer:
    """Base optimizer class that defines the interface for all optimizers"""
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
    
    def update(self, object, object_idx, param_names):
        """Update rule to be implemented by specific optimizers"""
        raise NotImplementedError

class SGD(Optimizer):
    """Standard Stochastic Gradient Descent optimizer"""
    def update(self, object, object_idx, param_names):
        """
        Update rule for standard SGD: θ = θ - α∇θ
        where α is the learning rate.
        
        This is the most basic form of gradient descent, which directly updates
        parameters in the opposite direction of the gradient, scaled by the learning rate.
        """
        new_params = tuple(
            getattr(object, name) - self.learning_rate * getattr(object, name).grad()
            for name in param_names
        )
        return new_params

class SGDMomentum(Optimizer):
    """
    Stochastic Gradient Descent optimizer with momentum.
    This optimizer accelerates SGD by accumulating a velocity vector in the direction of persistent gradients,
    helping to avoid local minima and speed up convergence.
    """
    def __init__(self, learning_rate: float = 0.01, momentum_coefficient: float = 0.9):
        super().__init__(learning_rate)
        self.momentum_coefficient = momentum_coefficient
        self.velocities = {}
        
    def update(self, object, object_idx, param_names):
        """
        Update rule for SGD with momentum: v = βv + α∇θ, θ = θ - v
        where β is the momentum coefficient and α is the learning rate.
        """
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

class AdaGrad(Optimizer):
    """
    Adaptive Gradient optimizer.
    Adapts the learning rate to parameters, performing smaller updates 
    for frequently updated parameters and larger updates for infrequent ones.
    """
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)
        self.epsilon = 1e-8
        self.squared_gradients = {}
        
    def update(self, object, object_idx, param_names):
        """
        Update rule for AdaGrad: θ = θ - (α / √(G + ε)) * ∇θ
        where G is the sum of squared gradients up to the current timestep
        """
        # Initialize squared gradients for this layer if not exist
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

class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.
    Combines the benefits of:
    1. Momentum: By keeping track of exponentially decaying gradient average
    2. RMSprop: By keeping track of exponentially decaying squared gradients
    Also includes bias correction terms to handle initialization.
    """
    def __init__(self, learning_rate: float = 0.01, exp_decay_gradients: float = 0.9, exp_decay_squared: float = 0.999):
        super().__init__(learning_rate)
        self.exp_decay_gradients = exp_decay_gradients
        self.exp_decay_squared = exp_decay_squared
        self.gradients_momentum = {}
        self.squared_gradients_momentum = {}
        self.epsilon = 1e-8
        self.timestep = 1
        
    def update(self, object, object_idx, param_names):
        """
        Update rule for Adam: θ = θ - α * m̂ / (√v̂ + ε)
        where:
        - m̂ is the bias-corrected first moment estimate
        - v̂ is the bias-corrected second moment estimate
        """
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