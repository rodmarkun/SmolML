from smolml.core.ml_array import MLArray
from smolml.models.nn import DenseLayer
import smolml.utils.memory as memory
import smolml.utils.losses as losses
import smolml.utils.activation as activation
import smolml.utils.optimizers as optimizers

"""
//////////////////////
/// NEURAL NETWORK ///
//////////////////////
"""

class NeuralNetwork:
    """
    Implementation of a feedforward neural network with customizable layers and loss function.
    Supports training through backpropagation and gradient descent.
    """
    def __init__(self, layers: list, loss_function: callable, optimizer: optimizers.Optimizer = optimizers.SGD()) -> None:
        """
        Initializes the network with a list of layers and a loss function for training.
        """
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer if optimizer is not None else optimizers.SGD()

    def forward(self, input_data):
        """
        Performs forward pass by sequentially applying each layer's transformation.
        """
        if not isinstance(input_data, MLArray):
            raise TypeError(f"Input data must be MLArray, not {type(input_data)}")
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def train(self, X, y, epochs, verbose=True, print_every=1):
        """
        Trains the network using gradient descent for the specified number of epochs.
        Prints loss every 100 epochs to monitor training progress.
        """
        X, y = MLArray.ensure_array(X, y)
        losses = []
        for epoch in range(epochs):
            # Forward pass through the network
            y_pred = self.forward(X)
            
            # Compute loss between predictions and targets
            loss = self.loss_function(y_pred, y)
            losses.append(loss.data.data)
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Update parameters in each layer
            for idx, layer in enumerate(self.layers):
                layer.update(self.optimizer, idx)
            
            # Reset gradients for next iteration
            X.restart()
            y.restart()
            for layer in self.layers:
                layer.weights.restart()
                layer.biases.restart()
                
            if verbose:
                # Print training progress
                if (epoch+1) % print_every == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.data}")

        return losses
        
    def __repr__(self):
        """
        Returns a string representation of the neural network architecture.
        Displays layer information, loss function, optimizer details, and detailed memory usage.
        """
        # Get terminal width for formatting
        try:
            import os
            terminal_width = os.get_terminal_size().columns
        except Exception:
            terminal_width = 80

        # Create header
        header = "Neural Network Architecture"
        separator = "=" * terminal_width
        
        # Get size information
        size_info = memory.calculate_neural_network_size(self)
        
        # Format layers information
        layers_info = []
        for i, (layer, layer_size) in enumerate(zip(self.layers, size_info['layers'])):
            if isinstance(layer, DenseLayer):
                input_size = layer.weights.shape[0]
                output_size = layer.weights.shape[1]
                activation_name = layer.activation_function.__name__
                layer_info = [
                    f"Layer {i+1}: Dense("
                    f"in={input_size}, "
                    f"out={output_size}, "
                    f"activation={activation_name})"
                ]
                
                # Parameters info
                params = input_size * output_size + output_size  # weights + biases
                layer_info.append(
                    f"    Parameters: {params:,} "
                    f"({input_size}×{output_size} weights + {output_size} biases)"
                )
                
                # Memory info
                layer_info.append(
                    f"    Memory: {memory.format_size(layer_size['total'])} "
                    f"(weights: {memory.format_size(layer_size['weights_size'])}, "
                    f"biases: {memory.format_size(layer_size['biases_size'])})"
                )
                
                layers_info.append("\n".join(layer_info))

        # Calculate total parameters
        total_params = sum(
            layer.weights.size() + layer.biases.size()
            for layer in self.layers
        )

        # Format optimizer information
        optimizer_info = [
            f"Optimizer: {self.optimizer.__class__.__name__}("
            f"learning_rate={self.optimizer.learning_rate})"
        ]
        
        # Add optimizer state information if it exists
        if size_info['optimizer']['state']:
            state_sizes = [
                f"    {key}: {memory.format_size(value)}"
                for key, value in size_info['optimizer']['state'].items()
            ]
            optimizer_info.extend(state_sizes)
        
        # Format loss function information
        loss_info = f"Loss Function: {self.loss_function.__name__}"

        # Detailed memory breakdown
        memory_info = ["Memory Usage:"]
        
        # Layer memory
        for i, layer_size in enumerate(size_info['layers']):
            memory_info.append(
                f"  Layer {i+1}: {memory.format_size(layer_size['total'])} "
                f"(weights: {memory.format_size(layer_size['weights_size'])}, "
                f"biases: {memory.format_size(layer_size['biases_size'])})"
            )
        
        # Optimizer memory
        if size_info['optimizer']['state']:
            opt_size = sum(size_info['optimizer']['state'].values())
            memory_info.append(f"  Optimizer State: {memory.format_size(opt_size)}")
        
        memory_info.append(f"  Base Objects: {memory.format_size(size_info['optimizer']['size'])}")
        memory_info.append(f"Total Memory: {memory.format_size(size_info['total'])}")

        # Combine all parts
        return (
            f"\n{header}\n{separator}\n\n"
            f"Architecture:\n"
            + "\n".join(f"  {layer}" for layer in layers_info)
            + f"\n\n"
            + "\n".join(optimizer_info)
            + f"\n{loss_info}\n\n"
            f"Total Parameters: {total_params:,}\n\n"
            + "\n".join(memory_info)
            + f"\n{separator}\n"
        )