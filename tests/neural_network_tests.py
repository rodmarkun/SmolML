import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import tensorflow as tf
import torch
import torch.nn as torch_nn
import torch.optim as torch_optim
from sklearn.datasets import make_moons, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from smolml.core.ml_array import MLArray
from smolml.models.nn.neural_network import NeuralNetwork
from smolml.models.nn.layer import DenseLayer
import smolml.utils.activation as activation
import smolml.utils.losses as losses
import smolml.utils.optimizers as optimizers

class TestXORProblem(unittest.TestCase):
    """
    Test XOR problem - comparing custom implementation against PyTorch
    """
    
    def setUp(self):
        """Set up XOR dataset and models"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # XOR data
        self.X_data = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        self.y_data = [[0.0], [1.0], [1.0], [0.0]]
        
        # Convert for custom model
        self.X_ml = MLArray(self.X_data)
        self.y_ml = MLArray(self.y_data)
        
        # Convert for PyTorch
        self.X_torch = torch.FloatTensor(self.X_data)
        self.y_torch = torch.FloatTensor(self.y_data)
        
        self.epochs = 100
        self.learning_rate = 0.1
        
        # Initialize models
        self.custom_model = self._create_custom_model()
        self.torch_model = self._create_torch_model()
        
    def _create_custom_model(self):
        """Create custom neural network"""
        return NeuralNetwork(
            layers=[
                DenseLayer(input_size=2, output_size=16, activation_function=activation.relu),
                DenseLayer(input_size=16, output_size=1, activation_function=activation.tanh)
            ],
            loss_function=losses.mse_loss,
            optimizer=optimizers.AdaGrad(learning_rate=self.learning_rate)
        )
    
    def _create_torch_model(self):
        """Create equivalent PyTorch model"""
        class XORNet(torch_nn.Module):
            def __init__(self):
                super(XORNet, self).__init__()
                self.fc1 = torch_nn.Linear(2, 16)
                self.fc2 = torch_nn.Linear(16, 1)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.tanh(self.fc2(x))
                return x
        
        return XORNet()
    
    def test_xor_comparison(self):
        """Train and compare both models on XOR problem"""
        print("\n" + "="*50)
        print("Testing XOR Problem")
        print("="*50)
        
        # Train custom model
        print("\nTraining custom model...")
        custom_history = self.custom_model.train(self.X_ml, self.y_ml, 
                                                  epochs=self.epochs, 
                                                  verbose=False)
        
        # Train PyTorch model
        print("Training PyTorch model...")
        optimizer = torch_optim.Adagrad(self.torch_model.parameters(), lr=self.learning_rate)
        criterion = torch_nn.MSELoss()
        torch_history = []
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.torch_model(self.X_torch)
            loss = criterion(outputs, self.y_torch)
            loss.backward()
            optimizer.step()
            torch_history.append(loss.item())
        
        # Plot comparison
        plt.figure(figsize=(12, 4))
        
        # Plot 1: Training Loss
        plt.subplot(1, 2, 1)
        plt.plot(range(self.epochs), custom_history, label='Custom NN', linewidth=2)
        plt.plot(range(self.epochs), torch_history, label='PyTorch', linewidth=2, linestyle='--')
        plt.title('XOR Problem - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Predictions
        plt.subplot(1, 2, 2)
        x_labels = ['[0,0]', '[0,1]', '[1,0]', '[1,1]']
        
        custom_preds = []
        torch_preds = []
        actuals = [y[0] for y in self.y_data]
        
        for i, x_sample in enumerate(self.X_data):
            custom_pred = self.custom_model.forward(MLArray([x_sample])).to_list()[0][0]
            custom_preds.append(custom_pred)
            
            torch_pred = self.torch_model(torch.FloatTensor([x_sample])).item()
            torch_preds.append(torch_pred)
        
        x_pos = np.arange(len(x_labels))
        width = 0.25
        
        plt.bar(x_pos - width, actuals, width, label='Actual', alpha=0.8)
        plt.bar(x_pos, custom_preds, width, label='Custom NN', alpha=0.8)
        plt.bar(x_pos + width, torch_preds, width, label='PyTorch', alpha=0.8)
        
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.title('XOR Problem - Predictions')
        plt.xticks(x_pos, x_labels)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('xor_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\nPredictions:")
        for i, x_sample in enumerate(self.X_data):
            print(f"Input: {x_sample} → Custom: {custom_preds[i]:.4f}, "
                  f"PyTorch: {torch_preds[i]:.4f}, Actual: {actuals[i]}")
        
        print("\nPlot saved as 'xor_comparison.png'")


class TestBinaryClassification(unittest.TestCase):
    """
    Compare custom neural network implementation against TensorFlow
    using the make_moons dataset
    """
    
    def setUp(self):
        """Set up dataset and models"""
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Generate moon dataset
        X, y = make_moons(n_samples=150, noise=0.2)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Convert data for custom implementation
        self.X_train_ml = MLArray([[float(x) for x in row] for row in self.X_train])
        self.y_train_ml = MLArray([[float(y)] for y in self.y_train])
        self.X_test_ml = MLArray([[float(x) for x in row] for row in self.X_test])
        self.y_test_ml = MLArray([[float(y)] for y in self.y_test])
        
        # Model parameters
        self.input_size = 2
        self.hidden_size = 32
        self.output_size = 1
        self.epochs = 100
        self.learning_rate = 0.1
        
        # Initialize models
        self.custom_model = self._create_custom_model()
        self.tf_model = self._create_tf_model()

    def _create_custom_model(self):
        """Create custom neural network with same architecture"""
        return NeuralNetwork([
            DenseLayer(self.input_size, self.hidden_size, activation.relu),
            DenseLayer(self.hidden_size, self.output_size, activation.sigmoid)
        ], losses.binary_cross_entropy, optimizer=optimizers.SGDMomentum(learning_rate=self.learning_rate))

    def _create_tf_model(self):
        """Create equivalent TensorFlow model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_size, activation='relu', input_shape=(self.input_size,)),
            tf.keras.layers.Dense(self.output_size, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model

    def _plot_decision_boundary(self, model, is_tf=False):
        """Plot decision boundary for either model"""
        x_min, x_max = self.X_test[:, 0].min() - 0.5, self.X_test[:, 0].max() + 0.5
        y_min, y_max = self.X_test[:, 1].min() - 0.5, self.X_test[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Get predictions for mesh grid points
        if is_tf:
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0)
        else:
            X_mesh = MLArray([[float(x), float(y)] for x, y in zip(xx.ravel(), yy.ravel())])
            Z = model.forward(X_mesh).to_list()
        Z = np.array(Z).reshape(xx.shape)
        
        return xx, yy, Z

    def test_binary_classification(self):
        """Train and compare both models"""
        print("\n" + "="*50)
        print("Testing Binary Classification (Moons Dataset)")
        print("="*50)
        print("IMPORTANT: SmolML implementation of NNs is very inefficient due to being written in Python.")
        
        # Train custom model
        print("\nTraining custom model...")
        custom_history = []
        for epoch in range(self.epochs):
            y_pred = self.custom_model.forward(self.X_train_ml)
            loss = self.custom_model.loss_function(y_pred, self.y_train_ml)
            loss.backward()
            
            for idx, layer in enumerate(self.custom_model.layers):
                layer.update(self.custom_model.optimizer, idx)
            
            # Reset computational graph
            self.X_train_ml = self.X_train_ml.restart()
            self.y_train_ml = self.y_train_ml.restart()
            for layer in self.custom_model.layers:
                layer.weights = layer.weights.restart()
                layer.biases = layer.biases.restart()
            
            custom_history.append(float(loss.data.data))
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.data.data}")
        
        # Train TensorFlow model
        print("\nTraining TensorFlow model...")
        tf_history = self.tf_model.fit(
            self.X_train, self.y_train,
            epochs=self.epochs,
            batch_size=len(self.X_train),
            verbose=0
        )
        
        # Plot training curves
        plt.figure(figsize=(12, 4))

        print("\nPlotting training loss...")
        # Plot 1: Training Loss
        plt.subplot(1, 2, 1)
        plt.plot(range(self.epochs), custom_history, label='Custom NN', linewidth=2)
        plt.plot(range(self.epochs), tf_history.history['loss'], label='TensorFlow', linewidth=2, linestyle='--')
        plt.title('Binary Classification - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Decision Boundaries
        plt.subplot(1, 2, 2)
        
        print("Plotting decision boundaries...")
        # Plot decision boundaries
        xx, yy, Z_custom = self._plot_decision_boundary(self.custom_model)
        plt.contourf(xx, yy, Z_custom > 0.5, alpha=0.4, cmap='RdYlBu')
        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, 
                   cmap='RdYlBu', edgecolors='black', alpha=0.8)
        plt.title('Decision Boundaries (Custom NN)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        plt.tight_layout()
        plt.savefig('binary_classification_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Compute and print accuracies
        custom_pred = np.array(self.custom_model.forward(self.X_test_ml).to_list()) > 0.5
        tf_pred = self.tf_model.predict(self.X_test, verbose=0) > 0.5
        
        custom_accuracy = np.mean(custom_pred.flatten() == self.y_test)
        tf_accuracy = np.mean(tf_pred.flatten() == self.y_test)
        
        print("\nTest Accuracies:")
        print(f"Custom NN: {custom_accuracy:.4f}")
        print(f"TensorFlow: {tf_accuracy:.4f}")
        print("\nPlot saved as 'binary_classification_comparison.png'")


class TestMultiClassClassification(unittest.TestCase):
    """
    Test multi-class classification on Iris dataset - comparing against TensorFlow
    """
    
    def setUp(self):
        """Set up Iris dataset and models"""
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Load Iris dataset
        iris = load_iris()
        X_data = iris.data
        y_data = iris.target
        
        # Normalize features
        self.scaler = StandardScaler()
        X_data = self.scaler.fit_transform(X_data)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42
        )
        
        # Convert to one-hot encoding for custom model
        def to_one_hot(labels, num_classes=3):
            one_hot = []
            for label in labels:
                oh = [0.0] * num_classes
                oh[label] = 1.0
                one_hot.append(oh)
            return one_hot
        
        self.y_train_onehot = to_one_hot(self.y_train)
        self.y_test_onehot = to_one_hot(self.y_test)
        
        # Convert for custom model
        self.X_train_ml = MLArray(self.X_train.tolist())
        self.y_train_ml = MLArray(self.y_train_onehot)
        self.X_test_ml = MLArray(self.X_test.tolist())
        
        self.class_names = iris.target_names
        self.epochs = 100
        self.learning_rate = 0.01
        
        # Initialize models
        self.custom_model = self._create_custom_model()
        self.tf_model = self._create_tf_model()
        
    def _create_custom_model(self):
        """Create custom neural network"""
        return NeuralNetwork(
            layers=[
                DenseLayer(input_size=4, output_size=16, activation_function=activation.relu),
                DenseLayer(input_size=16, output_size=8, activation_function=activation.relu),
                DenseLayer(input_size=8, output_size=3, activation_function=activation.softmax)
            ],
            loss_function=losses.categorical_cross_entropy,
            optimizer=optimizers.Adam(learning_rate=self.learning_rate)
        )
    
    def _create_tf_model(self):
        """Create equivalent TensorFlow model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def test_multiclass_classification(self):
        """Train and compare both models on Iris dataset"""
        print("\n" + "="*50)
        print("Testing Multi-Class Classification (Iris Dataset)")
        print("="*50)
        
        # Train custom model
        print("\nTraining custom model...")
        custom_history = self.custom_model.train(
            self.X_train_ml, self.y_train_ml,
            epochs=self.epochs,
            verbose=False
        )
        
        # Train TensorFlow model
        print("Training TensorFlow model...")
        tf_history = self.tf_model.fit(
            self.X_train, self.y_train,
            epochs=self.epochs,
            batch_size=len(self.X_train),
            verbose=0
        )
        
        # Plot comparison
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Training Loss
        plt.subplot(1, 3, 1)
        plt.plot(range(self.epochs), custom_history, label='Custom NN', linewidth=2)
        plt.plot(range(self.epochs), tf_history.history['loss'], label='TensorFlow', linewidth=2, linestyle='--')
        plt.title('Multi-Class Classification - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Confusion Matrix (Custom)
        plt.subplot(1, 3, 2)
        custom_preds = []
        for x in self.X_test_ml.to_list():
            pred = self.custom_model.forward(MLArray([x])).to_list()[0]
            custom_preds.append(pred.index(max(pred)))
        
        from sklearn.metrics import confusion_matrix
        cm_custom = confusion_matrix(self.y_test, custom_preds)
        
        plt.imshow(cm_custom, interpolation='nearest', cmap='Blues')
        plt.title('Custom NN - Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(3)
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)
        
        # Add text annotations
        for i in range(3):
            for j in range(3):
                plt.text(j, i, str(cm_custom[i, j]),
                        ha="center", va="center",
                        color="white" if cm_custom[i, j] > cm_custom.max() / 2 else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Plot 3: Prediction probabilities for a few samples
        plt.subplot(1, 3, 3)
        test_indices = [0, 15, 29]  # One from each class in test set
        
        for idx, test_idx in enumerate(test_indices):
            X_test_sample = MLArray([self.X_test_ml.to_list()[test_idx]])
            prediction = self.custom_model.forward(X_test_sample)
            pred_probs = prediction.to_list()[0]
            
            x_pos = np.arange(3) + idx * 4
            plt.bar(x_pos, pred_probs, alpha=0.7, label=f'Sample {test_idx}')
        
        plt.title('Prediction Probabilities (Custom NN)')
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.xticks([1, 5, 9], self.class_names)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('multiclass_classification_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Calculate accuracies
        custom_accuracy = np.mean(np.array(custom_preds) == self.y_test)
        
        tf_preds = np.argmax(self.tf_model.predict(self.X_test, verbose=0), axis=1)
        tf_accuracy = np.mean(tf_preds == self.y_test)
        
        print("\nTest Accuracies:")
        print(f"Custom NN: {custom_accuracy:.4f}")
        print(f"TensorFlow: {tf_accuracy:.4f}")
        
        print("\nSample predictions (Custom NN):")
        for idx in [0, 15, 29]:
            X_test_sample = MLArray([self.X_test_ml.to_list()[idx]])
            prediction = self.custom_model.forward(X_test_sample)
            pred_probs = prediction.to_list()[0]
            pred_class = pred_probs.index(max(pred_probs))
            actual_class = self.y_test[idx]
            
            print(f"\nSample {idx}:")
            print(f"  Predicted: {self.class_names[pred_class]} ({pred_probs[pred_class]:.3f})")
            print(f"  Actual: {self.class_names[actual_class]}")
        
        print("\nPlot saved as 'multiclass_classification_comparison.png'")


if __name__ == '__main__':
    unittest.main()