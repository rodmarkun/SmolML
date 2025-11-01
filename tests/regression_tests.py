import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from smolml.core.ml_array import MLArray, randn
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import smolml.utils.initializers as initializers
import smolml.utils.optimizers as optimizers
import smolml.utils.losses as losses
from smolml.models.regression import LinearRegression, PolynomialRegression

class TestRegressionVisualization(unittest.TestCase):
    """
    Test and visualize linear and polynomial regression implementations
    with interactive epoch slider (2D and 3D)
    """
    
    def setUp(self):
        """
        Set up common parameters and styling for tests
        """
        np.random.seed(42)
        
        # Training parameters
        self.iterations = 100
        
        # Initialize optimizer
        self.optimizer = optimizers.SGD(learning_rate=0.5)
        
        # Set plotting style with font fallback
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
        plt.rcParams['axes.facecolor'] = '#f0f0f0'
        plt.rcParams['axes.edgecolor'] = '#333333'
        plt.rcParams['axes.labelcolor'] = '#333333'
        plt.rcParams['xtick.color'] = '#333333'
        plt.rcParams['ytick.color'] = '#333333'
        plt.rcParams['grid.color'] = '#ffffff'
        plt.rcParams['grid.linestyle'] = '-'
        plt.rcParams['grid.linewidth'] = 1

    def generate_linear_data(self, size=25):
        """Generate data with linear relationship plus noise"""
        X = randn(size, 1)
        y = X * 2 + 1 + randn(size, 1) * 0.1
        return X, y

    def generate_nonlinear_data(self, size=25):
        """Generate data with polynomial relationship plus noise"""
        X = randn(size, 1)
        y = X * 2 + X * X * 3 + 1 + randn(size, 1) * 0.1
        return X, y

    def generate_cubic_data(self, size=30):
        """Generate cubic data: y = 2x³ - x² + 3x + 1"""
        X = randn(size, 1)
        y = X * X * X * 2 - X * X + X * 3 + 1 + randn(size, 1) * 0.2
        return X, y

    def train_and_visualize(self, model, X, y, title):
        """Train model and create interactive 2D visualization"""
        # Store predictions history for every epoch
        predictions_history = []
        losses_history = []
        
        # Store initial prediction (epoch 0)
        y_pred = model.predict(X)
        predictions_history.append(y_pred.to_list())
        
        # Modified training loop to store predictions at each epoch
        X_train, y_train = X, y
        for i in range(self.iterations):
            # Make prediction 
            y_pred = model.predict(X_train)
            # Compute loss
            loss = model.loss_function(y_train, y_pred)
            losses_history.append(loss.data.data)
            # Backward pass
            loss.backward()

            # Update parameters
            model.weights, model.bias = model.optimizer.update(
                model, model.__class__.__name__, param_names=("weights", "bias")
            )

            # Reset gradients
            X_train, y_train = model.restart(X_train, y_train)
            
            # Store prediction after this update
            X_eval = X.restart()
            y_pred_eval = model.predict(X_eval)
            predictions_history.append(y_pred_eval.to_list())

            if (i+1) % 10 == 0:
                print(f"Iteration {i + 1}/{self.iterations}, Loss: {loss.data}")
        
        # Convert to numpy for plotting
        X_np = np.array(X.to_list())
        y_np = np.array(y.to_list())
        
        # Sort for smooth curve plotting
        sort_idx = np.argsort(X_np.flatten())
        X_np = X_np[sort_idx]
        y_np = y_np[sort_idx]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25)
        
        scatter = ax.scatter(X_np, y_np, c='#1E88E5', alpha=0.6, s=50, label='Data')
        predictions_sorted = [np.array(pred)[sort_idx] for pred in predictions_history]
        line, = ax.plot(X_np, predictions_sorted[0], color='#D81B60', lw=2, label='Prediction')
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True)
        
        # Add slider
        slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='#d3d3d3')
        slider = Slider(slider_ax, 'Epoch', 0, self.iterations,
                       valinit=0, valstep=1, color='#FFC107')
        
        def update(val):
            epoch_index = int(slider.val)
            line.set_ydata(predictions_sorted[epoch_index])
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        
        # Add epoch and loss text
        info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                           fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def update_text(val):
            epoch_index = int(slider.val)
            loss_val = losses_history[epoch_index] if epoch_index < len(losses_history) else losses_history[-1]
            info_text.set_text(f'Epoch: {epoch_index}\nLoss: {loss_val:.6f}')
        
        slider.on_changed(update_text)
        update_text(0)  # Initialize text
        
        plt.show()
        
        # Print final parameters
        print("\nFinal Parameters:")
        print("Weights:", model.weights.data)
        print("Bias:", model.bias.data)
        
        return predictions_history, losses_history[-1]

    def train_and_visualize_3d(self, model, X, y, title, feature_names=None):
        """Train model and create interactive 3D visualization for 2-feature data"""
        if feature_names is None:
            feature_names = ['X₁', 'X₂']
            
        # Store predictions history for every epoch
        predictions_history = []
        losses_history = []
        
        # Store initial prediction (epoch 0)
        y_pred = model.predict(X)
        predictions_history.append(y_pred.to_list())
        
        # Modified training loop to store predictions at each epoch
        X_train, y_train = X, y
        for i in range(self.iterations):
            # Make prediction 
            y_pred = model.predict(X_train)
            # Compute loss
            loss = model.loss_function(y_train, y_pred)
            losses_history.append(loss.data.data)
            # Backward pass
            loss.backward()

            # Update parameters
            model.weights, model.bias = model.optimizer.update(
                model, model.__class__.__name__, param_names=("weights", "bias")
            )

            # Reset gradients
            X_train, y_train = model.restart(X_train, y_train)
            
            # Store prediction after this update
            X_eval = X.restart()
            y_pred_eval = model.predict(X_eval)
            predictions_history.append(y_pred_eval.to_list())

            if (i+1) % 10 == 0:
                print(f"Iteration {i + 1}/{self.iterations}, Loss: {loss.data}")
        
        # Convert to numpy for plotting
        X_np = np.array(X.to_list())
        y_np = np.array(y.to_list()).flatten()
        
        # Extract features
        X1 = X_np[:, 0]
        X2 = X_np[:, 1]
        
        # Create plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.25, right=0.95, left=0.05)
        
        # Plot actual data points
        scatter = ax.scatter(X1, X2, y_np, c='#1E88E5', s=80, alpha=0.7, 
                            label='Actual Data', edgecolors='black', linewidth=0.5)
        
        # Plot predictions 
        predictions_np = [np.array(pred).flatten() for pred in predictions_history]
        pred_scatter = ax.scatter(X1, X2, predictions_np[0], c='#D81B60', s=60, 
                                 alpha=0.8, label='Predictions', marker='^')
        
        # Add connecting lines between actual and predicted
        lines = []
        for i in range(len(X1)):
            line, = ax.plot([X1[i], X1[i]], [X2[i], X2[i]], 
                           [y_np[i], predictions_np[0][i]], 
                           'k--', alpha=0.3, linewidth=1)
            lines.append(line)
        
        ax.set_xlabel(feature_names[0], fontsize=12, labelpad=10)
        ax.set_ylabel(feature_names[1], fontsize=12, labelpad=10)
        ax.set_zlabel('Y', fontsize=12, labelpad=10)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=10, loc='lower left')
        
        # Set nice viewing angle
        ax.view_init(elev=20, azim=45)
        
        # Add slider
        slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='#d3d3d3')
        slider = Slider(slider_ax, 'Epoch', 0, self.iterations,
                       valinit=0, valstep=1, color='#FFC107')
        
        def update(val):
            epoch_index = int(slider.val)
            pred_scatter._offsets3d = (X1, X2, predictions_np[epoch_index])
            
            # Update connecting lines
            for i, line in enumerate(lines):
                line.set_data_3d([X1[i], X1[i]], [X2[i], X2[i]], 
                                [y_np[i], predictions_np[epoch_index][i]])
            
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        
        # Add epoch and loss text
        info_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, 
                             fontsize=11, fontweight='bold', verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        def update_text(val):
            epoch_index = int(slider.val)
            loss_val = losses_history[epoch_index] if epoch_index < len(losses_history) else losses_history[-1]
            info_text.set_text(f'Epoch: {epoch_index}\nLoss: {loss_val:.6f}')
        
        slider.on_changed(update_text)
        update_text(0) 
        
        plt.show()
        
        # Print final parameters
        print("\nFinal Parameters:")
        print("Weights:", model.weights.data)
        print("Bias:", model.bias.data)
        
        return predictions_history, losses_history[-1]

    def test_linear_regression(self):
        """Test linear regression with visualization"""
        print("\nTesting Linear Regression...")
        X, y = self.generate_linear_data()
        
        model = LinearRegression(
            input_size=1,
            loss_function=losses.mse_loss,
            optimizer=self.optimizer,
            initializer=initializers.XavierUniform()
        )

        print(model)
        predictions, final_loss = self.train_and_visualize(
            model, X, y, 'Linear Regression: Data vs Predictions'
        )
        
        self.assertIsNotNone(predictions)
        self.assertGreater(len(predictions), 0)
        self.assertLess(final_loss, 1.0)

    def test_polynomial_regression(self):
        """Test polynomial regression with visualization"""
        print("\nTesting Polynomial Regression...")
        X, y = self.generate_nonlinear_data()
        
        model = PolynomialRegression(
            input_size=1,
            degree=2,
            loss_function=losses.mse_loss,
            optimizer=self.optimizer,
            initializer=initializers.XavierUniform()
        )

        print(model)
        predictions, final_loss = self.train_and_visualize(
            model, X, y, 'Polynomial Regression (Degree 2): Data vs Predictions'
        )
        
        self.assertIsNotNone(predictions)
        self.assertGreater(len(predictions), 0)
        self.assertLess(final_loss, 1.0)
        
    def test_polynomial_regression_cubic(self):
        """Test polynomial regression on cubic data"""
        print("\nTesting Polynomial Regression on Cubic Data...")
        
        X, y = self.generate_cubic_data()
        
        model = PolynomialRegression(
            input_size=1,
            degree=3,
            loss_function=losses.mse_loss,
            optimizer=optimizers.SGD(learning_rate=0.05),
            initializer=initializers.XavierUniform()
        )

        print(model)
        predictions, final_loss = self.train_and_visualize(
            model, X, y, 'Polynomial Regression (Degree 3 - Cubic): Data vs Predictions'
        )
        
        self.assertIsNotNone(predictions)
        self.assertGreater(len(predictions), 0)
        self.assertLess(final_loss, 2.0)

    def test_polynomial_regression_multifeature(self):
        """Test polynomial regression with multiple input features using 3D visualization"""
        print("\nTesting Polynomial Regression with 2 Features (3D Visualization)...")
        
        # Generate 2-feature data: y = 2x₁ + 3x₂ + x₁² + x₁x₂
        X_1 = randn(30, 1)
        X_2 = randn(30, 1)
        
        X_data = []
        y_data = []
        for i in range(30):
            x1 = X_1.data[i][0].data
            x2 = X_2.data[i][0].data
            X_data.append([x1, x2])
            y_val = 2*x1 + 3*x2 + x1*x1 + x1*x2 + np.random.randn()*0.1
            y_data.append([y_val])
        
        X = MLArray(X_data)
        y = MLArray(y_data)
        
        model = PolynomialRegression(
            input_size=2,
            degree=2,
            loss_function=losses.mse_loss,
            optimizer=optimizers.SGD(learning_rate=0.05),
            initializer=initializers.XavierUniform()
        )

        print(model)
        
        # Use 3D visualization
        predictions, final_loss = self.train_and_visualize_3d(
            model, X, y, 
            'Polynomial Regression (2 Features, Degree 2): 3D Visualization',
            feature_names=['X₁', 'X₂']
        )
        
        self.assertIsNotNone(predictions)
        self.assertLess(final_loss, 1.0)

if __name__ == '__main__':
    unittest.main()