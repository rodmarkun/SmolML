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
import smolml.utils.initializers as initializers
import smolml.utils.optimizers as optimizers
import smolml.utils.losses as losses
from smolml.models.regression import LinearRegression, PolynomialRegression

class TestRegressionVisualization(unittest.TestCase):
    """
    Test and visualize linear and polynomial regression implementations
    with interactive epoch slider
    """
    
    def setUp(self):
        """
        Set up common parameters and styling for tests
        """
        np.random.seed(42)
        
        # Training parameters
        self.iterations = 100
        self.epochs_to_store = [0, 5, 10, 25, 50, 99]
        
        # Initialize optimizer
        self.optimizer = optimizers.SGD(learning_rate=0.1)
        
        # Set plotting style
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
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

    def train_and_visualize(self, model, X, y, title):
        """Train model and create interactive visualization"""
        # Store predictions history
        predictions_history = []
        losses_history = []
        
        # Initial prediction for storage
        y_pred = model.predict(X)
        predictions_history.append(y_pred.to_list())
        
        # Training loop using model's fit method
        losses = model.fit(X, y, iterations=self.iterations, verbose=True, print_every=10)
        
        # Store predictions at specified epochs
        X_eval = X.restart()  # Create fresh copy for evaluation
        for epoch in self.epochs_to_store[1:]:  # Skip 0 as we already stored it
            y_pred = model.predict(X_eval)
            predictions_history.append(y_pred.to_list())
        
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
        
        scatter = ax.scatter(X_np, y_np, c='#1E88E5', alpha=0.6, label='Data')
        predictions_sorted = [np.array(pred)[sort_idx] for pred in predictions_history]
        line, = ax.plot(X_np, predictions_sorted[0], color='#D81B60', lw=2, label='Prediction')
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True)
        
        # Add slider
        slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='#d3d3d3')
        slider = Slider(slider_ax, 'Epoch', 0, len(self.epochs_to_store) - 1,
                       valinit=0, valstep=1, color='#FFC107')
        
        def update(val):
            epoch_index = int(slider.val)
            line.set_ydata(predictions_sorted[epoch_index])
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        
        # Add epoch text
        epoch_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                           fontsize=12, fontweight='bold')
        
        def update_epoch_text(val):
            epoch_index = int(slider.val)
            epoch_text.set_text(f'Epoch: {self.epochs_to_store[epoch_index]}')
        
        slider.on_changed(update_epoch_text)
        update_epoch_text(0)  # Initialize text
        
        plt.show()
        
        # Print final parameters
        print("\nFinal Parameters:")
        print("Weights:", model.weights.data)
        print("Bias:", model.bias.data)
        
        return predictions_history, losses[-1]

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
        
        # Basic assertions
        self.assertIsNotNone(predictions)
        self.assertGreater(len(predictions), 0)
        self.assertLess(final_loss, 1.0)  # Assuming convergence

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
            model, X, y, 'Polynomial Regression: Data vs Predictions'
        )
        
        # Basic assertions
        self.assertIsNotNone(predictions)
        self.assertGreater(len(predictions), 0)
        self.assertLess(final_loss, 1.0)  # Assuming convergence
        
    def test_polynomial_regression_multidegree(self):
        """Test polynomial regression with different degrees"""
        print("\nTesting Polynomial Regression with Multiple Degrees...")
        X, y = self.generate_nonlinear_data()
        
        # Test with degree 3
        model_deg3 = PolynomialRegression(
            input_size=1,
            degree=3,
            loss_function=losses.mse_loss,
            optimizer=optimizers.SGD(learning_rate=0.05),
            initializer=initializers.XavierUniform()
        )

        print("\nDegree 3 Model:")
        print(model_deg3)
        predictions_deg3, final_loss_deg3 = self.train_and_visualize(
            model_deg3, X, y, 'Polynomial Regression (Degree 3): Data vs Predictions'
        )
        
        # Test with degree 4
        model_deg4 = PolynomialRegression(
            input_size=1,
            degree=4,
            loss_function=losses.mse_loss,
            optimizer=optimizers.SGD(learning_rate=0.03),
            initializer=initializers.XavierUniform()
        )

        print("\nDegree 4 Model:")
        print(model_deg4)
        predictions_deg4, final_loss_deg4 = self.train_and_visualize(
            model_deg4, X, y, 'Polynomial Regression (Degree 4): Data vs Predictions'
        )
        
        # Basic assertions
        self.assertIsNotNone(predictions_deg3)
        self.assertIsNotNone(predictions_deg4)
        self.assertLess(final_loss_deg3, 1.0)
        self.assertLess(final_loss_deg4, 1.0)
        
        print(f"\nComparison:")
        print(f"Degree 3 Final Loss: {final_loss_deg3:.6f}")
        print(f"Degree 4 Final Loss: {final_loss_deg4:.6f}")

    def test_polynomial_regression_cubic_data(self):
        """Test polynomial regression on cubic data"""
        print("\nTesting Polynomial Regression on Cubic Data...")
        
        # Generate cubic data: y = 2x³ - x² + 3x + 1
        X = randn(30, 1)
        y = X * X * X * 2 - X * X + X * 3 + 1 + randn(30, 1) * 0.2
        
        model = PolynomialRegression(
            input_size=1,
            degree=3,
            loss_function=losses.mse_loss,
            optimizer=optimizers.SGD(learning_rate=0.05),
            initializer=initializers.XavierUniform()
        )

        print(model)
        predictions, final_loss = self.train_and_visualize(
            model, X, y, 'Polynomial Regression (Cubic): Data vs Predictions'
        )
        
        # Basic assertions
        self.assertIsNotNone(predictions)
        self.assertGreater(len(predictions), 0)
        self.assertLess(final_loss, 2.0)

    def test_polynomial_regression_multifeature(self):
        """Test polynomial regression with multiple input features"""
        print("\nTesting Polynomial Regression with Multiple Features...")
        
        # Generate 2-feature data: y = 2x₁ + 3x₂ + x₁² + x₁x₂
        X_1 = randn(30, 1)
        X_2 = randn(30, 1)
        
        # Concatenate features
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
        
        # Train without visualization (since we can't easily plot 2D input)
        losses = model.fit(X, y, iterations=self.iterations, verbose=True, print_every=10)
        
        # Print final parameters
        print("\nFinal Parameters:")
        print("Weights:", model.weights.data)
        print("Bias:", model.bias.data)
        print(f"Final Loss: {losses[-1]:.6f}")
        
        # Basic assertions
        self.assertLess(losses[-1], 1.0)
        print("\n✓ Multi-feature polynomial regression converged successfully!")

    def test_polynomial_overfitting(self):
        """Test and visualize overfitting with high degree polynomial"""
        print("\nTesting Polynomial Overfitting (High Degree)...")
        
        # Generate simple quadratic data with noise
        X = randn(15, 1)  # Smaller dataset
        y = X * X * 2 + randn(15, 1) * 0.5  # More noise
        
        # Use very high degree
        model = PolynomialRegression(
            input_size=1,
            degree=6,
            loss_function=losses.mse_loss,
            optimizer=optimizers.SGD(learning_rate=0.01),
            initializer=initializers.XavierUniform()
        )

        print(model)
        predictions, final_loss = self.train_and_visualize(
            model, X, y, 'Polynomial Regression (Degree 6 - Overfitting): Data vs Predictions'
        )
        
        print("\n⚠ Note: High degree polynomial may show overfitting behavior")
        print(f"Final Loss: {final_loss:.6f}")
        
        # Basic assertions
        self.assertIsNotNone(predictions)

if __name__ == '__main__':
    unittest.main()