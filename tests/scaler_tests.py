import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from sklearn.preprocessing import MinMaxScaler as SKMinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from smolml.core.ml_array import MLArray
from smolml.preprocessing.scalers import StandardScaler, MinMaxScaler
from tests.img_utils import image_path

class TestScalers(unittest.TestCase):
    """
    Test custom scaler implementations against scikit-learn's
    """
    
    def setUp(self):
        """
        Set up test data and initialize scalers
        """
        # Simple test data
        self.simple_data = [
            [1, 4], 
            [100, 2], 
            [-20, 2],
            [0, 8],
            [50, -4]
        ]
        
        # Edge case data
        self.edge_data = [
            [0, 0],  # zeros
            [1e6, 1e-6],  # very large/small values
            [-1e6, -1e-6],  # negative large/small values
            [1, 1],  # same values
            [-1, -1]  # same negative values
        ]
        
        # Convert to proper formats
        self.simple_ml = MLArray(self.simple_data)
        self.simple_np = np.array(self.simple_data)
        self.edge_ml = MLArray(self.edge_data)
        self.edge_np = np.array(self.edge_data)
        
        # Initialize all scalers
        self.standard_ml = StandardScaler()
        self.standard_sk = SKStandardScaler()
        self.minmax_ml = MinMaxScaler()
        self.minmax_sk = SKMinMaxScaler()

    def plot_comparison(self, original_data, ml_scaled, sk_scaled, title, filename):
        """
        Create visualization comparing original and scaled data
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original data
        ax1.scatter(original_data[:, 0], original_data[:, 1], c='blue', alpha=0.6)
        ax1.set_title('Original Data')
        ax1.grid(True)
        
        # Custom implementation
        ax2.scatter(ml_scaled[:, 0], ml_scaled[:, 1], c='red', alpha=0.6)
        ax2.set_title('Custom Scaler')
        ax2.grid(True)
        
        # Scikit-learn implementation
        ax3.scatter(sk_scaled[:, 0], sk_scaled[:, 1], c='green', alpha=0.6)
        ax3.set_title('Scikit-learn Scaler')
        ax3.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(image_path(filename))
        plt.close()

    def test_standard_scaler_simple(self):
        """Test StandardScaler with simple data"""
        print("\nTesting StandardScaler with simple data...")
        
        # Fit and transform using both implementations
        ml_scaled = self.standard_ml.fit_transform(self.simple_ml)
        sk_scaled = self.standard_sk.fit_transform(self.simple_np)
        
        # Convert MLArray to numpy for comparison
        ml_scaled_np = np.array(ml_scaled.to_list())
        
        # Plot comparison
        self.plot_comparison(
            self.simple_np, 
            ml_scaled_np, 
            sk_scaled,
            'StandardScaler Comparison - Simple Data',
            'standard_scaler_simple.png'
        )
        
        # Basic statistical tests
        print("\nStandard Scaler Statistics (Simple Data):")
        print("Custom Implementation:")
        print(f"Mean: {ml_scaled_np.mean(axis=0)}")
        print(f"Std: {ml_scaled_np.std(axis=0)}")
        print("\nScikit-learn Implementation:")
        print(f"Mean: {sk_scaled.mean(axis=0)}")
        print(f"Std: {sk_scaled.std(axis=0)}")
        
        # Assertions
        np.testing.assert_array_almost_equal(
            ml_scaled_np.mean(axis=0), 
            np.zeros_like(ml_scaled_np.mean(axis=0)), 
            decimal=10
        )
        np.testing.assert_array_almost_equal(
            ml_scaled_np.std(axis=0), 
            np.ones_like(ml_scaled_np.std(axis=0)), 
            decimal=10
        )
        np.testing.assert_array_almost_equal(ml_scaled_np, sk_scaled, decimal=10)

    def test_standard_scaler_edge(self):
        """Test StandardScaler with edge cases"""
        print("\nTesting StandardScaler with edge cases...")
        
        # Fit and transform using both implementations
        ml_scaled = self.standard_ml.fit_transform(self.edge_ml)
        sk_scaled = self.standard_sk.fit_transform(self.edge_np)
        
        # Convert MLArray to numpy for comparison
        ml_scaled_np = np.array(ml_scaled.to_list())
        
        # Plot comparison
        self.plot_comparison(
            self.edge_np, 
            ml_scaled_np, 
            sk_scaled,
            'StandardScaler Comparison - Edge Cases',
            'standard_scaler_edge.png'
        )
        
        # Assertions for edge cases
        np.testing.assert_array_almost_equal(ml_scaled_np, sk_scaled, decimal=10)

    def test_minmax_scaler_simple(self):
        """Test MinMaxScaler with simple data"""
        print("\nTesting MinMaxScaler with simple data...")
        
        # Fit and transform using both implementations
        ml_scaled = self.minmax_ml.fit_transform(self.simple_ml)
        sk_scaled = self.minmax_sk.fit_transform(self.simple_np)
        
        # Convert MLArray to numpy for comparison
        ml_scaled_np = np.array(ml_scaled.to_list())
        
        # Plot comparison
        self.plot_comparison(
            self.simple_np, 
            ml_scaled_np, 
            sk_scaled,
            'MinMaxScaler Comparison - Simple Data',
            'minmax_scaler_simple.png'
        )
        
        # Basic range tests
        print("\nMinMax Scaler Statistics (Simple Data):")
        print("Custom Implementation:")
        print(f"Min: {ml_scaled_np.min(axis=0)}")
        print(f"Max: {ml_scaled_np.max(axis=0)}")
        print("\nScikit-learn Implementation:")
        print(f"Min: {sk_scaled.min(axis=0)}")
        print(f"Max: {sk_scaled.max(axis=0)}")
        
        # Assertions
        np.testing.assert_array_almost_equal(
            ml_scaled_np.min(axis=0), 
            np.zeros_like(ml_scaled_np.min(axis=0)), 
            decimal=10
        )
        np.testing.assert_array_almost_equal(
            ml_scaled_np.max(axis=0), 
            np.ones_like(ml_scaled_np.max(axis=0)), 
            decimal=10
        )
        np.testing.assert_array_almost_equal(ml_scaled_np, sk_scaled, decimal=10)

    def test_minmax_scaler_edge(self):
        """Test MinMaxScaler with edge cases"""
        print("\nTesting MinMaxScaler with edge cases...")
        
        # Fit and transform using both implementations
        ml_scaled = self.minmax_ml.fit_transform(self.edge_ml)
        sk_scaled = self.minmax_sk.fit_transform(self.edge_np)
        
        # Convert MLArray to numpy for comparison
        ml_scaled_np = np.array(ml_scaled.to_list())
        
        # Plot comparison
        self.plot_comparison(
            self.edge_np, 
            ml_scaled_np, 
            sk_scaled,
            'MinMaxScaler Comparison - Edge Cases',
            'minmax_scaler_edge.png'
        )
        
        # Assertions for edge cases
        np.testing.assert_array_almost_equal(ml_scaled_np, sk_scaled, decimal=10)

    def test_single_value(self):
        """Test scalers with single value"""
        single_data_ml = MLArray([[1.0, 2.0]])
        single_data_np = np.array([[1.0, 2.0]])
        
        # Test StandardScaler
        std_ml = self.standard_ml.fit_transform(single_data_ml)
        std_sk = self.standard_sk.fit_transform(single_data_np)
        
        # Test MinMaxScaler
        minmax_ml = self.minmax_ml.fit_transform(single_data_ml)
        minmax_sk = self.minmax_sk.fit_transform(single_data_np)
        
        # Convert to numpy for comparison
        std_ml_np = np.array(std_ml.to_list())
        minmax_ml_np = np.array(minmax_ml.to_list())
        
        # These should be either 0 or NaN for both implementations
        np.testing.assert_array_almost_equal(std_ml_np, std_sk, decimal=10)
        np.testing.assert_array_almost_equal(minmax_ml_np, minmax_sk, decimal=10)

if __name__ == '__main__':
    unittest.main()