import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from smolml.core.ml_array import MLArray
import smolml.utils.losses as losses
from tests.img_utils import image_path

class TestLossFunctions(unittest.TestCase):
    """
    Test custom loss functions against TensorFlow implementations
    """
    
    def setUp(self):
        """
        Set up test data and configure TensorFlow
        """
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Regression data (for MSE, MAE, Huber)
        self.y_true_reg = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        self.y_pred_reg = [[1.2], [1.8], [3.3], [3.9], [5.2]]
        
        # Binary classification data (for BCE)
        self.y_true_bin = [[0], [1], [1], [0], [1]]
        self.y_pred_bin = [[0.2], [0.8], [0.7], [0.1], [0.9]]
        
        # Multi-class classification data (for CCE)
        self.y_true_cat = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ]
        self.y_pred_cat = [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.2, 0.6],
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2]
        ]
        
        # Convert to MLArray format
        self.ml_true_reg = MLArray(self.y_true_reg)
        self.ml_pred_reg = MLArray(self.y_pred_reg)
        self.ml_true_bin = MLArray(self.y_true_bin)
        self.ml_pred_bin = MLArray(self.y_pred_bin)
        self.ml_true_cat = MLArray(self.y_true_cat)
        self.ml_pred_cat = MLArray(self.y_pred_cat)
        
        # Convert to TensorFlow format
        self.tf_true_reg = tf.constant(self.y_true_reg, dtype=tf.float32)
        self.tf_pred_reg = tf.constant(self.y_pred_reg, dtype=tf.float32)
        self.tf_true_bin = tf.constant(self.y_true_bin, dtype=tf.float32)
        self.tf_pred_bin = tf.constant(self.y_pred_bin, dtype=tf.float32)
        self.tf_true_cat = tf.constant(self.y_true_cat, dtype=tf.float32)
        self.tf_pred_cat = tf.constant(self.y_pred_cat, dtype=tf.float32)

    def test_losses(self):
        """
        Test all loss functions and create visualization
        """
        # Create figure
        fig = plt.figure(figsize=(20, 15))
        plt.suptitle('Loss Function Comparisons', fontsize=16, y=0.95)
        
        # Calculate all losses
        # 1. MSE Loss
        custom_mse = float(losses.mse_loss(self.ml_pred_reg, self.ml_true_reg).data.data)
        mse = tf.keras.losses.MeanSquaredError()
        tf_mse = float(mse(self.tf_true_reg, self.tf_pred_reg).numpy())
        print(f"\nMSE - Custom: {custom_mse:.6f}, TF: {tf_mse:.6f}")
        np.testing.assert_allclose(custom_mse, tf_mse, rtol=1e-5)
        
        # 2. MAE Loss
        custom_mae = float(losses.mae_loss(self.ml_pred_reg, self.ml_true_reg).data.data)
        mae = tf.keras.losses.MeanAbsoluteError()
        tf_mae = float(mae(self.tf_true_reg, self.tf_pred_reg).numpy())
        print(f"MAE - Custom: {custom_mae:.6f}, TF: {tf_mae:.6f}")
        np.testing.assert_allclose(custom_mae, tf_mae, rtol=1e-5)
        
        # 3. Binary Cross-Entropy
        custom_bce = float(losses.binary_cross_entropy(self.ml_pred_bin, self.ml_true_bin).data.data)
        bce = tf.keras.losses.BinaryCrossentropy()
        tf_bce = float(bce(self.tf_true_bin, self.tf_pred_bin).numpy())
        print(f"BCE - Custom: {custom_bce:.6f}, TF: {tf_bce:.6f}")
        np.testing.assert_allclose(custom_bce, tf_bce, rtol=1e-5)
        
        # 4. Categorical Cross-Entropy
        custom_cce = float(losses.categorical_cross_entropy(self.ml_pred_cat, self.ml_true_cat).data.data)
        cce = tf.keras.losses.CategoricalCrossentropy()
        tf_cce = float(cce(self.tf_true_cat, self.tf_pred_cat).numpy())
        print(f"CCE - Custom: {custom_cce:.6f}, TF: {tf_cce:.6f}")
        np.testing.assert_allclose(custom_cce, tf_cce, rtol=1e-5)
        
        # 5. Huber Loss
        delta = 1.0
        custom_huber = float(losses.huber_loss(self.ml_pred_reg, self.ml_true_reg, delta).data.data)
        huber = tf.keras.losses.Huber(delta=delta)
        tf_huber = float(huber(self.tf_true_reg, self.tf_pred_reg).numpy())
        print(f"Huber - Custom: {custom_huber:.6f}, TF: {tf_huber:.6f}")
        np.testing.assert_allclose(custom_huber, tf_huber, rtol=1e-5)

        # Plotting
        # 1. Regression predictions (top left)
        ax1 = plt.subplot(3, 2, 1)
        plt.scatter([y[0] for y in self.y_true_reg], [y[0] for y in self.y_pred_reg],
                   c='blue', alpha=0.6, label='Predictions')
        plt.plot([min(self.y_true_reg), max(self.y_true_reg)],
                [min(self.y_true_reg), max(self.y_true_reg)],
                'r--', label='Perfect Prediction')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Regression Predictions vs Truth')
        plt.legend()

        # 2. Regression Losses Comparison (top right)
        ax2 = plt.subplot(3, 2, 2)
        width = 0.35
        x = np.arange(3)
        plt.bar(x - width/2, [custom_mse, custom_mae, custom_huber],
                width, label='Custom', color='blue', alpha=0.6)
        plt.bar(x + width/2, [tf_mse, tf_mae, tf_huber],
                width, label='TensorFlow', color='green', alpha=0.6)
        plt.xticks(x, ['MSE', 'MAE', 'Huber'])
        plt.title('Regression Loss Values')
        plt.legend()

        # 3. Binary Cross-Entropy (middle left)
        ax3 = plt.subplot(3, 2, 3)
        x = np.arange(len(self.y_true_bin))
        width = 0.35
        plt.bar(x - width/2, [p[0] for p in self.y_pred_bin],
                width, label='Predicted', color='blue', alpha=0.6)
        plt.bar(x + width/2, [t[0] for t in self.y_true_bin],
                width, label='True', color='green', alpha=0.6)
        plt.xlabel('Sample')
        plt.ylabel('Probability')
        plt.title(f'Binary Cross-Entropy\nCustom: {custom_bce:.4f}, TF: {tf_bce:.4f}')
        plt.legend()

        # 4. Categorical Cross-Entropy (middle right)
        ax4 = plt.subplot(3, 2, 4)
        num_samples = len(self.y_true_cat)
        x = np.arange(num_samples)
        width = 0.35

        # True values on the left, predicted on the right for each sample
        for i in range(num_samples):
            plt.bar(x[i] - width/2, self.y_true_cat[i],
                   width, label='True' if i == 0 else "", bottom=np.sum(self.y_true_cat[i][:i]),
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.6)
            plt.bar(x[i] + width/2, self.y_pred_cat[i],
                   width, label='Predicted' if i == 0 else "", bottom=np.sum(self.y_pred_cat[i][:i]),
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.3)

        plt.xlabel('Sample')
        plt.ylabel('Class Distribution')
        plt.title(f'Categorical Cross-Entropy\nCustom: {custom_cce:.4f}, TF: {tf_cce:.4f}')
        plt.legend()

        # 5. All Losses Comparison (bottom)
        ax5 = plt.subplot(3, 2, (5, 6))
        losses_custom = [custom_mse, custom_mae, custom_huber, custom_bce, custom_cce]
        losses_tf = [tf_mse, tf_mae, tf_huber, tf_bce, tf_cce]
        x = np.arange(5)
        plt.bar(x - width/2, losses_custom, width, label='Custom', color='blue', alpha=0.6)
        plt.bar(x + width/2, losses_tf, width, label='TensorFlow', color='green', alpha=0.6)
        plt.xticks(x, ['MSE', 'MAE', 'Huber', 'BCE', 'CCE'])
        plt.title('All Loss Functions Comparison')
        plt.legend()

        plt.tight_layout()
        plt.savefig(image_path('loss_functions_comparison.png'), bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == '__main__':
    unittest.main()