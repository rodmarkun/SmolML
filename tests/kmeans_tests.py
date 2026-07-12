import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as SKLearnKMeans
import matplotlib.pyplot as plt
from smolml.core.ml_array import MLArray
from smolml.models.unsupervised.kmeans import KMeans
from tests.img_utils import image_path

class TestKMeansVsSklearn(unittest.TestCase):
    """
    Compare custom KMeans implementation against scikit-learn
    using synthetic clustered data
    """
    
    def setUp(self):
        """
        Set up dataset and models
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate synthetic clustered data
        n_samples = 300
        self.n_clusters = 3
        X, y = make_blobs(n_samples=n_samples, 
                         centers=self.n_clusters,
                         cluster_std=1.0,
                         random_state=42)
        
        # Scale the data
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        # Store the data
        self.X = X
        self.y = y
        
        # Convert data for custom implementation
        self.X_ml = MLArray([[float(x) for x in row] for row in self.X])
        
        # Initialize models
        self.custom_kmeans = KMeans(n_clusters=self.n_clusters, 
                                  max_iters=100, 
                                  tol=1e-4)
        self.sklearn_kmeans = SKLearnKMeans(n_clusters=self.n_clusters,
                                          max_iter=100,
                                          tol=1e-4,
                                          random_state=42)

    def _plot_clusters(self, custom_labels, sklearn_labels):
        """
        Plot clustering results from both implementations
        """
        plt.figure(figsize=(12, 5))
        
        # Plot custom implementation results
        plt.subplot(1, 2, 1)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=custom_labels.to_list(), 
                   cmap='viridis', alpha=0.6)
        custom_centroids = self.custom_kmeans.centroids.to_list()
        plt.scatter(np.array(custom_centroids)[:, 0], 
                   np.array(custom_centroids)[:, 1], 
                   c='red', marker='x', s=200, linewidth=3, 
                   label='Centroids')
        plt.title('Custom KMeans Clustering')
        plt.legend()
        
        # Plot scikit-learn results
        plt.subplot(1, 2, 2)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=sklearn_labels, 
                   cmap='viridis', alpha=0.6)
        plt.scatter(self.sklearn_kmeans.cluster_centers_[:, 0],
                   self.sklearn_kmeans.cluster_centers_[:, 1],
                   c='red', marker='x', s=200, linewidth=3,
                   label='Centroids')
        plt.title('Scikit-learn KMeans Clustering')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(image_path('kmeans_comparison.png'))
        plt.close()

    def _compute_inertia(self, X, labels, centroids):
        """
        Compute inertia (within-cluster sum of squares)
        """
        inertia = 0
        X_list = X.to_list() if isinstance(X, MLArray) else X
        centroids_list = centroids.to_list() if isinstance(centroids, MLArray) else centroids
        labels_list = labels.to_list() if isinstance(labels, MLArray) else labels
        
        for i, point in enumerate(X_list):
            centroid = centroids_list[labels_list[i]]
            diff = np.array(point) - np.array(centroid)
            inertia += np.sum(diff ** 2)
            
        return inertia

    def _plot_training_progress(self):
        """
        Visualizes the training progress showing how centroids moved during training
        """
        n_plots = min(5, len(self.custom_kmeans.centroid_history))
        fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
        
        # If we only have one subplot, wrap it in a list
        if n_plots == 1:
            axes = [axes]
        
        # Select iterations to plot
        if len(self.custom_kmeans.centroid_history) <= n_plots:
            plot_iterations = range(len(self.custom_kmeans.centroid_history))
        else:
            # Select evenly spaced iterations including first and last
            plot_iterations = np.linspace(0, len(self.custom_kmeans.centroid_history)-1, n_plots, dtype=int)
        
        # Convert data to numpy for easier plotting
        X_array = np.array(self.X_ml.to_list())
        
        # Plot each selected iteration
        for idx, iter_idx in enumerate(plot_iterations):
            ax = axes[idx]
            centroids = np.array(self.custom_kmeans.centroid_history[iter_idx])
            
            # Plot data points
            if iter_idx == len(self.custom_kmeans.centroid_history) - 1:
                # For the final iteration, color points by cluster
                labels = np.array(self.custom_kmeans.labels_.to_list())
                scatter = ax.scatter(X_array[:, 0], X_array[:, 1], c=labels, 
                                cmap='viridis', alpha=0.6, s=50)
            else:
                # For earlier iterations, show all points in grey
                ax.scatter(X_array[:, 0], X_array[:, 1], c='grey', 
                        alpha=0.3, s=50)
            
            # Plot centroids
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', 
                    marker='x', s=200, linewidth=3, label='Centroids')
            
            # Add iteration number
            ax.set_title(f'Iteration {iter_idx}')
            
            # If first subplot, add legend
            if idx == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(image_path('kmeans_training_progress.png'))
        plt.close()

    def test_compare_clustering(self):
        """
        Train and compare both implementations
        """
        print("\nFitting custom KMeans...")
        custom_labels = self.custom_kmeans.fit_predict(self.X_ml)
        
        # Plot training progress
        self._plot_training_progress()
        
        print("Fitting scikit-learn KMeans...")
        sklearn_labels = self.sklearn_kmeans.fit_predict(self.X)
        
        # Plot clustering results
        self._plot_clusters(custom_labels, sklearn_labels)

if __name__ == '__main__':
    unittest.main()