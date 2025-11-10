import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from smolml.core.ml_array import MLArray
from smolml.models.tree.decision_tree import DecisionTree

class TestDecisionTree(unittest.TestCase):
    """
    Test decision tree implementation using sklearn datasets
    """
    
    def setUp(self):
        """
        Set up common parameters and load datasets
        """
        np.random.seed(42)
        
        # Load and prepare iris dataset for classification
        iris = load_iris()
        X_iris, y_iris = iris.data, iris.target
        
        # Use only two features for visualization
        self.X_iris = X_iris[:, [0, 1]]  # sepal length and sepal width
        self.y_iris = y_iris
        self.feature_names_iris = [iris.feature_names[0], iris.feature_names[1]]
        self.class_names_iris = iris.target_names
        
        # Split iris data
        self.X_iris_train, self.X_iris_test, self.y_iris_train, self.y_iris_test = train_test_split(
            self.X_iris, self.y_iris, test_size=0.2, random_state=42
        )
        
        # Load and prepare diabetes dataset for regression
        diabetes = load_diabetes()
        X_diabetes, y_diabetes = diabetes.data, diabetes.target
        
        # Scale the data
        scaler = StandardScaler()
        X_diabetes = scaler.fit_transform(X_diabetes)
        y_diabetes = (y_diabetes - y_diabetes.mean()) / y_diabetes.std()
        
        # Use only two features for visualization
        self.X_diabetes = X_diabetes[:, [0, 2]]  # age and bmi
        self.y_diabetes = y_diabetes
        self.feature_names_diabetes = [diabetes.feature_names[0], diabetes.feature_names[2]]
        
        # Split diabetes data
        self.X_diabetes_train, self.X_diabetes_test, self.y_diabetes_train, self.y_diabetes_test = train_test_split(
            self.X_diabetes, self.y_diabetes, test_size=0.2, random_state=42
        )

    def plot_decision_boundary(self, X, y, tree, title, feature_names, class_names=None):
        """
        Plot decision boundary for classification or color-coded predictions for regression
        """
        plt.figure(figsize=(10, 8))
        
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        # Convert mesh points to list format for MLArray
        mesh_points = [[float(x), float(y)] for x, y in zip(xx.ravel(), yy.ravel())]
        
        # Make predictions
        Z = tree.predict(MLArray(mesh_points)).to_list()
        Z = np.array(Z).reshape(xx.shape)
        
        if class_names is not None:  # Classification
            # Plot decision boundary
            plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
            scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
            plt.colorbar(scatter)
            
            # Add legend
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=plt.cm.viridis(i/2.), 
                                        label=class_names[i], markersize=10)
                             for i in range(3)]
            plt.legend(handles=legend_elements)
        else:  # Regression
            # Plot predictions
            plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
            scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
            plt.colorbar(scatter)
        
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title(title)
        
        # Save plot
        plot_type = 'classification' if class_names is not None else 'regression'
        plt.savefig(f'decision_tree_{plot_type}.png')
        plt.close()

    def test_classification(self):
        """
        Test decision tree classification on iris dataset
        """
        print("\nTesting Classification on Iris Dataset...")
        
        # Convert numpy arrays to lists for MLArray
        X_train_list = [[float(x) for x in row] for row in self.X_iris_train]
        y_train_list = [float(y) for y in self.y_iris_train]
        X_test_list = [[float(x) for x in row] for row in self.X_iris_test]
        
        # Create and train tree
        clf = DecisionTree(max_depth=10, min_samples_split=5, task="classification")
        clf.fit(MLArray(X_train_list), MLArray(y_train_list))
        print(clf)
        
        # Make predictions
        y_pred = clf.predict(MLArray(X_test_list))
        y_pred = np.array(y_pred.to_list())
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == self.y_iris_test)
        print(f"Classification Accuracy: {accuracy:.3f}")
        
        # Prepare data for plotting
        X_plot_list = [[float(x) for x in row] for row in self.X_iris]
        y_plot_list = [float(y) for y in self.y_iris]
        
        # Plot decision boundary
        self.plot_decision_boundary(
            self.X_iris, self.y_iris, clf,
            "Iris Classification Decision Boundary",
            self.feature_names_iris, self.class_names_iris
        )
        
        # Show tree with feature names
        print("\nDecision Tree Structure:")
        clf.show_tree(feature_names=self.feature_names_iris)
        
        # Assertions
        self.assertGreater(accuracy, 0.7, "Classification accuracy should be > 70%")

    def test_regression(self):
        """
        Test decision tree regression on diabetes dataset
        """
        print("\nTesting Regression on Diabetes Dataset...")
        
        # Convert numpy arrays to lists for MLArray
        X_train_list = [[float(x) for x in row] for row in self.X_diabetes_train]
        y_train_list = [float(y) for y in self.y_diabetes_train]
        X_test_list = [[float(x) for x in row] for row in self.X_diabetes_test]
        
        # Create and train tree
        reg = DecisionTree(max_depth=5, min_samples_split=5, task="regression")
        reg.fit(MLArray(X_train_list), MLArray(y_train_list))
        print(reg)
        
        # Make predictions
        y_pred = reg.predict(MLArray(X_test_list))
        y_pred = np.array(y_pred.to_list())
        
        # Calculate MSE
        mse = np.mean((y_pred - self.y_diabetes_test) ** 2)
        print(f"Regression MSE: {mse:.3f}")
        
        # Prepare data for plotting
        X_plot_list = [[float(x) for x in row] for row in self.X_diabetes]
        y_plot_list = [float(y) for y in self.y_diabetes]
        
        # Plot predictions
        self.plot_decision_boundary(
            self.X_diabetes, self.y_diabetes, reg,
            "Diabetes Regression Predictions",
            self.feature_names_diabetes
        )
        
        # Show tree with feature names
        print("\nDecision Tree Structure:")
        reg.show_tree(feature_names=self.feature_names_diabetes)
        
        # Assertions
        self.assertLess(mse, 1.0, "MSE should be < 1.0 for scaled data")

    def test_edge_cases(self):
        """
        Test edge cases and parameter validation
        """
        # Test with minimal leaf samples
        tree = DecisionTree(max_depth=2, min_samples_leaf=1)
        tree.fit(MLArray([[1], [2]]), MLArray([1, 2]))
        self.assertIsNotNone(tree.root)
        
        print("\nEdge Case Tree Structure:")
        tree.show_tree(feature_names=['simple_feature'])
        
        # Test with single class
        tree = DecisionTree()
        tree.fit(MLArray([[1], [2]]), MLArray([1, 1]))
        self.assertEqual(tree.predict(MLArray([[1.5]])).to_list()[0], 1)
        
        print("\nSingle Class Tree Structure:")
        tree.show_tree(feature_names=['feature_x'])

if __name__ == '__main__':
    unittest.main()