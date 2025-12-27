import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from sklearn.datasets import load_iris, make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR as SklearnSVR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from smolml.core.ml_array import MLArray
from smolml.models.svm import SVM, SVMMulticlass, SVR


class TestSVMClassification(unittest.TestCase):
    """
    Test SVM classification implementation against sklearn
    """

    def setUp(self):
        """
        Set up datasets for testing
        """
        np.random.seed(42)

        # Linearly separable dataset
        X_simple, y_simple = make_classification(
            n_samples=100, n_features=2, n_informative=2,
            n_redundant=0, n_clusters_per_class=1, random_state=42
        )
        scaler = StandardScaler()
        self.X_simple = scaler.fit_transform(X_simple)
        self.y_simple = y_simple
        self.X_simple_train, self.X_simple_test, self.y_simple_train, self.y_simple_test = train_test_split(
            self.X_simple, self.y_simple, test_size=0.2, random_state=42
        )

        # Non-linearly separable dataset (moons)
        X_moons, y_moons = make_moons(n_samples=100, noise=0.1, random_state=42)
        scaler = StandardScaler()
        self.X_moons = scaler.fit_transform(X_moons)
        self.y_moons = y_moons
        self.X_moons_train, self.X_moons_test, self.y_moons_train, self.y_moons_test = train_test_split(
            self.X_moons, self.y_moons, test_size=0.2, random_state=42
        )

        # Iris dataset for multiclass
        iris = load_iris()
        X_iris, y_iris = iris.data[:, :2], iris.target
        scaler = StandardScaler()
        self.X_iris = scaler.fit_transform(X_iris)
        self.y_iris = y_iris
        self.X_iris_train, self.X_iris_test, self.y_iris_train, self.y_iris_test = train_test_split(
            self.X_iris, self.y_iris, test_size=0.2, random_state=42
        )

    def plot_comparison(self, X, y, smol_model, sklearn_model, title, filename):
        """Plot decision boundaries for SmolML vs sklearn"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))

        # SmolML predictions
        mesh_points = [[float(x), float(y)] for x, y in zip(xx.ravel(), yy.ravel())]
        Z_smol = np.array(smol_model.predict(MLArray(mesh_points)).to_list()).reshape(xx.shape)

        axes[0].contourf(xx, yy, Z_smol, alpha=0.4, cmap='RdYlBu')
        axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black', s=50)
        axes[0].set_title(f'SmolML: {title}')
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')

        # sklearn predictions
        Z_sklearn = sklearn_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        axes[1].contourf(xx, yy, Z_sklearn, alpha=0.4, cmap='RdYlBu')
        axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black', s=50)
        axes[1].set_title(f'sklearn: {title}')
        axes[1].set_xlabel('Feature 1')
        axes[1].set_ylabel('Feature 2')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"  Plot saved: {filename}")

    def test_linear_kernel(self):
        """Test linear kernel SVM against sklearn"""
        print("\n" + "="*60)
        print("Testing Linear Kernel SVM vs sklearn")
        print("="*60)

        X_train_list = [[float(x) for x in row] for row in self.X_simple_train]
        y_train_list = [int(y) for y in self.y_simple_train]
        X_test_list = [[float(x) for x in row] for row in self.X_simple_test]

        # SmolML
        svm = SVM(kernel='linear', C=1.0, max_iter=1000)
        svm.fit(MLArray(X_train_list), MLArray(y_train_list))
        smol_acc = np.mean(np.array(svm.predict(MLArray(X_test_list)).to_list()) == self.y_simple_test)

        # sklearn
        sklearn_svm = SVC(kernel='linear', C=1.0)
        sklearn_svm.fit(self.X_simple_train, self.y_simple_train)
        sklearn_acc = sklearn_svm.score(self.X_simple_test, self.y_simple_test)

        print(f"SmolML Accuracy:  {smol_acc:.3f}")
        print(f"sklearn Accuracy: {sklearn_acc:.3f}")

        self.plot_comparison(self.X_simple, self.y_simple, svm, sklearn_svm,
                           "Linear Kernel", "svm_linear.png")

        self.assertGreater(smol_acc, 0.8)

    def test_rbf_kernel(self):
        """Test RBF kernel SVM against sklearn"""
        print("\n" + "="*60)
        print("Testing RBF Kernel SVM vs sklearn (Moons)")
        print("="*60)

        X_train_list = [[float(x) for x in row] for row in self.X_moons_train]
        y_train_list = [int(y) for y in self.y_moons_train]
        X_test_list = [[float(x) for x in row] for row in self.X_moons_test]

        # SmolML
        svm = SVM(kernel='rbf', C=1.0, gamma='scale', max_iter=1000)
        svm.fit(MLArray(X_train_list), MLArray(y_train_list))
        smol_acc = np.mean(np.array(svm.predict(MLArray(X_test_list)).to_list()) == self.y_moons_test)

        # sklearn
        sklearn_svm = SVC(kernel='rbf', C=1.0, gamma='scale')
        sklearn_svm.fit(self.X_moons_train, self.y_moons_train)
        sklearn_acc = sklearn_svm.score(self.X_moons_test, self.y_moons_test)

        print(f"SmolML Accuracy:  {smol_acc:.3f}")
        print(f"sklearn Accuracy: {sklearn_acc:.3f}")

        self.plot_comparison(self.X_moons, self.y_moons, svm, sklearn_svm,
                           "RBF Kernel (Moons)", "svm_rbf.png")

        self.assertGreater(smol_acc, 0.7)

    def test_multiclass(self):
        """Test multiclass SVM against sklearn"""
        print("\n" + "="*60)
        print("Testing Multiclass SVM vs sklearn (Iris)")
        print("="*60)

        X_train_list = [[float(x) for x in row] for row in self.X_iris_train]
        y_train_list = [int(y) for y in self.y_iris_train]
        X_test_list = [[float(x) for x in row] for row in self.X_iris_test]

        # SmolML
        svm = SVMMulticlass(kernel='rbf', C=1.0, gamma='scale', max_iter=500)
        svm.fit(MLArray(X_train_list), MLArray(y_train_list))
        smol_acc = np.mean(np.array(svm.predict(MLArray(X_test_list)).to_list()) == self.y_iris_test)

        # sklearn
        sklearn_svm = SVC(kernel='rbf', C=1.0, gamma='scale')
        sklearn_svm.fit(self.X_iris_train, self.y_iris_train)
        sklearn_acc = sklearn_svm.score(self.X_iris_test, self.y_iris_test)

        print(f"SmolML Accuracy:  {smol_acc:.3f}")
        print(f"sklearn Accuracy: {sklearn_acc:.3f}")

        self.plot_comparison(self.X_iris, self.y_iris, svm, sklearn_svm,
                           "Multiclass (Iris)", "svm_multiclass.png")

        self.assertGreater(smol_acc, 0.6)


class TestSVRRegression(unittest.TestCase):
    """
    Test SVR implementation against sklearn
    """

    def setUp(self):
        """Set up regression datasets"""
        np.random.seed(42)

        # Sinusoidal data
        self.X_sin = np.linspace(0, 10, 50).reshape(-1, 1)
        self.y_sin = np.sin(self.X_sin).flatten() + np.random.randn(50) * 0.1
        self.X_sin_train, self.X_sin_test, self.y_sin_train, self.y_sin_test = train_test_split(
            self.X_sin, self.y_sin, test_size=0.2, random_state=42
        )

    def plot_comparison(self, X, y, smol_model, sklearn_model, title, filename):
        """Plot regression results for SmolML vs sklearn"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sort_idx = np.argsort(X.flatten())
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]

        # SmolML predictions
        X_list = [[float(x)] for x in X_sorted.flatten()]
        y_pred_smol = smol_model.predict(MLArray(X_list)).to_list()

        axes[0].scatter(X_sorted, y_sorted, c='#1E88E5', alpha=0.6, s=50, label='Data')
        axes[0].plot(X_sorted, y_pred_smol, color='#D81B60', lw=2, label='SVR')
        axes[0].set_title(f'SmolML: {title}')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # sklearn predictions
        y_pred_sklearn = sklearn_model.predict(X_sorted)

        axes[1].scatter(X_sorted, y_sorted, c='#1E88E5', alpha=0.6, s=50, label='Data')
        axes[1].plot(X_sorted, y_pred_sklearn, color='#D81B60', lw=2, label='SVR')
        axes[1].set_title(f'sklearn: {title}')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"  Plot saved: {filename}")

    def test_rbf_svr(self):
        """Test RBF kernel SVR against sklearn"""
        print("\n" + "="*60)
        print("Testing RBF Kernel SVR vs sklearn")
        print("="*60)

        X_train_list = [[float(x)] for x in self.X_sin_train.flatten()]
        y_train_list = [float(y) for y in self.y_sin_train]
        X_test_list = [[float(x)] for x in self.X_sin_test.flatten()]

        # SmolML
        svr = SVR(kernel='rbf', C=100.0, epsilon=0.1, gamma=0.5, max_iter=1000)
        svr.fit(MLArray(X_train_list), MLArray(y_train_list))
        smol_r2 = svr.score(MLArray(X_test_list), MLArray(self.y_sin_test.tolist()))

        # sklearn
        sklearn_svr = SklearnSVR(kernel='rbf', C=100.0, epsilon=0.1, gamma=0.5)
        sklearn_svr.fit(self.X_sin_train, self.y_sin_train)
        sklearn_r2 = sklearn_svr.score(self.X_sin_test, self.y_sin_test)

        print(f"SmolML R²:  {smol_r2:.3f}")
        print(f"sklearn R²: {sklearn_r2:.3f}")

        self.plot_comparison(self.X_sin, self.y_sin, svr, sklearn_svr,
                           "RBF Kernel (Sinusoidal)", "svr_rbf.png")

    def test_linear_svr(self):
        """Test linear kernel SVR against sklearn"""
        print("\n" + "="*60)
        print("Testing Linear Kernel SVR vs sklearn")
        print("="*60)

        X_train_list = [[float(x)] for x in self.X_sin_train.flatten()]
        y_train_list = [float(y) for y in self.y_sin_train]
        X_test_list = [[float(x)] for x in self.X_sin_test.flatten()]

        # SmolML
        svr = SVR(kernel='linear', C=10.0, epsilon=0.1, max_iter=1000)
        svr.fit(MLArray(X_train_list), MLArray(y_train_list))
        smol_r2 = svr.score(MLArray(X_test_list), MLArray(self.y_sin_test.tolist()))

        # sklearn
        sklearn_svr = SklearnSVR(kernel='linear', C=10.0, epsilon=0.1)
        sklearn_svr.fit(self.X_sin_train, self.y_sin_train)
        sklearn_r2 = sklearn_svr.score(self.X_sin_test, self.y_sin_test)

        print(f"SmolML R²:  {smol_r2:.3f}")
        print(f"sklearn R²: {sklearn_r2:.3f}")

        self.plot_comparison(self.X_sin, self.y_sin, svr, sklearn_svr,
                           "Linear Kernel", "svr_linear.png")


if __name__ == '__main__':
    unittest.main()
