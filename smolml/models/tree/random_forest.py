from smolml.core.ml_array import MLArray
import random
from collections import Counter
from smolml.models.tree.decision_tree import DecisionTree

"""
/////////////////////
/// RANDOM FOREST ///
/////////////////////
"""

class RandomForest:
    """
    Random Forest implementation supporting both classification and regression.
    Uses bagging (bootstrap aggregating) and random feature selection.
    """
    def __init__(self, n_trees=100, max_features=None, max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1, bootstrap=True, 
                 task="classification"):
        """
        Initialize random forest with parameters for trees and bagging.
        
        Parameters:
        n_trees: Number of trees in the forest
        max_features: Number of features to consider for each split (if None, use sqrt for classification, 1/3 for regression)
        bootstrap: Whether to use bootstrap samples for each tree
        task: "classification" or "regression"
        """
        self.n_trees = n_trees
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.task = task if task in ["classification", "regression"] else None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        
    def _bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample with replacement.
        """
        n_samples = len(X)
        indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
        
        bootstrap_X = [X[i] for i in indices]
        bootstrap_y = [y[i] for i in indices]
        
        return bootstrap_X, bootstrap_y
    
    def _get_max_features(self, n_features):
        """
        Determine number of features to consider at each split.
        """
        if self.max_features is None:
            # Use sqrt(n_features) for classification, n_features/3 for regression
            if self.task == "classification":
                return max(1, int(n_features ** 0.5))
            else:
                return max(1, n_features // 3)
        return min(self.max_features, n_features)
    
    def fit(self, X, y):
        """
        Build random forest by creating and training individual trees.
        """
        if not isinstance(X, MLArray):
            X = MLArray(X)
        if not isinstance(y, MLArray):
            y = MLArray(y)
            
        X_data, y_data = X.data, y.data
        n_features = len(X_data[0])
        max_features = self._get_max_features(n_features)
        
        # Create and train each tree
        for _ in range(self.n_trees):
            # Create bootstrap sample if enabled
            if self.bootstrap:
                sample_X, sample_y = self._bootstrap_sample(X_data, y_data)
            else:
                sample_X, sample_y = X_data, y_data
            
            # Create and train tree with random feature selection
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                task=self.task
            )
            
            # Modify tree's _find_best_split to use random feature subset
            original_find_best_split = tree._find_best_split
            def random_feature_find_best_split(self, X, y):
                n_features = len(X[0])
                feature_indices = random.sample(range(n_features), max_features)
                
                best_gain = -float('inf')
                best_feature = None
                best_threshold = None
                
                for feature_idx in feature_indices:
                    thresholds = sorted(set(row[feature_idx] for row in X))
                    
                    for threshold in thresholds:
                        left_idxs, right_idxs = tree._split_data(X, feature_idx, threshold)
                        
                        if len(left_idxs) < tree.min_samples_leaf or len(right_idxs) < tree.min_samples_leaf:
                            continue
                        
                        left_y = [y[i] for i in left_idxs]
                        right_y = [y[i] for i in right_idxs]
                        
                        gain = tree._calculate_gain(y, left_y, right_y)
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_feature = feature_idx
                            best_threshold = threshold
                
                return best_feature, best_threshold
            
            # Replace tree's _find_best_split with our random feature version
            tree._find_best_split = random_feature_find_best_split.__get__(tree)
            
            # Train the tree
            tree.fit(MLArray(sample_X), MLArray(sample_y))
            self.trees.append(tree)
    
    def predict(self, X):
        """
        Make predictions by aggregating predictions from all trees.
        For classification: majority vote
        For regression: mean prediction
        """
        if not isinstance(X, MLArray):
            X = MLArray(X)
        
        # Get predictions from all trees
        tree_predictions = [tree.predict(X) for tree in self.trees]
        
        # Aggregate predictions based on task
        if self.task == "classification":
            final_predictions = []
            for i in range(len(X)):
                # Get predictions for this sample from all trees
                sample_predictions = [tree_pred.data[i] for tree_pred in tree_predictions]
                # Take majority vote
                vote = Counter(sample_predictions).most_common(1)[0][0]
                final_predictions.append(vote)
        elif self.task == "regression.py":
            final_predictions = []
            for i in range(len(X)):
                # Get predictions for this sample from all trees
                sample_predictions = [tree_pred.data[i] for tree_pred in tree_predictions]
                # Take mean
                mean = sum(sample_predictions) / len(sample_predictions)
                final_predictions.append(mean)
        else:
            raise Exception(f"Task in Random Forest not assigned to either 'classification' or 'regression'")
        
        return MLArray(final_predictions)