from smolml.core.ml_array import MLArray
import smolml.utils.memory as memory
from collections import Counter
import math

"""
/////////////////////
/// DECISION TREE ///
/////////////////////
"""

class DecisionNode:
    """
    Node in decision tree that handles splitting logic.
    Can be either internal node (with split rule) or leaf (with prediction).
    """
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Index of feature to split on
        self.threshold = threshold      # Value to split feature on
        self.left = left               # Left subtree (feature <= threshold)
        self.right = right             # Right subtree (feature > threshold)
        self.value = value             # Prediction value (for leaf nodes)

    def __repr__(self):
        if self.value is not None:
            return f"Leaf(value={self.value})"
        return f"Node(feature={self.feature_idx}, threshold={self.threshold:.4f})"

class DecisionTree:
    """
    Decision Tree implementation supporting both classification and regression.
    Uses binary splitting based on feature thresholds.
    """
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, task="classification"):
        """
        Initialize decision tree with stopping criteria.
        
        max_depth: Maximum tree depth to prevent overfitting
        min_samples_split: Minimum samples required to split node
        min_samples_leaf: Minimum samples required in leaf nodes
        task: "classification" or "regression"
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.task = task
        self.root = None

    def fit(self, X, y):
        """
        Build decision tree by recursively splitting data.
        """
        if not isinstance(X, MLArray):
            X = MLArray(X)
        if not isinstance(y, MLArray):
            y = MLArray(y)
            
        self.n_classes = len(set(y.flatten(y.data))) if self.task == "classification" else None
        self.root = self._grow_tree(X.data, y.data)

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grows tree by finding best splits.
        """
        n_samples = len(X)
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            self._is_pure(y)):
            return DecisionNode(value=self._leaf_value(y))

        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:  # No valid split found
            return DecisionNode(value=self._leaf_value(y))

        # Split data
        left_idxs, right_idxs = self._split_data(X, best_feature, best_threshold)
        
        # Check min_samples_leaf
        if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
            return DecisionNode(value=self._leaf_value(y))

        # Create child nodes
        left_X = [X[i] for i in left_idxs]
        right_X = [X[i] for i in right_idxs]
        left_y = [y[i] for i in left_idxs]
        right_y = [y[i] for i in right_idxs]

        left = self._grow_tree(left_X, left_y, depth + 1)
        right = self._grow_tree(right_X, right_y, depth + 1)

        return DecisionNode(feature_idx=best_feature, threshold=best_threshold, left=left, right=right)

    def _find_best_split(self, X, y):
        """
        Finds best feature and threshold for splitting data.
        Uses information gain for classification, MSE for regression.
        """
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None

        n_features = len(X[0])
        
        for feature_idx in range(n_features):
            thresholds = sorted(set(row[feature_idx] for row in X))
            
            for threshold in thresholds:
                left_idxs, right_idxs = self._split_data(X, feature_idx, threshold)
                
                if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
                    continue

                left_y = [y[i] for i in left_idxs]
                right_y = [y[i] for i in right_idxs]
                
                gain = self._calculate_gain(y, left_y, right_y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _split_data(self, X, feature_idx, threshold):
        """
        Splits data based on feature and threshold.
        """
        left_idxs = []
        right_idxs = []
        
        for i, row in enumerate(X):
            if row[feature_idx] <= threshold:
                left_idxs.append(i)
            else:
                right_idxs.append(i)
                
        return left_idxs, right_idxs

    def _calculate_gain(self, parent, left, right):
        """
        Calculates gain from split:
        - Information gain for classification
        - Reduction in MSE for regression
        """
        if self.task == "classification":
            return self._information_gain_entropy(parent, left, right)
        return self._information_gain_mse(parent, left, right)

    def _information_gain_entropy(self, parent, left, right):
        """
        Calculates information gain using entropy.
        """
        def entropy(y):
            counts = Counter(y)
            probs = [count/len(y) for count in counts.values()]
            return -sum(p * math.log2(p) for p in probs)

        n = len(parent)
        entropy_parent = entropy(parent)
        entropy_children = (len(left)/n * entropy(left) + 
                          len(right)/n * entropy(right))
        return entropy_parent - entropy_children

    def _information_gain_mse(self, parent, left, right):
        """
        Calculates reduction in MSE.
        """
        def mse(y):
            mean = sum(y)/len(y)
            return sum((val - mean)**2 for val in y)/len(y)

        n = len(parent)
        mse_parent = mse(parent)
        mse_children = (len(left)/n * mse(left) + 
                       len(right)/n * mse(right))
        return mse_parent - mse_children

    def _split_data(self, X, feature_idx, threshold):
        """
        Splits data based on feature and threshold.
        """
        left_idxs = [i for i, row in enumerate(X) if row[feature_idx] <= threshold]
        right_idxs = [i for i, row in enumerate(X) if row[feature_idx] > threshold]
        return left_idxs, right_idxs

    def _is_pure(self, y):
        """
        Checks if node is pure (all same class/value).
        """
        return len(set(y)) == 1

    def _leaf_value(self, y):
        """
        Determines prediction value for leaf node:
        - Most common class for classification
        - Mean value for regression
        """
        if self.task == "classification":
            return max(set(y), key=y.count)
        return sum(y)/len(y)

    def predict(self, X):
        """
        Makes predictions using trained tree.
        """
        if not isinstance(X, MLArray):
            X = MLArray(X)
            
        return MLArray([self._traverse_tree(x, self.root) for x in X.data])

    def _traverse_tree(self, x, node):
        """
        Traverses tree to make prediction for single sample.
        """
        if node.value is not None:  # Leaf node
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def show_tree(self, node=None, prefix="", is_left=True, is_root=True, feature_names=None):
        """
        Prints a visual representation of the decision tree structure.
        """
        if node is None:
            if self.root is None:
                print("Tree not yet trained!")
                return
            node = self.root
        
        # Determine the connector symbol
        if is_root:
            connector = "Root: "
            branch = ""
        else:
            connector = "├─ Left: " if is_left else "└─ Right: "
            branch = prefix
        
        # Print current node
        if node.value is not None:  # Leaf node
            leaf_val = node.value
            if hasattr(leaf_val, 'data'):
                leaf_val = leaf_val.data
            
            if self.task == "classification":
                print(f"{branch}{connector}Leaf → Class {leaf_val}")
            else:
                print(f"{branch}{connector}Leaf → {leaf_val:.4f}")
        else:  # Internal node
            # Extract threshold value if it's wrapped in Value/MLArray
            threshold_val = node.threshold
            if hasattr(threshold_val, 'data'):
                threshold_val = threshold_val.data
            
            # Get feature name
            if feature_names is not None and node.feature_idx < len(feature_names):
                feature_name = feature_names[node.feature_idx]
            else:
                feature_name = f"feature_{node.feature_idx}"
            
            print(f"{branch}{connector}{feature_name} <= {threshold_val:.4f}")
            
            # Prepare prefix for children
            if is_root:
                new_prefix = ""
            else:
                new_prefix = prefix + ("│  " if is_left else "   ")
            
            # Recursively print children
            if node.left:
                self.show_tree(node.left, new_prefix, is_left=True, is_root=False, feature_names=feature_names)
            if node.right:
                self.show_tree(node.right, new_prefix, is_left=False, is_root=False, feature_names=feature_names)
    
    def __repr__(self):
        """
        Returns string representation of decision tree with structure and memory information.
        """
        try:
            import os
            terminal_width = os.get_terminal_size().columns
        except Exception:
            terminal_width = 80
            
        header = f"Decision Tree ({self.task.title()})"
        separator = "=" * terminal_width
        
        # Get size information
        size_info = memory.calculate_decision_tree_size(self)
        
        # Model parameters
        params = [
            f"Max Depth: {self.max_depth if self.max_depth is not None else 'None'}",
            f"Min Samples Split: {self.min_samples_split}",
            f"Min Samples Leaf: {self.min_samples_leaf}",
            f"Task: {self.task}"
        ]
        
        # Tree structure information
        if self.root:
            structure_info = [
                "Tree Structure:",
                f"  Internal Nodes: {size_info['tree_structure']['internal_nodes']}",
                f"  Leaf Nodes: {size_info['tree_structure']['leaf_nodes']}",
                f"  Max Depth: {size_info['tree_structure']['max_depth']}",
                f"  Total Nodes: {size_info['tree_structure']['internal_nodes'] + size_info['tree_structure']['leaf_nodes']}"
            ]
        else:
            structure_info = ["Tree not yet trained"]
        
        # Memory usage
        memory_info = ["Memory Usage:"]
        memory_info.append(f"  Base Tree: {memory.format_size(size_info['base_size'])}")
        if self.root:
            memory_info.append(f"  Tree Structure: {memory.format_size(size_info['tree_structure']['total'])}")
        memory_info.append(f"Total Memory: {memory.format_size(size_info['total'])}")
        
        return (
            f"\n{header}\n{separator}\n\n"
            + "Parameters:\n" + "\n".join(f"  {param}" for param in params)
            + "\n\n" + "\n".join(structure_info)
            + "\n\n" + "\n".join(memory_info)
            + f"\n{separator}\n"
        )