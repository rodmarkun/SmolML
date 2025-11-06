# SmolML - Tree Models: Decisions, Decisions!

Welcome to the *branch* of SmolML dealing with **Tree-Based Models**! Unlike the models we saw in `Regression` (which rely on smooth equations and gradient descent), Decision Trees and their powerful sibling, Random Forests, make predictions by learning a series of explicit **decision rules** from the data. Think of it like building a sophisticated flowchart to classify an email as spam or not spam, or to predict a house price.

Instead of finding a smooth mathematical function that fits the data, trees ask a series of yes/no questions. "_Is the email subject line longer than 50 characters?_" → "_Does it contain the word 'FREE'?_" → "_Was it sent after midnight?_" Each answer narrows down the prediction until we reach a final decision. This makes them incredibly interpretable as you can literally trace the path the model took to make a prediction!

These models are incredibly versatile, handling both **classification** (predicting categories) and **regression** (predicting numerical values) tasks. They **don't need feature scaling** and can capture complex, non-linear relationships. Let's dive into how they work!

## Decision Trees

Imagine you're trying to decide if you should play tennis today. You might ask:
1.  Is the outlook sunny?
    * Yes -> Is it scorching hot?
        * Yes -> Don't Play
        * No -> Play!
    * No -> Is it raining?
        * Yes -> Don't Play
        * No -> Play!

That's the essence of a **Decision Tree**! It's a structure that recursively splits the data based on simple questions about the input features. But how do we actually _train_ something like this?

We could model each of these decisions as a `DecisionNode`. Imagine it! We take today's data. Based on that data we choose **a feature to split on**. For example, in the first decision we check the outlook of the day, while in the next one we might check temperature or a boolean indicating rain.

Of course, we need some **threshold** to split our decisions. For example, a British person might find 30°C outrageously hot, or 40°C for a Spanish one. But that threshold is what determines whether we play or not.

Each decision might lead to other decisions, so we should store the **possible decisions** down our tree. Basically, other `DecisionNode`s that derive from the current one.

And finally, we need some **value** for the final (leaf) decisions. In this case, we might assume that if we play, the value is `1`, and if we don't, it's `0`.

We can implement a simple class based on this!

```python
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
```

<div align="center">
  <img src="https://github.com/user-attachments/assets/0b805169-fa57-4097-80e0-e841ea3246af" width="600">
</div>

Great! Now we have an object to represent our decisions. But we're missing the most important part: using this as part of a `DecisionTree`. The goal is for the model to automatically learn which feature and threshold to split on from a dataset, eventually reaching leaf nodes with prediction values.

Okay, let's think this through: we need a structure to hold these `DecisionNode`s. These are actually continuously generated during training, so... maybe we should have some kind of limit?

We must ask ourselves: how _big_ do we want our tree to be? Or better yet, how **deep**? That's the main parameter our tree is going to have! A `max_depth` parameter will control how many levels of decisions we allow. A deeper tree can capture more complex patterns but risks overfitting to the training data.

Of course, we need an starting `DecisionNode` that will serve as our **root**, from which all other `DecisionNode`s will be children. But we can save that for later, whenever we start actually training our model.

But depth isn't the only consideration! We also need **stopping criteria** to prevent the tree from making splits that don't help much:
- What's the **minimum number of samples** a node must have before we even consider splitting it? If a node only has a handful of data points, splitting might not be meaningful.
- How many **samples should each leaf node contain at minimum**? This ensures our predictions are based on a reasonable amount of data rather than just one or two examples. It's another guard against overfitting, we don't want leaves that memorize individual training examples!

Finally, for implementation purposes, we should know the **task** for which the `DecisionTree` is to be used (_regression_ or _classification_). This determines how we make predictions at leaf nodes: for classification, we typically use the most common class, while for regression, we use the average value of samples in that leaf.

We end up with a constructor like this:

```python
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
```

Fine, time to train this thing! I hope you like recursivity because boy, we are up to a ride.

When dealing with recursivity, we should first define when is our algorithm going to end:
- **We reach max depth**: If we've gone as deep as we specified in `max_depth`, we stop. This prevents the tree from becoming too complex and overfitting to the training data.
- **Too few samples to split**: If the current node has fewer samples than `min_samples_split`, we can't meaningfully split it further. Imagine trying to make a decision based on just one data point, that's just not good.
- **Too few samples for a leaf**: _Even_ if we split, if either resulting child would have fewer than `min_samples_leaf` samples, we don't bother. This ensures our predictions are based on enough data.
- **Pure node**: If all samples in the current node have the same label (for classification) or value (for regression), there's no point in splitting further, we've literally splitted it perfectly!

When any of these conditions are true, we create a leaf node with a prediction value. For classification, this is the most common class among the samples in that node. For regression, it's the average value.

Now, if none of these criteria are met, we need to actually split the data. We should define a method that will tell us which is the best **feature** and **threshold** to use. This method should loop over each **feature** and each **threshold** over our data. 

**How it's Built (The `fit` method):**

The magic happens in the `fit` method of the `DecisionTree` class (see `decision_tree.py`). It builds the tree structure, represented by interconnected `DecisionNode` objects, using a process called **recursive partitioning**:

1.  **Start with all data:** Begin at the root node with your entire training dataset.
2.  **Find the Best Question:** The core task is to find the *best* feature and *best* threshold value to split the current data into two groups (left and right branches). What's "best"? A split that makes the resulting groups as "pure" or homogeneous as possible regarding the target variable (e.g., all samples in a group belong to the same class, or have very similar numerical values).
    * **How? The `_find_best_split` and `_calculate_gain` methods:** The tree tries *every possible split* (each feature, each unique value as a threshold) and evaluates how much "purer" the resulting groups are compared to the parent group.
        * **For Classification:** It typically uses **Entropy** (a measure of disorder) and calculates the **Information Gain** (how much the entropy decreases after the split). A higher gain means a better split. (See `_information_gain`).
        * **For Regression:** It typically uses **Variance** or **Mean Squared Error (MSE)** and calculates how much this metric is reduced by the split. A larger reduction means a better split. (See `_mse_reduction`).
3.  **Split the Data:** Apply the best split found, dividing the data into two subsets.
4.  **Repeat Recursively:** Treat each subset as a new problem and repeat steps 2 and 3 for the left and right branches, creating child nodes (`_grow_tree` method calls itself).
5.  **Stop Splitting (Create a Leaf Node):** The recursion stops, and a **leaf node** (a `DecisionNode` with a `value` but no children) is created when certain conditions are met:
    * The node is perfectly "pure" (all samples belong to the same class/have very similar values - check `_is_pure`).
    * A predefined `max_depth` is reached.
    * The number of samples in a node falls below `min_samples_split`.
    * A potential split would result in a child node having fewer than `min_samples_leaf` samples.
    * No further split improves purity.
    These stopping criteria (hyperparameters set during `__init__`) are crucial to prevent the tree from growing too complex and **overfitting** (memorizing the training data instead of learning general patterns).

**Making Predictions (The `predict` method):**

Once the tree is built, predicting is straightforward! For a new data point:
1.  Start at the `root` node.
2.  Check the decision rule (feature and threshold) at the current node.
3.  Follow the corresponding branch (left if `feature_value <= threshold`, right otherwise).
4.  Repeat steps 2 and 3 until you reach a leaf node (`_traverse_tree` method).
5.  The prediction is the value stored in that leaf node (`node.value`). This value is determined during training (`_leaf_value`):
    * Classification: The most common class among the training samples that ended up in this leaf.
    * Regression: The average value of the training samples that ended up in this leaf.

Cool, right? A single tree is intuitive, but sometimes they can be a bit unstable and prone to overfitting. What if we could combine *many* trees?

## Random Forests: The Wisdom of Many Trees


<div align="center">
  <img src="https://github.com/user-attachments/assets/6a652774-4fc3-4ed1-89d4-e9eaf1410e2a" width="600">
</div>

A single Decision Tree can be sensitive to the specific data it's trained on. A slightly different dataset might produce a very different tree structure. **Random Forests** tackle this by building an *ensemble* (a "forest") of many Decision Trees and combining their predictions. It's like asking many different experts (trees) and going with the consensus!

**The "Random" Secrets (`RandomForest` class in `random_forest.py`):**

Random Forests introduce clever randomness during the training (`fit` method) of individual trees to make them diverse:

1.  **Bagging (Bootstrap Aggregating):** Each tree in the forest is trained on a slightly different dataset. This is done by **bootstrapping**: creating a random sample of the original training data *with replacement*. This means some data points might appear multiple times in a tree's training set, while others might be left out entirely. (Controlled by the `bootstrap` parameter and implemented in `_bootstrap_sample`). Why? It ensures each tree sees a slightly different perspective of the data.
2.  **Random Feature Subsets:** When finding the best split at each node within *each* tree, the algorithm doesn't consider *all* features. Instead, it only evaluates a **random subset** of the features (`max_features` parameter). (See `_get_max_features` and the modified `_find_best_split` logic injected during `RandomForest.fit`). Why? This prevents a few very strong features from dominating *all* the trees, forcing other features to be considered and leading to more diverse tree structures.

**Building the Forest (`fit`):**

The `RandomForest.fit` method essentially does this:
* Loop `n_trees` times:
    * Create a bootstrap sample of the data (if `bootstrap=True`).
    * Instantiate a `DecisionTree`.
    * Inject the "random feature subset" logic into the tree's splitting mechanism.
    * Train the `DecisionTree` on the sampled data with the modified splitting.
    * Store the trained tree in `self.trees`.

**Making Predictions (`predict`):**

To make a prediction for a new data point, the Random Forest asks *every tree* in its ensemble (`self.trees`) to make a prediction. Then, it combines them:
* **Classification:** It takes a **majority vote**. The class predicted by the most trees wins.
* **Regression:** It calculates the **average** of all the predictions from the individual trees.

This aggregation process typically leads to models that are much more robust, less prone to overfitting, and generalize better to new, unseen data compared to a single Decision Tree.

## Example Usage

Let's see how you might use a `RandomForestClassifier` (assuming classification task):

```python
from smolml.models.tree import RandomForest, DecisionTree
from smolml.core.ml_array import MLArray

# Sample Data (e.g., 4 features, 5 samples for classification)
X_data = [[5.1, 3.5, 1.4, 0.2],
          [4.9, 3.0, 1.4, 0.2],
          [6.7, 3.1, 4.4, 1.4],
          [6.0, 2.9, 4.5, 1.5],
          [5.8, 2.7, 5.1, 1.9]]
# Target classes (e.g., 0, 1, 2)
y_data = [0, 0, 1, 1, 2]

# Convert to MLArray (though the fit/predict methods handle conversion)
X = MLArray(X_data)
y = MLArray(y_data)

# --- Using a Decision Tree ---
print("--- Training Decision Tree ---")
dt = DecisionTree(max_depth=3, task="classification")
dt.fit(X, y)
print(dt) # Shows structure and stats
dt_pred = dt.predict(X)
print(f"DT Predictions on training data: {dt_pred.to_list()}")

# --- Using a Random Forest ---
print("\n--- Training Random Forest ---")
# Build a forest of 10 trees
rf = RandomForest(n_trees=10, max_depth=3, task="classification")
rf.fit(X, y)
print(rf) # Shows forest stats
rf_pred = rf.predict(X)
print(f"RF Predictions on training data: {rf_pred.to_list()}")

# Predict on new data
X_new = MLArray([[6.0, 3.0, 4.8, 1.8], [5.0, 3.4, 1.6, 0.4]])
rf_new_pred = rf.predict(X_new)
print(f"\nRF Predictions on new data {X_new.to_list()}: {rf_new_pred.to_list()}")
```

## From roots to leaves, from data to predictions

Decision Trees offer an interpretable, flowchart-like way to model data, while Random Forests leverage the power of ensemble learning (combining many diverse trees) to create highly accurate and robust models for both classification and regression. They represent a different paradigm from gradient-based optimization, but are a cornerstone of practical machine learning!
