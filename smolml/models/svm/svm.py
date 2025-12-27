from smolml.core.ml_array import MLArray
import smolml.utils.memory as memory
import random
import math

"""
/////////////////////////////
/// SUPPORT VECTOR MACHINE ///
/////////////////////////////
"""

class SVM:
    """
    Support Vector Machine for binary classification using SMO algorithm.
    Finds optimal hyperplane maximizing margin between two classes.
    """
    def __init__(self, C=1.0, kernel="linear", gamma="scale", degree=3, coef0=0.0,
                 tol=1e-3, max_iter=1000):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

        self.alphas = None
        self.b = 0.0
        self.support_vectors = None
        self.X_train = None
        self.y_train = None
        self._gamma_value = None

    def _compute_kernel(self, x1, x2):
        """Compute kernel function between two samples."""
        dot = sum(a * b for a, b in zip(x1, x2))

        if self.kernel == "linear":
            return dot
        elif self.kernel == "rbf":
            sq_dist = sum((a - b) ** 2 for a, b in zip(x1, x2))
            return math.exp(-self._gamma_value * sq_dist)
        elif self.kernel == "poly":
            return (self._gamma_value * dot + self.coef0) ** self.degree
        raise ValueError(f"Unknown kernel: {self.kernel}")

    def _compute_gamma(self, X_data, n_features):
        """Compute gamma value based on data."""
        n = len(X_data)
        if n == 0:
            return 1.0
        mean = [sum(X_data[i][j] for i in range(n)) / n for j in range(n_features)]
        var = sum((X_data[i][j] - mean[j]) ** 2 for i in range(n) for j in range(n_features)) / (n * n_features)
        return 1.0 / (n_features * var) if var > 0 else 1.0

    def _decision_function_single(self, x):
        """Compute f(x) = sum(alpha_i * y_i * K(x_i, x)) - b"""
        result = -self.b
        for i, alpha in enumerate(self.alphas):
            if alpha > 1e-8:
                result += alpha * self.y_train[i] * self._compute_kernel(self.X_train[i], x)
        return result

    def fit(self, X, y):
        """Train SVM using Sequential Minimal Optimization (SMO)."""
        if not isinstance(X, MLArray):
            X = MLArray(X)
        if not isinstance(y, MLArray):
            y = MLArray(y)

        X_data = X.to_list()
        y_data = y.to_list()
        if isinstance(y_data[0], list):
            y_data = [item[0] if isinstance(item, list) else item for item in y_data]

        # Convert labels to -1, +1
        unique_labels = sorted(set(y_data))
        if len(unique_labels) != 2:
            raise ValueError(f"SVM requires exactly 2 classes, got {len(unique_labels)}")
        self._label_map = {unique_labels[0]: -1, unique_labels[1]: 1}
        self._label_map_inverse = {-1: unique_labels[0], 1: unique_labels[1]}
        y_data = [self._label_map[label] for label in y_data]

        n_samples = len(X_data)
        n_features = len(X_data[0]) if X_data else 0
        self._gamma_value = self._compute_gamma(X_data, n_features)

        self.X_train = X_data
        self.y_train = y_data
        self.alphas = [0.0] * n_samples
        self.b = 0.0

        errors = [self._decision_function_single(X_data[i]) - y_data[i] for i in range(n_samples)]

        # SMO main loop
        num_changed = 0
        examine_all = True
        iteration = 0

        while (num_changed > 0 or examine_all) and iteration < self.max_iter:
            num_changed = 0
            indices = range(n_samples) if examine_all else [i for i in range(n_samples) if 0 < self.alphas[i] < self.C]

            for i2 in indices:
                num_changed += self._examine_example(i2, X_data, y_data, errors)

            examine_all = not examine_all if num_changed == 0 or examine_all else False
            iteration += 1

        self.support_vectors = [i for i in range(n_samples) if self.alphas[i] > 1e-8]
        return self

    def _examine_example(self, i2, X_data, y_data, errors):
        """Examine example and possibly update its alpha."""
        y2, E2 = y_data[i2], errors[i2]
        r2 = E2 * y2

        if not ((r2 < -self.tol and self.alphas[i2] < self.C) or (r2 > self.tol and self.alphas[i2] > 0)):
            return 0

        # Try to find best partner using heuristics
        # 1. Maximize |E1 - E2|
        i1 = self._select_partner(i2, E2, errors)
        if i1 >= 0 and self._take_step(i1, i2, X_data, y_data, errors):
            return 1

        # 2. Try non-bound examples
        non_bound = [i for i in range(len(X_data)) if 0 < self.alphas[i] < self.C and i != i2]
        random.shuffle(non_bound)
        for i1 in non_bound:
            if self._take_step(i1, i2, X_data, y_data, errors):
                return 1

        # 3. Try all examples
        all_indices = [i for i in range(len(X_data)) if i != i2]
        random.shuffle(all_indices)
        for i1 in all_indices:
            if self._take_step(i1, i2, X_data, y_data, errors):
                return 1

        return 0

    def _select_partner(self, i2, E2, errors):
        """Select partner that maximizes |E1 - E2|."""
        best_i, max_delta = -1, 0
        for i, alpha in enumerate(self.alphas):
            if 0 < alpha < self.C:
                delta = abs(errors[i] - E2)
                if delta > max_delta:
                    max_delta, best_i = delta, i
        return best_i

    def _take_step(self, i1, i2, X_data, y_data, errors):
        """Optimize alpha pair (i1, i2)."""
        if i1 == i2:
            return False

        a1_old, a2_old = self.alphas[i1], self.alphas[i2]
        y1, y2 = y_data[i1], y_data[i2]
        E1, E2 = errors[i1], errors[i2]
        s = y1 * y2

        # Compute bounds
        if y1 != y2:
            L, H = max(0, a2_old - a1_old), min(self.C, self.C + a2_old - a1_old)
        else:
            L, H = max(0, a2_old + a1_old - self.C), min(self.C, a2_old + a1_old)
        if L >= H:
            return False

        k11 = self._compute_kernel(X_data[i1], X_data[i1])
        k12 = self._compute_kernel(X_data[i1], X_data[i2])
        k22 = self._compute_kernel(X_data[i2], X_data[i2])
        eta = k11 + k22 - 2 * k12

        if eta > 0:
            a2_new = max(L, min(H, a2_old + y2 * (E1 - E2) / eta))
        else:
            # Evaluate objective at bounds
            f1 = y1 * (E1 + self.b) - a1_old * k11 - s * a2_old * k12
            f2 = y2 * (E2 + self.b) - s * a1_old * k12 - a2_old * k22
            L1 = a1_old + s * (a2_old - L)
            H1 = a1_old + s * (a2_old - H)
            Lobj = L1 * f1 + L * f2 + 0.5 * L1**2 * k11 + 0.5 * L**2 * k22 + s * L * L1 * k12
            Hobj = H1 * f1 + H * f2 + 0.5 * H1**2 * k11 + 0.5 * H**2 * k22 + s * H * H1 * k12
            a2_new = L if Lobj < Hobj - 1e-8 else (H if Hobj < Lobj - 1e-8 else a2_old)

        if abs(a2_new - a2_old) < 1e-8 * (a2_new + a2_old + 1e-8):
            return False

        a1_new = a1_old + s * (a2_old - a2_new)

        # Update bias
        b1 = E1 + y1 * (a1_new - a1_old) * k11 + y2 * (a2_new - a2_old) * k12 + self.b
        b2 = E2 + y1 * (a1_new - a1_old) * k12 + y2 * (a2_new - a2_old) * k22 + self.b
        self.b = b1 if 0 < a1_new < self.C else (b2 if 0 < a2_new < self.C else (b1 + b2) / 2)

        self.alphas[i1], self.alphas[i2] = a1_new, a2_new

        # Update errors
        for i in range(len(X_data)):
            errors[i] = self._decision_function_single(X_data[i]) - y_data[i]

        return True

    def decision_function(self, X):
        """Compute signed distance to hyperplane."""
        if not isinstance(X, MLArray):
            X = MLArray(X)
        X_data = X.to_list()
        if not isinstance(X_data[0], list):
            X_data = [X_data]
        return MLArray([self._decision_function_single(x) for x in X_data])

    def predict(self, X):
        """Predict class labels."""
        if not isinstance(X, MLArray):
            X = MLArray(X)
        X_data = X.to_list()
        if not isinstance(X_data[0], list):
            X_data = [X_data]
        return MLArray([self._label_map_inverse[1 if self._decision_function_single(x) >= 0 else -1] for x in X_data])

    def score(self, X, y):
        """Compute classification accuracy."""
        if not isinstance(y, MLArray):
            y = MLArray(y)
        preds = self.predict(X).to_list()
        y_true = y.to_list()
        if isinstance(y_true[0], list):
            y_true = [item[0] if isinstance(item, list) else item for item in y_true]
        return sum(1 for p, t in zip(preds, y_true) if p == t) / len(y_true)

    def __repr__(self):
        try:
            import os
            terminal_width = os.get_terminal_size().columns
        except Exception:
            terminal_width = 80

        header = "Support Vector Machine (SVM)"
        separator = "=" * terminal_width

        params = [
            f"Kernel: {self.kernel}",
            f"C (Regularization): {self.C}",
            f"Gamma: {self._gamma_value if self._gamma_value else self.gamma}",
        ]
        if self.kernel == "poly":
            params.extend([f"Degree: {self.degree}", f"Coef0: {self.coef0}"])
        params.extend([f"Tolerance: {self.tol}", f"Max Iterations: {self.max_iter}"])

        if self.support_vectors is not None:
            n_samples = len(self.X_train) if self.X_train else 0
            n_sv = len(self.support_vectors)
            sv_pct = f" ({100*n_sv/n_samples:.1f}% of training data)" if n_samples > 0 else ""
            training_info = [
                "Training Results:",
                f"  Training Samples: {n_samples}",
                f"  Support Vectors: {n_sv}{sv_pct}",
                f"  Bias (b): {self.b:.6f}"
            ]
        else:
            training_info = ["Model not yet trained"]

        import sys
        memory_info = ["Memory Usage:"]
        total_memory = sys.getsizeof(self)
        memory_info.append(f"  Base Object: {memory.format_size(total_memory)}")
        if self.alphas:
            alphas_size = sys.getsizeof(self.alphas) + len(self.alphas) * sys.getsizeof(0.0)
            total_memory += alphas_size
            memory_info.append(f"  Alphas: {memory.format_size(alphas_size)}")
        if self.X_train:
            n_samples = len(self.X_train)
            n_features = len(self.X_train[0]) if self.X_train else 0
            train_size = n_samples * n_features * sys.getsizeof(0.0)
            total_memory += train_size
            memory_info.append(f"  Training Data: {memory.format_size(train_size)}")
        memory_info.append(f"Total Memory: {memory.format_size(total_memory)}")

        return (
            f"\n{header}\n{separator}\n\n"
            + "Parameters:\n" + "\n".join(f"  {p}" for p in params)
            + "\n\n" + "\n".join(training_info)
            + "\n\n" + "\n".join(memory_info)
            + f"\n{separator}\n"
        )


class SVMMulticlass:
    """
    Multiclass SVM using One-vs-Rest (OvR) strategy.
    Trains K binary classifiers for K classes.
    """
    def __init__(self, C=1.0, kernel="linear", gamma="scale", degree=3, coef0=0.0,
                 tol=1e-3, max_iter=1000):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.classifiers = {}
        self.classes = None

    def fit(self, X, y):
        """Train one SVM per class using One-vs-Rest."""
        if not isinstance(X, MLArray):
            X = MLArray(X)
        if not isinstance(y, MLArray):
            y = MLArray(y)

        y_data = y.to_list()
        if isinstance(y_data[0], list):
            y_data = [item[0] if isinstance(item, list) else item for item in y_data]

        self.classes = sorted(set(y_data))

        for cls in self.classes:
            y_binary = [1 if label == cls else 0 for label in y_data]
            svm = SVM(C=self.C, kernel=self.kernel, gamma=self.gamma,
                     degree=self.degree, coef0=self.coef0, tol=self.tol, max_iter=self.max_iter)
            svm.fit(X, MLArray(y_binary))
            self.classifiers[cls] = svm

        return self

    def predict(self, X):
        """Predict class with highest decision function value."""
        if not isinstance(X, MLArray):
            X = MLArray(X)
        X_data = X.to_list()
        if not isinstance(X_data[0], list):
            X_data = [X_data]

        predictions = []
        for x in X_data:
            scores = {cls: svm._decision_function_single(x) for cls, svm in self.classifiers.items()}
            predictions.append(max(scores, key=scores.get))
        return MLArray(predictions)

    def score(self, X, y):
        """Compute classification accuracy."""
        if not isinstance(y, MLArray):
            y = MLArray(y)
        preds = self.predict(X).to_list()
        y_true = y.to_list()
        if isinstance(y_true[0], list):
            y_true = [item[0] if isinstance(item, list) else item for item in y_true]
        return sum(1 for p, t in zip(preds, y_true) if p == t) / len(y_true)

    def __repr__(self):
        try:
            import os
            terminal_width = os.get_terminal_size().columns
        except Exception:
            terminal_width = 80

        header = "Multiclass SVM (One-vs-Rest)"
        separator = "=" * terminal_width

        params = [
            f"Kernel: {self.kernel}",
            f"C (Regularization): {self.C}",
            f"Gamma: {self.gamma}",
        ]
        if self.kernel == "poly":
            params.extend([f"Degree: {self.degree}", f"Coef0: {self.coef0}"])

        if self.classes:
            training_info = [
                "Training Results:",
                f"  Number of Classes: {len(self.classes)}",
                f"  Classes: {self.classes}",
                f"  Binary Classifiers: {len(self.classifiers)}"
            ]
            for cls, svm in self.classifiers.items():
                n_sv = len(svm.support_vectors) if svm.support_vectors else 0
                training_info.append(f"    Class {cls}: {n_sv} support vectors")
        else:
            training_info = ["Model not yet trained"]

        return (
            f"\n{header}\n{separator}\n\n"
            + "Parameters:\n" + "\n".join(f"  {p}" for p in params)
            + "\n\n" + "\n".join(training_info)
            + f"\n{separator}\n"
        )


class SVR:
    """
    Support Vector Regression using SMO algorithm.
    Uses ε-insensitive loss: errors within ±ε are ignored.
    """
    def __init__(self, C=1.0, epsilon=0.1, kernel="linear", gamma="scale",
                 degree=3, coef0=0.0, tol=1e-3, max_iter=1000):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

        self.alphas = None
        self.alphas_star = None
        self.b = 0.0
        self.support_vectors = None
        self.X_train = None
        self.y_train = None
        self._gamma_value = None

    def _compute_kernel(self, x1, x2):
        """Compute kernel function between two samples."""
        dot = sum(a * b for a, b in zip(x1, x2))

        if self.kernel == "linear":
            return dot
        elif self.kernel == "rbf":
            sq_dist = sum((a - b) ** 2 for a, b in zip(x1, x2))
            return math.exp(-self._gamma_value * sq_dist)
        elif self.kernel == "poly":
            return (self._gamma_value * dot + self.coef0) ** self.degree
        raise ValueError(f"Unknown kernel: {self.kernel}")

    def _compute_gamma(self, X_data, n_features):
        """Compute gamma value based on data."""
        if self.gamma == "scale":
            n = len(X_data)
            if n == 0:
                return 1.0
            mean = [sum(X_data[i][j] for i in range(n)) / n for j in range(n_features)]
            var = sum((X_data[i][j] - mean[j]) ** 2 for i in range(n) for j in range(n_features)) / (n * n_features)
            return 1.0 / (n_features * var) if var > 0 else 1.0
        elif self.gamma == "auto":
            return 1.0 / n_features
        return self.gamma

    def _predict_single(self, x):
        """Compute f(x) = sum((alpha_i - alpha_i*) * K(x_i, x)) + b"""
        result = self.b
        for i in range(len(self.X_train)):
            coef = self.alphas[i] - self.alphas_star[i]
            if abs(coef) > 1e-8:
                result += coef * self._compute_kernel(self.X_train[i], x)
        return result

    def fit(self, X, y):
        """Train SVR using SMO algorithm."""
        if not isinstance(X, MLArray):
            X = MLArray(X)
        if not isinstance(y, MLArray):
            y = MLArray(y)

        X_data = X.to_list()
        y_data = y.to_list()
        if isinstance(y_data[0], list):
            y_data = [item[0] if isinstance(item, list) else item for item in y_data]

        n_samples = len(X_data)
        n_features = len(X_data[0]) if X_data else 0
        self._gamma_value = self._compute_gamma(X_data, n_features)

        self.X_train = X_data
        self.y_train = y_data
        self.alphas = [0.0] * n_samples
        self.alphas_star = [0.0] * n_samples
        self.b = 0.0

        errors = [self._predict_single(X_data[i]) - y_data[i] for i in range(n_samples)]

        # SMO main loop
        num_changed = 0
        examine_all = True
        iteration = 0

        while (num_changed > 0 or examine_all) and iteration < self.max_iter:
            num_changed = 0

            if examine_all:
                indices = range(n_samples)
            else:
                indices = [i for i in range(n_samples)
                          if (0 < self.alphas[i] < self.C) or (0 < self.alphas_star[i] < self.C)]

            for i2 in indices:
                num_changed += self._examine_example(i2, X_data, y_data, errors)

            examine_all = not examine_all if num_changed == 0 or examine_all else False
            iteration += 1

        self.support_vectors = [i for i in range(n_samples)
                               if abs(self.alphas[i] - self.alphas_star[i]) > 1e-8]
        return self

    def _examine_example(self, i2, X_data, y_data, errors):
        """Examine example and possibly update its multipliers."""
        E2 = errors[i2]
        a2, a2_star = self.alphas[i2], self.alphas_star[i2]

        # Check KKT violations
        kkt_violated = ((E2 < -self.epsilon - self.tol and a2 < self.C) or
                       (E2 > -self.epsilon + self.tol and a2 > 0) or
                       (E2 > self.epsilon + self.tol and a2_star < self.C) or
                       (E2 < self.epsilon - self.tol and a2_star > 0))

        if not kkt_violated:
            return 0

        # Try to find partner
        i1 = self._select_partner(i2, E2, errors)
        if i1 >= 0 and self._take_step(i1, i2, X_data, y_data, errors):
            return 1

        # Try non-bound examples
        non_bound = [i for i in range(len(X_data)) if i != i2 and
                    ((0 < self.alphas[i] < self.C) or (0 < self.alphas_star[i] < self.C))]
        random.shuffle(non_bound)
        for i1 in non_bound:
            if self._take_step(i1, i2, X_data, y_data, errors):
                return 1

        # Try all examples
        all_indices = [i for i in range(len(X_data)) if i != i2]
        random.shuffle(all_indices)
        for i1 in all_indices:
            if self._take_step(i1, i2, X_data, y_data, errors):
                return 1

        return 0

    def _select_partner(self, i2, E2, errors):
        """Select partner that maximizes |E1 - E2|."""
        best_i, max_delta = -1, 0
        for i in range(len(errors)):
            if (0 < self.alphas[i] < self.C) or (0 < self.alphas_star[i] < self.C):
                delta = abs(errors[i] - E2)
                if delta > max_delta:
                    max_delta, best_i = delta, i
        return best_i

    def _take_step(self, i1, i2, X_data, y_data, errors):
        """Optimize multiplier pair (i1, i2)."""
        if i1 == i2:
            return False

        E1, E2 = errors[i1], errors[i2]

        k11 = self._compute_kernel(X_data[i1], X_data[i1])
        k12 = self._compute_kernel(X_data[i1], X_data[i2])
        k22 = self._compute_kernel(X_data[i2], X_data[i2])
        eta = k11 + k22 - 2 * k12

        if eta <= 0:
            return False

        # Use β = α - α* formulation
        beta1 = self.alphas[i1] - self.alphas_star[i1]
        beta2 = self.alphas[i2] - self.alphas_star[i2]

        beta2_new = max(-self.C, min(self.C, beta2 + (E1 - E2) / eta))

        if abs(beta2_new - beta2) < 1e-8:
            return False

        beta1_new = max(-self.C, min(self.C, beta1 + (beta2 - beta2_new)))

        # Convert β back to α and α*
        if beta1_new >= 0:
            self.alphas[i1], self.alphas_star[i1] = beta1_new, 0.0
        else:
            self.alphas[i1], self.alphas_star[i1] = 0.0, -beta1_new

        if beta2_new >= 0:
            self.alphas[i2], self.alphas_star[i2] = beta2_new, 0.0
        else:
            self.alphas[i2], self.alphas_star[i2] = 0.0, -beta2_new

        self._update_bias(X_data, y_data)

        # Update errors
        for i in range(len(X_data)):
            errors[i] = self._predict_single(X_data[i]) - y_data[i]

        return True

    def _update_bias(self, X_data, y_data):
        """Update bias using support vectors on the margin."""
        b_sum, count = 0.0, 0

        for i in range(len(X_data)):
            f_no_b = sum((self.alphas[j] - self.alphas_star[j]) * self._compute_kernel(X_data[j], X_data[i])
                        for j in range(len(X_data)))

            if 0 < self.alphas[i] < self.C:
                b_sum += y_data[i] - f_no_b - self.epsilon
                count += 1
            elif 0 < self.alphas_star[i] < self.C:
                b_sum += y_data[i] - f_no_b + self.epsilon
                count += 1

        if count > 0:
            self.b = b_sum / count

    def predict(self, X):
        """Predict target values."""
        if not isinstance(X, MLArray):
            X = MLArray(X)
        X_data = X.to_list()
        if not isinstance(X_data[0], list):
            X_data = [X_data]
        return MLArray([self._predict_single(x) for x in X_data])

    def score(self, X, y):
        """Compute R² score."""
        if not isinstance(y, MLArray):
            y = MLArray(y)
        preds = self.predict(X).to_list()
        y_true = y.to_list()
        if isinstance(y_true[0], list):
            y_true = [item[0] if isinstance(item, list) else item for item in y_true]

        y_mean = sum(y_true) / len(y_true)
        ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, preds))
        ss_tot = sum((yt - y_mean) ** 2 for yt in y_true)

        return 1 - (ss_res / ss_tot) if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)

    def mse(self, X, y):
        """Compute Mean Squared Error."""
        if not isinstance(y, MLArray):
            y = MLArray(y)
        preds = self.predict(X).to_list()
        y_true = y.to_list()
        if isinstance(y_true[0], list):
            y_true = [item[0] if isinstance(item, list) else item for item in y_true]
        return sum((yt - yp) ** 2 for yt, yp in zip(y_true, preds)) / len(y_true)

    def __repr__(self):
        try:
            import os
            terminal_width = os.get_terminal_size().columns
        except Exception:
            terminal_width = 80

        header = "Support Vector Regression (SVR)"
        separator = "=" * terminal_width

        params = [
            f"Kernel: {self.kernel}",
            f"C (Regularization): {self.C}",
            f"Epsilon (ε-tube width): {self.epsilon}",
            f"Gamma: {self._gamma_value if self._gamma_value else self.gamma}",
        ]
        if self.kernel == "poly":
            params.extend([f"Degree: {self.degree}", f"Coef0: {self.coef0}"])
        params.extend([f"Tolerance: {self.tol}", f"Max Iterations: {self.max_iter}"])

        if self.support_vectors is not None:
            n_samples = len(self.X_train) if self.X_train else 0
            n_sv = len(self.support_vectors)
            sv_pct = f" ({100*n_sv/n_samples:.1f}% of training data)" if n_samples > 0 else ""
            training_info = [
                "Training Results:",
                f"  Training Samples: {n_samples}",
                f"  Support Vectors: {n_sv}{sv_pct}",
                f"  Bias (b): {self.b:.6f}"
            ]
        else:
            training_info = ["Model not yet trained"]

        import sys
        memory_info = ["Memory Usage:"]
        total_memory = sys.getsizeof(self)
        memory_info.append(f"  Base Object: {memory.format_size(total_memory)}")
        if self.alphas:
            alphas_size = (sys.getsizeof(self.alphas) + sys.getsizeof(self.alphas_star) +
                          2 * len(self.alphas) * sys.getsizeof(0.0))
            total_memory += alphas_size
            memory_info.append(f"  Alphas (α, α*): {memory.format_size(alphas_size)}")
        if self.X_train:
            n_samples = len(self.X_train)
            n_features = len(self.X_train[0]) if self.X_train else 0
            train_size = n_samples * n_features * sys.getsizeof(0.0)
            total_memory += train_size
            memory_info.append(f"  Training Data: {memory.format_size(train_size)}")
        memory_info.append(f"Total Memory: {memory.format_size(total_memory)}")

        return (
            f"\n{header}\n{separator}\n\n"
            + "Parameters:\n" + "\n".join(f"  {p}" for p in params)
            + "\n\n" + "\n".join(training_info)
            + "\n\n" + "\n".join(memory_info)
            + f"\n{separator}\n"
        )
