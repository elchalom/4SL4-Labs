
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from typing import Tuple, Dict, List
 
SEED = 4322   
NOISE_STD = 0.2  
K_MIN, K_MAX = 1, 60
N_TRAIN = 201
N_TEST = 101
N_FOLDS = 5
 
class Model:
    def __init__(self, seed: int = SEED, noise_std: float = NOISE_STD):
        """
        Builds the model object and generates the datasets.
        The datasets are accessible as:
          self.x_training_set, self.t_training_set  (N_TRAIN)
          self.x_test_set, self.t_test_set          (N_TEST)
        """
        self.seed = seed
        self.noise_std = noise_std
        np.random.seed(self.seed)

        # Generate training and test datasets per assignment
        self.x_training_set = self.generate_set(N_TRAIN)
        self.t_training_set = self.generate_targets(self.x_training_set)

        self.x_test_set = self.generate_set(N_TEST)
        self.t_test_set = self.generate_targets(self.x_test_set)

        # Storage for CV and results
        self.k_values = list(range(K_MIN, K_MAX + 1))
        self.cv_mean_rmse: Dict[int, float] = {}
        self.train_mean_rmse_cv: Dict[int, float] = {}
        self.train_rmse_full: Dict[int, float] = {}
        self.best_k: int = -1
        self.best_cv_rmse: float = np.inf
        self.test_rmse_best: float = np.inf
        self.t_test_pred_best: np.ndarray = None

    # -----------------------
    # Data generation helpers
    # -----------------------
    @staticmethod
    def generate_set(N: int, lower_bound: float = 0.0, upper_bound: float = 1.0) -> np.ndarray:
        """Return N evenly spaced x values in [lower_bound, upper_bound]."""
        return np.linspace(lower_bound, upper_bound, N)

    @staticmethod
    def f_opt(x: np.ndarray) -> np.ndarray:
        """True function f_opt(x) = sin(2*pi*x)."""
        return np.sin(2 * np.pi * x)

    def generate_targets(self, x: np.ndarray) -> np.ndarray:
        """Generate noisy targets t = sin(2πx) + Gaussian noise (std = noise_std)."""
        return self.f_opt(x) + self.noise_std * np.random.randn(len(x))

    # -----------------------
    # k-NN prediction
    # -----------------------
    @staticmethod
    def knn_predict(x_train: np.ndarray, t_train: np.ndarray, x_query: np.ndarray, k: int) -> np.ndarray:
        """
        Vectorized k-NN for 1D x.
        Returns predictions for x_query using x_train/t_train and averaging k nearest neighbors.
        """
        # pairwise absolute distances: shape (N_query, N_train)
        dists = np.abs(x_query[:, None] - x_train[None, :])
        # argsort along train axis -> indices of neighbors
        idx_sorted = np.argsort(dists, axis=1)
        idx_k = idx_sorted[:, :k]  # shape (N_query, k)
        # gather neighbor targets and average
        neighbors = t_train[idx_k]  # advanced indexing
        preds = np.mean(neighbors, axis=1)
        return preds

    # -----------------------
    # Error metric
    # -----------------------
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # -----------------------
    # Cross-validation
    # -----------------------
    def cross_validate(self, k_values: List[int], n_folds: int = N_FOLDS, shuffle: bool = True) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Perform n_folds cross-validation over the training set for each k in k_values.
        Returns:
          (cv_mean_rmse, train_mean_rmse_cv) as dictionaries keyed by k.
        """
        x = self.x_training_set
        t = self.t_training_set
        kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=self.seed)

        cv_rmse_store: Dict[int, List[float]] = {k: [] for k in k_values}
        train_rmse_store: Dict[int, List[float]] = {k: [] for k in k_values}

        for train_idx, val_idx in kf.split(x):
            x_tr, t_tr = x[train_idx], t[train_idx]
            x_val, t_val = x[val_idx], t[val_idx]

            # For each k, predict on val and compute RMSE; also compute training RMSE on x_tr
            # (Vectorizing across k is possible but simpler and still fast for k<=60)
            for k in k_values:
                t_val_pred = self.knn_predict(x_tr, t_tr, x_val, k)
                cv_rmse_store[k].append(self.rmse(t_val, t_val_pred))

                # training error on the fold's training subset
                t_tr_pred = self.knn_predict(x_tr, t_tr, x_tr, k)
                train_rmse_store[k].append(self.rmse(t_tr, t_tr_pred))

        # compute means across folds
        cv_mean = {k: float(np.mean(cv_rmse_store[k])) for k in k_values}
        train_mean = {k: float(np.mean(train_rmse_store[k])) for k in k_values}

        # store in object
        self.cv_mean_rmse = cv_mean
        self.train_mean_rmse_cv = train_mean
        return cv_mean, train_mean

    # -----------------------
    # Evaluate train-on-full for all k
    # -----------------------
    def compute_train_rmse_full(self, k_values: List[int]) -> Dict[int, float]:
        """
        Compute RMSE on the full training set when model is trained on the full training set (train-on-all).
        """
        x_tr = self.x_training_set
        t_tr = self.t_training_set
        out = {}
        for k in k_values:
            t_pred = self.knn_predict(x_tr, t_tr, x_tr, k)
            out[k] = self.rmse(t_tr, t_pred)
        self.train_rmse_full = out
        return out

    # -----------------------
    # Full experiment pipeline
    # -----------------------
    def run_knn_cv_experiment(self, k_min: int = K_MIN, k_max: int = K_MAX, n_folds: int = N_FOLDS, verbose: bool = True):
        """
        High-level method that runs CV for k in [k_min, k_max], selects best k by CV,
        evaluates on test set, and stores results in object fields.
        """
        k_values = list(range(k_min, k_max + 1))
        self.k_values = k_values

        # 1) CV
        cv_mean, train_mean_cv = self.cross_validate(k_values, n_folds=n_folds, shuffle=True)
        # 2) train-on-all error
        train_full = self.compute_train_rmse_full(k_values)

        # 3) select best k (lowest CV mean RMSE)
        cv_array = np.array([cv_mean[k] for k in k_values])
        best_idx = int(np.argmin(cv_array))
        self.best_k = k_values[best_idx]
        self.best_cv_rmse = float(cv_array[best_idx])

        # 4) Evaluate best k on test set (predict using full training set)
        t_test_pred = self.knn_predict(self.x_training_set, self.t_training_set, self.x_test_set, self.best_k)
        self.t_test_pred_best = t_test_pred
        self.test_rmse_best = self.rmse(self.t_test_set, t_test_pred)

        if verbose:
            print(f"Best k by 5-fold CV: {self.best_k} (CV RMSE = {self.best_cv_rmse:.5f})")
            print(f"Test RMSE of k={self.best_k}: {self.test_rmse_best:.5f}")
            # f_opt RMSE for comparison
            fopt_rmse = self.rmse(self.t_test_set, self.f_opt(self.x_test_set))
            print(f"RMSE of f_opt (sin(2πx)) on noisy test set: {fopt_rmse:.5f}")

        return {
            "k_values": k_values,
            "cv_mean_rmse": cv_mean,
            "train_mean_rmse_cv": train_mean_cv,
            "train_rmse_full": train_full,
            "best_k": self.best_k,
            "best_cv_rmse": self.best_cv_rmse,
            "test_rmse_best": self.test_rmse_best,
            "t_test_pred_best": self.t_test_pred_best
        }

    # -----------------------
    # Plot helpers (inside class for cohesion)
    # -----------------------
    def plot_errors_vs_k(self):
        """Plot training-on-full, training mean (CV-train), and CV validation error vs k."""
        k = self.k_values
        train_full_arr = np.array([self.train_rmse_full[k_] for k_ in k])
        train_cv_arr = np.array([self.train_mean_rmse_cv[k_] for k_ in k])
        cv_arr = np.array([self.cv_mean_rmse[k_] for k_ in k])

        plt.figure(figsize=(9, 5))
        plt.plot(k, train_full_arr, marker='o', label="Training error (train-on-full)")
        plt.plot(k, train_cv_arr, marker='s', label="Training error (CV-train mean)")
        plt.plot(k, cv_arr, marker='^', label="CV validation error (5-fold mean)")
        plt.xlabel("k (number of neighbors)")
        plt.ylabel("RMSE")
        plt.title("k-NN: Training RMSE vs CV RMSE (k = {}..{})".format(k[0], k[-1]))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_predictions(self):
        """Plot predictions of best_k vs x along with the true f_opt and noisy points."""
        if self.best_k < 1:
            raise RuntimeError("Best k not set. Run run_knn_cv_experiment() first.")

        x_dense = np.linspace(0, 1, 1000)
        t_dense = self.f_opt(x_dense)

        # predictions
        t_test_pred = self.t_test_pred_best
        t_train_pred_on_train = self.knn_predict(self.x_training_set, self.t_training_set, self.x_training_set, self.best_k)

        plt.figure(figsize=(9, 5))
        plt.plot(x_dense, t_dense, label="True f(x) = sin(2πx)", linewidth=2)
        plt.scatter(self.x_test_set, t_test_pred, s=30, marker='o',color = 'red', label=f"Predicted (test), k={self.best_k}")
        plt.scatter(self.x_test_set, self.t_test_set, s=18, marker='x', alpha=0.6, label="Noisy test targets")
        plt.scatter(self.x_training_set, t_train_pred_on_train, s=18, marker='s',color = 'blue',  alpha=0.9, label="Predicted (train)")
        plt.scatter(self.x_training_set, self.t_training_set, s=8, marker='.', color='black', alpha=0.6, label="Noisy training targets")
        plt.xlabel("x")
        plt.ylabel("t / predictions")
        plt.title(f"k-NN predictions vs x (best k = {self.best_k})")
        plt.legend(ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# -----------------------
# Run when executed as script
# -----------------------
def main():
    model = Model(seed=SEED, noise_std=NOISE_STD)
    results = model.run_knn_cv_experiment(k_min=K_MIN, k_max=K_MAX, n_folds=N_FOLDS, verbose=True)

    # Print table-like summary for inclusion in the report
    print("\nk    CV_RMSE    TrainFull_RMSE    TrainCV_RMSE")
    for k in results["k_values"]:
        print(f"{k:2d}  {results['cv_mean_rmse'][k]:8.4f}    {results['train_rmse_full'][k]:8.4f}       {results['train_mean_rmse_cv'][k]:8.4f}")

    # Plot errors vs k
    model.plot_errors_vs_k()

    # Plot predictions with best k
    model.plot_predictions()

    # Concise summary
    print("\nSummary for report:")
    print(f"- Seed used: {model.seed}")
    print(f"- Training set size: {len(model.x_training_set)}")
    print(f"- Test set size: {len(model.x_test_set)}")
    print(f"- Best k (by 5-fold CV): {model.best_k}")
    print(f"- CV RMSE (best k): {model.best_cv_rmse:.5f}")
    print(f"- Test RMSE (best k): {model.test_rmse_best:.5f}")
    print(f"- RMSE of f_opt on test set: {model.rmse(model.t_test_set, model.f_opt(model.x_test_set)):.5f}")
    print("\nUse the errors-vs-k plot to identify ranges of underfitting (high train & high CV),")
    print("overfitting (low train & higher CV), and the region of optimal capacity (low CV).")

if __name__ == "__main__":
    main()

