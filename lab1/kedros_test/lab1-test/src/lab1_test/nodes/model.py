import numpy as np
from sklearn.preprocessing import StandardScaler

def design_matrix(x: np.ndarray, M: int) -> np.ndarray:
    # Columns: 1, x, x^2, ..., x^M  (shape: [len(x), M+1])
    return np.vstack([x**m for m in range(M+1)]).T

def fit_least_squares(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    # Solve (X^T X)w = X^T t
    XT_X = X.T @ X
    XT_t = X.T @ t
    return np.linalg.solve(XT_X, XT_t)

def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X @ w

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# —— Ridge-specific helpers (M = N-1 case) ——
def standardize_design(train_X: np.ndarray, valid_X: np.ndarray):
    sc = StandardScaler(with_mean=True, with_std=True)
    Xtr = sc.fit_transform(train_X)
    Xva = sc.transform(valid_X)
    return Xtr, Xva, sc

def fit_ridge(X: np.ndarray, t: np.ndarray, lam: float) -> np.ndarray:
    # Solve (X^T X + λI)w = X^T t
    M1 = X.shape[1]
    XT_X = X.T @ X
    return np.linalg.solve(XT_X + lam * np.eye(M1), X.T @ t)
