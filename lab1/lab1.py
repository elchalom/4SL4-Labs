import numpy as np

np.random.seed(4322)

def generate_set(N : int, lower_bound : float = 0, upper_bound : float = 1.0) -> np.ndarray:
    values = np.linspace(lower_bound, upper_bound, N)
    return values

class Model:
    def __init__(self):
        self.x_training_set = generate_set(N=9)
        self.x_validation_set = generate_set(N=100)
        self.x_test_set = generate_set(N=100)

        self.t_training_set = np.sin(2*np.pi*self.x_training_set) + 0.2*np.random.randn(9)
        self.t_validation_set = np.sin(2*np.pi*self.x_validation_set) + 0.2*np.random.randn(100)
        self.t_test_set = np.sin(2*np.pi*self.x_test_set) + 0.2*np.random.randn(100)


    def design_matrix(self, x : np.ndarray, M : int) -> np.ndarray:
        # columns: 1, x, x^2, ..., x^M
        # result shape: (len(x), M+1)
        return np.vstack([x**m for m in range(M+1)]).T
    
    def fit_least_squares(self, X : np.ndarray, t : np.ndarray) -> np.ndarray:
        # (X^T X)w = (X^T)t
        # solve for w
        
        X_transpose_X = X.T @ X
        X_transpose_t = X.T @ t
        
        w = np.linalg.solve(X_transpose_X, X_transpose_t)
        
        return w
    
    def predict(self, X : np.ndarray, w : np.ndarray) -> np.ndarray:
        return X @ w

    def rmse(self, actual : np.ndarray, predicted : np.ndarray) -> float:
        return np.sqrt(np.mean((actual-predicted)**2))
