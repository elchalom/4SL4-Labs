import numpy as np
from dataclasses import dataclass

@dataclass
class DataSets:
    x_train: np.ndarray
    t_train: np.ndarray
    x_valid: np.ndarray
    t_valid: np.ndarray
    x_test: np.ndarray
    t_test: np.ndarray

def generate_linspace(n: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    return np.linspace(lo, hi, n)

def make_datasets(N: int, seed: int, noise_var: float = 0.04) -> DataSets:
    # Data per assignment spec (sin(2πx) + ε, ε ~ N(0, 0.04))
    rng = np.random.default_rng(seed)
    x_train = generate_linspace(N)
    x_valid = generate_linspace(100)
    x_test  = generate_linspace(100)

    noise = lambda n: rng.normal(0.0, np.sqrt(noise_var), size=n)
    t_train = np.sin(2*np.pi*x_train) + noise(N)
    t_valid = np.sin(2*np.pi*x_valid) + noise(100)
    t_test  = np.sin(2*np.pi*x_test)  + noise(100)

    return DataSets(x_train, t_train, x_valid, t_valid, x_test, t_test)
