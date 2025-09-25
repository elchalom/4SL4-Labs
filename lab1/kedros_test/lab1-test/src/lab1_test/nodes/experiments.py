import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .model import design_matrix, fit_least_squares, fit_ridge, rmse, predict, standardize_design

@dataclass
class DegreeSweepResult:
    ms: List[int]
    train_rmse: List[float]
    valid_rmse: List[float]
    best_m: int
    best_valid_rmse: float

def degree_sweep(dsets, m_min: int, m_max: int) -> DegreeSweepResult:
    ms, tr, va = [], [], []
    for M in range(m_min, m_max + 1):
        Xtr = design_matrix(dsets.x_train, M)
        Xva = design_matrix(dsets.x_valid, M)

        w = fit_least_squares(Xtr, dsets.t_train)
        tr.append(rmse(dsets.t_train, predict(Xtr, w)))
        va.append(rmse(dsets.t_valid,  predict(Xva, w)))
        ms.append(M)

    best_idx = int(np.argmin(va))
    return DegreeSweepResult(
        ms=ms,
        train_rmse=tr,
        valid_rmse=va,
        best_m=ms[best_idx],
        best_valid_rmse=va[best_idx],
    )

def plot_degree_curves(dsets, result: DegreeSweepResult, out_png: str):
    plt.figure()
    plt.plot(result.ms, result.train_rmse, marker="o", label="Train RMSE")
    plt.plot(result.ms, result.valid_rmse, marker="o", label="Valid RMSE")
    plt.xlabel("Polynomial degree M")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_fit_for_m(dsets, M: int, out_png: str):
    Xtr = design_matrix(dsets.x_train, M)
    Xva = design_matrix(dsets.x_valid, M)
    w   = fit_least_squares(Xtr, dsets.t_train)

    xs = np.linspace(0, 1, 400)
    Xs = design_matrix(xs, M)
    y_hat = predict(Xs, w)
    y_opt = np.sin(2*np.pi*xs)

    plt.figure()
    plt.scatter(dsets.x_train, dsets.t_train, s=20, label="Train")
    plt.scatter(dsets.x_valid, dsets.t_valid, s=20, label="Valid")
    plt.plot(xs, y_hat, label=f"f_M (M={M})")
    plt.plot(xs, y_opt, label="f_opt = sin(2πx)")
    plt.xlabel("x"); plt.ylabel("t")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

@dataclass
class RidgeSweepResult:
    lambdas: List[float]
    train_rmse: List[float]
    valid_rmse: List[float]
    best_lambda: float
    best_valid_rmse: float
    best_w: np.ndarray

def ridge_sweep(dsets, M: int, lambdas: List[float]) -> RidgeSweepResult:
    Xtr_raw = design_matrix(dsets.x_train, M)
    Xva_raw = design_matrix(dsets.x_valid, M)
    Xtr, Xva, _ = standardize_design(Xtr_raw, Xva_raw)

    tr, va, ws = [], [], []
    for lam in lambdas:
        w = fit_ridge(Xtr, dsets.t_train, lam)
        tr.append(rmse(dsets.t_train, predict(Xtr, w)))
        va.append(rmse(dsets.t_valid,  predict(Xva, w)))
        ws.append(w)

    best_idx = int(np.argmin(va))
    return RidgeSweepResult(
        lambdas=lambdas,
        train_rmse=tr,
        valid_rmse=va,
        best_lambda=lambdas[best_idx],
        best_valid_rmse=va[best_idx],
        best_w=ws[best_idx],
    )

def plot_ridge_curves(result: RidgeSweepResult, out_png: str):
    import matplotlib.pyplot as plt
    xs = [np.log10(l) for l in result.lambdas]
    plt.figure()
    plt.plot(xs, result.train_rmse, marker="o", label="Train RMSE")
    plt.plot(xs, result.valid_rmse, marker="o", label="Valid RMSE")
    plt.xlabel("log10(λ)"); plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_ridge_fits(dsets, M: int, lambdas_show: List[float], out_pngs: List[str]):
    Xtr_raw = design_matrix(dsets.x_train, M)
    Xva_raw = design_matrix(dsets.x_valid, M)
    Xtr, Xva, _ = standardize_design(Xtr_raw, Xva_raw)

    xs = np.linspace(0, 1, 400)
    Xs_raw = design_matrix(xs, M)
    # Use training scaler for evaluation to be consistent with spec
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler(with_mean=True, with_std=True).fit(Xtr_raw)
    Xs = sc.transform(Xs_raw)

    for lam, path in zip(lambdas_show, out_pngs):
        w = fit_ridge(Xtr, dsets.t_train, lam)
        y_hat = (Xs @ w)
        y_opt = np.sin(2*np.pi*xs)

        plt.figure()
        plt.scatter(dsets.x_train, dsets.t_train, s=20, label="Train")
        plt.scatter(dsets.x_valid, dsets.t_valid, s=20, label="Valid")
        plt.plot(xs, y_hat, label=f"Ridge λ={lam:g}")
        plt.plot(xs, y_opt, label="f_opt = sin(2πx)")
        plt.xlabel("x"); plt.ylabel("t")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
