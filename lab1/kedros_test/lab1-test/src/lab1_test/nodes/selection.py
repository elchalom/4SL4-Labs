import numpy as np
from .model import design_matrix, fit_least_squares, predict, rmse, standardize_design, fit_ridge

def select_and_test_best(dsets, degree_sweep_result, ridge_sweep_result, M_full: int):
    # Compare the best validation RMSE from degree sweep vs ridge sweep (M=N-1)
    best_deg_rmse = degree_sweep_result.best_valid_rmse
    best_rdg_rmse = ridge_sweep_result.best_valid_rmse

    if best_deg_rmse <= best_rdg_rmse:
        M = degree_sweep_result.best_m
        Xtr = design_matrix(dsets.x_train, M)
        Xte = design_matrix(dsets.x_test,  M)
        w   = fit_least_squares(Xtr, dsets.t_train)
        test_rmse = rmse(dsets.t_test, predict(Xte, w))
        return {"chosen": "OLS", "M": M, "lambda": None, "test_rmse": float(test_rmse)}
    else:
        M = M_full
        Xtr_raw = design_matrix(dsets.x_train, M)
        Xte_raw = design_matrix(dsets.x_test,  M)
        Xtr, Xte, _ = standardize_design(Xtr_raw, Xte_raw)
        w = fit_ridge(Xtr, dsets.t_train, ridge_sweep_result.best_lambda)
        test_rmse = rmse(dsets.t_test, predict(Xte, w))
        return {"chosen": "Ridge", "M": M, "lambda": float(ridge_sweep_result.best_lambda), "test_rmse": float(test_rmse)}
