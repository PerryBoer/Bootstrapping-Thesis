import numpy as np
from pprint import pprint
from config import Config
from dgp import DGP
from LassoEstimatorTheory import LassoEstimatorTheory

def compute_lambda_alpha_grid(signal_type, error_type, num_lambdas=3):
    # 1. Generate one DGP draw to base the grid on
    dgp = DGP(signal_type=signal_type, error_type=error_type, seed=Config.seed)
    data = dgp.generate()
    X, y, beta_true, support_true, snr = data["X"], data["y"], data["beta"], data["support"], data["snr"]

    # 2. Fit Lasso with fixed λ (theory-based)
    estimator = LassoEstimatorTheory()
    rough_lambda = 0.5 * np.sqrt(np.log(Config.p) / Config.n)
    estimator.fit(X, y, lam=rough_lambda, thresholding_level=0.0, apply_threshold=False, support_true=support_true)
    
    beta_hat = estimator.beta_hat.copy()

    # 3. Estimate sigma from residuals
    residuals = estimator.residuals
    sigma_hat = np.std(residuals)

    # 4. Theoretical lambda base
    lambda_base = sigma_hat * np.sqrt(np.log(Config.p) / Config.n)
    lambda_grid = np.round(np.array([0.5, 1.0, 2.0]) * lambda_base, 5)

    # 5. Alpha threshold (based on 95% quantile of nulls)
    null_indices = list(set(range(Config.p)) - set(support_true))
    null_magnitudes = np.abs(beta_hat[null_indices])
    threshold_95 = np.quantile(null_magnitudes, 0.95)
    alpha_null = threshold_95 / np.sqrt(np.log(Config.p) / Config.n)

    # 6. Alpha grid based on alpha_null
    alpha_grid = [round(alpha_null * factor, 3) for factor in [0.5, 1.0, 2.0]]

    return {
        "lambda_base": lambda_base,
        "lambda_grid": lambda_grid.tolist(),
        "alpha_null": round(alpha_null, 3),
        "alpha_grid": alpha_grid,
        "sigma_hat": sigma_hat,
        # "support": support_true,
        "snr": snr
    }

# Compute and return results for all 9 configurations
configurations = {}
for signal in ["strong", "weak", "nearzero"]:
    for error in ["gaussian", "heteroskedastic", "ar1"]:
        key = (signal, error)
        configurations[key] = compute_lambda_alpha_grid(signal, error)

configurations[("strong", "gaussian")]  # Example output to verify

pprint(configurations)