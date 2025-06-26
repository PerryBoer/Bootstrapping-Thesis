import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from config import Config
from dgp import DGP
from MonteCarloRunner import MonteCarloRunner
from LassoEstimatorTheory import LassoEstimatorTheory
from joblib import Parallel, delayed
from config import Config


def compute_lambda_alpha(config: Config, signal_type="strong", error_type="heteroskedastic"):
    dgp = DGP(config=config, signal_type=signal_type, error_type=error_type)
    data = dgp.generate()
    X, y, beta_true, support_true = data["X"], data["y"], data["beta"], data["support"]

    estimator = LassoEstimatorTheory()
    rough_lambda = 0.5 * np.sqrt(np.log(config.p) / config.n)
    estimator.fit(X, y, lam=rough_lambda, thresholding_level=0.0, apply_threshold=False, support_true=support_true)

    beta_hat = estimator.beta_hat
    residuals = estimator.residuals
    sigma_hat = np.std(residuals)

    lambda_val = sigma_hat * np.sqrt(np.log(config.p) / config.n)

    null_indices = list(set(range(config.p)) - set(support_true))
    null_magnitudes = np.abs(beta_hat[null_indices])
    threshold_95 = np.quantile(null_magnitudes, 0.95)
    alpha_val = threshold_95 / np.sqrt(np.log(config.p) / config.n)

    return lambda_val, alpha_val



def run_mc_for_config(method, lam, alpha, config: Config, tracked_indices, base_dir="results/dim_sensitivity"):
    runner = MonteCarloRunner(
        method=method,
        lambda_val=lam,
        threshold_val=alpha,
        signal_type="weak",
        error_type="heteroskedastic",
        config=config,
        tracked_indices=tracked_indices,
    )
    results = runner.run()

    subdir = os.path.join(base_dir, method, f"p{config.p}")
    os.makedirs(subdir, exist_ok=True)
    suffix = f"lambda{lam:.4f}_alpha{alpha:.3f}"

    pd.DataFrame([results["summary"]]).to_csv(os.path.join(subdir, f"summary_{suffix}.csv"), index=False)

    for j, tracked in results["tracked"].items():
        df = pd.DataFrame(tracked)
        df.to_csv(os.path.join(subdir, f"tracked_j{j}_{suffix}.csv"), index=False)

    raw_df = pd.DataFrame(results["raw_runs"])
    raw_df.to_csv(os.path.join(subdir, f"raw_runs_{suffix}.csv"), index=False)

    return f"Finished: method={method} | p={config.p} | λ={lam:.4f} | α={alpha:.4f}"




class ParallelDimensionalitySweep:
    def __init__(self, methods, p_grid, base_results_dir="results/dim_sensitivity", n_jobs=4):
        self.methods = methods
        self.p_grid = p_grid
        self.base_results_dir = base_results_dir
        self.n_jobs = n_jobs
        self.tracked_indices = [1, 2]  # Track all support

    def run(self):
        joblist = []

        for p in self.p_grid:
            config = Config(p=p, n=250)
            lam, alpha = compute_lambda_alpha(config, signal_type="weak", error_type="heteroskedastic")

            for method in self.methods:
                joblist.append((method, lam, alpha, config, self.tracked_indices, self.base_results_dir))

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(run_mc_for_config)(*args) for args in joblist
        )
        return results

