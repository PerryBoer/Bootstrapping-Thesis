import os
import numpy as np
from joblib import Parallel, delayed
from config import Config
from dgp import DGP
from MonteCarloRunner import MonteCarloRunner
from LassoEstimatorTheory import LassoEstimatorTheory
import pandas as pd


def compute_lambda_alpha_grid(signal_type, error_type):
    dgp = DGP(signal_type=signal_type, error_type=error_type, seed=Config.seed)
    data = dgp.generate()
    X, y, beta_true, support_true, snr = data["X"], data["y"], data["beta"], data["support"], data["snr"]

    estimator = LassoEstimatorTheory()
    rough_lambda = 0.5 * np.sqrt(np.log(Config.p) / Config.n)
    estimator.fit(X, y, lam=rough_lambda, thresholding_level=0.0, apply_threshold=False, support_true=support_true)
    beta_hat = estimator.beta_hat.copy()
    residuals = estimator.residuals
    sigma_hat = np.std(residuals)

    lambda_base = sigma_hat * np.sqrt(np.log(Config.p) / Config.n)
    lambda_grid = np.round(np.array([0.5, 1.0, 2.0]) * lambda_base, 5)

    null_indices = list(set(range(Config.p)) - set(support_true))
    null_magnitudes = np.abs(beta_hat[null_indices])
    threshold_95 = np.quantile(null_magnitudes, 0.95)
    alpha_null = threshold_95 / np.sqrt(np.log(Config.p) / Config.n)
    alpha_grid = [round(alpha_null * factor, 3) for factor in [0.5, 1.0, 2.0]]

    return {
        "lambda_base": lambda_base,
        "lambda_grid": lambda_grid.tolist(),
        "alpha_null": round(alpha_null, 3),
        "alpha_grid": alpha_grid,
        "sigma_hat": sigma_hat,
        "snr": snr
    }


def run_mc_for_config(method, lam, alpha, signal, error, tracked_indices, base_dir="results"):
    runner = MonteCarloRunner(
        method=method,
        lambda_val=lam,
        threshold_val=alpha,
        signal_type=signal,
        error_type=error,
        tracked_indices=tracked_indices,
        R=Config.num_mc
    )
    results = runner.run()

    subdir = os.path.join(base_dir, error, signal, method)
    os.makedirs(subdir, exist_ok=True)
    suffix = f"lambda{lam:.4f}_alpha{alpha:.3f}"

    pd.DataFrame([results["summary"]]).to_csv(os.path.join(subdir, f"summary_{suffix}.csv"), index=False)

    for j, tracked in results["tracked"].items():
        df = pd.DataFrame(tracked)
        df.to_csv(os.path.join(subdir, f"tracked_j{j}_{suffix}.csv"), index=False)

    raw_df = pd.DataFrame(results["raw_runs"])
    raw_df.to_csv(os.path.join(subdir, f"raw_runs_{suffix}.csv"), index=False)

    return f"Finished: {method} | λ={lam} | α={alpha} | signal={signal} | error={error}"


class ParallelSimulationGrid:
    def __init__(self, methods, signal_types, error_types, tracked_indices=[5, 22], base_results_dir="results", n_jobs=4):
        self.methods = methods
        self.signal_types = signal_types
        self.error_types = error_types
        self.tracked_indices = tracked_indices
        self.base_results_dir = base_results_dir
        self.n_jobs = n_jobs

    def run(self):
        configurations = {(s, e): compute_lambda_alpha_grid(s, e)
                          for s in self.signal_types for e in self.error_types}

        joblist = []
        for (sig, err), cfg in configurations.items():
            for lam in cfg["lambda_grid"]:
                for alpha in cfg["alpha_grid"]:
                    for method in self.methods:
                        print(f"Preparing job: {method} | λ={lam} | α={alpha} | signal={sig} | error={err}")
                        joblist.append((method, lam, alpha, sig, err, self.tracked_indices, self.base_results_dir))

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(run_mc_for_config)(*args) for args in joblist
        )
        return results

