import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from config import Config
from dgp import DGP
from MonteCarloRunner import MonteCarloRunner
from LassoEstimatorTheory import LassoEstimatorTheory


def compute_lambda_alpha(config: Config, signal_type: str, error_type: str):
    """
    Simulate one dataset, fit a rough Lasso to get residual std,
    then compute λ and α on the null betas.
    """
    dgp = DGP(config=config, signal_type=signal_type, error_type=error_type)
    data = dgp.generate()
    X, y, _, support_true = data["X"], data["y"], data["beta"], data["support"]

    # fit rough Lasso
    estimator = LassoEstimatorTheory()
    rough_lambda = 0.5 * np.sqrt(np.log(config.p) / config.n)
    estimator.fit(
        X, y,
        lam=rough_lambda,
        thresholding_level=0.0,
        apply_threshold=False,
        support_true=support_true
    )

    beta_hat = estimator.beta_hat
    residuals = estimator.residuals
    sigma_hat = np.std(residuals)

    # final λ
    lam = sigma_hat * np.sqrt(np.log(config.p) / config.n)

    # final α = 95% quantile of null |β̂|, scaled
    null_idx = list(set(range(config.p)) - set(support_true))
    null_mags = np.abs(beta_hat[null_idx])
    alpha = np.quantile(null_mags, 0.95) / np.sqrt(np.log(config.p) / config.n)

    return lam, alpha


def run_one(method: str,
            lam: float,
            alpha: float,
            config: Config,
            tracked_indices,
            base_dir: str,
            signal_type: str,
            error_type: str):
    """
    Run Monte Carlo for one combination,
    save into base_dir/error/signal/method/p<config.p>/...
    and include lam/alpha as columns in summary.csv.
    """
    # run it
    runner = MonteCarloRunner(
        method=method,
        lambda_val=lam,
        threshold_val=alpha,
        signal_type=signal_type,
        error_type=error_type,
        config=config,
        tracked_indices=tracked_indices
    )
    results = runner.run()

    # make the 4-deep folder
    subdir = os.path.join(
        base_dir,
        error_type,
        signal_type,
        method,
        f"p{config.p}"
    )
    os.makedirs(subdir, exist_ok=True)

    # summary.csv with λ & α columns
    summary = results["summary"].copy()
    summary["lambda"] = lam
    summary["alpha"] = alpha
    pd.DataFrame([summary]).to_csv(
        os.path.join(subdir, "summary.csv"), index=False
    )

    # one tracked_j*.csv per tracked index
    for j, tracked in results["tracked"].items():
        pd.DataFrame(tracked).to_csv(
            os.path.join(subdir, f"tracked_j{j}.csv"), index=False
        )

    # raw_runs.csv
    pd.DataFrame(results["raw_runs"]).to_csv(
        os.path.join(subdir, "raw_runs.csv"), index=False
    )

    return f"Done: {error_type}/{signal_type}/{method}/p{config.p}"


class FullSweep:
    def __init__(self,
                 methods,
                 p_grid,
                 error_types,
                 signal_types,
                 base_results_dir="results/full_sweep",
                 n_jobs=4):
        self.methods = methods
        self.p_grid = p_grid
        self.error_types = error_types
        self.signal_types = signal_types
        self.base_results_dir = base_results_dir
        self.n_jobs = n_jobs
        self.tracked_indices = [1, 2]

    def run(self):
        tasks = []
        for err in self.error_types:
            for sig in self.signal_types:
                for p in self.p_grid:
                    cfg = Config(p=p, n=250)
                    lam, alpha = compute_lambda_alpha(cfg, sig, err)
                    for method in self.methods:
                        tasks.append((
                            method, lam, alpha, cfg,
                            self.tracked_indices,
                            self.base_results_dir,
                            sig, err
                        ))

        # parallel execution
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(run_one)(*t) for t in tasks
        )

        # print a quick pass/fail list
        for msg in results:
            print(msg)
        return results


if __name__ == "__main__":
    METHODS = ["naive", "modified", "wild", "block"]
    P_GRID = [10, 25, 50, 100, 150, 200, 300, 400]
    ERRORS = ["gaussian", "heteroskedastic", "ar1"]
    SIGNALS = ["strong", "weak", "nearzero"]

    sweep = FullSweep(
        methods=METHODS,
        p_grid=P_GRID,
        error_types=ERRORS,
        signal_types=SIGNALS,
        base_results_dir="results/dimensionality_sensitivity",
        n_jobs=8
    )
    sweep.run()
