import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import Config
from dgp import DGP
from MonteCarloRunner import MonteCarloRunner
from LassoEstimatorTheory import LassoEstimatorTheory


def compute_lambda_alpha(config: Config, signal_type: str, error_type: str):
    # (unchanged)
    dgp = DGP(config=config, signal_type=signal_type, error_type=error_type)
    data = dgp.generate()
    X, y, _, support_true = data["X"], data["y"], data["beta"], data["support"]

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

    lam = sigma_hat * np.sqrt(np.log(config.p) / config.n)
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
    # (unchanged)
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

    subdir = os.path.join(
        base_dir,
        error_type,
        signal_type,
        method,
        f"s{config.s}",
        f"p{config.p}"
    )
    os.makedirs(subdir, exist_ok=True)

    summary = results["summary"].copy()
    summary["lambda"] = lam
    summary["alpha"] = alpha
    pd.DataFrame([summary]).to_csv(
        os.path.join(subdir, "summary.csv"), index=False
    )

    for j, tracked in results["tracked"].items():
        pd.DataFrame(tracked).to_csv(
            os.path.join(subdir, f"tracked_j{j}.csv"), index=False
        )

    # pd.DataFrame(results["raw_runs"]).to_csv(
    #     os.path.join(subdir, "raw_runs.csv"), index=False
    # )

    return f"Done: {error_type}/{signal_type}/s{config.s}/p{config.p}"


class SparsitySweep:
    def __init__(self,
                 methods,
                 p_fixed,
                 s_grid,
                 error_types,
                 signal_types,
                 base_results_dir="results/sparsity_sensitivity",
                 n_jobs=4):
        self.methods = methods
        self.p_fixed = p_fixed
        self.s_grid = s_grid
        self.error_types = error_types
        self.signal_types = signal_types
        self.base_results_dir = base_results_dir
        self.n_jobs = n_jobs
        self.tracked_indices = [1, 2]

    def run(self):
        tasks = []
        for err in self.error_types:
            for sig in self.signal_types:
                for s in self.s_grid:
                    cfg = Config(p=self.p_fixed, n=250, s=s)
                    lam, alpha = compute_lambda_alpha(cfg, sig, err)
                    for method in self.methods:
                        tasks.append((
                            method, lam, alpha, cfg,
                            self.tracked_indices,
                            self.base_results_dir,
                            sig, err
                        ))

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(run_one)(*t) for t in tasks
        )

        for msg in results:
            print(msg)
        return results


if __name__ == "__main__":
    METHODS = ["naive", "modified", "wild", "block"]
    P_FIXED = 200
    S_GRID = [10, 20, 40, 60, 80, 100]
    ERRORS = ["gaussian", "heteroskedastic", "ar1"]
    SIGNALS = ["strong", "weak", "nearzero"]

    sweep = SparsitySweep(
        methods=METHODS,
        p_fixed=P_FIXED,
        s_grid=S_GRID,
        error_types=ERRORS,
        signal_types=SIGNALS,
        base_results_dir="results/sparsity_sensitivity",
        n_jobs=8
    )
    sweep.run()


