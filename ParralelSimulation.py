import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import Config
from dgp import DGP
from MonteCarloRunner import MonteCarloRunner
from LassoEstimatorTheory import LassoEstimatorTheory

def compute_lambda_alpha_grid(config: Config, signal_type: str, error_type: str):
    """
    One DGP draw → rough Lasso to get σ̂ → base λ & α → {0.5,1,2}× grids
    """
    # 1. Generate data
    dgp = DGP(config=config, signal_type=signal_type, error_type=error_type)
    data = dgp.generate()
    X, y, _, support_true, snr = (
        data["X"], data["y"], data["beta"], data["support"], data.get("snr")
    )

    # 2. Rough Lasso fit
    estimator = LassoEstimatorTheory()
    rough_lambda = 0.5 * np.sqrt(np.log(config.p) / config.n)
    estimator.fit(
        X, y,
        lam=rough_lambda,
        thresholding_level=0.0,
        apply_threshold=False,
        support_true=support_true
    )
    beta_hat = estimator.beta_hat.copy()
    sigma_hat = np.std(estimator.residuals, ddof=1)

    # 3. Base λ & grid
    lambda_base = sigma_hat * np.sqrt(np.log(config.p) / config.n)
    factors = np.array([0.5, 1.0, 2.0])
    lambda_grid = np.round(factors * lambda_base, 5)

    # 4. Base α & grid
    null_idx = list(set(range(config.p)) - set(support_true))
    null_mags = np.abs(beta_hat[null_idx])
    threshold_95 = np.quantile(null_mags, 0.95)
    alpha_null = threshold_95 / np.sqrt(np.log(config.p) / config.n)
    alpha_grid = [round(alpha_null * f, 3) for f in factors]

    return {
        "lambda_base": lambda_base,
        "lambda_grid": lambda_grid.tolist(),
        "alpha_null": round(alpha_null, 3),
        "alpha_grid": alpha_grid,
        "sigma_hat": sigma_hat,
        "snr": snr
    }

def run_mc_for_config(method: str,
                      lam: float,
                      alpha: float,
                      signal: str,
                      error: str,
                      tracked_indices,
                      config: Config,
                      base_dir="results"):
    """
    Runs Monte Carlo for (λ,α) on one config, saves under …/p<config.p>/…
    """
    runner = MonteCarloRunner(
        method=method,
        lambda_val=lam,
        threshold_val=alpha,
        signal_type=signal,
        error_type=error,
        config=config,
        tracked_indices=tracked_indices,
        R=config.num_mc
    )
    results = runner.run()

    # structure: base_dir/error/signal/method/p<config.p>/
    subdir = os.path.join(base_dir, error, signal, method, f"p{config.p}")
    os.makedirs(subdir, exist_ok=True)

    suffix = f"lambda{lam:.5f}_alpha{alpha:.3f}"
    pd.DataFrame([results["summary"]]) \
      .to_csv(os.path.join(subdir, f"summary_{suffix}.csv"), index=False)

    for j, tracked in results["tracked"].items():
        pd.DataFrame(tracked) \
          .to_csv(os.path.join(subdir, f"tracked_j{j}_{suffix}.csv"), index=False)

    # pd.DataFrame(results["raw_runs"]) \
    #   .to_csv(os.path.join(subdir, f"raw_runs_{suffix}.csv"), index=False)

    return f"Finished: {method} | λ={lam} | α={alpha} | signal={signal} | error={error}"

class ParallelSimulationGrid:
    def __init__(self,
                 methods,
                 signal_types,
                 error_types,
                 tracked_indices=[5, 22],
                 base_results_dir="results",
                 n_jobs=4,
                 p=300,
                 n=250):
        self.methods = methods
        self.signal_types = signal_types
        self.error_types = error_types
        self.tracked_indices = tracked_indices
        self.base_results_dir = base_results_dir
        self.n_jobs = n_jobs
        # fix dimensionality here
        self.config = Config(p=p, n=n)

    def run(self):
        # build λ/α grids once per scenario
        grids = {
            (s, e): compute_lambda_alpha_grid(self.config, s, e)
            for s in self.signal_types
            for e in self.error_types
        }

        jobs = []
        for (sig, err), info in grids.items():
            for lam in info["lambda_grid"]:
                for alpha in info["alpha_grid"]:
                    for method in self.methods:
                        print(f"Job: {method} | λ={lam} | α={alpha} | {sig}/{err}")
                        jobs.append((
                            method, lam, alpha, sig, err,
                            self.tracked_indices,
                            self.config,
                            self.base_results_dir
                        ))

        # execute in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(run_mc_for_config)(*job) for job in jobs
        )
        return results

if __name__ == "__main__":
    METHODS = ["wild"]
    ERRORS  = ["heteroskedastic"]
    SIGNALS = ["strong", "weak", "nearzero"]

    sweep = ParallelSimulationGrid(
        methods=METHODS,
        signal_types=SIGNALS,
        error_types=ERRORS,
        base_results_dir="results/symmetric_sweep/wild_weights/normal",
        n_jobs=3,
        p=300,
        n=200
    )
    sweep.run()
