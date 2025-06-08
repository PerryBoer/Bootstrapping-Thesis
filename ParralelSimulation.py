from joblib import Parallel, delayed
import numpy as np
import os
from config import Config
from simulation import SimulationRunner
from SimulationTables import generate_summary_tables


def run_single_configuration(method, lam, alpha_th, signal_type, error_type, tracked_indices, base_dir="results"):
    threshold_val = alpha_th * np.sqrt(np.log(Config.p) / Config.n)
    subdir = os.path.join(base_dir, error_type, signal_type, method)
    os.makedirs(subdir, exist_ok=True)

    runner = SimulationRunner(
        method=method,
        lambda_val=lam,
        threshold_val=threshold_val,
        signal_type=signal_type,
        error_type=error_type,
        subdir=subdir,
        tracked_indices=tracked_indices
    )
    results = runner.run()

    # Safe filenames
    suffix = f"lambda{lam:.4f}_alpha{alpha_th:.3f}"
    results["summary_df"].to_csv(os.path.join(subdir, f"summary_{suffix}.csv"), index=False)
    results["raw_df"].to_csv(os.path.join(subdir, f"raw_{suffix}.csv"), index=False)
    results["pointwise_variance_df"].to_csv(os.path.join(subdir, f"pointwise_variance_{suffix}.csv"), index=False)

    tables = generate_summary_tables(results["summary_df"], results["pointwise_variance_df"])
    for name, df in tables.items():
        df.to_csv(os.path.join(subdir, f"{name}_table_{suffix}.csv"))

    return f"Finished: {method} | λ={lam} | α={alpha_th} | signal={signal_type} | error={error_type}"


class ParallelSimulationGrid:
    def __init__(
        self,
        methods,
        lambda_grid,
        alpha_grid,
        signal_types,
        error_types,
        tracked_indices=[5, 20],
        base_results_dir="results",
        n_jobs=4
    ):
        self.methods = methods
        self.lambda_grid = lambda_grid
        self.alpha_grid = alpha_grid
        self.signal_types = signal_types
        self.error_types = error_types
        self.tracked_indices = tracked_indices
        self.base_results_dir = base_results_dir
        self.n_jobs = n_jobs

    def run(self):
        joblist = [
            (m, lam, alpha, sig, err, self.tracked_indices, self.base_results_dir)
            for m in self.methods
            for lam in self.lambda_grid
            for alpha in self.alpha_grid
            for sig in self.signal_types
            for err in self.error_types
        ]
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(run_single_configuration)(*args) for args in joblist
        )
        return results
