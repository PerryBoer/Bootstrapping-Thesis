import os
import numpy as np
from typing import List
from config import Config
from simulation import SimulationRunner
from SimulationTables import generate_summary_tables


class GenerateSimulations:
    def __init__(
        self,
        methods: List[str],
        lambda_grid: List[float],
        alpha_grid: List[float],
        signal_types: List[str],
        error_types: List[str],
        tracked_indices: List[int] = [5, 20],
        base_results_dir: str = "results"
    ):
        self.methods = methods
        self.lambda_grid = lambda_grid
        self.alpha_grid = alpha_grid
        self.signal_types = signal_types
        self.error_types = error_types
        self.tracked_indices = tracked_indices
        self.base_results_dir = base_results_dir

    def run_all(self):
        for signal in self.signal_types:
            for error in self.error_types:
                for lam in self.lambda_grid:
                    for alpha_th in self.alpha_grid:
                        threshold_val = alpha_th * np.sqrt(np.log(Config.p) / Config.n)

                        for method in self.methods:
                            subdir = os.path.join(self.base_results_dir, error, signal, method)
                            os.makedirs(subdir, exist_ok=True)

                            runner = SimulationRunner(
                                method=method,
                                lambda_val=lam,
                                threshold_val=threshold_val,
                                signal_type=signal,
                                error_type=error,
                                subdir=subdir,
                                tracked_indices=self.tracked_indices
                            )
                            results = runner.run()

                            # Save results
                            results["summary_df"].to_csv(os.path.join(subdir, f"summary_lambda{lam}_alpha{alpha_th}.csv"), index=False)
                            results["raw_df"].to_csv(os.path.join(subdir, f"raw_lambda{lam}_alpha{alpha_th}.csv"), index=False)
                            results["pointwise_variance_df"].to_csv(os.path.join(subdir, f"pointwise_variance_lambda{lam}_alpha{alpha_th}.csv"), index=False)

                            # Save diagnostic tables
                            tables = generate_summary_tables(results["summary_df"], results["pointwise_variance_df"])
                            for name, df in tables.items():
                                df.to_csv(os.path.join(subdir, f"{name}_lambda{lam}_alpha{alpha_th}.csv"))


