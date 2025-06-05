import os
import numpy as np
import pandas as pd
from simulation import SimulationRunner


class GenerateSimulationResults:
    def __init__(
        self,
        signal_types: list,
        error_types: list,
        methods: list,
        alpha_vals: list,
        level: float = 0.90,
        tracked_indices: list = [5, 20],
        base_dir: str = "results/simulation_outputs"
    ):
        self.signal_types = signal_types
        self.error_types = error_types
        self.methods = methods
        self.alpha_vals = alpha_vals
        self.level = level
        self.tracked_indices = tracked_indices
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def run_all(self):
        for signal in self.signal_types:
            for error in self.error_types:
                for method in self.methods:
                    for alpha in self.alpha_vals:
                        print(f"Running: {method} | α = {alpha} | signal = {signal} | error = {error}")
                        subdir = os.path.join(
                            self.base_dir,
                            f"{method}/alpha_{str(alpha).replace('.', '')}/{signal}_{error}"
                        )
                        os.makedirs(subdir, exist_ok=True)

                        runner = SimulationRunner(
                            method=method,
                            alpha_th=alpha,
                            signal_type=signal,
                            error_type=error,
                            level=self.level,
                            tracked_indices=self.tracked_indices,
                            subdir=subdir
                        )

                        results = runner.run()

                        results["summary_df"].to_csv(os.path.join(subdir, "summary.csv"), index=False)
                        results["raw_df"].to_csv(os.path.join(subdir, "raw.csv"), index=False)
                        results["pointwise_variance_df"].to_csv(os.path.join(subdir, "pointwise_variance_diagnostics.csv"), index=False)

                        print(f"Saved to {subdir}\n")
                        print("*" * 50)
