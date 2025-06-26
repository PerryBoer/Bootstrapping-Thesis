from collections import defaultdict
import numpy as np
from simulation import SingleSimulationRun
from typing import List, Dict, Any
from config import Config
import random

class MonteCarloRunner:
    def __init__(
        self,
        method: str,
        lambda_val: float,
        threshold_val: float,
        signal_type: str,
        error_type: str,
        config: Config,
        level: float = 0.90,
        tracked_indices: List[int] = [1, 2],
        R: int = None,
    ):
        self.method = method
        self.lambda_val = lambda_val
        self.threshold_val = threshold_val
        self.signal_type = signal_type
        self.error_type = error_type
        self.config = config
        self.level = level
        self.tracked_indices = tracked_indices
        self.R = R or config.num_mc

    def run(self) -> Dict[str, Any]:
        all_results = []
        run_cov_support = []
        run_cov_null = []
        run_cov_overall = []
        tracked_metrics = {j: defaultdict(list) for j in self.tracked_indices}

        boot_var_acc = []
        boot_bias_acc = []
        tail_asym_acc = []
        stable_support_sizes = []

        for _ in range(self.R):
            sim = SingleSimulationRun(
                method=self.method,
                lambda_val=self.lambda_val,
                threshold_val=self.threshold_val,
                signal_type=self.signal_type,
                error_type=self.error_type,
                config=self.config,
                level=self.level,
                tracked_indices=self.tracked_indices,
            )
            sim.seed = self.config.seed + random.randint(0, 10_000)
            result = sim.run()
            all_results.append(result)

            # collect coverage metrics
            run_cov_support.append(result["coverage_support"])
            run_cov_null.append(result["coverage_null"])
            run_cov_overall.append(result["coverage_overall"])

            # existing diagnostics
            stable_support = result["stable_indices"]
            stable_support_sizes.append(len(stable_support))
            if len(stable_support) > 0:
                boot_var_acc.append(np.mean([result["boot_var"][j] for j in stable_support]))
                boot_bias_acc.append(np.mean([result["boot_bias"][j] for j in stable_support]))
                tail_asym_acc.append(np.mean([result["tail_asymmetry"][j] for j in stable_support]))

            for j in self.tracked_indices:
                tracked_metrics[j]["coverage"].append(result["coverage"][j])
                tracked_metrics[j]["ci_width"].append(result["ci_length"][j])
                tracked_metrics[j]["bias"].append(result["bias_hat"][j])
                tracked_metrics[j]["boot_var"].append(result["boot_var"][j])
                tracked_metrics[j]["boot_bias"].append(result["boot_bias"][j])
                tracked_metrics[j]["tail_asym"].append(result["tail_asymmetry"][j])

        def avg(key: str):
            vals = [r[key] for r in all_results if key in r and not np.isnan(r[key])]
            return np.mean(vals) if vals else np.nan

        summary = {
            "method": self.method,
            "lambda_val": self.lambda_val,
            "threshold_val": self.threshold_val,
            "signal_type": self.signal_type,
            "error_type": self.error_type,
            "coverage_support": np.mean(run_cov_support),
            "coverage_null": np.mean(run_cov_null),
            "coverage_overall": np.mean(run_cov_overall),
            "ci_width_support": avg("avg_ci_width_support"),
            "mean_abs_bias": avg("mean_abs_bias_support"),
            "precision": avg("precision"),
            "recall": avg("recall"),
            "fdr": avg("fdr"),
            "jaccard": avg("jaccard"),
            "snr": avg("snr"),
            "boot_var_support": np.mean(boot_var_acc) if boot_var_acc else np.nan,
            "boot_bias_support": np.mean(boot_bias_acc) if boot_bias_acc else np.nan,
            "tail_asym_support": np.mean(tail_asym_acc) if tail_asym_acc else np.nan,
            "stable_support_avg_size": np.mean(stable_support_sizes) if stable_support_sizes else np.nan,
        }

        return {
            "summary": summary,
            "tracked": {
                j: {k: np.array(v) for k, v in d.items()}
                for j, d in tracked_metrics.items()
            },
            "raw_runs": all_results
        }