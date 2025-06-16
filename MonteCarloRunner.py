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
        level: float = 0.90,
        tracked_indices: List[int] = [5, 20],
        R: int = Config.num_mc
    ):
        self.method = method
        self.lambda_val = lambda_val
        self.threshold_val = threshold_val
        self.signal_type = signal_type
        self.error_type = error_type
        self.level = level
        self.tracked_indices = tracked_indices
        self.R = R

    def run(self) -> Dict[str, Any]:
        all_results = []
        tracked_metrics = {j: defaultdict(list) for j in self.tracked_indices}

        boot_var_acc = []
        boot_bias_acc = []
        tail_asym_acc = []

        for r in range(self.R):
            sim = SingleSimulationRun(
                method=self.method,
                lambda_val=self.lambda_val,
                threshold_val=self.threshold_val,
                signal_type=self.signal_type,
                error_type=self.error_type,
                level=self.level,
                tracked_indices=self.tracked_indices
            )
            sim.seed = Config.seed + random.randint(0, 10000)  # Ensure different seed for each run
            # print seed for debugging
            
            result = sim.run()
            all_results.append(result)

            # Collect tracked indices diagnostics
            for j in self.tracked_indices:
                tracked_metrics[j]["coverage"].append(result["coverage"][j])
                tracked_metrics[j]["ci_width"].append(result["ci_length"][j])
                tracked_metrics[j]["bias"].append(result["bias_hat"][j])
                tracked_metrics[j]["boot_var"].append(result["boot_var"][j])
                tracked_metrics[j]["boot_bias"].append(result["boot_bias"][j])
                tracked_metrics[j]["tail_asym"].append(result["tail_asymmetry"][j])

            # Track optional diagnostics over full support
            boot_var_acc.append(np.mean([result["boot_var"][j] for j in result["support_true"]]))
            boot_bias_acc.append(np.mean([result["boot_bias"][j] for j in result["support_true"]]))
            tail_asym_acc.append(np.mean([result["tail_asymmetry"][j] for j in result["support_true"]]))

        def avg(key):
            return np.mean([r[key] for r in all_results])

        summary = {
            "method": self.method,
            "lambda_val": self.lambda_val,
            "threshold_val": self.threshold_val,
            "signal_type": self.signal_type,
            "error_type": self.error_type,
            "coverage_support": avg("coverage_rate_support"),
            "coverage_null": avg("null_coverage_rate"),
            "ci_width_support": avg("avg_ci_width_support"),
            "mean_abs_bias": avg("mean_abs_bias_support"),
            "precision": avg("precision"),
            "recall": avg("recall"),
            "fdr": avg("fdr"),
            "jaccard": avg("jaccard"),
            "snr": avg("snr"),
            "boot_var_support": np.mean(boot_var_acc),
            "boot_bias_support": np.mean(boot_bias_acc),
            "tail_asym_support": np.mean(tail_asym_acc)
        }

        return {
            "summary": summary,
            "tracked": {j: {k: np.array(v) for k, v in d.items()} for j, d in tracked_metrics.items()},
            "raw_runs": all_results
        }
    
