import numpy as np
import pandas as pd
from typing import Dict, Any, List
from config import Config
from dgp import DGP
from LassoEstimatorTheory import LassoEstimatorTheory
from bootstraps.naive import NaiveBootstrap
from bootstraps.modified import ModifiedBootstrap
from bootstraps.wild import WildBootstrap
from bootstraps.block import BlockBootstrap
from SimulationPlotter import SimulationPlotter


class SimulationRunner:
    def __init__(
        self,
        method: str,
        lambda_val: float,
        threshold_val: float,
        signal_type: str,
        error_type: str,
        level: float = 0.90,
        tracked_indices: List[int] = [5, 22],
        subdir: str = "results/plots"
    ):
        self.method = method
        self.lambda_val = lambda_val
        self.threshold_val = threshold_val
        self.signal_type = signal_type
        self.error_type = error_type
        self.level = level
        self.subdir = subdir

        self.n = Config.n
        self.p = Config.p
        self.s = Config.s
        self.R = Config.num_mc
        self.B = Config.num_bootstrap
        self.tracked_indices = tracked_indices

        self.raw_results = []
        self.summary_records = []
        self.beta_hat_matrix = {j: [] for j in tracked_indices}
        self.boot_var_matrix = {j: [] for j in tracked_indices}

    def jaccard_index(self, set1: set, set2: set) -> float:
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def run(self) -> Dict[str, Any]:
        for r in range(self.R):
            np.random.seed(Config.seed + r)
            dgp = DGP(signal_type=self.signal_type, error_type=self.error_type, seed=Config.seed + r)
            data = dgp.generate()

            X, y, beta, support, snr = data["X"], data["y"], data["beta"], data["support"], data["snr"]

            estimator = LassoEstimatorTheory()
            estimator.fit(
                X=X,
                y=y,
                lam=self.lambda_val,
                thresholding_level=self.threshold_val,
                apply_threshold=(self.method != "naive"),
                support_true=support
            )

            beta_hat = estimator.beta_hat.copy()
            beta_tilde = estimator.beta_tilde.copy()
            residuals = estimator.residuals.copy()

            if self.method == "naive":
                bootstrap = NaiveBootstrap(X, y, beta_hat, beta, fit_intercept=False)
            elif self.method == "modified":
                bootstrap = ModifiedBootstrap(X, y, beta_hat, beta, self.threshold_val)
            elif self.method == "wild":
                bootstrap = WildBootstrap(X, y, beta_hat, beta, self.threshold_val)
            elif self.method == "block":
                bootstrap = BlockBootstrap(X, y, beta_hat, beta, self.threshold_val)
            else:
                raise ValueError(f"Unknown bootstrap method: {self.method}")

            boot_results = bootstrap.generate_bootstrap_distribution(B=self.B, lam=self.lambda_val, level=self.level)

            for j in self.tracked_indices:
                self.beta_hat_matrix[j].append(beta_hat[j])
                self.boot_var_matrix[j].append(np.var(boot_results["beta_star"][:, j]))

            self.raw_results.append({
                "rep": r,
                "method": self.method,
                "lambda_val": self.lambda_val,
                "threshold_val": self.threshold_val,
                "snr": snr,
                "beta_true": beta.copy(),
                "beta_hat": beta_hat,
                "beta_tilde": beta_tilde,
                "support": support.copy(),
                "residuals": residuals,
                "beta_star": boot_results["beta_star"],
                "ci_lower": boot_results["ci_lower"],
                "ci_upper": boot_results["ci_upper"],
                "ci_length": boot_results["ci_length"],
                "coverage": boot_results["coverage"],
                "var_boot": boot_results["var"],
                "support_size": estimator.support_size,
                "perfect_match": estimator.perfect_match,
                "estimated_support": list(estimator.active_set)
            })

        for result in self.raw_results:
            true_support = set(result["support"])
            est_support = set(result["estimated_support"])

            TP = len(true_support & est_support)
            FP = len(est_support - true_support)
            FN = len(true_support - est_support)

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            fdr = FP / (TP + FP) if (TP + FP) > 0 else 0.0
            fnr = FN / (TP + FN) if (TP + FN) > 0 else 0.0
            jaccard = self.jaccard_index(est_support, true_support)

            bias_hat = result["beta_hat"] - result["beta_true"]
            bias_tilde = result["beta_tilde"] - result["beta_true"]
            mse_boot_var = (result["var_boot"] - np.var(result["beta_star"], axis=0)) ** 2

            self.summary_records.append({
                "rep": result["rep"],
                "method": result["method"],
                "lambda_val": result["lambda_val"],
                "threshold_val": result["threshold_val"],
                "snr": result["snr"],
                "mean_coverage": np.mean(result["coverage"][result["support"]]),
                "mean_ci_length": np.mean(result["ci_length"][result["support"]]),
                "bias_hat": np.mean(bias_hat[result["support"]]),
                "bias_tilde": np.mean(bias_tilde[result["support"]]),
                "var_boot_mean": np.mean(result["var_boot"][result["support"]]),
                "mse_var_boot": np.mean(mse_boot_var[result["support"]]),
                "support_size": result["support_size"],
                "perfect_match": result["perfect_match"],
                "jaccard": jaccard,
                "fdr": fdr,
                "fnr": fnr,
                "precision": precision,
                "recall": recall
            })

        pointwise_var_records = []
        for j in self.tracked_indices:
            beta_hat_j = np.array(self.beta_hat_matrix[j])
            boot_var_j = np.array(self.boot_var_matrix[j])
            var_mc = np.var(beta_hat_j)
            var_boot = np.mean(boot_var_j)
            mse_var_boot = np.mean((boot_var_j - var_mc) ** 2)
            pointwise_var_records.append({
                "beta_index": j,
                "mc_variance": var_mc,
                "boot_variance_mean": var_boot,
                "mse_boot_variance": mse_var_boot,
                "method": self.method,
                "lambda_val": self.lambda_val,
                "threshold_val": self.threshold_val
            })

        df_summary = pd.DataFrame(self.summary_records)
        df_raw = pd.DataFrame(self.raw_results)
        df_pointwise_var = pd.DataFrame(pointwise_var_records)

        plotter = SimulationPlotter(raw_df=df_raw, summary_df=df_summary, save_dir=self.subdir)
        plotter.generate_all_plots(beta_indices=self.tracked_indices)

        return {
            "summary_df": df_summary,
            "raw_df": df_raw,
            "pointwise_variance_df": df_pointwise_var
        }


def jaccard_index(set1: set, set2: set) -> float:
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def jaccard_index(set1: set, set2: set) -> float:
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


class SingleSimulationRun:
    def __init__(
        self,
        method: str,
        lambda_val: float,
        threshold_val: float,
        signal_type: str,
        error_type: str,
        level: float = 0.95,
        tracked_indices: List[int] = [5, 20],
        seed: int = Config.seed,
    ):
        self.method = method
        self.lambda_val = lambda_val
        self.threshold_val = threshold_val
        self.signal_type = signal_type
        self.error_type = error_type
        self.level = level
        self.tracked_indices = tracked_indices
        self.seed = seed

        self.n = Config.n
        self.p = Config.p
        self.s = Config.s
        self.B = Config.num_bootstrap

    def run(self) -> Dict[str, Any]:
        np.random.seed(self.seed)
        # print seed
        print(f"Running simulation with seed: {self.seed}")

        dgp = DGP(signal_type=self.signal_type, error_type=self.error_type, seed=self.seed)
        data = dgp.generate()
        X, y, beta_true, support_true, snr = data["X"], data["y"], data["beta"], data["support"], data["snr"]

        estimator = LassoEstimatorTheory()
        estimator.fit(
            X=X,
            y=y,
            lam=self.lambda_val,
            thresholding_level=self.threshold_val,
            apply_threshold=(self.method != "naive"),
            support_true=support_true
        )

        beta_hat = estimator.beta_hat.copy()
        beta_tilde = estimator.beta_tilde.copy()
        residuals = estimator.residuals.copy()
        support_est = list(estimator.active_set)

        if self.method == "naive":
            bootstrap = NaiveBootstrap(X, y, beta_hat, beta_true, fit_intercept=False)
        elif self.method == "modified":
            bootstrap = ModifiedBootstrap(X, y, beta_hat, beta_true, self.threshold_val)
        elif self.method == "wild":
            bootstrap = WildBootstrap(X, y, beta_hat, beta_true, self.threshold_val)
        elif self.method == "block":
            bootstrap = BlockBootstrap(X, y, beta_hat, beta_true, self.threshold_val)
        else:
            raise ValueError(f"Unknown bootstrap method: {self.method}")

        boot_results = bootstrap.generate_bootstrap_distribution(B=self.B, lam=self.lambda_val, level=self.level)

        beta_star = boot_results["beta_star"]
        ci_lower = boot_results["ci_lower"]
        ci_upper = boot_results["ci_upper"]
        ci_length = boot_results["ci_length"]
        coverage = (ci_lower <= beta_true) & (beta_true <= ci_upper)

        bias_hat = beta_hat - beta_true
        bias_tilde = beta_tilde - beta_true
        abs_bias_hat = np.abs(bias_hat)
        boot_mean = np.mean(beta_star, axis=0)
        boot_bias = boot_mean - beta_tilde
        boot_var = np.var(beta_star, axis=0)

        support_est_set = set(support_est)
        support_true_set = set(support_true)
        TP = len(support_est_set & support_true_set)
        FP = len(support_est_set - support_true_set)
        FN = len(support_true_set - support_est_set)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fdr = FP / (TP + FP) if (TP + FP) > 0 else 0.0
        jaccard = jaccard_index(support_est_set, support_true_set)

        ci_width_support = ci_length[support_true]
        coverage_support = coverage[support_true]
        abs_bias_support = abs_bias_hat[support_true]
        avg_ci_width_support = np.mean(ci_width_support)
        mean_abs_bias_support = np.mean(abs_bias_support)
        coverage_rate_support = np.mean(coverage_support)

        # Tail asymmetry
        tail_asymmetry = (ci_upper - boot_mean) - (boot_mean - ci_lower)

        # Quantiles
        ci_quantiles = {
            "q_025": np.quantile(beta_star, 0.025, axis=0),
            "q_975": np.quantile(beta_star, 0.975, axis=0),
            "q_05": np.quantile(beta_star, 0.05, axis=0),
            "q_95": np.quantile(beta_star, 0.95, axis=0),
        }

        # Null coverage
        null_indices = list(set(range(self.p)) - support_true_set)
        null_coverage = np.mean((ci_lower[null_indices] <= beta_true[null_indices]) &
                                (beta_true[null_indices] <= ci_upper[null_indices]))

        return {
            "X": X, "y": y,
            "beta_true": beta_true,
            "beta_hat": beta_hat,
            "beta_tilde": beta_tilde,
            "residuals": residuals,
            "beta_star": beta_star,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_length": ci_length,
            "coverage": coverage,
            "support_true": support_true,
            "support_est": support_est,
            "support_size": len(support_est),
            "snr": snr,
            "lambda_val": self.lambda_val,
            "threshold_val": self.threshold_val,
            "method": self.method,
            "tracked_indices": self.tracked_indices,
            "bias_hat": bias_hat,
            "bias_tilde": bias_tilde,
            "abs_bias_hat": abs_bias_hat,
            "boot_bias": boot_bias,
            "boot_var": boot_var,
            "precision": precision,
            "recall": recall,
            "fdr": fdr,
            "jaccard": jaccard,
            "avg_ci_width_support": avg_ci_width_support,
            "mean_abs_bias_support": mean_abs_bias_support,
            "coverage_rate_support": coverage_rate_support,
            "null_coverage_rate": null_coverage,
            "tail_asymmetry": tail_asymmetry,
            "ci_quantiles": ci_quantiles
        }


