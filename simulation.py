import numpy as np
import pandas as pd
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from config import Config
from dgp import DGP
from lasso_estimator import LassoEstimator
from bootstraps.naive import NaiveBootstrap
from bootstraps.modified import ModifiedBootstrap
from bootstraps.wild import WildBootstrap
from bootstraps.block import BlockBootstrap


class SimulationRunner:
    def __init__(
        self,
        method: str,
        alpha_th: float,
        signal_type: str,
        error_type: str,
        lambda_grid: np.ndarray,
        level: float = 0.90,
        tracked_indices: List[int] = [5, 20],
    ):
        self.method = method
        self.alpha_th = alpha_th
        self.signal_type = signal_type
        self.error_type = error_type
        self.lambda_grid = lambda_grid
        self.level = level

        self.n = Config.n
        self.p = Config.p
        self.s = Config.s
        self.R = Config.num_mc
        self.B = Config.num_bootstrap
        self.tracked_indices = tracked_indices

        # For collecting raw and processed results
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
            a_n = Config.get_threshold(self.alpha_th)

            estimator = LassoEstimator()
            lam_star, _ = estimator.select_lambda_bootstrap_mse(
                X=X,
                y=y,
                lambdas=self.lambda_grid,
                B=self.B,
                a_n=a_n,
                method=self.method,
            )

            estimator.fit(
                X=X,
                y=y,
                lam=lam_star,
                a_n=a_n,
                threshold=(self.method != "naive"),
                support_true=support
            )

            beta_hat = estimator.beta_hat.copy()
            beta_tilde = estimator.beta_tilde.copy()
            residuals = estimator.residuals.copy()

            # Bootstrap method selection
            if self.method == "naive":
                bootstrap = NaiveBootstrap(X, y, beta_hat, beta, fit_intercept=False)
            elif self.method == "modified":
                bootstrap = ModifiedBootstrap(X, y, beta_hat, beta, a_n)
            elif self.method == "wild":
                bootstrap = WildBootstrap(X, y, beta_hat, beta, a_n)
            elif self.method == "block":
                bootstrap = BlockBootstrap(X, y, beta_hat, beta, a_n)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            boot_results = bootstrap.generate_bootstrap_distribution(B=self.B, lam=lam_star, level=self.level)

            for j in self.tracked_indices:
                self.beta_hat_matrix[j].append(beta_hat[j])
                self.boot_var_matrix[j].append(np.var(boot_results["beta_star"][:, j]))

            # Store raw result
            self.raw_results.append({
                "rep": r,
                "method": self.method,
                "alpha_th": self.alpha_th,
                "lambda_star": lam_star,
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
                "var_boot": np.var(boot_results["beta_star"], axis=0),
                "support_size": estimator.support_size,
                "perfect_match": estimator.perfect_match,
                "estimated_support": list(estimator.active_set)
            })

        # Summary computation
        for result in self.raw_results:
            support_indices = result["support"]
            est_support_set = set(result["estimated_support"])
            true_support_set = set(result["support"])
            jaccard = self.jaccard_index(est_support_set, true_support_set)

            bias_hat = result["beta_hat"] - result["beta_true"]
            bias_tilde = result["beta_tilde"] - result["beta_true"]
            mse_boot_var = (result["var_boot"] - np.var(result["beta_star"], axis=0)) ** 2

            self.summary_records.append({
                "rep": result["rep"],
                "method": result["method"],
                "alpha_th": result["alpha_th"],
                "lambda_star": result["lambda_star"],
                "snr": result["snr"],
                "mean_coverage": np.mean(result["coverage"][support_indices]),
                "mean_ci_length": np.mean(result["ci_length"][support_indices]),
                "bias_hat": np.mean(bias_hat[support_indices]),
                "bias_tilde": np.mean(bias_tilde[support_indices]),
                "var_boot_mean": np.mean(result["var_boot"][support_indices]),
                "mse_var_boot": np.mean(mse_boot_var[support_indices]),
                "support_size": result["support_size"],
                "perfect_match": result["perfect_match"],
                "jaccard": jaccard
            })

        # Pointwise variance fidelity
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
                "alpha_th": self.alpha_th,
            })

        df_summary = pd.DataFrame(self.summary_records)
        df_raw = pd.DataFrame(self.raw_results)
        df_pointwise_var = pd.DataFrame(pointwise_var_records)

        return {
            "summary_df": df_summary,
            "raw_df": df_raw,
            "pointwise_variance_df": df_pointwise_var
        }
