import numpy as np
from typing import List, Dict
from lasso_estimator import LassoEstimator
from dgp import DGP
from bootstraps.naive import NaiveBootstrap
from bootstraps.modified import ModifiedBootstrap
from bootstraps.wild import WildBootstrap
from bootstraps.block import BlockBootstrap

class Simulation:
    def __init__(
        self,
        n: int,
        p: int,
        s: int,
        R: int,
        B: int,
        signal_type: str,
        error_type: str,
        lambdas: np.ndarray,
        a_n: float,
        block_length: int = None,
        seed: int = 42
    ):
        self.n = n
        self.p = p
        self.s = s
        self.R = R
        self.B = B
        self.signal_type = signal_type
        self.error_type = error_type
        self.lambdas = lambdas
        self.a_n = a_n
        self.block_length = block_length
        self.seed = seed
        np.random.seed(seed)

        self.methods = {
            'naive': NaiveBootstrap,
            'modified': ModifiedBootstrap,
            'wild': WildBootstrap,
            'block': BlockBootstrap,
        }

    def run(self) -> Dict[str, Dict[str, np.ndarray]]:
        results = {
            method: {
                'coverage': [],
                'ci_length': [],
                'bias': [],
                'bias_tilde': [],
                'variance': [],
                'mse_var': [],
                'jaccard': [],
                'lambda_star': [],
                'snr': [],
                'support_size': [],
                'perfect_match': []
            } for method in self.methods
        }

        all_beta_hats = {method: [] for method in self.methods}
        lambda_list = []
        for r in range(self.R):
            # --- Data Generation ---
            dgp = DGP(n=self.n, p=self.p, s=self.s, signal_type=self.signal_type,
                      error_type=self.error_type, seed=self.seed + r)
            data = dgp.generate()
            X, y = data["X"], data["y"]
            beta_true, eps, support_true, snr = data["beta"], data["errors"], data["support"], data["snr"]

            for method_name, method_cls in self.methods.items():
                # --- Method-specific multiplier ---
                multiplier = 'rademacher' if method_name == 'wild' else 'normal'

                # --- Lambda selection via bootstrap-MSE (method-specific) ---
                estimator = LassoEstimator()
                lambda_star = estimator.select_lambda_bootstrap_mse(
                    X, y, self.lambdas, B=50, a_n=self.a_n, multiplier=multiplier
                )

                # --- Fit LASSO with method-specific λ* ---
                estimator.fit_lasso(X, y, lambda_star, a_n=self.a_n, support_true=support_true)
                beta_hat = estimator.beta_hat
                beta_tilde = estimator.get_thresholded_beta()
                active_set = estimator.get_active_set()

                all_beta_hats[method_name].append(beta_hat)

                # --- Shared support diagnostics ---
                support_true_set = set(support_true)
                active_set_set = set(active_set)
                intersection = len(support_true_set & active_set_set)
                union = len(support_true_set | active_set_set)
                jaccard_index = intersection / union if union > 0 else 0
                support_size = len(active_set)
                perfect_match = int(support_true_set == active_set_set)

                # --- Determine center of bootstrap ---
                beta_center = beta_hat if method_name == "naive" else beta_tilde

                if method_name == 'block':
                    bootstrapper = method_cls(
                        X, y, beta_center, a_n=self.a_n,
                        block_length=self.block_length,
                        seed=self.seed + r,
                        true_beta=beta_true
                    )
                else:
                    bootstrapper = method_cls(
                        X, y, beta_center, a_n=self.a_n,
                        seed=self.seed + r,
                        true_beta=beta_true
                    )

                # --- Run bootstrap inference ---
                out = bootstrapper.generate_bootstrap_distribution(B=self.B, lam=lambda_star)
                beta_star = out['beta_star']
                ci_len = out['ci_upper'] - out['ci_lower']
                bias_true = out['mean'] - beta_true
                bias_tilde = out['mean'] - beta_tilde
                variance = out['std'] ** 2
                coverage = out['coverage']

                # --- Store metrics ---
                results[method_name]['lambda_star'].append(lambda_star)
                results[method_name]['coverage'].append(coverage)
                results[method_name]['ci_length'].append(ci_len)
                results[method_name]['bias'].append(bias_true)
                results[method_name]['bias_tilde'].append(bias_tilde)
                results[method_name]['variance'].append(variance)
                results[method_name]['jaccard'].append(jaccard_index)
                results[method_name]['snr'].append(snr)
                results[method_name]['support_size'].append(support_size)
                results[method_name]['perfect_match'].append(perfect_match)

        # --- Compute method-specific empirical MSE(Var) ---
        for method in self.methods:
            all_beta = np.array(all_beta_hats[method])
            empirical_var = np.var(all_beta, axis=0)
            method_variances = np.array(results[method]['variance'])  # shape (R, p)
            mse_var = np.mean((method_variances - empirical_var) ** 2)
            results[method]['mse_var'] = mse_var

            # Convert lists to arrays
            for key in results[method]:
                if key != 'mse_var':
                    results[method][key] = np.array(results[method][key])

        return results
