# modified_bootstrap.py
import numpy as np
from sklearn.linear_model import Lasso

class ModifiedBootstrap:
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 beta_hat: np.ndarray,
                 beta_true: np.ndarray,
                 a_n: float,
                 fit_intercept: bool = False,
                ):
        """
        X, y must already be centered & scaled (as in DGP.generate()).
        beta_hat: initial Lasso estimate from Lasso(alpha=lam, fit_intercept=False)
        beta_true: ground truth
        a_n: threshold sequence (e.g. n**(-1/3))
        """
        self.X = X
        self.y = y
        self.beta_hat = beta_hat
        self.beta_true = beta_true
        self.a_n = a_n
        self.fit_intercept = fit_intercept
        self.n, self.p = X.shape

    def threshold(self, beta: np.ndarray) -> np.ndarray:
        """Hard‐threshold at ±a_n."""
        return beta * (np.abs(beta) > self.a_n)

    def center_residuals(self, residuals: np.ndarray) -> np.ndarray:
        """Make residuals mean‐zero."""
        return residuals - np.mean(residuals)

    def compute_ci(self, T_star: np.ndarray, beta_tilde: np.ndarray,
                   level: float = 0.90):
        """Quantile‐based CIs for thresholded parameter."""
        # note: percentiles reversed for two‐sided
        q_lower = np.percentile(T_star, (1 + level) / 2 * 100, axis=0)
        q_upper = np.percentile(T_star, (1 - level) / 2 * 100, axis=0)
        ci_lower = beta_tilde - q_lower / np.sqrt(self.n)
        ci_upper = beta_tilde - q_upper / np.sqrt(self.n)
        return ci_lower, ci_upper

    def compute_coverage(self, ci_lower: np.ndarray,
                         ci_upper: np.ndarray) -> np.ndarray:
        """Indicator if true β falls in each CI."""
        return ((self.beta_true >= ci_lower) &
                (self.beta_true <= ci_upper)).astype(int)

    def generate_bootstrap_distribution(self,
                                        B: int,
                                        lam: float,
                                        level: float = 0.90):
        """
        B: number of bootstrap draws
        lam: theoretical λ_n passed straight into Lasso(alpha=lam,…)
        """
        # 1. Threshold the original estimator
        beta_tilde = self.threshold(self.beta_hat)

        # 2. Residuals (centered)
        resid = self.y - self.X @ beta_tilde
        resid = self.center_residuals(resid)

        # 3. Bootstrap draws
        beta_star = np.zeros((B, self.p))
        for b in range(B):
            e_star = np.random.choice(resid, size=self.n, replace=True)
            y_star = self.X @ beta_tilde + e_star
            model = Lasso(alpha=lam,
                          fit_intercept=self.fit_intercept,
                          max_iter=10000)
            model.fit(self.X, y_star)
            beta_star[b, :] = self.threshold(model.coef_)

        # 4. Studentized deviations
        T_star = np.sqrt(self.n) * (beta_star - beta_tilde)

        # 5. CIs, coverage, diagnostics
        ci_lower, ci_upper = self.compute_ci(T_star, beta_tilde, level)
        coverage = self.compute_coverage(ci_lower, ci_upper)
        ci_length = ci_upper - ci_lower
        mean_bs = beta_star.mean(axis=0)
        std_bs = beta_star.std(axis=0)
        var_bs = beta_star.var(axis=0)

        # (Optional) detailed diagnostic print
        support = np.where(self.beta_true != 0)[0]
        # print("\n[MODIFIED BOOTSTRAP DIAGNOSTICS]")
        # for j in support:
        #     print(f"\n--- j = {j} ---")
        #     print(f"β_true[{j}]: {self.beta_true[j]:.4f}")
        #     print(f"CI: [{ci_lower[j]:.4f}, {ci_upper[j]:.4f}]")
        #     print(
        #         f"→ Contains β_true? {'YES' if ci_lower[j] <= self.beta_true[j] <= ci_upper[j] else 'NO'}"
        #     )
        #     print(f"→ β̂ bias: {self.beta_hat[j] - self.beta_true[j]:.4f}")
        #     print(f"→ Bootstrap mean: {mean_bs[j]:.4f}")
        #     print(f"→ Bootstrap std: {std_bs[j]:.5f}")
        #     print(f"→ CI width: {ci_length[j]:.4f}")

        return {
            "beta_star": beta_star,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_length": ci_length,
            "coverage": coverage,
            "mean": mean_bs,
            "std": std_bs,
            "var": var_bs
        }
