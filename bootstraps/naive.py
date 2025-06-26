import numpy as np
from sklearn.linear_model import Lasso

class NaiveBootstrap:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        beta_hat: np.ndarray,
        beta_true: np.ndarray,
        fit_intercept: bool = False
    ):
        self.X = X
        self.y = y
        self.beta_hat = beta_hat
        self.beta_true = beta_true
        self.fit_intercept = fit_intercept
        self.n, self.p = X.shape

    def center_residuals(self, residuals: np.ndarray) -> np.ndarray:
        return residuals - residuals.mean()

    def compute_ci(
        self,
        T_star: np.ndarray,
        beta_ref: np.ndarray,
        level: float = 0.90
    ):
        q_lower = np.percentile(T_star, (1 + level) / 2 * 100, axis=0)
        q_upper = np.percentile(T_star, (1 - level) / 2 * 100, axis=0)
        ci_lower = beta_ref - q_lower / np.sqrt(self.n)
        ci_upper = beta_ref - q_upper / np.sqrt(self.n)
        return ci_lower, ci_upper

    def generate_bootstrap_distribution(
        self,
        B: int,
        lam: float,
        level: float = 0.90
    ):
        beta_ref = self.beta_hat
        resid = self.center_residuals(self.y - self.X @ beta_ref)
        beta_star = np.zeros((B, self.p))

        for b in range(B):
            e_star = np.random.choice(resid, size=self.n, replace=True)
            y_star = self.X @ beta_ref + e_star
            model = Lasso(alpha=lam, fit_intercept=self.fit_intercept, max_iter=10000)
            model.fit(self.X, y_star)
            beta_star[b, :] = model.coef_

        T_star = np.sqrt(self.n) * (beta_star - beta_ref)
        ci_lower, ci_upper = self.compute_ci(T_star, beta_ref, level)
        coverage = ((self.beta_true >= ci_lower) & (self.beta_true <= ci_upper)).astype(int)
        ci_length = ci_upper - ci_lower

        return {
            "beta_star": beta_star,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_length": ci_length,
            "coverage": coverage,
            "mean": beta_star.mean(axis=0),
            "std": beta_star.std(axis=0),
            "var": beta_star.var(axis=0),
        }