import numpy as np
from sklearn.linear_model import Lasso
from bootstraps.base import BasisBootstrap

class WildBootstrap(BasisBootstrap):
    def generate_bootstrap_distribution(
        self,
        B: int,
        lam: float
    ) -> dict:
        beta_tilde = self.threshold_beta()
        residuals = self.y - self.X @ beta_tilde
        residuals = self.center_residuals(residuals)

        n, p = self.n, self.p
        beta_star = np.zeros((B, p))

        for b in range(B):
            v = self._sample_rademacher_weights(n)
            e_star = v * residuals
            y_star = self.X @ beta_tilde + e_star

            lasso = Lasso(alpha=lam, fit_intercept=self.fit_intercept)
            lasso.fit(self.X, y_star)
            beta_star[b] = lasso.coef_

        mean = beta_star.mean(axis=0)
        std = beta_star.std(axis=0)
        ci_lower, ci_upper = self.compute_ci(beta_star, level=0.95)
        coverage = self.compute_coverage(beta_star, level=0.95)

        return {
            "beta_star": beta_star,
            "mean": mean,
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "coverage": coverage
        }

    def _sample_rademacher_weights(self, n: int) -> np.ndarray:
        return np.random.choice([-1, 1], size=n)
