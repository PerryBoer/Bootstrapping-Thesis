import numpy as np
from sklearn.linear_model import Lasso
from bootstraps.base import BasisBootstrap

class NaiveBootstrap(BasisBootstrap):
    def generate_bootstrap_distribution(
        self,
        B: int,
        lam: float
    ) -> dict:
        residuals = self.y - self.X @ self.beta_hat
        residuals = self.center_residuals(residuals)

        n, p = self.n, self.p
        beta_star = np.zeros((B, p))

        for b in range(B):
            e_star = np.random.choice(residuals, size=n, replace=True)
            y_star = self.X @ self.beta_hat + e_star

            lasso = Lasso(alpha=lam, fit_intercept=self.fit_intercept)
            lasso.fit(self.X, y_star)
            beta_star[b] = lasso.coef_

        # Compute summary metrics
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