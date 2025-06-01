import numpy as np
from sklearn.linear_model import Lasso
from bootstraps.base import BasisBootstrap

class ModifiedBootstrap(BasisBootstrap):
    def generate_bootstrap_distribution(
        self,
        B: int,
        lam: float
    ) -> dict:
        # Step 1: Threshold to get beta_tilde
        beta_tilde = self.threshold_beta()

        # Step 2: Compute centered residuals
        residuals = self.y - self.X @ beta_tilde
        residuals = self.center_residuals(residuals)

        n, p = self.n, self.p
        beta_star = np.zeros((B, p))

        # Step 3: Bootstrap samples around beta_tilde
        for b in range(B):
            e_star = np.random.choice(residuals, size=n, replace=True)
            y_star = self.X @ beta_tilde + e_star

            lasso = Lasso(alpha=lam, fit_intercept=self.fit_intercept, max_iter=5000)
            lasso.fit(self.X, y_star)
            beta_star[b] = lasso.coef_

        # Step 4: Diagnostics
        mean = beta_star.mean(axis=0)
        std = beta_star.std(axis=0)
        ci_lower, ci_upper = self.compute_ci(beta_star, level=0.95)
        coverage = self.compute_coverage(beta_star, level=0.95)

        # Step 5: Store internally
        self.beta_star_dist = beta_star
        self.ci_bounds = (ci_lower, ci_upper)
        self.coverage_vec = coverage

        return {
            "beta_star": beta_star,
            "mean": mean,
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "coverage": coverage
        }

