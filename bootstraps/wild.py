import numpy as np
from sklearn.linear_model import Lasso
from bootstraps.base import BasisBootstrap

class WildBootstrap(BasisBootstrap):
    def generate_bootstrap_distribution(
        self,
        B: int,
        lam: float
    ) -> dict:
        # Step 1: Compute thresholded beta and residuals
        beta_tilde = self.threshold_beta()
        residuals = self.y - self.X @ beta_tilde
        residuals = self.center_residuals(residuals)

        n, p = self.n, self.p
        beta_star = np.zeros((B, p))

        # Step 2: Wild bootstrap with Rademacher weights
        for b in range(B):
            v = self._sample_rademacher_weights(n)
            e_star = v * residuals
            y_star = self.X @ beta_tilde + e_star

            lasso = Lasso(alpha=lam, fit_intercept=self.fit_intercept, max_iter=5000)
            lasso.fit(self.X, y_star)
            beta_star[b] = lasso.coef_

        # Step 3: Summary statistics
        mean = beta_star.mean(axis=0)
        std = beta_star.std(axis=0)
        ci_lower, ci_upper = self.compute_ci(beta_star, level=0.95)
        coverage = self.compute_coverage(beta_star, level=0.95)

        # Step 4: Store for introspection
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

    def _sample_rademacher_weights(self, n: int) -> np.ndarray:
        """Generates Rademacher weights: +-1 with equal probability."""
        return np.random.choice([-1, 1], size=n)