import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

class WildBootstrap:
    def __init__(self, X, y, beta_hat, beta_true, threshold_level, standardize=True, fit_intercept=False, seed=42):
        self.X_raw = X
        self.y = y
        self.beta_hat = beta_hat
        self.beta_true = beta_true
        self.threshold_level = threshold_level
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.seed = seed

        # Standardize design matrix
        if self.standardize:
            self.scaler = StandardScaler().fit(X)
            self.X = self.scaler.transform(X)
        else:
            self.X = X

        self.n, self.p = self.X.shape
        np.random.seed(seed)

    def threshold(self, beta):
        beta_tilde = beta.copy()
        beta_tilde[np.abs(beta_tilde) < self.threshold_level] = 0.0
        return beta_tilde

    def center_residuals(self, residuals):
        return residuals - np.mean(residuals)

    def compute_ci(self, beta_star, level=0.90):
        lower = np.percentile(beta_star, (1 - level) / 2 * 100, axis=0)
        upper = np.percentile(beta_star, (1 + level) / 2 * 100, axis=0)
        return lower, upper

    def compute_coverage(self, beta_star, level=0.90):
        lower, upper = self.compute_ci(beta_star, level)
        return ((self.beta_true >= lower) & (self.beta_true <= upper)).astype(int)

    def _sample_rademacher_weights(self, n):
        return np.random.choice([-1, 1], size=n)

    def generate_bootstrap_distribution(self, B, lam, level=0.90):
        beta_tilde = self.threshold(self.beta_hat)
        residuals = self.y - self.X @ beta_tilde
        residuals = self.center_residuals(residuals)

        beta_star = np.zeros((B, self.p))

        for b in range(B):
            v = self._sample_rademacher_weights(self.n)
            e_star = v * residuals
            y_star = self.X @ beta_tilde + e_star

            model = Lasso(alpha=lam, fit_intercept=self.fit_intercept, max_iter=5000)
            model.fit(self.X, y_star)
            beta_star[b] = model.coef_

        ci_lower, ci_upper = self.compute_ci(beta_star, level)
        coverage = self.compute_coverage(beta_star, level)
        ci_length = ci_upper - ci_lower

        return {
            "beta_star": beta_star,
            "mean": beta_star.mean(axis=0),
            "std": beta_star.std(axis=0),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_length": ci_length,
            "coverage": coverage,
            "var": beta_star.var(axis=0),
        }