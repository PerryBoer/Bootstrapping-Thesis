import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

class NaiveBootstrap:
    def __init__(self, X, y, beta_hat, beta_true, fit_intercept=False, standardize=True, seed=42):
        self.X_raw = X
        self.y = y
        self.beta_hat = beta_hat
        self.beta_true = beta_true
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.seed = seed

        if self.standardize:
            self.scaler = StandardScaler().fit(X)
            self.X = self.scaler.transform(X)
        else:
            self.X = X

        self.n, self.p = self.X.shape
        np.random.seed(seed)

    def center_residuals(self, residuals):
        return residuals - np.mean(residuals)

    def compute_ci(self, beta_star, level=0.90):
        lower = np.percentile(beta_star, (1 - level) / 2 * 100, axis=0)
        upper = np.percentile(beta_star, (1 + level) / 2 * 100, axis=0)
        return lower, upper

    def compute_coverage(self, beta_star, level=0.90):
        lower, upper = self.compute_ci(beta_star, level)
        return ((self.beta_true >= lower) & (self.beta_true <= upper)).astype(int)

    def generate_bootstrap_distribution(self, B, lam, level=0.90):
        residuals = self.center_residuals(self.y - self.X @ self.beta_hat)
        beta_star = np.zeros((B, self.p))

        for b in range(B):
            e_star = np.random.choice(residuals, size=self.n, replace=True)
            y_star = self.X @ self.beta_hat + e_star

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
            "var": beta_star.var(axis=0)
        }
