import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

class ModifiedBootstrap:
    def __init__(self, X, y, beta_hat, beta_true, a_n, standardize=True, fit_intercept=False, seed=42):
        self.X_raw = X
        self.y = y
        self.beta_hat = beta_hat
        self.beta_true = beta_true
        self.a_n = a_n
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.seed = seed

        if self.standardize:
            self.scaler = StandardScaler().fit(X)
            self.X = self.scaler.transform(X)
        else:
            self.X = X

        self.n, self.p = self.X.shape
        np.random.seed(self.seed)

    def threshold(self, beta):
        return beta * (np.abs(beta) > self.a_n)

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
        beta_tilde = self.threshold(self.beta_hat)
        residuals = self.center_residuals(self.y - self.X @ beta_tilde)

        beta_star = np.zeros((B, self.p))

        for b in range(B):
            e_star = np.random.choice(residuals, size=self.n, replace=True)
            y_star = self.X @ beta_tilde + e_star
            model = Lasso(alpha=lam, fit_intercept=self.fit_intercept, max_iter=5000)
            model.fit(self.X, y_star)
            beta_star[b] = model.coef_

        ci_lower, ci_upper = self.compute_ci(beta_star, level)
        coverage = self.compute_coverage(beta_star, level)
        ci_length = ci_upper - ci_lower
        mean = np.mean(beta_star, axis=0)
        std = np.std(beta_star, axis=0)
        var = np.var(beta_star, axis=0)

        return {
            "beta_star": beta_star,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_length": ci_length,
            "coverage": coverage,
            "mean": mean,
            "std": std,
            "var": var
        }
