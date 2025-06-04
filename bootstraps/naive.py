import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

class NaiveBootstrap:
    def __init__(self, X, y, beta_hat, beta_true, fit_intercept=False, standardize=True):
        # iinitialize parameters
        self.X_raw = X
        self.y = y
        self.beta_hat = beta_hat
        self.beta_true = beta_true
        self.fit_intercept = fit_intercept
        self.standardize = standardize

        # standardize design matrix if needed
        if self.standardize:
            self.scaler = StandardScaler().fit(X)
            self.X = self.scaler.transform(X)
        else:
            self.X = X

        self.n, self.p = self.X.shape
    
    # center residuals
    def center_residuals(self, residuals):
        return residuals - np.mean(residuals)

    # Percentile confidence intervals
    def compute_ci(self, beta_star, level=0.90):
        lower = np.percentile(beta_star, (1 - level) / 2 * 100, axis=0)
        upper = np.percentile(beta_star, (1 + level) / 2 * 100, axis=0)
        return lower, upper
    
    # return coverage indicator returning 1 if beta_true is within the confidence interval
    def compute_coverage(self, beta_star, level=0.90):
        ci_lower, ci_upper = self.compute_ci(beta_star, level)
        coverage = (self.beta_true >= ci_lower) & (self.beta_true <= ci_upper)
        return coverage.astype(int)

    def generate_bootstrap_distribution(self, B, lam, level=0.90):
        # compute residuals from the fitted model
        residuals = self.y - self.X @ self.beta_hat
        residuals = self.center_residuals(residuals)

        # init bootstrap array
        beta_star = np.zeros((B, self.p))

        # bootstrap loop
        for b in range(B):
            e_star = np.random.choice(residuals, size=self.n, replace=True)
            y_star = self.X @ self.beta_hat + e_star

            lasso = Lasso(alpha=lam, fit_intercept=self.fit_intercept, max_iter=5000)
            lasso.fit(self.X, y_star)
            beta_star[b] = lasso.coef_

        # compute bootstrap statistics
        mean = beta_star.mean(axis=0)
        std = beta_star.std(axis=0)
        ci_lower, ci_upper = self.compute_ci(beta_star, level)
        coverage = self.compute_coverage(beta_star, level)
        ci_length = ci_upper - ci_lower

        # return results as a dictionary
        return {
            "beta_star": beta_star,
            "mean": mean,
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_length": ci_length,
            "coverage": coverage
        }
