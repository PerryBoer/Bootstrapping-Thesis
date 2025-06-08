import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

class ModifiedBootstrap:
    def __init__(self, X, y, beta_hat, beta_true, a_n, standardize=True, fit_intercept=False, seed=42):
        # init parameters
        self.X_raw = X
        self.y = y
        self.beta_hat = beta_hat
        self.beta_true = beta_true
        self.a_n = a_n
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.seed = seed

        # standardize design matrix if needed
        if self.standardize:
            self.scaler = StandardScaler().fit(X)
            self.X = self.scaler.transform(X)
        else:
            self.X = X

        self.n, self.p = self.X.shape
        np.random.seed(self.seed)

    # apply thresholding to get beta_tilde from beta_hat
    def threshold(self, beta):
        beta_tilde = beta.copy()
        beta_tilde[np.abs(beta_tilde) < self.a_n] = 0.0
        return beta_tilde

    # center residuals
    def center_residuals(self, residuals):
        return residuals - np.mean(residuals)

    # compute percentile confidence intervals
    def compute_ci(self, beta_star, level=0.90):
        lower = np.percentile(beta_star, (1 - level) / 2 * 100, axis=0)
        upper = np.percentile(beta_star, (1 + level) / 2 * 100, axis=0)
        return lower, upper

    # compute coverage indicator returning 1 if beta_true is within the confidence interval
    def compute_coverage(self, beta_star, level=0.90):
        lower, upper = self.compute_ci(beta_star, level)
        coverage = (self.beta_true >= lower) & (self.beta_true <= upper)
        return coverage.astype(int)

    def generate_bootstrap_distribution(self, B, lam, level=0.90):
        # threshold original estimator to form beta_tilde
        beta_tilde = self.threshold(self.beta_hat)

        # compute centered residuals from beta_tilde
        residuals = self.y - self.X @ beta_tilde
        residuals = self.center_residuals(residuals)

        beta_star = np.zeros((B, self.p))

        # bootstrap loop
        for b in range(B):
            # generate bootstrap sample
            e_star = np.random.choice(residuals, size=self.n, replace=True)

            # create bootstrap response variable
            y_star = self.X @ beta_tilde + e_star

            # fit LASSO model to bootstrap sample
            model = Lasso(alpha=lam, fit_intercept=self.fit_intercept, max_iter=5000)
            model.fit(self.X, y_star)
            beta_star[b] = model.coef_ 

        # Step 4: Summary diagnostics
        mean = beta_star.mean(axis=0)
        std = beta_star.std(axis=0)
        ci_lower, ci_upper = self.compute_ci(beta_star, level)
        coverage = self.compute_coverage(beta_star, level)
        ci_length = ci_upper - ci_lower

        return {
            "beta_star": beta_star,
            "mean": mean,
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_length": ci_length,
            "coverage": coverage
        }
