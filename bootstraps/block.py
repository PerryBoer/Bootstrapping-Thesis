import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from config import Config


class BlockBootstrap:
    def __init__(self, X, y, beta_hat, beta_true, a_n, standardize=True, fit_intercept=False, seed=42):
        # Store initial inputs
        self.X_raw = X
        self.y = y
        self.beta_hat = beta_hat
        self.beta_true = beta_true
        self.a_n = a_n
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.seed = seed

        # Standardize design matrix if requested
        if self.standardize:
            self.scaler = StandardScaler().fit(X)
            self.X = self.scaler.transform(X)
        else:
            self.X = X

        self.n, self.p = self.X.shape
        np.random.seed(seed)

    # Apply hard thresholding to get beta_tilde
    def threshold(self, beta):
        beta_tilde = beta.copy()
        beta_tilde[np.abs(beta_tilde) < self.a_n] = 0.0
        return beta_tilde

    # Center residuals to have mean zero
    def center_residuals(self, residuals):
        return residuals - np.mean(residuals)

    # Compute percentile-based confidence intervals
    def compute_ci(self, beta_star, level=0.90):
        lower = np.percentile(beta_star, (1 - level) / 2 * 100, axis=0)
        upper = np.percentile(beta_star, (1 + level) / 2 * 100, axis=0)
        return lower, upper

    # Compute per-coordinate coverage based on true beta
    def compute_coverage(self, beta_star, level=0.90):
        lower, upper = self.compute_ci(beta_star, level)
        return ((self.beta_true >= lower) & (self.beta_true <= upper)).astype(int)

    # Non-overlapping block resampling function
    def _nonoverlapping_block_resample(self, residuals):
        self.block_size = int(np.floor(Config.n ** (1 / 3)))
        n = self.n
        l = self.block_size
        m = int(np.floor(n / l))  # number of full non-overlapping blocks

        # Form full non-overlapping blocks
        blocks = [residuals[i * l:(i + 1) * l] for i in range(m)]

        # Sample m blocks with replacement
        sampled_blocks = [blocks[i] for i in np.random.choice(m, size=m, replace=True)]

        # Stitch together sampled blocks and truncate to original length n
        resampled = np.concatenate(sampled_blocks)[:n]
        return resampled

    # Generate the full bootstrap distribution
    def generate_bootstrap_distribution(self, B, lam, level=0.90):
        # Threshold the initial LASSO estimator
        beta_tilde = self.threshold(self.beta_hat)

        # Compute and center residuals
        residuals = self.y - self.X @ beta_tilde
        residuals = self.center_residuals(residuals)

        beta_star = np.zeros((B, self.p))

        # Bootstrap loop
        for b in range(B):
            e_star = self._nonoverlapping_block_resample(residuals)
            y_star = self.X @ beta_tilde + e_star

            model = Lasso(alpha=lam, fit_intercept=self.fit_intercept, max_iter=5000)
            model.fit(self.X, y_star)
            beta_star[b] = self.threshold(model.coef_)  # Apply thresholding again as in CL

        # Compute summary statistics
        mean = beta_star.mean(axis=0)
        std = beta_star.std(axis=0)
        ci_lower, ci_upper = self.compute_ci(beta_star, level)
        ci_length = ci_upper - ci_lower
        coverage = self.compute_coverage(beta_star, level)

        return {
            "beta_star": beta_star,
            "mean": mean,
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_length": ci_length,
            "coverage": coverage
        }
