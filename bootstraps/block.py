import numpy as np
from sklearn.linear_model import Lasso
from statsmodels.tsa.stattools import acf
from bootstraps.base import BasisBootstrap
from typing import Optional


class BlockBootstrap(BasisBootstrap):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        beta_hat: np.ndarray,
        a_n: float = 0.5,
        block_length: Optional[int] = None,  # None triggers automatic ACF-based selection
        standardize: bool = True,
        seed: int = 42,
        true_beta: Optional[np.ndarray] = None,
        fit_intercept: bool = False
    ):
        super().__init__(
            X=X,
            y=y,
            beta_hat=beta_hat,
            a_n=a_n,
            standardize=standardize,
            fit_intercept=fit_intercept,
            seed=seed,
            true_beta=true_beta
        )
        self.block_length = block_length

    def generate_bootstrap_distribution(
        self,
        B: int,
        lam: float
    ) -> dict:
        beta_tilde = self.threshold_beta()
        residuals = self.y - self.X @ beta_tilde
        residuals = self.center_residuals(residuals)

        # Determine block length if not fixed
        block_len = self._determine_block_length(residuals) if self.block_length is None else self.block_length

        n, p = self.n, self.p
        beta_star = np.zeros((B, p))

        for b in range(B):
            r_star = self._resample_blocks(residuals, block_len, n)
            y_star = self.X @ beta_tilde + r_star

            lasso = Lasso(alpha=lam, fit_intercept=self.fit_intercept, max_iter=5000)
            lasso.fit(self.X, y_star)
            beta_star[b] = lasso.coef_

        mean = beta_star.mean(axis=0)
        std = beta_star.std(axis=0)
        ci_lower, ci_upper = self.compute_ci(beta_star, level=0.95)
        coverage = self.compute_coverage(beta_star, level=0.95)

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

    def _determine_block_length(self, residuals: np.ndarray) -> int:
        """Estimate block length using ACF fall-off and theoretical rate."""
        acf_vals = acf(residuals, nlags=min(40, self.n // 2), fft=True)
        k = next((i for i, val in enumerate(acf_vals[1:], start=1) if np.abs(val) < 0.1), self.n)
        return min(k, int(np.ceil(self.n ** (1 / 3))))

    def _resample_blocks(self, residuals: np.ndarray, l: int, n: int) -> np.ndarray:
        """Draw non-overlapping blocks with replacement and stitch them together."""
        blocks = [residuals[i:i + l] for i in range(0, n - l + 1, l)]
        num_blocks = int(np.ceil(n / l))
        selected_blocks = np.random.choice(len(blocks), size=num_blocks, replace=True)
        r_star = np.concatenate([blocks[i] for i in selected_blocks])[:n]
        return r_star