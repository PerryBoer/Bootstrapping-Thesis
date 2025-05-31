import numpy as np
from sklearn.linear_model import Lasso
from statsmodels.tsa.stattools import acf
from bootstraps.base import BasisBootstrap


class BlockBootstrap(BasisBootstrap):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        beta_hat: np.ndarray,
        a_n: float = 0.5,
        block_length: Optional[int] = None,
        standardize: bool = True,
        seed: int = 42
    ):
        super().__init__(X, y, beta_hat, a_n, standardize, seed)
        self.block_length = block_length  # If None, it will be determined from ACF

    def generate_bootstrap_distribution(
        self,
        B: int,
        lam: float
    ) -> np.ndarray:
        beta_tilde = self.threshold_beta()
        residuals = self.y - self.X @ beta_tilde
        residuals = self.center_residuals(residuals)

        # Choose block length from ACF if not given
        block_len = self._determine_block_length(residuals) if self.block_length is None else self.block_length
        n, p = self.n, self.p
        beta_star = np.zeros((B, p))

        for b in range(B):
            r_star = self._resample_blocks(residuals, block_len, n)
            y_star = self.X @ beta_tilde + r_star

            lasso = Lasso(alpha=lam, fit_intercept=False)
            lasso.fit(self.X, y_star)
            beta_star[b] = lasso.coef_

        return beta_star

    def _determine_block_length(self, residuals: np.ndarray) -> int:
        acf_vals = acf(residuals, nlags=min(40, self.n // 2), fft=True)
        k = next((i for i, val in enumerate(acf_vals[1:], start=1) if np.abs(val) < 0.1), self.n)
        return min(k, int(np.ceil(self.n ** (1/3))))

    def _resample_blocks(self, residuals: np.ndarray, l: int, n: int) -> np.ndarray:
        # Use non-overlapping blocks of length l
        blocks = [residuals[i:i + l] for i in range(0, n - l + 1, l)]
        num_blocks = int(np.ceil(n / l))
        selected_blocks = np.random.choice(len(blocks), size=num_blocks, replace=True)
        r_star = np.concatenate([blocks[i] for i in selected_blocks])[:n]
        return r_star
