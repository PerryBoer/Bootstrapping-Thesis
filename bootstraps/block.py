import numpy as np
from sklearn.linear_model import Lasso


class BlockBootstrap:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        beta_hat: np.ndarray,
        beta_true: np.ndarray,
        a_n: float,
        fit_intercept: bool = False
    ):
        """
        X, y must already be centered & scaled (as in DGP.generate()).
        beta_hat: initial Lasso estimate from Lasso(alpha=lam, fit_intercept=False)
        beta_true: ground truth
        a_n: threshold sequence (e.g. n**(-1/3))
        """
        self.X = X
        self.y = y
        self.beta_hat = beta_hat
        self.beta_true = beta_true
        self.a_n = a_n
        self.fit_intercept = fit_intercept
        self.n, self.p = X.shape

    def threshold(self, beta: np.ndarray) -> np.ndarray:
        """Hard‐threshold at ±a_n."""
        return beta * (np.abs(beta) > self.a_n)

    def center_residuals(self, residuals: np.ndarray) -> np.ndarray:
        """Make residuals mean‐zero."""
        return residuals - residuals.mean()

    def _block_resample(self, resid: np.ndarray) -> np.ndarray:
        """
        Non‐overlapping block resampling of residuals.
        Block size ≈ n^(1/3), sample blocks with replacement.
        """
        b = max(1, int(self.n ** (1/3)))
        m = int(np.ceil(self.n / b))
        starts = np.random.randint(0, self.n - b + 1, size=m)
        blocks = [resid[s:(s + b)] for s in starts]
        return np.concatenate(blocks)[:self.n]

    def compute_ci(
        self,
        T_star: np.ndarray,
        beta_tilde: np.ndarray,
        level: float = 0.90
    ):
        """Quantile‐based CIs, same as ModifiedBootstrap."""
        q_lower = np.percentile(T_star, (1 + level) / 2 * 100, axis=0)
        q_upper = np.percentile(T_star, (1 - level) / 2 * 100, axis=0)
        ci_lower = beta_tilde - q_lower / np.sqrt(self.n)
        ci_upper = beta_tilde - q_upper / np.sqrt(self.n)
        return ci_lower, ci_upper

    def generate_bootstrap_distribution(
        self,
        B: int,
        lam: float,
        level: float = 0.90
    ):
        # 1. Threshold original estimator
        beta_tilde = self.threshold(self.beta_hat)

        # 2. Compute & center residuals
        resid = self.y - self.X @ beta_tilde
        resid = self.center_residuals(resid)

        # 3. Block‐bootstrap draws
        beta_star = np.zeros((B, self.p))
        for b in range(B):
            e_star = self._block_resample(resid)
            y_star = self.X @ beta_tilde + e_star

            model = Lasso(alpha=lam,
                          fit_intercept=self.fit_intercept,
                          max_iter=10000)
            model.fit(self.X, y_star)
            beta_star[b, :] = self.threshold(model.coef_)

        # 4. Studentized deviations
        T_star = np.sqrt(self.n) * (beta_star - beta_tilde)

        # 5. CIs & coverage
        ci_lower, ci_upper = self.compute_ci(T_star, beta_tilde, level)
        coverage = ((self.beta_true >= ci_lower) & (self.beta_true <= ci_upper)).astype(int)
        ci_length = ci_upper - ci_lower

        return {
            "beta_star": beta_star,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_length": ci_length,
            "coverage": coverage,
            "mean": beta_star.mean(axis=0),
            "std": beta_star.std(axis=0),
            "var": beta_star.var(axis=0),
        }






