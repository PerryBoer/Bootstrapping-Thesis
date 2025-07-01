import numpy as np
from sklearn.linear_model import Lasso

import numpy as np
from sklearn.linear_model import Lasso

class WildBootstrap:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        beta_hat: np.ndarray,
        beta_true: np.ndarray,
        a_n: float,
        lam: float,
        weight: str = "rademacher",
        fit_intercept: bool = False
    ):
        self.X = X
        self.y = y
        self.beta_hat = beta_hat
        self.beta_true = beta_true
        self.a_n = a_n
        self.lam = lam
        self.weight = weight
        self.fit_intercept = fit_intercept
        self.n, self.p = X.shape

    def threshold(self, beta: np.ndarray) -> np.ndarray:
        """Apply hard-threshold at ±a_n."""
        return beta * (np.abs(beta) > self.a_n)

    def center_residuals(self, resid: np.ndarray) -> np.ndarray:
        """Center residuals to have mean zero."""
        return resid - np.mean(resid)

    def _draw_weights(self) -> np.ndarray:
        """Draw wild-bootstrap weights according to specified distribution."""
        if self.weight == "rademacher":
            return np.random.choice([-1, 1], size=self.n)

        if self.weight == "mammen":
            # Mammen (1992) two-point distribution:
            s5 = np.sqrt(5)
            w1 = (1 + s5) / 2
            w2 = (1 - s5) / 2
            p1 = (s5 + 1) / (2 * s5)
            p2 = 1 - p1
            return np.random.choice([w1, w2], size=self.n, p=[p1, p2])

        if self.weight == "normal":
            # Standard normal weights
            return np.random.randn(self.n)

        raise ValueError(f"Unknown weight type: {self.weight}")

    def compute_ci(
        self,
        T_star: np.ndarray,
        beta_tilde: np.ndarray,
        level: float = 0.90
    ):
        """Quantile-based confidence intervals."""
        alpha_low = (1 + level) / 2 * 100
        alpha_high = (1 - level) / 2 * 100
        q_low = np.percentile(T_star, alpha_low, axis=0)
        q_high = np.percentile(T_star, alpha_high, axis=0)
        ci_lower = beta_tilde - q_low / np.sqrt(self.n)
        ci_upper = beta_tilde - q_high / np.sqrt(self.n)
        return ci_lower, ci_upper

    def generate_bootstrap_distribution(
        self,
        B: int,
        lam: float,
        level: float = 0.90
    ):
        # 1. Hard-threshold the initial estimate
        beta_tilde = self.threshold(self.beta_hat)

        # 2. Compute and center residuals
        resid = self.y - self.X @ beta_tilde
        resid = self.center_residuals(resid)

        # 3. Draw bootstrap distributions
        beta_star = np.zeros((B, self.p))
        for b in range(B):
            v = self._draw_weights()
            y_star = self.X @ beta_tilde + v * resid
            model = Lasso(alpha=self.lam, fit_intercept=self.fit_intercept, max_iter=10000)
            model.fit(self.X, y_star)
            beta_star[b, :] = self.threshold(model.coef_)

        # 4. Studentized deviations
        T_star = np.sqrt(self.n) * (beta_star - beta_tilde)

        # 5. Confidence intervals and coverage
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

