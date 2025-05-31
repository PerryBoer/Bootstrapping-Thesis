import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple


class BasisBootstrap(ABC):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        beta_hat: np.ndarray,
        a_n: float = 0.5,
        standardize: bool = True,
        fit_intercept: bool = False,
        seed: int = 42,
        true_beta: Optional[np.ndarray] = None  # For coverage computation (optional)
    ):
        self.X = X
        self.y = y
        self.beta_hat = beta_hat.copy()
        self.a_n = a_n
        self.seed = seed
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.true_beta = true_beta
        self.scaler = None
        self.n, self.p = X.shape

        if self.standardize:
            self.scaler = StandardScaler().fit(X)
            self.X = self.scaler.transform(X)

        np.random.seed(seed)

    def threshold_beta(self) -> np.ndarray:
        beta_mod = self.beta_hat.copy()
        beta_mod[np.abs(beta_mod) < self.a_n] = 0.0
        return beta_mod

    def center_residuals(self, residuals: np.ndarray) -> np.ndarray:
        return residuals - np.mean(residuals)

    def compute_ci(
        self,
        beta_star: np.ndarray,
        level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        alpha = (1 - level) / 2
        lower = np.percentile(beta_star, 100 * alpha, axis=0)
        upper = np.percentile(beta_star, 100 * (1 - alpha), axis=0)
        return lower, upper

    def compute_coverage(
        self,
        beta_star: np.ndarray,
        level: float = 0.95
    ) -> Optional[np.ndarray]:
        if self.true_beta is None:
            return None
        lower, upper = self.compute_ci(beta_star, level)
        return (self.true_beta >= lower) & (self.true_beta <= upper)

    @abstractmethod
    def generate_bootstrap_distribution(
        self,
        B: int,
        lam: float
    ) -> dict:
        pass

