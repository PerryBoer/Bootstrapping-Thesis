import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Tuple

class LassoEstimator:
    def __init__(
        self,
        standardize: bool = True,
        fit_intercept: bool = False,
        random_state: int = 42
    ):
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.scaler = None

        # Store results
        self.beta_hat = None
        self.beta_tilde = None
        self.active_set = None
        self.residuals = None
        self.predictions = None
        self.bootstrap_betas = None

    def _fit_core(self, X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
        model = Lasso(alpha=lam, fit_intercept=self.fit_intercept,
                      random_state=self.random_state, max_iter=5000)
        model.fit(X, y)
        return model.coef_

    def _threshold_beta(self, beta: np.ndarray, a_n: float) -> np.ndarray:
        return beta * (np.abs(beta) > a_n)

    def fit_lasso(self, X: np.ndarray, y: np.ndarray, lam: float, a_n: float) -> np.ndarray:
        if self.standardize:
            self.scaler = StandardScaler().fit(X)
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        self.beta_hat = self._fit_core(X_scaled, y, lam)
        self.beta_tilde = self._threshold_beta(self.beta_hat, a_n)
        self.active_set = np.flatnonzero(self.beta_hat)
        self.predictions = X_scaled @ self.beta_tilde
        self.residuals = y - self.predictions
        return self.beta_hat

    def get_active_set(self) -> Optional[np.ndarray]:
        return self.active_set

    def get_thresholded_beta(self) -> Optional[np.ndarray]:
        return self.beta_tilde

    def get_bootstrap_replicates(self) -> Optional[np.ndarray]:
        return self.bootstrap_betas

    def select_lambda_bootstrap_mse(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lambdas: np.ndarray,
        B: int = 100,
        a_n: float = 0.05,
        multiplier: str = "rademacher",
        return_full_path: bool = False
    ) -> float | Tuple[float, List[float]]:
        """
        Selects lambda via bootstrap MSE minimization using residual resampling
        """
        if self.standardize:
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)

        n, p = X.shape
        mse_per_lambda = []

        for lam in lambdas:
            # Step 1: Fit original LASSO + threshold
            beta_hat = self._fit_core(X, y, lam)
            beta_tilde = self._threshold_beta(beta_hat, a_n)
            residuals = y - X @ beta_tilde

            # Step 2: Generate bootstrap estimates around beta_tilde
            beta_b = np.zeros((B, p))

            for b in range(B):
                if multiplier == "rademacher":
                    v = np.random.choice([-1, 1], size=n)
                elif multiplier == "normal":
                    v = np.random.normal(0, 1, size=n)
                else:
                    raise ValueError("Unknown multiplier type")

                y_star = X @ beta_tilde + residuals * v
                beta_b[b] = self._fit_core(X, y_star, lam)

            mse = np.mean(np.linalg.norm(beta_b - beta_tilde, axis=1) ** 2)
            mse_per_lambda.append(mse)

        best_idx = np.argmin(mse_per_lambda)
        best_lambda = lambdas[best_idx]

        if return_full_path:
            return best_lambda, mse_per_lambda
        return best_lambda


