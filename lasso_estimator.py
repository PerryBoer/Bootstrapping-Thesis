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
        self.support_size = None
        self.perfect_match = None
        self.test_mse = None

    def _fit_core(self, X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
        model = Lasso(alpha=lam, fit_intercept=self.fit_intercept,
                      random_state=self.random_state, max_iter=5000)
        model.fit(X, y)
        return model.coef_

    def _threshold_beta(self, beta: np.ndarray, a_n: float) -> np.ndarray:
        return beta * (np.abs(beta) > a_n)

    def fit_lasso(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lam: float,
        a_n: float,
        support_true: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit LASSO and store thresholded estimator, active set, predictions,
        and optionally test MSE and perfect match indicator.
        """
        if self.standardize:
            self.scaler = StandardScaler().fit(X)
            X_scaled = self.scaler.transform(X)
            if X_test is not None:
                X_test = self.scaler.transform(X_test)
        else:
            X_scaled = X

        self.beta_hat = self._fit_core(X_scaled, y, lam)
        self.beta_tilde = self._threshold_beta(self.beta_hat, a_n)
        self.active_set = np.flatnonzero(self.beta_hat)
        self.support_size = len(self.active_set)

        self.predictions = X_scaled @ self.beta_tilde
        self.residuals = y - self.predictions

        if support_true is not None:
            self.perfect_match = set(self.active_set) == set(support_true)

        if X_test is not None and y_test is not None:
            y_pred = X_test @ self.beta_tilde
            self.test_mse = np.mean((y_test - y_pred) ** 2)

        return self.beta_hat

    def get_active_set(self) -> Optional[np.ndarray]:
        return self.active_set

    def get_thresholded_beta(self) -> Optional[np.ndarray]:
        return self.beta_tilde

    def get_bootstrap_replicates(self) -> Optional[np.ndarray]:
        return self.bootstrap_betas

    def get_test_mse(self) -> Optional[float]:
        return self.test_mse

    def get_support_size(self) -> Optional[int]:
        return self.support_size

    def is_perfect_match(self) -> Optional[bool]:
        return self.perfect_match

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
        Select lambda via bootstrap MSE minimization using residual resampling.
        Uses method-dependent multiplier logic:
        - 'rademacher': wild bootstrap (±1)
        - 'normal': standard residual resampling (i.i.d. errors)
        """
        if self.standardize:
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)

        n, p = X.shape
        mse_per_lambda = []

        for lam in lambdas:
            beta_hat = self._fit_core(X, y, lam)
            beta_tilde = self._threshold_beta(beta_hat, a_n)
            residuals = y - X @ beta_tilde

            beta_b = np.zeros((B, p))
            for b in range(B):
                if multiplier == "rademacher":
                    v = np.random.choice([-1, 1], size=n)
                    y_star = X @ beta_tilde + residuals * v
                elif multiplier == "normal":
                    e_star = np.random.choice(residuals, size=n, replace=True)
                    y_star = X @ beta_tilde + e_star
                else:
                    raise ValueError("Unknown multiplier type")

                beta_b[b] = self._fit_core(X, y_star, lam)

            mse = np.mean(np.linalg.norm(beta_b - beta_tilde, axis=1) ** 2)
            mse_per_lambda.append(mse)

        best_idx = np.argmin(mse_per_lambda)
        best_lambda = lambdas[best_idx]

        if return_full_path:
            return best_lambda, mse_per_lambda
        return best_lambda

