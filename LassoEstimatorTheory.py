import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any


class LassoEstimatorTheory:
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

        # Outputs
        self.beta_hat = None
        self.beta_tilde = None
        self.active_set = None
        self.residuals = None
        self.predictions = None
        self.support_size = None
        self.perfect_match = None
        self.test_mse = None

    def _fit_lasso(self, X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
        model = Lasso(alpha=lam, fit_intercept=self.fit_intercept,
                      random_state=self.random_state, max_iter=5000)
        model.fit(X, y)
        return model.coef_

    def _threshold(self, beta: np.ndarray, threshold_level: float) -> np.ndarray:
        return beta * (np.abs(beta) > threshold_level)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lam: float,
        thresholding_level: Optional[float] = None,
        apply_threshold: bool = True,
        support_true: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ):
        # Standardize data
        if self.standardize:
            self.scaler = StandardScaler().fit(X)
            X_scaled = self.scaler.transform(X)
            if X_test is not None:
                X_test = self.scaler.transform(X_test)
        else:
            X_scaled = X

        # Fit LASSO
        self.beta_hat = self._fit_lasso(X_scaled, y, lam)

        # Threshold if required
        if apply_threshold and thresholding_level is not None:
            self.beta_tilde = self._threshold(self.beta_hat, thresholding_level)
        else:
            self.beta_tilde = self.beta_hat

        # Predictions and residuals
        self.predictions = X_scaled @ self.beta_tilde
        self.residuals = y - self.predictions

        # Active set
        self.active_set = np.flatnonzero(self.beta_hat)
        self.support_size = len(self.active_set)

        # Match true support
        if support_true is not None:
            self.perfect_match = set(self.active_set) == set(support_true)

        # Optional test error
        if X_test is not None and y_test is not None:
            y_pred = X_test @ self.beta_tilde
            self.test_mse = np.mean((y_test - y_pred) ** 2)

        return self.beta_hat

    def get_outputs(self) -> Dict[str, Any]:
        return {
            "beta_hat": self.beta_hat,
            "beta_tilde": self.beta_tilde,
            "active_set": self.active_set,
            "support_size": self.support_size,
            "residuals": self.residuals,
            "predictions": self.predictions,
            "perfect_match": self.perfect_match,
            "test_mse": self.test_mse,
        }
