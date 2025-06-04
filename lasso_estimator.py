import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, List
from config import Config


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

        # outputs
        self.beta_hat = None
        self.beta_tilde = None
        self.active_set = None
        self.residuals = None
        self.predictions = None
        self.support_size = None
        self.perfect_match = None
        self.test_mse = None

    # fit LASSO model and return coefficients based on sklearn's Lasso implementation
    def _fit_lasso(self, X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
        model = Lasso(alpha=lam, fit_intercept=self.fit_intercept,
                      random_state=self.random_state, max_iter=5000)
        model.fit(X, y)
        return model.coef_

    # thresholding function to apply after LASSO fit --- core functionality
    def _threshold(self, beta: np.ndarray, a_n: float) -> np.ndarray:
        return beta * (np.abs(beta) > a_n)

    # main fitting function that applies LASSO and thresholdinh
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lam: float,
        a_n: float,
        threshold: bool = True,
        support_true: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ):
        # Standardize the data if set to true
        if self.standardize:
            self.scaler = StandardScaler().fit(X)
            X_scaled = self.scaler.transform(X)
            if X_test is not None:
                X_test = self.scaler.transform(X_test)
        else:
            X_scaled = X

        # fit beta hat
        self.beta_hat = self._fit_lasso(X_scaled, y, lam)

        # apply thresholding if required -- not needed for naive
        self.beta_tilde = self._threshold(self.beta_hat, a_n) if threshold else self.beta_hat

        # compute predictions and residuals
        self.predictions = X_scaled @ self.beta_tilde
        self.residuals = y - self.predictions

        # active set and support size
        self.active_set = np.flatnonzero(self.beta_hat)
        self.support_size = len(self.active_set)

        # check if perfect match with true support
        if support_true is not None:
            self.perfect_match = set(self.active_set) == set(support_true)

        # if test data is provided, compute test MSE
        if X_test is not None and y_test is not None:
            y_pred = X_test @ self.beta_tilde
            self.test_mse = np.mean((y_test - y_pred) ** 2)

        return self.beta_hat

    def select_lambda_bootstrap_mse(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lambdas: np.ndarray,
        B: int,
        a_n: float,
        method: str = "cl",  # 'naive', 'wild', 'cl'
        multipliers: Optional[np.ndarray] = None
    ) -> Tuple[float, List[float]]:
        """
        Select lambda* via bootstrap MSE. Applies thresholding for modified/wild. Skips for naive.
        Block bootstrap must be handled externally (different y_star handling due to resampling blocks).
        """
        if self.standardize:
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)

        # store the number of samples and features
        n, p = X.shape

        # placeholder for MSE values
        mse_list = []

        # go over each lambda value in the grid
        for lam in lambdas:

            # fit beta for current lambda
            beta_hat = self._fit_lasso(X, y, lam)

            # apply thresholding if needed
            beta_center = self._threshold(beta_hat, a_n) if method in ["modified", "wild", "block"] else beta_hat

            # compute residuals
            residuals = y - X @ beta_center

            # bootstrap placeholder
            beta_boot = np.zeros((B, p))

            for b in range(B):
                # generate bootstrap responses per method
                if method == "wild":
                    v = multipliers[b] if multipliers is not None else np.random.choice([-1, 1], size=n)
                    y_star = X @ beta_center + residuals * v
                elif method in ["naive", "modified"]:
                    e_star = np.random.choice(residuals, size=n, replace=True)
                    y_star = X @ beta_center + e_star
                elif method == "block":
                    block_size = int(np.floor(Config.n ** (1 / 3)))
                    m = n // block_size
                    blocks = [residuals[i * block_size:(i + 1) * block_size] for i in range(m)]
                    sampled = [blocks[i] for i in np.random.choice(m, m, replace=True)]
                    e_star = np.concatenate(sampled)[:n]
                    y_star = X @ beta_center + e_star
                else:
                    raise ValueError(f"Unknown bootstrap method: {method}")

                # fit LASSO on bootstrap sample
                beta_boot[b] = self._fit_lasso(X, y_star, lam)

            
            # calculate the MSE between bootstrap estimates and the centered beta
            mse = np.mean(np.linalg.norm(beta_boot - beta_center, axis=1) ** 2)
            mse_list.append(mse)

        # return lambda with minimum MSE and the list of MSEs
        best_idx = np.argmin(mse_list)
        return lambdas[best_idx], mse_list
