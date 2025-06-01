import numpy as np
from typing import Literal, Optional

class DGP:
    def __init__(
        self,
        n: int,
        p: int,
        s: int,
        signal_type: Literal["strong", "weak", "near_zero", "custom"] = "strong",
        error_type: Literal["gaussian", "heteroskedastic", "ar1"] = "gaussian",
        rho_cov_X: float = 0.0,
        fixed_X: Optional[np.ndarray] = None,
        seed: int = 42
    ):
        self.n = n
        self.p = p
        self.s = s
        self.signal_type = signal_type
        self.error_type = error_type
        self.rho = rho_cov_X
        self.fixed_X = fixed_X
        self.seed = seed
        np.random.seed(seed)

    def _generate_beta(self):
        beta = np.zeros(self.p)
        if self.signal_type == "strong":
            values = np.random.choice([3, 5, -3, -5], size=self.s)
        elif self.signal_type == "weak":
            values = np.random.choice([0.25, 0.5, -0.25, -0.5], size=self.s)
        elif self.signal_type == "near_zero":
            scale = 1 / np.sqrt(self.n)
            values = np.random.normal(0, scale, self.s)
        elif self.signal_type == "custom":
            values = np.random.uniform(1, 2, self.s) * np.random.choice([-1, 1], self.s)
        else:
            raise ValueError(f"Invalid signal_type: {self.signal_type}")
        beta[:self.s] = values
        return beta

    def _generate_X(self):
        if self.fixed_X is not None:
            return self.fixed_X
        if self.rho == 0.0:
            return np.random.normal(0, 1, size=(self.n, self.p))
        cov = self.rho ** np.abs(np.subtract.outer(np.arange(self.p), np.arange(self.p)))
        return np.random.multivariate_normal(np.zeros(self.p), cov, size=self.n)

    def _generate_errors(self, X):
        u = np.random.normal(0, 1, self.n)

        if self.error_type == "gaussian":
            return u

        elif self.error_type == "heteroskedastic":
            sigma = 0.5 + 0.5 * np.abs(X @ np.random.uniform(-1, 1, self.p))
            return sigma * u

        elif self.error_type == "ar1":
            errors = np.zeros(self.n)
            phi = 0.5
            errors[0] = u[0]
            for t in range(1, self.n):
                errors[t] = phi * errors[t - 1] + u[t]
            return errors

        else:
            raise ValueError(f"Unknown error_type: {self.error_type}")

    def generate(self) -> dict:
        X = self._generate_X()
        beta = self._generate_beta()
        errors = self._generate_errors(X)
        y = X @ beta + errors
        support = np.flatnonzero(beta)
        snr = np.var(X @ beta) / np.var(errors)

        return {
            "X": X,
            "y": y,
            "beta": beta,
            "errors": errors,
            "support": support,
            "snr": snr
        }



# # Example usage
# dgp = DGP(
#     n=200,
#     p=100,
#     s=10,
#     signal_type="strong",
#     error_type="heteroskedastic",
#     rho_cov_X=0.2,
#     seed=42
# )

# data = dgp.generate()

# # Unpack for inspection
# X = data["X"]
# y = data["y"]
# beta = data["beta"]
# errors = data["errors"]
# support = data["support"]
# snr = data["snr"]

# # Print diagnostics
# print("X shape:", X.shape)
# print("y shape:", y.shape)
# print("beta shape:", beta.shape)
# print("Support indices:", support)
# print("First 5 rows of X:\n", X[:5])
# print("First 5 values of y:", y[:5])
# print("First 5 values of beta:", beta[:5])
# print("First 5 values of errors:", errors[:5])
# print("SNR:", snr)