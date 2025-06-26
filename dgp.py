import numpy as np
from typing import Literal
from config import Config

class DGP:
    def __init__(
        self,
        config: Config,
        signal_type: Literal["strong", "weak", "nearzero"] = "strong",
        error_type: Literal["gaussian", "heteroskedastic", "ar1"] = "gaussian",
    ):
        self.config = config
        self.n = config.n
        self.p = config.p
        self.s = config.s
        self.support = config.support
        self.signal_type = signal_type
        self.error_type = error_type

    def generate_X(self):
        return np.random.normal(0, 1, size=(self.n, self.p))

    def generate_beta(self):
        beta = np.zeros(self.p)
        if self.signal_type == "nearzero":
            values = self.config.signal_vectors["nearzero"](self.n)
        else:
            values = self.config.signal_vectors[self.signal_type]
        beta[self.support] = values[: self.s]
        return beta

    def generate_errors(self, X):
        u = np.random.normal(0, 1, self.n)
        if self.error_type == "gaussian":
            return u
        elif self.error_type == "heteroskedastic":
            w = np.random.uniform(-1, 1, self.p)
            sigma = 0.5 + 0.5 * np.abs(X @ w)
            return sigma * u
        elif self.error_type == "ar1":
            eps = np.zeros(self.n)
            phi = 0.9
            sigma_u = np.sqrt(1 - phi**2)
            innovations = np.random.normal(0, sigma_u, self.n)
            eps[0] = innovations[0]
            for t in range(1, self.n):
                eps[t] = phi * eps[t - 1] + innovations[t]
            return eps
        else:
            raise ValueError(f"Unknown error_type: {self.error_type}")

    def generate(self) -> dict:
        X = self.generate_X()
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
        beta = self.generate_beta()
        eps = self.generate_errors(X)
        y = X @ beta + eps
        y = y - np.mean(y)
        snr = np.var(X @ beta) / np.var(eps)
        return {
            "X": X,
            "y": y,
            "beta": beta,
            "errors": eps,
            "support": self.support,
            "snr": snr,
        }