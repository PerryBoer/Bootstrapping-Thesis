import numpy as np
from typing import Literal
from config import Config 


class DGP:
    def __init__(
        self,
        signal_type: Literal["strong", "weak", "nearzero"] = "strong",
        error_type: Literal["gaussian", "heteroskedastic", "ar1"] = "gaussian",
        seed: int = None,
    ):
        self.n = Config.n
        self.p = Config.p
        self.s = Config.s
        self.support = Config.support
        self.signal_type = signal_type
        self.error_type = error_type

        self.seed = seed or Config.seed
        np.random.seed(self.seed)

    def generate_X(self):
        """generate design matrix"""
        return np.random.normal(0, 1, size=(self.n, self.p))

    def generate_beta(self):
        """generate sparse coefficient vector beta based on fixed support and signal type"""
        beta = np.zeros(self.p)
        if self.signal_type == "nearzero":
            values = Config.signal_vectors["nearzero"](self.n)
        else:
            values = Config.signal_vectors[self.signal_type]
        beta[np.array(self.support)] = values
        return beta

    def generate_errors(self, X):
        """generate error vector based on specified structure"""
        u = np.random.normal(0, 1, self.n)

        if self.error_type == "gaussian":
            return u

        elif self.error_type == "heteroskedastic":
            w = np.random.uniform(-1, 1, self.p)
            sigma = 0.5 + 0.5 * np.abs(X @ w)
            return sigma * u

        elif self.error_type == "ar1":
            eps = np.zeros(self.n)
            phi = 0.5
            eps[0] = u[0]
            for t in range(1, self.n):
                eps[t] = phi * eps[t - 1] + u[t]
            return eps

        else:
            raise ValueError(f"Unknown error_type: {self.error_type}")

    def generate(self) -> dict:
        """DGP pipeline"""
        X = self.generate_X()
        beta = self.generate_beta()
        eps = self.generate_errors(X)
        y = X @ beta + eps
        snr = np.var(X @ beta) / np.var(eps)

        return {
            "X": X,
            "y": y,
            "beta": beta,
            "errors": eps,
            "support": self.support,
            "snr": snr,
        }
