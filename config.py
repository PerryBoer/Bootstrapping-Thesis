import numpy as np

class Config:
    """
    Dimensionality-aware configuration for the simulation study.
    Determines s and support based on p. Excludes λ/α placeholder machinery.
    """

    def __init__(self, p, n=250, seed=42):
        self.p = p
        self.n = n
        self.s = 5 if p < 25 else 10
        self.support = list(range(self.s))  # Always first s indices
        self.seed = seed

        # Simulation settings
        self.num_bootstrap = 500
        self.num_mc = 200

        # Confidence interval level
        self.alpha_ci = 0.10
        self.ci_quantiles = [self.alpha_ci / 2, 1 - self.alpha_ci / 2]

        # Signal vectors
        self.signal_vectors = {
            "strong": np.array([5, -4, 10, -6, 2, -5, 6, -3, 7, -1]),
            "weak":   np.array([4.0, -0.25, 0.75, 0.35, 1.0, -0.8, 1, 1.5, -2.0, 0.65]),
            "nearzero": lambda n: np.array([
                1 / np.sqrt(n), -3 / np.sqrt(n), 1, 0.35, 0.8, -2,
                -2 / np.sqrt(n), -0.5, -2.0, 1.2 / np.sqrt(n)
            ])
        }

        # Supported configurations
        self.error_types = ["gaussian", "heteroskedastic", "ar1"]
        self.bootstrap_methods = ["naive", "modified", "wild", "block"]

    def get_threshold(self, alpha_th: float) -> float:
        return alpha_th * np.sqrt(np.log(self.p) / self.n)

    def set_random_seed(self, seed=None):
        np.random.seed(self.seed if seed is None else seed)
