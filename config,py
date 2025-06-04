import numpy as np

class Config:
    """
    Shared configuration for the simulation study, including constants, thresholds,
    signal vectors, bootstrap method names, and lambda grid.
    """

    # constant
    n = 200
    p = 300
    s = 10
    num_bootstrap = 500
    num_mc = 1000
    alpha_ci = 0.10  # confidence level
    ci_quantiles = [alpha_ci / 2, 1 - alpha_ci / 2]  # [0.05, 0.95]

    # fixed support indices
    support = [5, 20, 45, 70, 100, 130, 160, 200, 240, 280]

    # thresholding constants
    alpha_th_vals = [0.125, 0.25, 0.75]

    # compute thresholds based on alpha values
    @staticmethod
    def get_threshold(alpha_th):
        return alpha_th * np.sqrt(np.log(Config.p) / Config.n)

    # signal strength vectors
    signal_vectors = {
        "strong": np.array([5, -4, 10, -6, 2, -5, 6, -3, 7, -1]),
        "weak": np.array([4.0, -0.25, 0.75, 0.35, 1.0, -0.8, 1, 1.5, -2.0, 0.65]),
        "nearzero": lambda n: np.array([
            1 / np.sqrt(n), -3 / np.sqrt(n), 1, 0.35, 0.8, -2,
            -2 / np.sqrt(n), -0.5, -2.0, 1.2 / np.sqrt(n)
        ])
    }

    # lambda grid 
    lambda_grid = np.logspace(-3, 1, 50)

    # methods and error structures
    error_types = ["gaussian", "heteroskedastic", "ar1"]
    bootstrap_methods = ["naive", "cl", "wild", "block"]

    # For reproducibility
    seed = 42

    @staticmethod
    def set_random_seed(seed=None):
        if seed is None:
            seed = Config.seed
        np.random.seed(seed)
