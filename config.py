import numpy as np

class Config:
    """
    Central configuration object for the simulation study.
    Encodes DGP settings, bootstrap controls, thresholding, and λ₀ theory-aligned grid.
    """

    # Data dimensions
    n = 200
    p = 300
    s = 10

    # Simulation controls
    num_bootstrap = 500
    num_mc = 100
    seed = 42

    # Confidence interval level
    alpha_ci = 0.10
    ci_quantiles = [alpha_ci / 2, 1 - alpha_ci / 2]

    # True support (fixed for all reps)
    support = [5, 20, 45, 70, 100, 130, 160, 200, 240, 280]

    # Thresholding constants
    alpha_th_vals = [0.125, 0.25, 0.75]

    @staticmethod
    def get_threshold(alpha_th: float) -> float:
        return alpha_th * np.sqrt(np.log(Config.p) / Config.n)

    # Signal configurations
    signal_vectors = {
        "strong": np.array([5, -4, 10, -6, 2, -5, 6, -3, 7, -1]),
        "weak":   np.array([4.0, -0.25, 0.75, 0.35, 1.0, -0.8, 1, 1.5, -2.0, 0.65]),
        "nearzero": lambda n: np.array([
            1 / np.sqrt(n), -3 / np.sqrt(n), 1, 0.35, 0.8, -2,
            -2 / np.sqrt(n), -0.5, -2.0, 1.2 / np.sqrt(n)
        ])
    }

    # Theory-backed λ₀ grid (used to compute λₙ = λ₀ √(log p / n))
    lambda0_grid = [0.25, 0.5, 1.0, 2.0]

    @staticmethod
    def get_lambda_from_lambda0(lambda0: float) -> float:
        return lambda0 * np.sqrt(np.log(Config.p) / Config.n)

    # Alternative: empirical λ grid (for bootstrap-MSE etc)
    lambda_grid_mse = np.logspace(-3, 1, 50)  # fallback if needed for diagnostics

    # Supported methods
    error_types = ["gaussian", "heteroskedastic", "ar1"]
    bootstrap_methods = ["naive", "cl", "wild", "block"]

    @staticmethod
    def set_random_seed(seed=None):
        if seed is None:
            seed = Config.seed
        np.random.seed(seed)
