from generate_simulation_results import GenerateSimulationResults

generator = GenerateSimulationResults(
    signal_types=["strong", "weak", "nearzero"],
    error_types=["gaussian", "heteroskedastic", "ar1"],
    methods=["naive", "modified", "wild", "block"],
    alpha_vals=[0.25, 0.5, 0.75],
    level=0.90
)

generator.run_all()

# https://glmnet.stanford.edu/articles/glmnet.html#ref-block # for lmambda grid selection 