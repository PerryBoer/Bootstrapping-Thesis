from GenerateSimulations import GenerateSimulations
from ParralelSimulation import ParallelSimulationGrid


# grid_runner = GenerateSimulations(
#     methods=["naive", "modified", "wild", "block"],
#     lambda_grid=[0.2, 0.4, 0.6, 0.8],
#     alpha_grid=[0.125, 0.25, 0.5, 0.75],
#     signal_types=["strong", "weak", "near-zero"],
#     error_types=["gaussian", "heteroskedastic", "ar1"]
# )
# grid_runner.run_all()


grid = ParallelSimulationGrid(
    methods=["naive", "modified", "wild", "block"],
    lambda_grid=[0.01, 0.025, 0.05, 0.1],
    alpha_grid=[0.125, 0.25, 0.75],
    signal_types=["strong", "weak", "nearzero"],
    error_types=["gaussian", "heteroskedastic", "ar1"],
    n_jobs=4  # use 4 or tune based on Task Manager load
)

results = grid.run()