from ParralelSimulation import ParallelSimulationGrid
import time
# start_time = time.time()

# simgrid = ParallelSimulationGrid(
#     methods=["naive", "modified", "wild", "block"],
#     signal_types=["strong", "weak", "nearzero"],	
#     error_types=["gaussian", "heteroskedastic", "ar1"],
#     tracked_indices=[5, 22],
#     n_jobs=4
# )
# simgrid.run()

# print(f"Total computation time: {time.time() - start_time:.2f} seconds")





simgrid = ParallelSimulationGrid(
    methods=["naive", "modified", "wild", "block"],
    signal_types=["strong", "weak", "nearzero"],	
    error_types=["ar1"],
    tracked_indices=[5, 22],
    n_jobs=4
)
simgrid.run()

