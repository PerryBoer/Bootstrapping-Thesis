The code is structured as follows.

First, all bootstrapped methods are modularized in classes, which makes debugging and data/code control easier. 

The data-generating process class is found under DGP.py which contains the DGP Class, which uses the Config class which contains the hyperparameters for sample size, sparsity, signal strenght.

Simulation.py contains the code for running a simulation, whereas the montecarlorunner.py uses this to aggregate the results. Lastly, the notebooks are used for visualisation and tables.

All Monte Carlo runs are run in parrallel. 