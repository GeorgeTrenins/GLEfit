* implement simulated annealing and/or simplex optimization

* implement numerical spectral density

* implement harmonic bath discretization ( -> can be used to initialize oscillatory Prony embedders ) 

* pytorch?

Code development
================

 * The config file speciefies an output directory -> set up output accordingly or remove from config handler

 * Currently configuration files do not have an optimization state (rename to optimizer state), whereas checkpoint files do. Allow configuration files to have that section as well, and initialize to None by default

 * Add final result output to the optimization script

 * Add helper script for visualizing the optimization output

 * In data parsing, add option to have a third column to specify the array of weight factors
