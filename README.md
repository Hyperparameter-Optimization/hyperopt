# hyperOpt [![Build Status](https://travis-ci.org/Hyperparameter-Optimization/hyperOpt.svg?branch=master)](https://travis-ci.org/Hyperparameter-Optimization/hyperOpt)
Evolutionary algorithms for hyperparameter optimization. In this package Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) are the two selected evolutionary algorithms for the hyperparameter optimization task.



## Package info & Installation

The analysis & comparison tools are collected to the **examples** folder.


To install this package run the following:

````
git clone https://github.com/Hyperparameter-Optimization/hyperOpt.git
cd hyperopt
pip install -r requirements.txt
pip install .
````

Run the unit tests:

````
pytest hyperOpt/tests/.
````

## Possible improvements:

- culling for PSO
- optimal culling point
- optimal amount of elites for GA
- cross-over for the whole gene, not for bits inside the chromosome
- Different cross-over functions for GA (e.g shuffle & uniform cross-over)
- 'migration' for GA
- better stopping criteria for reduction of computing time.