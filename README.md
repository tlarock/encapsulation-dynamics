# Encapsulation Dynamics
This repository contains code implementing the measurements and dynamical processes on hypergraphs described in:

* [LaRock, T. and Lambiotte, R. Encapsulation Structure and Dynamics in Hypergraphs. arXiv preprint. July 2023. arXiv:2307.04613.](https://arxiv.org/abs/2307.04613)

# Requirements
The code was written under python 3.9.15 and relies on standard libraries including `numpy`, `scipy`, `matplotlib`, and `networkx`. It also relies heavily on the `xgi`, the compleX Group Interactions package.

# Installation
This repository does not implement a package, therefore beyond installing the required packages, there is no installation.

# Usage
There are two main ways to run dynamical simulations. The file `run_simulations.py` can be used in conjunction with a configuration file to run simulations from the command line. Alternatively, simulation parameters can be specified directly in a dictionary and passed to one of the `run_*_simulations` functions found in `simulations.py`.

The file `encapsulation_dag.py` implements separate computation of the encapsulation and overlap line graphs. 

*Note:* This README is a work in progress. If you have any questions, open an issue or email me at [larock@maths.ox.ac.uk](mailto:larock@maths.ox.ac.uk).
