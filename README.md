# Note on Maintenance Status

This repository resulted from an initial project exploring the combination of Model Predictive Control (MPC) and Reinforcement Learning (RL). It is no longer actively maintained.

For recent developments and software implementations combining MPC and RL, please visit:
- [leap-c](https://github.com/leap-c/leap-c) - A newer project enabled by
- [differentiable acados](https://github.com/FreyJo/differentiable_nmpc) - The underlying differentiable NMPC framework (ocp solver)



# rlmpc

Reinforcement Learning with Model Predictive Control as function approximator.

![Overview](/assets/img/overview.png "Overview")

## Installation

Create python virtual environment

``` bash
    sudo pip3 install virtualenv
    cd <PATH_TO_VENV_DIRECTORY>
    virtualenv rlmpc_venv --python=/usr/bin/python3.11
    source rlmpc_venv/bin/activate
    python -m pip install -e .
```

## Dependencies

The repository depends on the following packages:

- [casadi](https://web.casadi.org/) for symbolic computations
- [acados](https://docs.acados.org/index.html) for generating MPC solvers
- [stable baselines 3](https://stable-baselines3.readthedocs.io/en/master/) for implementations of RL algorithms and utilities.
- [gymnasium](https://gymnasium.farama.org/) for reference environments.

assuming the binary python3.11 exists. Replace with some other 3.8+ version possible.

Install acados as described [here](https://github.com/acados/acados) and the python interface to rlmpc_venv following the instructions in [acados](https://docs.acados.org/python_interface/index.html). All other dependencies are being handled automatically.

## Usage

Try an example with the following command:

``` bash
    python examples/linear_system_mpc_qlearning.py
```
