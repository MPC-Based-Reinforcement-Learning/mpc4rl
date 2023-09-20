# rlmpc


## Installation

The repository depends on the following submodules
- [acados](https://docs.acados.org/index.html) for generating MPC solvers
- [stable baselines 3](https://stable-baselines3.readthedocs.io/en/master/) for implementations of RL algorithms
- [gymnasium](https://gymnasium.farama.org/) for reference environments.

Initialize submodules via running
``` bash
git submodule update --recursive --init
```

Create python virtual environment

``` bash
    sudo pip3 install virtualenv
    cd <PATH_TO_VENV_DIRECTORY>
    virtualenv rlmpc_venv --python=/usr/bin/python3.10
    source rlmpc_venv/bin/activate
```

Install acados interface to rlmpc_venv

``` bash
    python -m pip install ~/software/acados/interfaces/acados_template
```
