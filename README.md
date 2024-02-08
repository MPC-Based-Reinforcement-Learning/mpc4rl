# rlmpc

![Overview](/assets/img/overview.png "Overview")

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
    virtualenv rlmpc_venv --python=/usr/bin/python3.11
    source rlmpc_venv/bin/activate
    python -m pip install -e .
```

assuming the binary python3.11 exists. Replace with some other 3.8+ version possible.


(Necessary for development?): With (rlmpc_venv) active, install stable_baselines3 with optional dependencies and make it editable

``` bash
    python -m pip install -e <rlmpc_root>/external/stable_baselines3/.[extra,tests,docs]
```


Install acados interface to rlmpc_venv following the instructions in [acados](https://docs.acados.org/python_interface/index.html)

``` bash
    python -m pip install -e external/acados/interfaces/acados_template/

```
