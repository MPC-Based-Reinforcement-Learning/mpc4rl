import pytest
from rlmpc.mpc.linear_system.acados import AcadosMPC
from rlmpc.gym.linear_system.environment import LinearSystemEnv as Env

import numpy as np


def test_mpc_initialization():
    param = {
        "A": np.array([[1.0, 0.25], [0.0, 1.0]]),
        "B": np.array([[0.03125], [0.25]]),
        "Q": np.identity(2),
        "R": np.identity(1),
        "b": np.array([[0.0], [0.0]]),
        "f": np.array([[0.0], [0.0], [0.0]]),
        "V_0": np.array([1e-3]),
    }

    mpc = AcadosMPC(param, discount_factor=0.99)
    assert mpc is not None
    assert mpc.ocp_solver is not None
    assert mpc.ocp_solver.acados_ocp is not None
    assert mpc.ocp_solver.acados_ocp.model is not None
    assert mpc.ocp_solver.acados_ocp.dims is not None
    assert mpc.ocp_solver.acados_ocp.cost is not None
    assert mpc.ocp_solver.acados_ocp.constraints
