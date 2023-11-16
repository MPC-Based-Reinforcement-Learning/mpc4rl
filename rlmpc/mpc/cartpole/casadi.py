from typing import Union
import numpy as np
import casadi as cs

from rlmpc.common.mpc import MPC
from rlmpc.mpc.utils import ERK4

import matplotlib.pyplot as plt

from rlmpc.mpc.cartpole.common import (
    Config,
    define_model_expressions,
    define_dimensions,
    define_cost,
    define_constraints,
    define_parameters,
)

from acados_template import AcadosOcpOptions


def build_discrete_dynamics_functions(
    config: Config,
) -> tuple((Union[cs.SX, cs.MX], cs.Function, cs.Function)):
    """
    Build the discrete dynamics functions for the OCP.

    Parameters:
        acados_ocp: acados OCP object

    Returns:
        fun_f: discrete dynamics function
        fun_df_dp: derivative of discrete dynamics function with respect to parameters
    """

    model = define_model_expressions(config=config)
    dims = define_dimensions(config=config)
    ocp_options = config.ocp_options
    p = define_parameters(config=config)

    x = model["x"]
    u = model["u"]
    # p = model["p"]
    f = cs.Function("f", [x, u, p], model["f_expl_expr"], ["x", "u", "p"], ["xf"])

    h = ocp_options.tf / dims["N"]

    # TODO: Add support for other integrator types
    # TODO: Use the integrator type from the config file (independent of acados)
    # Next state expression.

    if ocp_options.integrator_type == "ERK" and ocp_options.sim_method_num_stages == 4:
        xf = ERK4(f, x, u, p, h)
    else:
        raise NotImplementedError(
            "Only ERK4 integrator types are supported at the moment."
        )

    # Integrator function.
    # fun_f = cs.Function("f", [x, u, p], [xf], ["x", "u", "p"], ["xf"])

    # Jacobian of the integrator function with respect to the parameters.
    df_dp = cs.Function(
        "df_dp", [x, u, p], [cs.jacobian(xf, p)], ["x", "u", "p"], ["dxf_dp"]
    )

    return (xf, f, df_dp)


def build_nlp_solver(config: Config) -> cs.nlpsol:
    model = define_model_expressions(config=config)

    # ocp_options = config.ocp_options.to_dict()

    xf, f, df_dp = build_discrete_dynamics_functions(config=config)

    print("hallo")


class CasadiOcp:
    """docstring for CasadiOcp."""

    _model: dict
    _cost: cs.Function
    _constraints: cs.Function
    _ocp: cs.Opti

    def __init__(self, config: Config):
        super().__init__()

        # self._ocp = cs.Opti()

        nlp_solver = build_nlp_solver(config=config)

        self._model = define_model_expressions(config=config)

        self._cost = define_cost(config=config)

        self._constraints = define_constraints(config=config)

    def solve(self, x0: np.ndarray, u0: np.ndarray, p: np.ndarray) -> np.ndarray:
        pass


class CasadiMPC(MPC):
    """docstring for CartpoleMPC."""

    _parameters: np.ndarray

    def __init__(self, config: Config, build: bool = True):
        super().__init__()

        self._ocp = CasadiOcp(config=config)

        self._parameters = define_parameters(config=config)

    def get_parameters(self) -> np.ndarray:
        return self._parameters

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def get_action(self, x0: np.ndarray) -> np.ndarray:
        """
        Update the solution of the OCP solver.

        Args:
            x0: Initial state.

        Returns:
            u: Optimal control action.
        """
        raise NotImplementedError()

    def get_parameters(self) -> np.ndarray:
        return self._parameters
