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
    # define_parameters,
)

from acados_template import AcadosOcpOptions


def define_casadi_dims(model: dict, config: Config) -> dict:
    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise TypeError("config must be an instance of Config")

    try:
        dims = define_dimensions(config)
    except Exception as e:
        # Handle or re-raise exception from define_constraints
        raise RuntimeError("Error in define_casadi_dims: " + str(e))

    # for key, val in dims.items():

    # Set the attribute, assuming the value is correct
    # TODO: Add validation for the value here
    # setattr(ocp.dims, key, val)

    return dims


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

    model, parameter_values = define_model_expressions(config=config)

    dims = define_dimensions(config=config)
    dims["np"] = parameter_values.shape[0]

    ocp_options = config.ocp_options
    # p = define_parameters(config=config)

    x = model["x"]
    u = model["u"]
    p = model["p"]
    f_expl = model["f_expl_expr"]
    f = cs.Function("f_expl", [x, u, p], [f_expl], ["x", "u", "p"], ["xf"])

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


def integrator_function(model: dict, config: Config) -> cs.Function:
    # Continuous dynamics function.
    f = cs.Function(
        "f",
        [model["x"], model["u"], model["p"]],
        [model["f_expl_expr"]],
        ["x", "u", "p"],
        ["xf"],
    )

    # ERK4 steps per interval.
    M = config.ocp_options.sim_method_num_stages

    # Horizon length.
    T = config.ocp_options.tf

    # Number of intervals.
    N = config.dimensions.N

    # Step size.
    h = T / N / M

    # Initial state.
    x0 = cs.SX.sym("x0", model["x"].size()[0])

    # Control input.
    u = cs.SX.sym("u", model["u"].size()[0])

    # Parameters.
    p = cs.SX.sym("p", model["p"].size()[0])

    # State at the beginning of the interval.
    x = x0

    # Integrate over the interval.
    if config.ocp_options.integrator_type == "ERK":
        for j in range(M):
            k1 = f(x, u, p)
            k2 = f(x + h / 2 * k1, u, p)
            k3 = f(x + h / 2 * k2, u, p)
            k4 = f(x + h * k3, u, p)

            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        F = cs.Function("F", [x0, u, p], [x], ["x0", "u", "p"], ["xf"])
    else:
        raise NotImplementedError(
            "Only ERK integrator types are supported at the moment."
        )

    return F


def define_stage_cost_function(model: dict, config: Config) -> cs.Function:
    cost = define_cost(config=config)

    # TODO: Add support for other cost types
    # TODO: Add yref as a parameter
    if config.cost.cost_type == "LINEAR_LS":
        Vx = cost["Vx"]
        Vu = cost["Vu"]
        yref = cost["yref"].reshape(-1, 1)
        W = cost["W"]

        x = model["x"]
        u = model["u"]
        y = cs.mtimes([Vx, x]) + cs.mtimes([Vu, u])

        # Stage cost function.
        stage_cost = cs.Function(
            "stage_cost",
            [model["x"], model["u"]],
            [cs.mtimes([(y - yref).T, W, (y - yref)])],
            ["x", "u"],
            ["l"],
        )

        return stage_cost


def define_terminal_cost_function(model: dict, config: Config) -> cs.Function:
    cost = define_cost(config=config)

    # TODO: Add support for other cost types
    # TODO: Add yref as a parameter
    if config.cost.cost_type_e == "LINEAR_LS":
        Vx_e = cost["Vx_e"]
        yref_e = cost["yref_e"].reshape(-1, 1)
        W_e = cost["W_e"]

        x = model["x"]
        y_e = cs.mtimes([Vx_e, x])

        # Stage cost function.
        stage_cost = cs.Function(
            "terminal_cost",
            [model["x"]],
            [cs.mtimes([(y_e - yref_e).T, W_e, (y_e - yref_e)])],
            ["x"],
            ["m"],
        )

        return stage_cost


def build_nlp_solver(config: Config) -> cs.nlpsol:
    model, parameter_values = define_model_expressions(config=config)

    dims = define_dimensions(config=config)
    # dims["np"] = parameter_values

    F = integrator_function(model=model, config=config)

    constraints = define_constraints(config=config)

    l = define_stage_cost_function(model=model, config=config)
    m = define_terminal_cost_function(model=model, config=config)

    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []

    # "Lift" initial conditions
    xk = cs.SX.sym("x0", dims["nx"])
    w += [xk]
    lbw += constraints["lbx"].tolist()
    ubw += constraints["ubx"].tolist()
    w0 += constraints["x0"].tolist()

    # Formulate the NLP
    for k in range(dims["N"]):
        # New NLP variable for the control
        uk = cs.SX.sym("u_" + str(k))
        w += [uk]
        lbw += constraints["lbu"].tolist()
        ubw += constraints["ubu"].tolist()
        w0 += [0]

        J = J + l(xk, uk)

        # Integrate till the end of the interval
        Fk = F(x0=xk, p=uk)
        xk_end = Fk["xf"]

        # New NLP variable for state at end of interval
        xk = cs.SX.sym("x_" + str(k + 1), dims["nx"])
        w += [xk]
        lbw += constraints["lbx"].tolist()
        ubw += constraints["ubx"].tolist()
        w0 += constraints["x0"].tolist()

        # Add equality constraint
        g += [xk_end - xk]
        lbg += np.zeros((dims["nx"],)).tolist()
        ubg += np.zeros((dims["nx"],)).tolist()

    # Add terminal cost
    J = J + m(xk_end)

    # Create an NLP solver
    prob = {"f": J, "x": cs.vertcat(*w), "g": cs.vertcat(*g)}

    solver = cs.nlpsol("solver", "ipopt", prob)

    return solver, parameter_values


class CasadiOcpDims:
    """docstring for CasadiOcpDims."""

    def __init__(self, config: Config):
        super().__init__()

        pass


class CasadiOcpConstraints:
    """docstring for CasadiOcpConstraints."""

    def __init__(self, config: Config):
        super().__init__()

        pass


class CasadiOcpCost:
    """docstring for CasadiOcpCost."""

    def __init__(self, config: Config):
        super().__init__()

        pass


class CasadiOcpOptions:
    """docstring for CasadiOcpOptions."""

    def __init__(self, config: Config):
        super().__init__()

        pass


class CasadiOcp:
    """docstring for CasadiOcp."""

    def __init__(self, config: Config):
        super().__init__()

        pass


class CasadiOcpSolver:
    """docstring for CasadiOcp."""

    _model: dict
    _cost: cs.Function
    _constraints: cs.Function
    _nlp_solver: cs.nlpsol

    def __init__(self, config: Config):
        super().__init__()

        # self._ocp = cs.Opti()

        self._nlp_solver = build_nlp_solver(config=config)

        # self._model = define_model_expressions(config=config)

        # self._cost = define_cost(config=config)

        # self._constraints = define_constraints(config=config)

    def set(self, stage: int, field: str, value: np.ndarray) -> None:
        """
        Set a field of the OCP solver.

        Args:
            stage: Stage index.
            field: Field name.
            value: Field value.
        """
        raise NotImplementedError()

    def get(self, stage: int, field: str) -> np.ndarray:
        """
        Get a field of the OCP solver.

        Args:
            stage: Stage index.
            field: Field name.

        Returns:
            Field value.
        """
        raise NotImplementedError()

    def solve(self, x0: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Solve the OCP.

        Args:
            x0: Initial state.
            p: Parameters.

        Returns:
            Solution.
        """
        raise NotImplementedError()


class CasadiMPC(MPC):
    """docstring for CartpoleMPC."""

    _parameters: np.ndarray

    def __init__(self, config: Config, build: bool = True):
        super().__init__()

        self._nlp_solver, self._parameters = build_nlp_solver(config=config)

        # self._ocp = CasadiOcp(config=config)

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
