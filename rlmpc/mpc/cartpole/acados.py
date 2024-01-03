import os
from acados_template.acados_ocp_solver import ocp_generate_external_functions
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosOcpConstraints, AcadosOcpCost, AcadosOcpDims, AcadosModel

from typing import Union

import casadi as cs


import scipy

from rlmpc.common.mpc import MPC

import matplotlib.pyplot as plt

from rlmpc.mpc.cartpole.common import (
    CasadiNLP,
    Config,
    define_dimensions,
    define_cost,
    define_constraints,
    define_parameter_values,
    define_discrete_dynamics_function,
    build_nlp,
)


def define_acados_model(ocp: AcadosOcp, config: Config) -> AcadosModel:
    # Check if ocp is an instance of AcadosOcp
    if not isinstance(ocp, AcadosOcp):
        raise TypeError("ocp must be an instance of AcadosOcp")

    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise TypeError("config must be an instance of Config")

    # try:
    #     model = define_model_expressions(config)
    # except Exception as e:
    #     # Handle or re-raise exception from define_constraints
    #     raise RuntimeError("Error in define_acados_model: " + str(e))

    # for key, val in model.items():
    #     # Check if the attribute exists in ocp.constraints
    #     if not hasattr(ocp.model, key):
    #         raise AttributeError(f"Attribute {key} does not exist in ocp.model")

    #     # Set the attribute, assuming the value is correct
    #     # TODO: Add validation for the value here
    #     setattr(ocp.model, key, val)

    name = config.model_name

    # set up states & controls
    s = cs.SX.sym("x")
    s_dot = cs.SX.sym("x_dot")
    theta = cs.SX.sym("theta")
    theta_dot = cs.SX.sym("theta_dot")

    x = cs.vertcat(s, s_dot, theta, theta_dot)

    F = cs.SX.sym("F")
    u = cs.vertcat(F)

    x_dot = cs.SX.sym("xdot", 4, 1)

    # algebraic variables
    z = None

    # parameters
    p_sym = []

    # Set up parameters to nominal values
    p = {key: param["value"] for key, param in config.model_params.to_dict().items()}

    parameter_values = []
    # Set up parameters to symbolic variables if not fixed
    for key, param in config.model_params.to_dict().items():
        if not param["fixed"]:
            p_sym += [cs.SX.sym(key)]
            p[key] = p_sym[-1]
            parameter_values += [param["value"]]

    p_sym = cs.vertcat(*p_sym)
    parameter_values = np.array(parameter_values)

    # Define model dynamics
    cos_theta = cs.cos(theta)
    sin_theta = cs.sin(theta)
    temp = (u + p["m"] * theta_dot**2 * sin_theta) / (p["m"] + p["M"])

    theta_ddot = (p["g"] * sin_theta - cos_theta * temp) / (p["l"] * (4.0 / 3.0 - p["m"] * cos_theta**2 / (p["m"] + p["M"])))

    f_expl = cs.vertcat(
        s_dot,
        temp - p["m"] * theta_ddot * cos_theta / (p["m"] + p["M"]),  # x_ddot
        theta_dot,
        (p["g"] * sin_theta - cos_theta * temp) / (p["l"] * (4.0 / 3.0 - p["m"] * cos_theta**2 / (p["m"] + p["M"]))),
    )

    f_impl = x_dot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = x_dot
    model.p = p_sym
    model.u = u
    model.z = z
    model.name = name

    return model


def define_acados_dims(ocp: AcadosOcp, config: Config) -> AcadosOcpDims:
    # Check if ocp is an instance of AcadosOcp
    if not isinstance(ocp, AcadosOcp):
        raise TypeError("ocp must be an instance of AcadosOcp")

    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise TypeError("config must be an instance of Config")

    try:
        dims = define_dimensions(config)
    except Exception as e:
        # Handle or re-raise exception from define_constraints
        raise RuntimeError("Error in define_acados_dims: " + str(e))

    for key, val in dims.items():
        # Check if the attribute exists in ocp.constraints
        if not hasattr(ocp.dims, key):
            raise AttributeError(f"Attribute {key} does not exist in ocp.dims")

        # Set the attribute, assuming the value is correct
        # TODO: Add validation for the value here
        setattr(ocp.dims, key, val)

    ocp.dims.np = ocp.model.p.size()[0]

    # TODO: Add other slack variable dimensions
    ocp.dims.nsbx = ocp.constraints.idxsbx.shape[0]
    ocp.dims.nsbu = ocp.constraints.idxsbu.shape[0]
    ocp.dims.ns = ocp.dims.nsbx + ocp.dims.nsbu

    ocp.dims.nsbx_e = ocp.constraints.idxsbx_e.shape[0]
    ocp.dims.ns_e = ocp.dims.nsbx_e

    ocp.dims.nbx_e = ocp.constraints.idxbx_e.shape[0]

    return ocp.dims


def define_acados_cost(ocp: AcadosOcp, config: Config) -> AcadosOcpCost:
    # Check if ocp is an instance of AcadosOcp
    if not isinstance(ocp, AcadosOcp):
        raise TypeError("ocp must be an instance of AcadosOcp")

    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise TypeError("config must be an instance of Config")

    try:
        cost = define_cost(config)
    except Exception as e:
        # Handle or re-raise exception from define_constraints
        raise RuntimeError("Error in define_acados_cost: " + str(e))

    for key, val in cost.items():
        # Check if the attribute exists in ocp.constraints
        if not hasattr(ocp.cost, key):
            raise AttributeError(f"Attribute {key} does not exist in ocp.cost")

        # Set the attribute, assuming the value is correct
        # TODO: Add validation for the value here
        setattr(ocp.cost, key, val)

    return ocp.cost


def define_acados_constraints(ocp: AcadosOcp, config: Config) -> AcadosOcpConstraints:
    # Check if ocp is an instance of AcadosOcp
    if not isinstance(ocp, AcadosOcp):
        raise TypeError("ocp must be an instance of AcadosOcp")

    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise TypeError("config must be an instance of Config")

    try:
        constraints = define_constraints(config)
    except Exception as e:
        # Handle or re-raise exception from define_constraints
        raise RuntimeError("Error in define_constraints: " + str(e))

    for key, val in constraints.items():
        # Check if the attribute exists in ocp.constraints
        if not hasattr(ocp.constraints, key):
            raise AttributeError(f"Attribute {key} does not exist in ocp.constraints")

        # Set the attribute, assuming the value is correct
        # TODO: Add validation for the value here
        setattr(ocp.constraints, key, val)

    return ocp.constraints


def ERK4(
    f: Union[cs.SX, cs.Function],
    x: Union[cs.SX, np.ndarray],
    u: Union[cs.SX, np.ndarray],
    p: Union[cs.SX, np.ndarray],
    h: float,
) -> Union[cs.SX, np.ndarray]:
    """
    Explicit Runge-Kutta 4 integrator

    TODO: Works for numeric values as well as for symbolic values. Type hinting is a bit misleading.

    Parameters:
        f: function to integrate
        x: state
        u: control
        p: parameters
        h: step size

        Returns:
            xf: integrated state
    """
    k1 = f(x, u, p)
    k2 = f(x + h / 2 * k1, u, p)
    k3 = f(x + h / 2 * k2, u, p)
    k4 = f(x + h * k3, u, p)
    xf = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return xf


class AcadosMPC(MPC):
    """docstring for CartpoleMPC."""

    _parameters: np.ndarray
    ocp_solver: AcadosOcpSolver
    nlp: CasadiNLP
    idx: dict

    def __init__(self, config: Config, build: bool = True):
        super().__init__()

        ocp = AcadosOcp()

        ocp.model = define_acados_model(ocp=ocp, config=config)

        ocp.model.disc_dyn_expr = ERK4(
            cs.Function("ode", [ocp.model.x, ocp.model.u, ocp.model.p], [ocp.model.f_expl_expr]),
            ocp.model.x,
            ocp.model.u,
            ocp.model.p,
            config.ocp_options.tf / config.dimensions.N / config.ocp_options.sim_method_num_stages,
        )

        ocp.parameter_values = define_parameter_values(ocp=ocp, config=config)

        ocp.constraints = define_acados_constraints(ocp=ocp, config=config)

        ocp.dims = define_acados_dims(ocp=ocp, config=config)

        ocp.cost = define_acados_cost(ocp=ocp, config=config)

        ocp.cost.W_0 = ocp.cost.W
        ocp.dims.ny_0 = ocp.dims.ny
        ocp.cost.yref_0 = ocp.cost.yref

        ocp.model.cost_y_expr_0 = cs.vertcat(ocp.model.x, ocp.model.u)
        ocp.model.cost_y_expr = cs.vertcat(ocp.model.x, ocp.model.u)
        ocp.model.cost_y_expr_e = ocp.model.x

        # Build cost function

        ocp.solver_options = config.ocp_options

        ocp.code_export_directory = config.meta.code_export_dir

        self.ocp = ocp

        ocp_generate_external_functions(ocp, ocp.model)

        nlp, self.idx = build_nlp(ocp=self.ocp)

        # nlp.L.sym = nlp.cost.sym + cs.dot(nlp.pi.sym, nlp.g.sym) + cs.dot(nlp.lam.sym, nlp.h.sym)

        # nlp.dL_dw.sym = cs.jacobian(nlp.L.sym, nlp.w.sym)

        # nlp.R.sym = cs.vertcat(cs.transpose(nlp.dL_dw.sym), nlp.g.sym, nlp.lam.sym * nlp.h.sym)

        # nlp.R.fun = cs.Function(
        #     "R",
        #     [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym],
        #     [nlp.R.sym],
        #     ["w", "lbw", "ubw", "pi", "lam", "p"],
        #     ["R"],
        # )

        self.nlp = nlp

        # Check path to config.meta.json file. Create the directory if it does not exist.
        if not os.path.exists(os.path.dirname(config.meta.json_file)):
            os.makedirs(os.path.dirname(config.meta.json_file))

        # TODO: Add config entries for json file and c_generated_code folder, and build, generate flags
        if build:
            self.ocp_solver = AcadosOcpSolver(ocp, json_file=config.meta.json_file)
        else:
            # Assumes json file and c_generated_code folder already exists
            self.ocp_solver = AcadosOcpSolver(ocp, json_file=config.meta.json_file, build=False, generate=False)

        self._parameters = ocp.parameter_values

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low = self.ocp.constraints.lbu
        high = self.ocp.constraints.ubu

        return 2.0 * ((action - low) / (high - low)) - 1.0

    def get_action(self, x0: np.ndarray) -> np.ndarray:
        """
        Update the solution of the OCP solver.

        Args:
            x0: Initial state.

        Returns:
            u: Optimal control action.
        """
        # Set initial state
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        # Solve the optimization problem
        self.ocp_solver.solve()

        # Get solution
        action = self.ocp_solver.get(0, "u")

        # Scale to [-1, 1] for gym
        action = self.scale_action(action)

        return action

    def get_parameters(self) -> np.ndarray:
        return self._parameters

    def get_predicted_state_trajectory(self) -> np.ndarray:
        """
        Get the predicted state trajectory.

        Returns:
            x: Predicted state trajectory.
        """
        x = np.zeros((self.ocp.dims.N + 1, self.ocp.dims.nx))

        for i in range(self.ocp.dims.N + 1):
            x[i, :] = self.ocp_solver.get(i, "x")

        return x

    def get_predicted_control_trajectory(self) -> np.ndarray:
        """
        Get the predicted control trajectory.

        Returns:
            u: Predicted control trajectory.
        """
        u = np.zeros((self.ocp.dims.N, self.ocp.dims.nu))

        for i in range(self.ocp.dims.N):
            u[i, :] = self.ocp_solver.get(i, "u")

        return u

    def plot_prediction(self) -> None:
        """
        Plot the predicted trajectory.
        """

        x = self.get_predicted_state_trajectory()
        u = self.get_predicted_control_trajectory()

        _, ax = plt.subplots(self.ocp.dims.nx + self.ocp.dims.nu, 1, figsize=(10, 7))

        for i in range(self.ocp.dims.nx):
            ax[i].plot(x[:, i], "-o")
            ax[i].grid(True)
            ax[i].set_ylabel(f"x_{i}")

        # Make a stairs plot for u
        ax[self.ocp.dims.nx].step(np.arange(0, u.shape[0]), u.flatten(), where="post")
        ax[self.ocp.dims.nx].grid(True)
        ax[self.ocp.dims.nx].set_ylabel("u")

        # Draw bounds
        lbx = self.ocp.constraints.lbx
        ubx = self.ocp.constraints.ubx
        lbu = self.ocp.constraints.lbu
        ubu = self.ocp.constraints.ubu

        for i in range(self.ocp.dims.nx):
            ax[i].plot(np.ones_like(x[:, i]) * lbx[i], "--", color="grey")
            ax[i].plot(np.ones_like(x[:, i]) * ubx[i], "--", color="gray")

        ax[self.ocp.dims.nx].plot(np.ones_like(u) * lbu, "--", color="gray")
        ax[self.ocp.dims.nx].plot(np.ones_like(u) * ubu, "--", color="gray")

        plt.show()

    def print_header(self) -> None:
        """
        Print the header for the data table.
        """
        print("{:>8} {:>8} {:>8} {:>8} {:>8}".format("x", "x_dot", "theta", "theta_dot", "u"))

    def print_data(self, x: np.ndarray, u: np.ndarray) -> None:
        """
        Print the data table.

        Args:
            x: State.
            u: Control.
        """
        print("{:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f}".format(x[0], x[1], x[2], x[3], u))
