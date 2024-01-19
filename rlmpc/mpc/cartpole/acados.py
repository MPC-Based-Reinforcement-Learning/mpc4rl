import os

# from acados_template.acados_ocp_solver import ocp_generate_external_functions
import numpy as np
from acados_template import (
    AcadosOcp,
    AcadosOcpSolver,
    AcadosOcpConstraints,
    AcadosOcpCost,
    AcadosOcpDims,
    AcadosModel,
    AcadosOcpOptions,
)

import casadi as cs

from rlmpc.mpc.common.mpc import MPC

import matplotlib.pyplot as plt

from rlmpc.mpc.cartpole.common import Config, ModelParams, define_parameter_values

from rlmpc.common.integrator import ERK4

from rlmpc.mpc.nlp import NLP, build_nlp


def define_acados_model(ocp: AcadosOcp, config: dict) -> AcadosModel:
    # Check if ocp is an instance of AcadosOcp
    if not isinstance(ocp, AcadosOcp):
        raise TypeError("ocp must be an instance of AcadosOcp")

    name = config["model"]["name"]

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

    model_params = ModelParams.from_dict(config["model"]["params"])

    # Set up parameters to nominal values
    p = {key: param["value"] for key, param in model_params.to_dict().items()}

    parameter_values = []
    # Set up parameters to symbolic variables if not fixed
    for key, param in model_params.to_dict().items():
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

    disc_dyn_expr = ERK4(
        cs.Function("ode", [x, u, p_sym], [f_expl]),
        x,
        u,
        p_sym,
        config["ocp_options"]["tf"] / config["dimensions"]["N"] / config["ocp_options"]["sim_method_num_stages"],
    )

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.disc_dyn_expr = disc_dyn_expr
    model.x = x
    model.xdot = x_dot
    model.p = p_sym
    model.u = u
    model.z = z
    model.name = name
    model.cost_y_expr_0 = cs.vertcat(model.x, model.u)
    model.cost_y_expr = cs.vertcat(model.x, model.u)
    model.cost_y_expr_e = model.x

    return model


def define_acados_dims(config: Config) -> AcadosOcpDims:
    dims = AcadosOcpDims()

    for key, val in config.items():
        hasattr(dims, key), f"AcadosOcpDims does not have attribute {key}"

        # Set the attribute, assuming the value is correct
        # TODO: Add validation for the value here
        setattr(dims, key, val)

    return dims


def define_acados_cost(config: dict) -> AcadosOcpCost:
    cost = AcadosOcpCost()
    for key, val in config.items():
        assert hasattr(cost, key), f"AcadosOcpCost does not have attribute {key}"

        # Set the attribute, assuming the value is correct
        # TODO: Add validation for the value here
        if isinstance(val, list):
            setattr(cost, key, np.array(val))
        if isinstance(val, str):
            setattr(cost, key, val)

    return cost


def define_acados_constraints(config: dict) -> AcadosOcpConstraints:
    constraints = AcadosOcpConstraints()
    for key, val in config.items():
        hasattr(constraints, key), f"AcadosOcpConstraints does not have attribute {key}"

        if isinstance(val, list):
            setattr(constraints, key, np.array(val))
        if isinstance(val, str):
            setattr(constraints, key, val)

    return constraints


def define_acados_ocp_options(config: dict) -> AcadosOcpOptions:
    ocp_options = AcadosOcpOptions()
    for key, val in config.items():
        hasattr(ocp_options, key), f"AcadosOcpOptions does not have attribute {key}"

        setattr(ocp_options, key, val)

    return ocp_options


class AcadosMPC(MPC):
    """docstring for CartpoleMPC."""

    _parameters: np.ndarray
    ocp_solver: AcadosOcpSolver
    nlp: NLP

    def __init__(self, config: Config, build: bool = True):
        super().__init__()

        ocp = AcadosOcp()

        ocp.model = define_acados_model(ocp=ocp, config=config)

        ocp.parameter_values = define_parameter_values(config=config)

        ocp.constraints = define_acados_constraints(config=config["constraints"])

        ocp.dims = define_acados_dims(config=config["dimensions"])

        ocp.cost = define_acados_cost(config=config["cost"])

        ocp.solver_options = define_acados_ocp_options(config=config["ocp_options"])

        ocp.code_export_directory = config["meta"]["code_export_dir"]

        self.ocp = ocp

        # ocp_generate_external_functions(ocp, ocp.model)

        self.nlp = build_nlp(ocp=self.ocp)

        # Check path to config.meta.json file. Create the directory if it does not exist.
        if not os.path.exists(os.path.dirname(config["meta"]["json_file"])):
            os.makedirs(os.path.dirname(config["meta"]["json_file"]))

        # TODO: Add config entries for json file and c_generated_code folder, and build, generate flags
        if build:
            self.ocp_solver = AcadosOcpSolver(ocp, json_file=config["meta"]["json_file"])
        else:
            # Assumes json file and c_generated_code folder already exists
            self.ocp_solver = AcadosOcpSolver(ocp, json_file=config["meta"]["json_file"], build=False, generate=False)

    def assert_nlp_kkt_conditions(self) -> None:
        """
        Assert that the NLP is sane.
        """
        # Check if the NLP is sane

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

    def get_action(self, x0: np.ndarray) -> np.ndarray:
        """
        Get the optimal action.

        Args:
            x0: Initial state.

        Returns:
            u: Optimal action.
        """
        return self.scale_action(super().get_action(x0))

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
