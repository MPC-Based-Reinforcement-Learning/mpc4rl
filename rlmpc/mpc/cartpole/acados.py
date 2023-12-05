## For MPC
import numpy as np
from acados_template import (
    AcadosOcp,
    AcadosOcpSolver,
    AcadosOcpConstraints,
    AcadosOcpCost,
    AcadosOcpDims,
)

from rlmpc.common.mpc import MPC

import matplotlib.pyplot as plt

from rlmpc.mpc.cartpole.common import (
    Config,
    define_model_expressions,
    define_dimensions,
    define_cost,
    define_constraints,
)


# TODO: Define a function to get the model, dims, cost, constraints, and solver options from a config file. Check if refactor to one function


def define_acados_model(ocp: AcadosOcp, config: Config) -> (AcadosOcpCost, np.ndarray):
    # Check if ocp is an instance of AcadosOcp
    if not isinstance(ocp, AcadosOcp):
        raise TypeError("ocp must be an instance of AcadosOcp")

    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise TypeError("config must be an instance of Config")

    try:
        model, ocp.parameter_values = define_model_expressions(config)
    except Exception as e:
        # Handle or re-raise exception from define_constraints
        raise RuntimeError("Error in define_acados_model: " + str(e))

    for key, val in model.items():
        # Check if the attribute exists in ocp.constraints
        if not hasattr(ocp.model, key):
            raise AttributeError(f"Attribute {key} does not exist in ocp.model")

        # Set the attribute, assuming the value is correct
        # TODO: Add validation for the value here
        setattr(ocp.model, key, val)

    return ocp.model, ocp.parameter_values


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


class AcadosMPC(MPC):
    """docstring for CartpoleMPC."""

    _parameters: np.ndarray

    def __init__(self, config: Config, build: bool = True):
        super().__init__()

        ocp = AcadosOcp()

        ocp.model, ocp.parameter_values = define_acados_model(ocp=ocp, config=config)

        ocp.dims = define_acados_dims(ocp=ocp, config=config)

        ocp.cost = define_acados_cost(ocp=ocp, config=config)

        ocp.constraints = define_acados_constraints(ocp=ocp, config=config)

        ocp.solver_options = config.ocp_options

        self.ocp = ocp

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
