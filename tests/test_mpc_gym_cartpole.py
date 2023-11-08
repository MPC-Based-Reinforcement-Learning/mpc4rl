"""
    Test MPC for a cartpole in gym. No learning.
"""

from typing import Optional
import gymnasium as gym
import scipy

from stable_baselines3 import PPO
from rlmpc.gym.continuous_cartpole.environment import (
    ContinuousCartPoleBalanceEnv,
    ContinuousCartPoleSwingUpEnv,
)

# from stable_baselines3.common.policies import MPC
from rlmpc.common.mpc import MPC

from rlmpc.common.utils import read_config


from stable_baselines3.common.env_util import make_vec_env


## For MPC
import casadi as cs
import numpy as np
import scipy
from acados_template import (
    AcadosOcp,
    AcadosOcpSolver,
    AcadosOcpConstraints,
    AcadosOcpCost,
    AcadosOcpDims,
    AcadosModel,
    AcadosOcpOptions,
)

from dataclasses import asdict, dataclass, field

import matplotlib.pyplot as plt


@dataclass
class CartpoleModelParams:
    """
    Parameter class for Cartpole model in MPC.
    """

    M: float  # mass of the cart
    m: float  # mass of the pole
    l: float  # length of the pole
    g: float  # gravity
    name: str = "cartpole_model"

    @classmethod
    def from_dict(cls, config_dict: dict):
        return CartpoleModelParams(**config_dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class CartpoleCostParams:
    """
    Parameter class for Cartpole cost in MPC.
    """

    cost_type: str
    cost_type_e: str
    Q: np.ndarray
    R: np.ndarray
    Q_e: np.ndarray

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            cost_type=config_dict["cost_type"],
            cost_type_e=config_dict["cost_type_e"],
            Q=np.diag(config_dict["Q"]),
            R=np.diag(config_dict["R"]),
            Q_e=np.diag(config_dict["Q_e"]),
        )

    def to_dict(self):
        return asdict(self)


@dataclass
class CartpoleConstraintParams:
    """
    Parameter class for Cartpole constraints in MPC.
    """

    constraint_type: str
    x0: np.ndarray
    lbu: np.ndarray
    ubu: np.ndarray

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            constraint_type=config_dict["constraint_type"],
            x0=np.array(config_dict["x0"]),
            lbu=np.array(config_dict["lbu"]),
            ubu=np.array(config_dict["ubu"]),
        )

    def to_dict(self):
        return asdict(self)


@dataclass
class CartpoleDimensions:
    """
    Parameter class for Cartpole dimensions in MPC.
    """

    nx: int  # number of states
    nu: int  # number of inputs
    N: int  # horizon length

    @classmethod
    def from_dict(cls, config_dict: dict):
        return CartpoleDimensions(**config_dict)

    def to_dict(self):
        return asdict(self)


class CartpoleOcpOptions(AcadosOcpOptions):
    """
    Parameter class for Cartpole solver options in MPC.
    """

    def __init__(self):
        super().__init__()

    # tf: float
    # integrator_type: Optional[str]
    # nlp_solver_type: Optional[str]
    # qp_solver: Optional[str]
    # hessian_approx: Optional[str]
    # nlp_solver_max_iter: Optional[int]
    # qp_solver_iter_max: Optional[int]

    # TODO: Add more options to cover all AcadosOcpOptions. See
    # https://docs.acados.org/interfaces/acados_python_interface/#acadosocpoptions
    # for more info. Reconsider this solution, it requires more work to maintain
    # when the AcadosOcpOptions class changes.

    @classmethod
    def from_dict(cls, config_dict: dict):
        instance = cls()
        for key, value in config_dict.items():
            setattr(instance, key, value)

        return instance

    # def to_dict(self):
    #     return asdict(self)


@dataclass
class CartpoleMeta:
    """
    Parameter class for Cartpole meta parameters in MPC.
    """

    json_file: str = "acados_ocp.json"

    @classmethod
    def from_dict(cls, config_dict: dict):
        return CartpoleMeta(**config_dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class Config:
    """Configuration class for managing mpc parameters."""

    model: Optional[CartpoleModelParams]
    cost: Optional[CartpoleCostParams]
    constraints: Optional[CartpoleConstraintParams]
    dimensions: Optional[CartpoleDimensions]
    ocp_options: Optional[CartpoleOcpOptions]
    meta: Optional[CartpoleMeta]

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            model=CartpoleModelParams.from_dict(config_dict["model"]),
            cost=CartpoleCostParams.from_dict(config_dict["cost"]),
            constraints=CartpoleConstraintParams.from_dict(config_dict["constraints"]),
            dimensions=CartpoleDimensions.from_dict(config_dict["dimensions"]),
            ocp_options=CartpoleOcpOptions.from_dict(config_dict["ocp_options"]),
            meta=CartpoleMeta.from_dict(config_dict["meta"]),
        )

    def to_dict(self) -> dict:
        config_dict = {}

        if self.model is not None:
            config_dict["model"] = self.model.to_dict()

        if self.cost is not None:
            config_dict["cost"] = self.cost.to_dict()

        if self.constraints is not None:
            config_dict["constraints"] = self.constraints.to_dict()

        if self.dimensions is not None:
            config_dict["dimensions"] = self.dimensions.to_dict()

        if self.ocp_options is not None:
            config_dict["ocp_options"] = self.ocp_options.to_dict()

        if self.meta is not None:
            config_dict["meta"] = self.meta.to_dict()

        return config_dict


class CartpoleMPC(MPC):
    """docstring for CartpoleMPC."""

    def __init__(self, config: Config):
        super().__init__()

        model = AcadosModel()
        model.name = config.model.name

        # set up states & controls
        s = cs.SX.sym("x")
        s_dot = cs.SX.sym("x_dot")
        theta = cs.SX.sym("theta")
        theta_dot = cs.SX.sym("theta_dot")

        x = cs.vertcat(s, s_dot, theta, theta_dot)

        F = cs.SX.sym("F")
        u = cs.vertcat(F)

        # xdot
        # s_ddot = cs.SX.sym("x_ddot")
        # theta_ddot = cs.SX.sym("theta_ddot")

        # TODO: The naming x of cartpole distance is confusing, change to something else
        # x_dot = cs.vertcat(s_dot, s_ddot, theta_dot, theta_ddot)
        x_dot = cs.SX.sym("xdot", 4, 1)

        # algebraic variables
        z = None

        # parameters
        # p = {}

        # parameter_symbols = []
        # parameter_values = []
        # for param_name, value_dict in param.items():
        #     if value_dict["fixed"]:
        #         p[param_name] = value_dict["value"]
        #     else:
        #         p[param_name] = cs.SX.sym(param_name)
        #         parameter_symbols += [p[param_name]]
        #         parameter_values += [value_dict["value"]]

        # p = []

        p = config.model.to_dict()

        # Define model dynamics
        cos_theta = cs.cos(theta)
        sin_theta = cs.sin(theta)
        temp = (u + p["m"] * theta_dot**2 * sin_theta) / (p["m"] + p["M"])

        theta_ddot = (p["g"] * sin_theta - cos_theta * temp) / (
            p["l"] * (4.0 / 3.0 - p["m"] * cos_theta**2 / (p["m"] + p["M"]))
        )

        f_expl = cs.vertcat(
            s_dot,
            temp - p["m"] * theta_ddot * cos_theta / (p["m"] + p["M"]),  # x_ddot
            theta_dot,
            (p["g"] * sin_theta - cos_theta * temp)  # theta_ddot
            / (p["l"] * (4.0 / 3.0 - p["m"] * cos_theta**2 / (p["m"] + p["M"]))),
        )

        f_impl = x_dot - f_expl

        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = x_dot
        model.u = u
        model.z = z

        dims = AcadosOcpDims()
        dims.nx = config.dimensions.nx
        dims.nu = config.dimensions.nu
        dims.N = config.dimensions.N
        dims.ny = dims.nx + dims.nu
        dims.ny_e = dims.nx

        cost = AcadosOcpCost()
        cost.cost_type = config.cost.cost_type
        cost.cost_type_e = config.cost.cost_type_e

        # Q = np.array((cost_params["Q"]), dtype=np.float32)
        # R = np.array((cost_params["R"]), dtype=np.float32)
        # cost.W = np.diag(np.concatenate((config.cost.Q, config.cost.R)))

        # Make a blockdiagonal matrix from Q and R

        cost.W = scipy.linalg.block_diag(config.cost.Q, config.cost.R)

        cost.W_e = config.cost.Q_e

        cost.Vx = np.zeros((dims.ny, dims.nx), dtype=np.float32)
        cost.Vx[: dims.nx, : dims.nx] = np.eye(dims.nx)

        Vu = np.zeros((dims.ny, dims.nu))
        Vu[4, 0] = 1.0

        cost.Vu = Vu
        cost.Vx_e = np.eye(dims.nx)
        cost.yref = np.zeros((dims.ny,))
        cost.yref_e = np.zeros((dims.ny_e,))

        constraints = AcadosOcpConstraints()
        constraints.constr_type = config.constraints.constraint_type
        constraints.x0 = config.constraints.x0.reshape(-1)
        constraints.lbu = config.constraints.lbu.reshape(-1)
        constraints.ubu = config.constraints.ubu.reshape(-1)
        constraints.idxbu = np.array([0]).reshape(-1)

        ocp = AcadosOcp()
        ocp.dims = dims
        ocp.cost = cost
        ocp.model = model
        ocp.constraints = constraints
        ocp.solver_options = config.ocp_options

        self._ocp = ocp
        self._ocp_solver = AcadosOcpSolver(ocp, json_file=config.meta.json_file)

        # self._model = model
        # self._cost = cost
        # self._constraints = constraints
        # self._dims = dims
        # self._solver_options = config.ocp_options

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
        # Set initial state
        self._ocp_solver.set(0, "lbx", x0)
        self._ocp_solver.set(0, "ubx", x0)
        self._ocp_solver.set(0, "x", x0)

        # Solve the optimization problem
        self._ocp_solver.solve()

        # Get solution
        u = self._ocp_solver.get(0, "u")

        # Scale to [-1, 1] for gym
        action = (
            2.0
            * (
                (u - self._ocp.constraints.lbu)
                / (self._ocp.constraints.ubu - self._ocp.constraints.lbu)
            )
            - 1.0
        )

        return action

    def _get_predicted_state_trajectory(self) -> np.ndarray:
        """
        Get the predicted state trajectory.

        Returns:
            x: Predicted state trajectory.
        """
        x = np.zeros((self._ocp.dims.N + 1, self._ocp.dims.nx))

        for i in range(self._ocp.dims.N + 1):
            x[i, :] = self._ocp_solver.get(i, "x")

        return x

    def _get_predicted_control_trajectory(self) -> np.ndarray:
        """
        Get the predicted control trajectory.

        Returns:
            u: Predicted control trajectory.
        """
        u = np.zeros((self._ocp.dims.N, self._ocp.dims.nu))

        for i in range(self._ocp.dims.N):
            u[i, :] = self._ocp_solver.get(i, "u")

        return u

    def plot_prediction(self) -> None:
        """
        Plot the predicted trajectory.
        """

        x = self._get_predicted_state_trajectory()
        u = self._get_predicted_control_trajectory()

        fig, ax = plt.subplots(
            self._ocp.dims.nx + self._ocp.dims.nu, 1, figsize=(10, 7)
        )

        for i in range(self._ocp.dims.nx):
            ax[i].plot(x[:, i], "-o")
            ax[i].grid(True)
            ax[i].set_ylabel(f"x_{i}")

        # Make a stairs plot for u
        ax[self._ocp.dims.nx].step(np.arange(0, u.shape[0]), u.flatten(), where="post")
        ax[self._ocp.dims.nx].grid(True)
        ax[self._ocp.dims.nx].set_ylabel("u")

        plt.show()

    def print_header(self) -> None:
        """
        Print the header for the data table.
        """
        print(
            "{:>8} {:>8} {:>8} {:>8} {:>8}".format(
                "x", "x_dot", "theta", "theta_dot", "u"
            )
        )

    def print_data(self, x: np.ndarray, u: np.ndarray) -> None:
        """
        Print the data table.

        Args:
            x: State.
            u: Control.
        """
        print(
            "{:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f}".format(x[0], x[1], x[2], x[3], u)
        )


if __name__ == "__main__":
    config = read_config("config/test_mpc_gym_cartpole.yaml")

    env = gym.make(
        config["environment"]["id"],
        render_mode=config["environment"]["render_mode"],
        min_action=-1.0,
        max_action=1.0,
        force_mag=config["environment"]["force_mag"],
    )

    mpc = CartpoleMPC(config=Config.from_dict(config["mpc"]))

    model = PPO(
        "ModelPredictiveControlPolicy",
        env,
        verbose=1,
        policy_kwargs={"mpc": mpc},
    )

    # Insert training here

    vec_env = model.get_env()

    obs = vec_env.reset()

    for i in range(200):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = vec_env.step(action)

        vec_env.render("human")
