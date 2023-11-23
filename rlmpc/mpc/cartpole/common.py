import casadi as cs
from dataclasses import asdict, dataclass, field
import numpy as np
from acados_template import AcadosOcpOptions
from typing import Optional
import scipy


@dataclass
class Param:
    value: float
    fixed: bool = True

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class ModelParams:
    """
    Parameter class for Cartpole model in MPC.
    """

    M: Param  # mass of the cart
    m: Param  # mass of the pole
    l: Param  # length of the pole
    g: Param  # gravity

    @classmethod
    def from_dict(cls, config_dict: dict):
        # return ModelParams(**config_dict)
        return cls(
            M=Param.from_dict(config_dict["M"]),
            m=Param.from_dict(config_dict["m"]),
            l=Param.from_dict(config_dict["l"]),
            g=Param.from_dict(config_dict["g"]),
        )

    def to_dict(self):
        return asdict(self)


@dataclass
class CostParams:
    """
    Parameter class for Cartpole cost in MPC.
    """

    cost_type: str
    cost_type_e: str
    Q: np.ndarray
    R: np.ndarray
    Q_e: np.ndarray
    Zl: np.ndarray
    Zu: np.ndarray
    zl: np.ndarray
    zu: np.ndarray
    Zl_e: np.ndarray
    Zu_e: np.ndarray
    zl_e: np.ndarray
    zu_e: np.ndarray

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            cost_type=config_dict["cost_type"],
            cost_type_e=config_dict["cost_type_e"],
            Q=np.diag(config_dict["Q"]),
            R=np.diag(config_dict["R"]),
            Q_e=np.diag(config_dict["Q_e"]),
            Zl=np.diag(config_dict["Zl"]),
            Zu=np.diag(config_dict["Zu"]),
            zl=np.array(config_dict["zl"]),
            zu=np.array(config_dict["zu"]),
            Zl_e=np.diag(config_dict["Zl_e"]),
            Zu_e=np.diag(config_dict["Zu_e"]),
            zl_e=np.array(config_dict["zl_e"]),
            zu_e=np.array(config_dict["zu_e"]),
        )

    def to_dict(self):
        return asdict(self)


@dataclass
class ConstraintParams:
    """
    Parameter class for Cartpole constraints in MPC.
    """

    constraint_type: str
    x0: np.ndarray
    lbu: np.ndarray
    ubu: np.ndarray
    lbx: np.ndarray
    ubx: np.ndarray
    idxbx: np.ndarray
    idxbu: np.ndarray
    idxsbx: np.ndarray = field(default_factory=lambda: np.array([]))
    idxsbu: np.ndarray = field(default_factory=lambda: np.array([]))

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            constraint_type=config_dict["constraint_type"],
            x0=np.array(config_dict["x0"]),
            lbu=np.array(config_dict["lbu"]),
            ubu=np.array(config_dict["ubu"]),
            lbx=np.array(config_dict["lbx"]),
            ubx=np.array(config_dict["ubx"]),
            idxbx=np.array(config_dict["idxbx"]),
            idxbu=np.array(config_dict["idxbu"]),
            idxsbx=np.array(config_dict["idxsbx"]),
            idxsbu=np.array(config_dict["idxsbu"]),
        )

    def to_dict(self):
        return asdict(self)


@dataclass
class Dimensions:
    """
    Parameter class for Cartpole dimensions in MPC.
    """

    nx: int  # number of states
    nu: int  # number of inputs
    N: int  # horizon length

    @classmethod
    def from_dict(cls, config_dict: dict):
        return Dimensions(**config_dict)

    def to_dict(self):
        return asdict(self)


class OcpOptions(AcadosOcpOptions):
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
class Meta:
    """
    Parameter class for Cartpole meta parameters in MPC.
    """

    json_file: str = "acados_ocp.json"

    @classmethod
    def from_dict(cls, config_dict: dict):
        return Meta(**config_dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class Config:
    """Configuration class for managing mpc parameters."""

    model_name: Optional[str]
    model_params: Optional[ModelParams]
    cost: Optional[CostParams]
    constraints: Optional[ConstraintParams]
    dimensions: Optional[Dimensions]
    ocp_options: Optional[OcpOptions]
    meta: Optional[Meta]

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            model_name=config_dict["model"]["name"],
            model_params=ModelParams.from_dict(config_dict["model"]["params"]),
            cost=CostParams.from_dict(config_dict["cost"]),
            constraints=ConstraintParams.from_dict(config_dict["constraints"]),
            dimensions=Dimensions.from_dict(config_dict["dimensions"]),
            ocp_options=OcpOptions.from_dict(config_dict["ocp_options"]),
            meta=Meta.from_dict(config_dict["meta"]),
        )

    def to_dict(self) -> dict:
        config_dict = {}

        if self.model_name is not None:
            config_dict["model_name"] = self.model_name

        if self.model_params is not None:
            config_dict["model_params"] = self.model_params.to_dict()

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


def define_model_expressions(config: Config) -> (dict, np.ndarray):
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
        (p["g"] * sin_theta - cos_theta * temp)  # theta_ddot
        / (p["l"] * (4.0 / 3.0 - p["m"] * cos_theta**2 / (p["m"] + p["M"]))),
    )

    f_impl = x_dot - f_expl

    model = dict()
    model["f_impl_expr"] = f_impl
    model["f_expl_expr"] = f_expl
    model["x"] = x
    model["xdot"] = x_dot
    model["p"] = p_sym
    model["u"] = u
    model["z"] = z
    model["name"] = name

    return model, parameter_values


def define_dimensions(config: Config) -> dict:
    dims = dict()
    dims["nx"] = config.dimensions.nx
    dims["nu"] = config.dimensions.nu
    dims["N"] = config.dimensions.N
    dims["ny"] = dims["nx"] + dims["nu"]
    dims["ny_e"] = dims["nx"]

    return dims


def define_cost(config: Config) -> dict:
    cost = dict()

    dims = define_dimensions(config)

    cost["cost_type"] = config.cost.cost_type
    cost["cost_type_e"] = config.cost.cost_type_e

    cost["W"] = scipy.linalg.block_diag(config.cost.Q, config.cost.R)
    cost["W_e"] = config.cost.Q_e
    cost["yref"] = np.zeros((dims["ny"],))
    cost["yref_e"] = np.zeros((dims["ny_e"],))
    cost["cost_type"] = config.cost.cost_type
    cost["cost_type_e"] = config.cost.cost_type_e

    cost["Vx"] = np.zeros((dims["ny"], dims["nx"]), dtype=np.float32)
    cost["Vx"][: dims["nx"], : dims["nx"]] = np.eye(dims["nx"])

    cost["Vu"] = np.zeros((dims["ny"], dims["nu"]))
    cost["Vu"][-1, 0] = 1.0

    cost["Vx_e"] = np.eye(dims["nx"])

    cost["Zl"] = config.cost.Zl
    cost["Zu"] = config.cost.Zu
    cost["zl"] = config.cost.zl
    cost["zu"] = config.cost.zu
    cost["Zl_e"] = config.cost.Zl_e
    cost["Zu_e"] = config.cost.Zu_e
    cost["zl_e"] = config.cost.zl_e
    cost["zu_e"] = config.cost.zu_e

    return cost


def define_constraints(config: Config) -> dict:
    constraints = dict()

    constraints["constr_type"] = config.constraints.constraint_type
    constraints["x0"] = config.constraints.x0.reshape(-1)
    constraints["lbu"] = config.constraints.lbu.reshape(-1)
    constraints["ubu"] = config.constraints.ubu.reshape(-1)
    constraints["lbx"] = config.constraints.lbx.reshape(-1)
    constraints["ubx"] = config.constraints.ubx.reshape(-1)
    constraints["idxbx"] = config.constraints.idxbx.reshape(-1)
    constraints["idxbu"] = config.constraints.idxbu.reshape(-1)
    constraints["idxsbx"] = config.constraints.idxsbx.reshape(-1)
    constraints["idxsbu"] = config.constraints.idxsbu.reshape(-1)

    return constraints


# def define_parameters(config: Config) -> np.array:
#     return np.array(
#         [
#             config.model.M,
#             config.model.m,
#             config.model.l,
#             config.model.g,
#         ]
#     )
