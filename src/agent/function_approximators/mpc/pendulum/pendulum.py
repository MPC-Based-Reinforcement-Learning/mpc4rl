import scipy
import numpy as np
import casadi as cs
from acados_template import (
    AcadosOcpCost,
    AcadosOcpDims,
    AcadosModel,
    AcadosOcpConstraints,
    AcadosOcpOptions,
    AcadosOcpSolver,
    AcadosOcp,
)


def stack_columns(matrix: np.ndarray) -> np.ndarray:
    return np.hstack([matrix[:, i] for i in range(matrix.shape[1])])


def initialize_solver(
    acados_solver: AcadosOcpSolver, acados_ocp: AcadosOcp, param: dict
) -> int:
    for stage in range(acados_ocp.dims.N):
        acados_solver.cost_set(stage, "yref", np.array(param["yref"]))
        acados_solver.set(stage, "x", np.array(param["x0"]))
        acados_solver.set(stage, "u", np.array(param["u0"]))

    return 0


def export_solver_options(dims: AcadosOcpDims, param: dict) -> AcadosOcpOptions:
    solver_options = AcadosOcpOptions()
    for key, val in param.items():
        setattr(solver_options, key, val)

    return solver_options


def export_constraints(param: dict) -> AcadosOcpConstraints:
    constraints = AcadosOcpConstraints()
    constraints.constr_type = "BGH"
    constraints.lbu = np.array([param["lbu"]]).reshape(-1)
    constraints.ubu = np.array([param["ubu"]]).reshape(-1)
    constraints.idxbu = np.array([0]).reshape(-1)

    constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])

    return constraints


def export_cost(dims: AcadosOcpDims, param: dict):
    cost = AcadosOcpCost()
    cost.cost_type = "LINEAR_LS"
    cost.cost_type_e = "LINEAR_LS"

    # if param["Q"]["fixed"]:
    #     Q = np.array((param["Q"]["value"]), dtype=np.float64)
    # else:
    #     Q = cs.SX.sym("Q", 4, 4)
    #     # p_sym += [Q[:]]
    #     # cost.W_e = Q

    # if param["R"]["fixed"]:
    #     R = np.array((param["R"]["value"]), dtype=np.float64)
    # else:
    #     R = cs.SX.sym("Q", 1, 1)
    # p_sym += [R[:]]

    # if param["Q"]["fixed"] and param["R"]["fixed"]:
    # cost.W = scipy.linalg.block_diag(Q, R)
    # else:
    #     cost.W = cs.SX.zeros("W", 5, 5)
    #     cost.W[:4, :4] = Q
    #     cost.W[4, 4] = R

    Q = np.array((param["Q"]["value"]), dtype=np.float64)
    R = np.array((param["R"]["value"]), dtype=np.float64)
    cost.W = scipy.linalg.block_diag(Q, R)
    cost.W_e = Q

    cost.Vx = np.zeros((dims.ny, dims.nx), dtype=np.float64)
    cost.Vx[: dims.nx, : dims.nx] = np.eye(dims.nx)

    Vu = np.zeros((dims.ny, dims.nu))
    Vu[4, 0] = 1.0

    cost.Vu = Vu
    cost.Vx_e = np.eye(dims.nx)
    cost.yref = np.zeros((dims.ny,))
    cost.yref_e = np.zeros((dims.ny_e,))

    return cost


def export_dims(param: dict) -> AcadosOcpDims:
    dims = AcadosOcpDims()
    dims.nx = param["nx"]
    dims.nu = param["nu"]
    dims.N = param["N"]
    dims.ny = dims.nx + dims.nu
    dims.ny_e = dims.nx
    return dims


def export_model(
    param: dict = {
        "M": {"value": 1.0, "fixed": True},  # mass of the cart [kg]
        "m": {"value": 0.1, "fixed": True},  # mass of the ball [kg]
        "g": {"value": 9.81, "fixed": True},  # gravity constant [m/s^2]
        "l": {"value": 0.8, "fixed": True},  # length of the rod [m]
    }
) -> AcadosModel:
    # Define model variables
    model = AcadosModel()
    model.name = "pendulum_ode"

    # set up states & controls
    x1 = cs.SX.sym("x1")
    theta = cs.SX.sym("theta")
    v1 = cs.SX.sym("v1")
    dtheta = cs.SX.sym("dtheta")

    x = cs.vertcat(x1, theta, v1, dtheta)

    F = cs.SX.sym("F")
    u = cs.vertcat(F)

    # xdot
    x1_dot = cs.SX.sym("x1_dot")
    theta_dot = cs.SX.sym("theta_dot")
    v1_dot = cs.SX.sym("v1_dot")
    dtheta_dot = cs.SX.sym("dtheta_dot")

    xdot = cs.vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot)

    # algebraic variables
    z = None

    # parameters
    p = {}

    parameter_symbols = []
    parameter_values = []
    for param_name, value_dict in param.items():
        if value_dict["fixed"]:
            p[param_name] = value_dict["value"]
        else:
            p[param_name] = cs.SX.sym(param_name)
            parameter_symbols += [p[param_name]]
            parameter_values += [value_dict["value"]]

    # p = []

    # Define model dynamics
    cos_theta = cs.cos(theta)
    sin_theta = cs.sin(theta)
    denominator = p["M"] + p["m"] - p["m"] * cos_theta * cos_theta
    f_expl = cs.vertcat(
        v1,
        dtheta,
        (
            -p["m"] * p["l"] * sin_theta * dtheta * dtheta
            + p["m"] * p["g"] * cos_theta * sin_theta
            + F
        )
        / denominator,
        (
            -p["m"] * p["l"] * cos_theta * sin_theta * dtheta * dtheta
            + F * cos_theta
            + (p["M"] + p["m"]) * p["g"] * sin_theta
        )
        / (p["l"] * denominator),
    )

    f_impl = xdot - f_expl

    # Define output for cost

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z

    # ocp.cost = AcadosOcpCost()
    # ocp.cost.cost_type = "EXTERNAL"
    # ocp.cost.cost_type_e = "EXTERNAL"

    # if param["cost"]["Q"]["fixed"]:
    #     Q_mat = np.array((param["cost"]["Q"]["value"]), dtype=np.float64)
    # else:
    #     Q_mat = cs.SX.sym("Q", 4, 4)
    #     parameter_symbols += [Q_mat[:]]

    #     parameter_values += stack_columns(
    #         np.array(param["cost"]["Q"]["value"], dtype=np.float64)
    #     ).tolist()

    # if param["cost"]["R"]["fixed"]:
    #     R_mat = np.array((param["cost"]["R"]["value"]), dtype=np.float64)
    # else:
    #     R_mat = cs.SX.sym("R", 1, 1)
    #     parameter_symbols += [R_mat[:]]

    #     # R is scalar
    #     parameter_values += param["cost"]["R"]["value"]

    # ocp.model.cost_expr_ext_cost = (
    #     ocp.model.x.T @ Q_mat @ ocp.model.x + ocp.model.u.T @ R_mat @ ocp.model.u
    # )
    # ocp.model.cost_expr_ext_cost_e = ocp.model.x.T @ Q_mat @ ocp.model.x

    # ocp.constraints = AcadosOcpConstraints()
    # ocp.constraints.constr_type = "BGH"
    # ocp.constraints.lbx = np.array(param["constraints"]["lbx"], dtype=np.float64)
    # ocp.constraints.ubx = np.array(param["constraints"]["ubx"], dtype=np.float64)
    # ocp.constraints.idxbx = np.array([0, 1, 2, 3], dtype=np.int64)

    # ocp.model.p = cs.vertcat(*parameter_symbols)

    # ocp.parameter_values = np.array(parameter_values, dtype=np.float64)

    # ocp.dims = AcadosOcpDims()
    # ocp.dims.N = param["dims"]["N"]
    # ocp.dims.nx = param["dims"]["nx"]
    # ocp.dims.nu = param["dims"]["nu"]
    # ocp.dims.ny = param["dims"]["ny"]
    # ocp.dims.ny_e = param["dims"]["nx"]

    # parameter_symbols = cs.vertcat(*parameter_symbols)

    return model


def export_ocp(
    param: dict = {
        "M": {"value": 1.0, "fixed": True},  # mass of the cart [kg]
        "m": {"value": 0.1, "fixed": True},  # mass of the ball [kg]
        "g": {"value": 9.81, "fixed": True},  # gravity constant [m/s^2]
        "l": {"value": 0.8, "fixed": True},  # length of the rod [m]
        "Q": {
            "value": np.diag((100, 100, 1, 1)),
            "fixed": True,
        },  # length of the rod [m]
        "R": {"value": np.array((0.001)), "fixed": True},  # length of the rod [m]
        "lbu": {"value": -80, "fixed": True},  # lower bound on control input [N]
        "ubu": {"value": 80, "fixed": True},  # upper bound on control input [N]
    }
) -> AcadosOcp:
    ocp = AcadosOcp("/usr/local")

    # Define model variables
    ocp.model = AcadosModel()
    ocp.model.name = "pendulum_ode"

    # set up states & controls
    x1 = cs.SX.sym("x1")
    theta = cs.SX.sym("theta")
    v1 = cs.SX.sym("v1")
    dtheta = cs.SX.sym("dtheta")

    x = cs.vertcat(x1, theta, v1, dtheta)

    F = cs.SX.sym("F")
    u = cs.vertcat(F)

    # xdot
    x1_dot = cs.SX.sym("x1_dot")
    theta_dot = cs.SX.sym("theta_dot")
    v1_dot = cs.SX.sym("v1_dot")
    dtheta_dot = cs.SX.sym("dtheta_dot")

    xdot = cs.vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot)

    # algebraic variables
    # z = None

    # parameters
    p = {}
    parameter_symbols = []
    parameter_values = []
    for param_name, value_dict in param["model"].items():
        if value_dict["fixed"]:
            p[param_name] = value_dict["value"]
        else:
            p[param_name] = cs.SX.sym(param_name)
            parameter_symbols += [p[param_name]]
            parameter_values += [value_dict["value"]]

    # p = []

    # Define model dynamics
    cos_theta = cs.cos(theta)
    sin_theta = cs.sin(theta)
    denominator = p["M"] + p["m"] - p["m"] * cos_theta * cos_theta
    f_expl = cs.vertcat(
        v1,
        dtheta,
        (
            -p["m"] * p["l"] * sin_theta * dtheta * dtheta
            + p["m"] * p["g"] * cos_theta * sin_theta
            + F
        )
        / denominator,
        (
            -p["m"] * p["l"] * cos_theta * sin_theta * dtheta * dtheta
            + F * cos_theta
            + (p["M"] + p["m"]) * p["g"] * sin_theta
        )
        / (p["l"] * denominator),
    )

    f_impl = xdot - f_expl

    # Define output for cost

    ocp.model.f_impl_expr = f_impl
    ocp.model.f_expl_expr = f_expl
    ocp.model.x = x
    ocp.model.xdot = xdot
    ocp.model.u = u
    # model.z = z

    ocp.cost = AcadosOcpCost()
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    if param["cost"]["Q"]["fixed"]:
        Q_mat = np.array((param["cost"]["Q"]["value"]), dtype=np.float64)
    else:
        Q_mat = cs.SX.sym("Q", 4, 4)
        parameter_symbols += [Q_mat[:]]

        parameter_values += stack_columns(
            np.array(param["cost"]["Q"]["value"], dtype=np.float64)
        ).tolist()

    if param["cost"]["R"]["fixed"]:
        R_mat = np.array((param["cost"]["R"]["value"]), dtype=np.float64)
    else:
        R_mat = cs.SX.sym("R", 1, 1)
        parameter_symbols += [R_mat[:]]

        # R is scalar
        parameter_values += param["cost"]["R"]["value"]

    ocp.model.cost_expr_ext_cost = (
        ocp.model.x.T @ Q_mat @ ocp.model.x + ocp.model.u.T @ R_mat @ ocp.model.u
    )
    ocp.model.cost_expr_ext_cost_e = ocp.model.x.T @ Q_mat @ ocp.model.x

    ocp.constraints = AcadosOcpConstraints()
    ocp.constraints.constr_type = "BGH"
    ocp.constraints.lbx = np.array(param["constraints"]["lbx"], dtype=np.float64)
    ocp.constraints.ubx = np.array(param["constraints"]["ubx"], dtype=np.float64)
    ocp.constraints.idxbx = np.array([0, 1, 2, 3], dtype=np.int64)

    ocp.model.p = cs.vertcat(*parameter_symbols)

    ocp.parameter_values = np.array(parameter_values, dtype=np.float64)

    ocp.dims = AcadosOcpDims()
    ocp.dims.N = param["dims"]["N"]
    ocp.dims.nx = param["dims"]["nx"]
    ocp.dims.nu = param["dims"]["nu"]
    ocp.dims.ny = param["dims"]["ny"]
    ocp.dims.ny_e = param["dims"]["nx"]

    parameter_symbols = cs.vertcat(*parameter_symbols)

    return ocp
