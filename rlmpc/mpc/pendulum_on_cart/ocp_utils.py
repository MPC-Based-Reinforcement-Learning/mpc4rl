import casadi as ca
from casadi.tools.structure3 import DMStruct
from casadi.tools import struct_symSX, entry
import numpy as np
from acados_template import AcadosOcp, AcadosModel

from rlmpc.mpc.common.ocp_utils import export_discrete_erk4_integrator_step
import scipy

from casadi import SX, vertcat, sin, cos, Function
import casadi as ca


def export_pendulum_ode_model() -> AcadosModel:
    model_name = "pendulum"

    # constants
    m_cart = 1.0  # mass of the cart [kg]
    m = 0.1  # mass of the ball [kg]
    g = 9.81  # gravity constant [m/s^2]
    l = 0.8  # length of the rod [m]

    # set up states & controls
    x1 = SX.sym("x1")
    theta = SX.sym("theta")
    v1 = SX.sym("v1")
    dtheta = SX.sym("dtheta")

    x = vertcat(x1, theta, v1, dtheta)

    F = SX.sym("F")
    u = vertcat(F)

    # xdot
    x1_dot = SX.sym("x1_dot")
    theta_dot = SX.sym("theta_dot")
    v1_dot = SX.sym("v1_dot")
    dtheta_dot = SX.sym("dtheta_dot")

    xdot = vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot)

    # parameters
    p = []

    # dynamics
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    denominator = m_cart + m - m * cos_theta * cos_theta
    f_expl = vertcat(
        v1,
        dtheta,
        (-m * l * sin_theta * dtheta * dtheta + m * g * cos_theta * sin_theta + F) / denominator,
        (-m * l * cos_theta * sin_theta * dtheta * dtheta + F * cos_theta + (m_cart + m) * g * sin_theta) / (l * denominator),
    )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.name = model_name

    return model


def export_linearized_pendulum(xbar, ubar):
    model = export_pendulum_ode_model()

    val = ca.substitute(ca.substitute(model.f_expl_expr, model.x, xbar), model.u, ubar)
    jac_x = ca.substitute(ca.substitute(ca.jacobian(model.f_expl_expr, model.x), model.x, xbar), model.u, ubar)
    jac_u = ca.substitute(ca.substitute(ca.jacobian(model.f_expl_expr, model.u), model.x, xbar), model.u, ubar)

    model.f_expl_expr = val + jac_x @ (model.x - xbar) + jac_u @ (model.u - ubar)
    model.f_impl_expr = model.f_expl_expr - model.xdot
    model.name += "_linearized"
    return model


def export_pendulum_ode_model_with_discrete_rk4(dT):
    model = export_pendulum_ode_model()

    x = model.x
    u = model.u

    ode = Function("ode", [x, u], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x, u)
    k2 = ode(x + dT / 2 * k1, u)
    k3 = ode(x + dT / 2 * k2, u)
    k4 = ode(x + dT * k3, u)
    xf = x + dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    model.disc_dyn_expr = xf
    # print("built RK4 for pendulum model with dT = ", dT)
    # print(xf)
    return model


def export_linearized_pendulum_ode_model_with_discrete_rk4(dT, xbar, ubar):
    model = export_linearized_pendulum(xbar, ubar)

    x = model.x
    u = model.u

    ode = Function("ode", [x, u], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x, u)
    k2 = ode(x + dT / 2 * k1, u)
    k3 = ode(x + dT / 2 * k2, u)
    k4 = ode(x + dT * k3, u)
    xf = x + dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    model.disc_dyn_expr = xf
    # print("built RK4 for pendulum model with dT = ", dT)
    # print(xf)
    return model


def export_augmented_pendulum_model():
    # pendulum model augmented with algebraic variable just for testing
    model = export_pendulum_ode_model()
    model_name = "augmented_pendulum"

    z = SX.sym("z", 2, 1)

    f_impl = vertcat(model.xdot - model.f_expl_expr, z - vertcat(model.x[0], model.u**2))

    model.f_impl_expr = f_impl
    model.z = z
    model.name = model_name

    return model


def define_param_struct_symSX() -> DMStruct:
    """Define parameter struct."""

    param_entries = [
        entry("M", shape=1),
        entry("m", shape=1),
        entry("g", shape=1),
        entry("l", shape=1),
        entry("Q", shape=(4, 4)),
        entry("R", shape=(1, 1)),
    ]

    return struct_symSX(param_entries)


def export_parametric_model(param: dict) -> AcadosModel:
    # set up states & controls
    x1 = ca.SX.sym("x1")
    theta = ca.SX.sym("theta")
    v1 = ca.SX.sym("v1")
    dtheta = ca.SX.sym("dtheta")

    x = ca.vertcat(x1, theta, v1, dtheta)

    F = ca.SX.sym("F")
    u = F

    # xdot
    x1_dot = ca.SX.sym("x1_dot")
    theta_dot = ca.SX.sym("theta_dot")
    v1_dot = ca.SX.sym("v1_dot")
    dtheta_dot = ca.SX.sym("dtheta_dot")

    xdot = ca.vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot)

    # parameters
    p = define_param_struct_symSX()

    # dynamics
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    denominator = p["M"] + p["m"] - p["m"] * cos_theta * cos_theta
    f_expl = ca.vertcat(
        v1,
        dtheta,
        (-p["m"] * p["l"] * sin_theta * dtheta * dtheta + p["m"] * p["g"] * cos_theta * sin_theta + F) / denominator,
        (-p["m"] * p["l"] * cos_theta * sin_theta * dtheta * dtheta + F * cos_theta + (p["M"] + p["m"]) * p["g"] * sin_theta)
        / (p["l"] * denominator),
    )

    f_impl = xdot - f_expl

    f_disc = export_discrete_erk4_integrator_step(f_expl=f_expl, x=x, u=u, p=p, h=param["Tf"] / param["N"])

    model = AcadosModel()

    model.disc_dyn_expr = f_disc
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p

    return model


def cost_expr_ext_cost_0(x, u, Q_mat, R_mat):
    return 0.5 * (x.T @ Q_mat @ x + u.T @ R_mat @ u)


def cost_expr_ext_cost(x, u, Q_mat, R_mat):
    return 0.5 * (x.T @ Q_mat @ x + u.T @ R_mat @ u)


def cost_expr_ext_cost_e(x, Q_mat):
    return 0.5 * (x.T @ Q_mat @ x)


def export_parametric_ocp(
    param: dict,
    x0=np.array([0.0, np.pi, 0.0, 0.0]),
    N_horizon=20,
    T_horizon=1.0,
    Fmax=80.0,
    qp_solver_ric_alg: int = 1,
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM",
    hessian_approx: str = "GAUSS_NEWTON",
    integrator_type: str = "IRK",
    nlp_solver_type: str = "SQP",
    name: str = "pendulum_on_a_cart",
) -> AcadosOcp:
    ocp = AcadosOcp()

    integrator_type = "DISCRETE"

    Tf = param["Tf"]
    N = param["N"]

    model = export_parametric_model(param=param)

    model.name = name

    f_expl = model.f_expl_expr
    x = model.x
    u = model.u

    p = model.p(0)

    p["Q"] = param["Q"]
    p["R"] = param["R"]
    p["M"] = param["M"]
    p["m"] = param["m"]
    p["g"] = param["g"]
    p["l"] = param["l"]

    model.disc_dyn_expr = export_discrete_erk4_integrator_step(f_expl=f_expl, x=x, u=u, p=model.p, h=Tf / N)
    ocp.parameter_values = p.cat.full().flatten()

    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    # set dimensions
    ocp.dims.N = N

    # set cost
    cost_version = "EXTERNAL"
    if cost_version == "LS":
        Q_mat = 2 * np.diag([1e3, 1e3, 1e-2, 1e-2])
        R_mat = 2 * np.diag([1e-2])
        ocp.cost.W_e = Q_mat
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)

        Vu = np.zeros((ny, nu))
        Vu[4, 0] = 1.0
        ocp.cost.Vu = Vu

        ocp.cost.Vx_e = np.eye(nx)

        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))
    elif cost_version == "EXTERNAL":
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        Q_mat = ocp.model.p["Q"]
        R_mat = ocp.model.p["R"]
        x = ocp.model.x
        u = ocp.model.u

        ocp.model.cost_expr_ext_cost_0 = cost_expr_ext_cost_0(x, u, Q_mat, R_mat)
        ocp.model.cost_expr_ext_cost = cost_expr_ext_cost(x, u, Q_mat, R_mat)
        ocp.model.cost_expr_ext_cost_e = cost_expr_ext_cost_e(x, Q_mat)

    # Need to convert from ssymStruct to SX/MX
    ocp.model.p = ocp.model.p.cat
    # set constraints
    # Fmax = 80
    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.x0 = x0
    ocp.constraints.idxbu = np.array([0])

    ocp.solver_options.qp_solver = qp_solver
    ocp.solver_options.hessian_approx = hessian_approx
    ocp.solver_options.qp_solver_cond_ric_alg = qp_solver_ric_alg
    ocp.solver_options.integrator_type = integrator_type

    # set prediction horizon
    ocp.solver_options.tf = Tf
    ocp.solver_options.nlp_solver_type = nlp_solver_type

    return ocp
