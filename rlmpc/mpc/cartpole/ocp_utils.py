import casadi as ca
from casadi.tools.structure3 import DMStruct
from casadi.tools import struct_symSX, entry
import numpy as np
from acados_template import AcadosOcp, AcadosModel

from rlmpc.mpc.common.ocp_utils import export_discrete_erk4_integrator_step


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

    f_disc = export_discrete_erk4_integrator_step(f_expl=f_expl, x=x, u=u, p=p, h=param["Ts"])

    model = AcadosModel()

    model.disc_dyn_expr = f_disc
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p
    model.name = param["model_name"]

    return model


def export_parametric_ocp(
    param: dict,
    x0=np.array([0.0, np.pi / 6, 0.0, 0.0]),
    # N_horizon=50,
    T_horizon=2.0,
    Fmax=80.0,
    qp_solver_ric_alg: int = 1,
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM",
    hessian_approx: str = "GAUSS_NEWTON",
    integrator_type: str = "DISCRETE",
    nlp_solver_type: str = "SQP",
    name: str = "pendulum_on_a_cart",
) -> AcadosOcp:
    ocp = AcadosOcp()
    dt = param["Ts"]
    ocp.dims.N = int(T_horizon / dt)

    ocp.model = export_parametric_model(param=param)

    p = ocp.model.p(0)

    p["Q"] = param["Q"]
    p["R"] = param["R"]
    p["M"] = param["M"]
    p["m"] = param["m"]
    p["g"] = param["g"]
    p["l"] = param["l"]

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.model.cost_expr_ext_cost = 0.5 * (
        ocp.model.x.T @ ocp.model.p["Q"] @ ocp.model.x + ocp.model.u.T @ ocp.model.p["R"] @ ocp.model.u
    )
    ocp.model.cost_expr_ext_cost_e = 0.5 * ocp.model.x.T @ ocp.model.p["Q"] @ ocp.model.x

    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.x0 = x0

    # set mass to one
    ocp.model.p = ocp.model.p.cat
    ocp.parameter_values = p.cat.full().flatten()

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.qp_solver_cond_N = ocp.dims.N

    ocp.solver_options.tf = T_horizon

    ocp.solver_options.qp_solver_ric_alg = qp_solver_ric_alg
    ocp.solver_options.hessian_approx = hessian_approx
    if hessian_approx == "EXACT":
        ocp.solver_options.nlp_solver_step_length = 0.0
        ocp.solver_options.nlp_solver_max_iter = 1
        ocp.solver_options.qp_solver_iter_max = 200
        ocp.solver_options.tol = 1e-10
        ocp.solver_options.with_solution_sens_wrt_params = True
        ocp.solver_options.with_value_sens_wrt_params = True
    else:
        ocp.solver_options.nlp_solver_max_iter = 400
        ocp.solver_options.tol = 1e-8

    return ocp
