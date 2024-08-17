import copy
import numpy as np
import casadi as cs
from scipy.linalg import solve_discrete_are
from acados_template import AcadosOcp, AcadosOcpSolver
import scipy.linalg as linalg
# from rlmpc.mpc.common.acados import set_discount_factor


def disc_dyn_expr(x, u, param):
    """
    Define the discrete dynamics function expression.
    """
    return param["A"] @ x + param["B"] @ u + param["b"]


def cost_expr_ext_cost(x, u, p):
    """
    Define the external cost function expression.
    """
    f = get_parameter("f", p)
    y = cs.vertcat(x, u)
    expr = 0.5 * (cs.mtimes([y.T, y])) + cs.mtimes([f.T, y])
    return expr


def cost_expr_ext_cost_0(x, u, p):
    """
    Define the external cost function expression at stage 0.
    """
    return get_parameter("V_0", p) + cost_expr_ext_cost(x, u, p)


def cost_expr_ext_cost_e(x, param, N):
    """
    Define the external cost function expression at the terminal stage as the solution of the discrete-time algebraic Riccati
    equation.
    """

    return 0.5 * cs.mtimes([x.T, solve_discrete_are(param["A"], param["B"], param["Q"], param["R"]), x])


def get_parameter(field, p):
    if field == "A":
        return cs.reshape(p[:4], 2, 2)
    elif field == "B":
        return cs.reshape(p[4:6], 2, 1)
    elif field == "b":
        return cs.reshape(p[6:8], 2, 1)
    elif field == "V_0":
        return p[8]
    elif field == "f":
        return cs.reshape(p[9:12], 3, 1)


def setup_ocp_solver(
    param: dict,
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM",
    hessian_approx: str = "GAUSS_NEWTON",
    integrator_type: str = "IRK",
    nlp_solver_type: str = "SQP",
    name: str = "lti",
    **ocp_solver_kwargs,
) -> AcadosOcpSolver:
    ocp = export_parametric_ocp(
        param=param,
        qp_solver=qp_solver,
        hessian_approx=hessian_approx,
        integrator_type=integrator_type,
        nlp_solver_type=nlp_solver_type,
        name=name,
    )

    return AcadosOcpSolver(ocp, **ocp_solver_kwargs)


def setup_ocp_sensitivity_solver(ocp_solver: AcadosOcpSolver, discount_factor: float = 0.99, **kwargs) -> AcadosOcpSolver:
    ocp = copy.deepcopy(ocp_solver.acados_ocp)
    ocp.name = ocp.name + "_sensitivity"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver_ric_alg = 0

    return AcadosOcpSolver(ocp, **kwargs)


def export_parametric_ocp(
    param: dict,
    cost_type="EXTERNAL",
    qp_solver_ric_alg: int = 0,
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM",
    hessian_approx: str = "GAUSS_NEWTON",
    integrator_type: str = "DISCRETE",
    nlp_solver_type: str = "SQP",
    name: str = "lti",
) -> AcadosOcp:
    ocp = AcadosOcp()

    ocp.model.name = name

    ocp.model.x = cs.SX.sym("x", 2)
    ocp.model.u = cs.SX.sym("u", 1)

    ocp.dims.N = 40
    ocp.dims.nx = 2
    ocp.dims.nu = 1

    A = cs.SX.sym("A", 2, 2)
    B = cs.SX.sym("B", 2, 1)
    b = cs.SX.sym("b", 2, 1)
    V_0 = cs.SX.sym("V_0", 1, 1)
    f = cs.SX.sym("f", 3, 1)

    ocp.model.p = cs.vertcat(cs.reshape(A, -1, 1), cs.reshape(B, -1, 1), cs.reshape(b, -1, 1), V_0, cs.reshape(f, -1, 1))

    ocp.parameter_values = np.concatenate([param[key].T.reshape(-1, 1) for key in ["A", "B", "b", "V_0", "f"]])

    ocp.model.disc_dyn_expr = A @ ocp.model.x + B @ ocp.model.u + b
    # ocp.model.disc_dyn_expr = param["A"] @ ocp.model.x + param["B"] @ ocp.model.u + param["b"]

    # f_disc = cs.Function("f", [ocp.model.x, ocp.model.u], [ocp.model.disc_dyn_expr])

    # print(f_disc(np.array([0.5, 0.5], 0)))

    if cost_type == "LINEAR_LS":
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.Vx_0 = np.zeros((ocp.dims.nx + ocp.dims.nu, ocp.dims.nx))
        ocp.cost.Vx_0[: ocp.dims.nx, : ocp.dims.nx] = np.identity(ocp.dims.nx)
        ocp.cost.Vu_0 = np.zeros((ocp.dims.nx + ocp.dims.nu, ocp.dims.nu))
        ocp.cost.Vu_0[-1, -1] = 1

        ocp.cost.Vx = np.zeros((ocp.dims.nx + ocp.dims.nu, ocp.dims.nx))
        ocp.cost.Vx[: ocp.dims.nx, : ocp.dims.nx] = np.identity(ocp.dims.nx)
        ocp.cost.Vu = np.zeros((ocp.dims.nx + ocp.dims.nu, ocp.dims.nu))
        ocp.cost.Vu[-1, -1] = 1
        ocp.cost.Vx_e = np.identity(ocp.dims.nx)

        ocp.cost.W_0 = linalg.block_diag(param["Q"], param["R"])
        ocp.cost.W = linalg.block_diag(param["Q"], param["R"])
        ocp.cost.W_e = param["Q"]

        ocp.cost.yref_0 = np.zeros(ocp.dims.nx + ocp.dims.nu)
        ocp.cost.yref = np.zeros(ocp.dims.nx + ocp.dims.nu)
        ocp.cost.yref_e = np.zeros(ocp.dims.nx)

    # :math:`l(x,u,z) = 0.5 \cdot || V_x \, x + V_u \, u + V_z \, z - y_\\text{ref}||^2_W`,

    elif cost_type == "EXTERNAL":
        ocp.cost.cost_type_0 = "EXTERNAL"
        ocp.model.cost_expr_ext_cost_0 = cost_expr_ext_cost_0(ocp.model.x, ocp.model.u, ocp.model.p)

        ocp.cost.cost_type = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = cost_expr_ext_cost(ocp.model.x, ocp.model.u, ocp.model.p)

        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost_e = cost_expr_ext_cost_e(ocp.model.x, param, ocp.dims.N)

    ocp.constraints.idxbx_0 = np.array([0, 1])
    ocp.constraints.lbx_0 = np.array([-1.0, -1.0])
    ocp.constraints.ubx_0 = np.array([1.0, 1.0])

    ocp.constraints.idxbx = np.array([0, 1])
    ocp.constraints.lbx = np.array([-0.0, -1.0])
    ocp.constraints.ubx = np.array([+1.0, +1.0])

    ocp.constraints.idxsbx = np.array([0])
    ocp.cost.zl = np.array([1e2])
    ocp.cost.zu = np.array([1e2])
    ocp.cost.Zl = np.diag([0])
    ocp.cost.Zu = np.diag([0])

    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.lbu = np.array([-1.0])
    ocp.constraints.ubu = np.array([+1.0])

    ocp.solver_options.tf = ocp.dims.N
    ocp.solver_options.integrator_type = integrator_type
    ocp.solver_options.nlp_solver_type = nlp_solver_type
    ocp.solver_options.hessian_approx = hessian_approx
    ocp.solver_options.qp_solver = qp_solver
    ocp.solver_options.qp_solver_ric_alg = qp_solver_ric_alg

    return ocp
