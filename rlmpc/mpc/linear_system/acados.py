import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
import casadi as cs

from scipy.linalg import solve_discrete_are

from rlmpc.mpc.common.mpc_nlp_sensitivities import MPC

from rlmpc.mpc.common.nlp import NLP, build_nlp


class AcadosMPC(MPC):
    """docstring for MPC."""

    nlp: NLP

    def __init__(self, param, discount_factor=0.99):
        super(AcadosMPC, self).__init__()

        self.ocp_solver = setup_acados_ocp_solver(param, integrator_type="DISCRETE")

        self.ocp_sensitivity_solver = setup_acados_ocp_solver(param, hessian_approx="EXACT", integrator_type="DISCRETE")

        self.nlp = build_nlp(self.ocp_solver.acados_ocp, gamma=discount_factor)

        self.set_discount_factor(discount_factor)


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


def setup_acados_ocp_solver(
    param: dict,
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM",
    hessian_approx: str = "GAUSS_NEWTON",
    integrator_type: str = "IRK",
    nlp_solver_type: str = "SQP",
    name: str = "lti",
) -> AcadosOcpSolver:
    ocp = export_parametric_ocp(
        param=param,
        qp_solver=qp_solver,
        hessian_approx=hessian_approx,
        integrator_type=integrator_type,
        nlp_solver_type=nlp_solver_type,
        name=name,
    )

    ocp_solver = AcadosOcpSolver(ocp)

    # TODO Is this necessary?
    for stage in range(ocp.dims.N + 1):
        ocp_solver.set(stage, "p", ocp.parameter_values)

    return ocp_solver


def export_parametric_ocp(
    param: dict,
    qp_solver_ric_alg: int = 0,
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM",
    hessian_approx: str = "GAUSS_NEWTON",
    integrator_type: str = "IRK",
    nlp_solver_type: str = "SQP",
    name: str = "lti",
) -> AcadosOcp:
    ocp = AcadosOcp()

    ocp.model.name = name

    ocp.model.x = cs.SX.sym("x", 2)
    ocp.model.u = cs.SX.sym("u", 1)

    ocp.dims.N = 40

    A = cs.SX.sym("A", 2, 2)
    B = cs.SX.sym("B", 2, 1)
    b = cs.SX.sym("b", 2, 1)
    V_0 = cs.SX.sym("V_0", 1, 1)
    f = cs.SX.sym("f", 3, 1)

    ocp.model.p = cs.vertcat(cs.reshape(A, -1, 1), cs.reshape(B, -1, 1), cs.reshape(b, -1, 1), V_0, cs.reshape(f, -1, 1))
    ocp.parameter_values = np.concatenate([param[key].T.reshape(-1, 1) for key in ["A", "B", "b", "V_0", "f"]])

    ocp.model.disc_dyn_expr = A @ ocp.model.x + B @ ocp.model.u + b

    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_0 = cost_expr_ext_cost_0(ocp.model.x, ocp.model.u, ocp.model.p)

    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = cost_expr_ext_cost(ocp.model.x, ocp.model.u, ocp.model.p)

    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = cost_expr_ext_cost_e(ocp.model.x, param, ocp.dims.N)

    ocp.constraints.idxbx_0 = np.array([0, 1])
    ocp.constraints.lbx_0 = np.array([0.0, 0.0])
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
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"

    return ocp
