import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
import casadi as cs

from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt


def disc_dyn_expr(x, u, param):
    return param["A"] @ x + param["B"] @ u + param["b"]


def cost_expr_ext_cost(x, u, p):
    f = get_parameter("f", p)
    return 0.5 * (cs.mtimes([x.T, x]) + 0.5 * cs.mtimes([u.T, u])) + cs.mtimes([f.T, cs.vertcat(x, u)])


def cost_expr_ext_cost_0(x, u, p):
    V_0 = get_parameter("V_0", p)
    return V_0 + cost_expr_ext_cost(x, u, p)


def cost_expr_ext_cost_e(x, param, N):
    return 0.5 * param["gamma"] ** N * cs.mtimes([x.T, solve_discrete_are(param["A"], param["B"], param["Q"], param["R"]), x])


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


def setup_acados_ocp_solver(param: dict) -> AcadosOcpSolver:
    ocp = AcadosOcp()

    ocp.model.name = "lti"

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
    ocp.constraints.lbx = np.array([-1.0, -1.0])
    ocp.constraints.ubx = np.array([1.0, 1.0])

    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.lbu = np.array([-1.0])
    ocp.constraints.ubu = np.array([1.0])

    ocp.solver_options.tf = ocp.dims.N
    ocp.solver_options.integrator_type = "DISCRETE"

    return AcadosOcpSolver(ocp)


def test_acados_ocp_solver(param: dict) -> None:
    ocp_solver = setup_acados_ocp_solver(param)

    x0 = np.array([[0.5], [0.5]])

    for stage in range(ocp_solver.acados_ocp.dims.N + 1):
        ocp_solver.set(stage, "p", ocp_solver.acados_ocp.parameter_values)

    ocp_solver.solve_for_x0(x0)

    X = np.vstack([ocp_solver.get(stage, "x") for stage in range(ocp_solver.acados_ocp.dims.N + 1)])
    U = np.vstack([ocp_solver.get(stage, "u") for stage in range(ocp_solver.acados_ocp.dims.N)])

    plt.figure(1)
    plt.subplot(211)
    plt.grid()
    plt.plot(X)
    plt.subplot(212)
    plt.stairs(edges=np.arange(ocp_solver.acados_ocp.dims.N + 1), values=U[:, 0])
    plt.grid()
    plt.show()


if __name__ == "__main__":
    param = {
        "A": np.array([[1.0, 0.25], [0.0, 1.0]]),
        "B": np.array([[0.03125], [0.25]]),
        "Q": np.identity(2),
        "R": np.identity(1),
        "b": np.array([[0.0], [0.0]]),
        "gamma": 0.9,
        "f": np.array([[0.0], [0.0], [0.0]]),
        "V_0": np.array([1e-3]),
    }

    test_acados_ocp_solver(param)
