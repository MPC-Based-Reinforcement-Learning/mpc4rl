import numpy as np

import scipy.integrate as integrate
import matplotlib.pyplot as plt

import casadi as cs

from acados_template import AcadosOcp, AcadosOcpSolver


def erk4_integrator_step(f, x, u, p, h) -> np.ndarray:
    """
    Explicit Runge-Kutta 4 integrator


    Parameters:
        f: function to integrate
        x: state
        u: control
        p: parameters
        h: step size

        Returns:
            xf: integrated state
    """
    k1 = f(x, u, p)
    k2 = f(x + h / 2 * k1, u, p)
    k3 = f(x + h / 2 * k2, u, p)
    k4 = f(x + h * k3, u, p)
    xf = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return xf


def sym_ode(x, u, p):
    k_r = p["k_r"]
    c_Af = p["c_Af"]
    c_Bf = p["c_Bf"]
    V_R = p["V_R"]

    c_A = x[0]
    c_B = x[1]

    q = u

    c_A_dot = (q / V_R) * (c_Af - c_A) - k_r * c_A
    c_B_dot = (q / V_R) * (c_Bf - c_B) + k_r * c_A

    return [c_A_dot, c_B_dot]


def num_ode(x, u, p):
    k_r = p["k_r"]
    c_Af = p["c_Af"]
    c_Bf = p["c_Bf"]
    V_R = p["V_R"]

    c_A = x[0]
    c_B = x[1]

    q = u

    c_A_dot = (q / V_R) * (c_Af - c_A) - k_r * c_A
    c_B_dot = (q / V_R) * (c_Bf - c_B) + k_r * c_A

    return np.array([c_A_dot, c_B_dot])


def test_cstr(param: dict):
    yref = np.array([0.5, 0.5, 4.0])

    rk45 = integrate.RK45(
        lambda t, y: num_ode(y, yref[2], param),
        t0=0,
        # y0=np.array([0.5, 0.5]),
        y0=yref[:2],
        t_bound=10,
        max_step=0.1,
    )

    t = []
    x = []
    u = []
    while rk45.status == "running":
        t.append(rk45.t)
        x.append(rk45.y)
        u.append(yref[2])
        rk45.step()

    t = np.vstack(t)
    x = np.vstack(x)
    u = np.vstack(u)

    figure, axes = plt.subplots()
    axes.plot(t, x[:, 0], label="c_A")
    axes.plot(t, x[:, 1], label="c_B")
    axes.plot(t, u, label="q")
    axes.legend()
    axes.grid()
    plt.show()


def define_acados_ocp_solver(param):
    c_A = cs.SX.sym("c_A")
    c_B = cs.SX.sym("c_B")
    q = cs.SX.sym("q")

    c_A_dot = cs.SX.sym("c_A_dot")
    c_B_dot = cs.SX.sym("c_B_dot")

    x_dot = cs.vertcat(c_A_dot, c_B_dot)

    # k_r = param["k_r"]
    # c_Af = param["c_Af"]
    # c_Bf = param["c_Bf"]
    # V_R = param["V_R"]

    x = cs.vertcat(c_A, c_B)
    u = cs.vertcat(q)

    f_expl_expr = cs.vertcat(*sym_ode(x, u, param))

    ocp = AcadosOcp()

    ocp.dims.N = 100
    ocp.dims.nx = 2
    ocp.dims.nu = 1
    ocp.dims.ny = 3
    ocp.dims.np = 0

    ocp.model.name = "cstr"
    ocp.model.x = x
    ocp.model.u = u
    ocp.model.xdot = x_dot
    ocp.model.z = cs.SX.sym("z", 0, 0)
    ocp.model.f_expl_expr = f_expl_expr
    ocp.model.f_impl_expr = f_expl_expr - x_dot

    if False:
        Lam = cs.SX.sym("Lambda", 2, 2)
        H = cs.SX.sym("H", 2, 2)
        lam = cs.SX.sym("lambda", 2)
        l = cs.SX.sym("l", 1)  # noqa F841
    else:
        Lam = np.identity(2)
        H = np.identity(3)
        lam = np.ones(2)
        l = 1.0  # noqa F841

    if False:
        x_ss = cs.SX.sym("x_ss", 2)
        u_ss = cs.SX.sym("u_ss", 1)
    else:
        x_ss = np.array([0.5, 0.5])
        u_ss = np.array([4.0])

    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.cost.cost_ext_fun_type_0 = "casadi"
    ocp.model.cost_expr_ext_cost_0 = (x - x_ss).T @ Lam @ (x - x_ss) + cs.dot(lam, x - x_ss) + l

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_ext_fun_type = "casadi"
    ocp.model.cost_expr_ext_cost = (x - x_ss).T @ H[:-1, :-1] @ (x - x_ss) + (u - u_ss).T @ H[-1, -1] @ (u - u_ss)
    # ocp.model.cost_y_expr = (x - x_ss).T @ H[:-1, :-1] @ (x - x_ss) + (u - u_ss).T @ H[-1, -1] @ (u - u_ss)

    ocp.solver_options.tf = 100.0

    ocp_solver = AcadosOcpSolver(ocp, json_file="cstr_ocp.json")

    return ocp_solver


if __name__ == "__main__":
    param = {"k_r": 0.4, "c_Af": 1.0, "c_Bf": 0.0, "V_R": 100.0}

    ocp_solver = define_acados_ocp_solver(param)
