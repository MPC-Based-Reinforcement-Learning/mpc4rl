import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
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
    x = cs.vertcat(c_A, c_B)

    q = cs.SX.sym("q")
    u = cs.vertcat(q)

    c_A_dot = cs.SX.sym("c_A_dot")
    c_B_dot = cs.SX.sym("c_B_dot")
    x_dot = cs.vertcat(c_A_dot, c_B_dot)

    f_expl_expr = cs.vertcat(*sym_ode(x, u, param))

    ocp = AcadosOcp()

    ocp.dims.N = 30
    ocp.solver_options.time_steps = np.array([0.5] * ocp.dims.N)
    ocp.solver_options.tf = ocp.solver_options.time_steps.sum()

    ocp.dims.nx = 2
    ocp.dims.nu = 1
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
        H = np.diag([1.0, 1.0, 1e-4])
        lam = np.ones(2)
        l = 1.0  # noqa F841

    if False:
        x_ss = cs.SX.sym("x_ss", 2)
        u_ss = cs.SX.sym("u_ss", 1)
    else:
        x_ss = np.array([0.5, 0.5])
        u_ss = np.array([4.0])

    c_A_ss = 0.5
    c_B_ss = 0.5
    Q_ss = param["k_r"] * param["V_R"]

    if False:
        ocp.cost.cost_type_0 = "EXTERNAL"
        ocp.cost.cost_ext_fun_type_0 = "casadi"
        ocp.model.cost_expr_ext_cost_0 = (x - x_ss).T @ Lam @ (x - x_ss) + cs.dot(lam, x - x_ss) + l

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_ext_fun_type = "casadi"
        ocp.model.cost_expr_ext_cost = (x - x_ss).T @ H[:-1, :-1] @ (x - x_ss) + (u - u_ss).T @ H[-1, -1] @ (u - u_ss)

        # W_e = 0.5 * np.array([[0.11, 0.04], [0.04, 0.15]])
        # lam_e = np.array([-19.82, -44])
        # ocp.cost.cost_type_e = "EXTERNAL"
        # ocp.cost.cost_ext_fun_type_e = "casadi"
        # ocp.model.cost_expr_ext_cost_e = (x - x_ss).T @ W_e @ (x - x_ss) + cs.dot(lam_e, x - x_ss)

        ocp.constraints.idxbx_e = np.array([0, 1])
        ocp.constraints.lbx_e = np.array([0.5, 0.5])
        ocp.constraints.ubx_e = np.array([0.5, 0.5])
        ocp.constraints.idxsbx_e = np.array([0, 1])
        ocp.cost.Zu_e = np.diag([2e1, 2e1])
        ocp.cost.zu_e = np.array([1e1, 1e1])
        ocp.cost.Zl_e = np.diag([2e1, 2e1])
        ocp.cost.zl_e = np.array([1e1, 1e1])
        # ocp.constraints.lsbx_e = np.array([0.0, 0.0])
        # ocp.constrains.idxsbx_e = np.array([0, 1])
        # ocp.constraints.lsbx_e = np.array([0.0, 0.0])
        # ocp.constraints.usbx_e = np.array([0.0, 0.0])
    else:
        ocp.cost.cost_type_0 = "EXTERNAL"
        ocp.cost.cost_ext_fun_type_0 = "casadi"
        # ocp.model.cost_expr_ext_cost_0 = -(2 * u * x[1] - 0.5 * u) + 0.1 * (u - u_ss) ** 2
        ocp.model.cost_expr_ext_cost_0 = (
            -(2 * q * c_B - 0.5 * q) + 0.505 * (c_A - c_A_ss) ** 2 + 0.505 * (c_B - c_B_ss) ** 2 + 0.505 * (q - Q_ss) ** 2
        )

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_ext_fun_type = "casadi"
        # ocp.model.cost_expr_ext_cost = -(2 * u * x[1] - 0.5 * u) + 0.1 * (u - u_ss) ** 2
        ocp.model.cost_expr_ext_cost = (
            -(2 * q * c_B - 0.5 * q) + 0.505 * (c_A - c_A_ss) ** 2 + 0.505 * (c_B - c_B_ss) ** 2 + 0.505 * (q - Q_ss) ** 2
        )

        W_e = 0.5 * np.array([[0.11, 0.04], [0.04, 0.15]])
        lam_e = np.array([-19.82, -44])
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.cost.cost_ext_fun_type_e = "casadi"
        ocp.model.cost_expr_ext_cost_e = (x - x_ss).T @ W_e @ (x - x_ss) + cs.dot(lam_e, x - x_ss)

    ocp.constraints.lbu = np.array([0.0])
    ocp.constraints.ubu = np.array([20.0])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.x0 = np.array([0.5, 0.5])
    ocp.constraints.idxbx_0 = np.array([0, 1])

    ocp_solver = AcadosOcpSolver(ocp, json_file="cstr_ocp.json")

    return ocp_solver


def test_cstr_prediction(param):
    ocp_solver = define_acados_ocp_solver(param)

    x0 = np.array([1.0, 0.0])

    ocp_solver.solve_for_x0(x0)

    dt = ocp_solver.acados_ocp.solver_options.tf / ocp_solver.acados_ocp.dims.N
    Tf = ocp_solver.acados_ocp.solver_options.tf
    t = np.arange(0.0, Tf + dt, dt)
    X = np.vstack([ocp_solver.get(stage, "x") for stage in range(ocp_solver.acados_ocp.dims.N + 1)])
    U = np.vstack([ocp_solver.get(stage, "u") for stage in range(ocp_solver.acados_ocp.dims.N)])

    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].plot(t, X)
    axes[1].plot(t[:-1], U)
    axes[0].grid()
    axes[1].grid()
    plt.show()


def test_cstr_closed_loop_phase(param):
    ocp_solver = define_acados_ocp_solver(param)

    X0 = [0.5 + 0.5 * np.array([np.sin(phi), np.cos(phi)]) for phi in np.arange(0.0, 2 * np.pi, 2 * np.pi / 9)]
    N_sim = 30

    plt.figure(1)
    for x0 in X0:
        x = [x0]

        for _ in range(N_sim):
            ocp_solver.solve_for_x0(x[-1])
            x.append(ocp_solver.get(1, "x"))

        X = np.vstack(x)

        plt.plot(X[:, 0], X[:, 1])

    plt.grid()
    plt.show()


def test_cstr_closed_loop_time(param):
    ocp_solver = define_acados_ocp_solver(param)

    N_sim = 10

    x = [np.array([1.0, 0.0])]
    u = []

    dt = ocp_solver.acados_ocp.solver_options.tf / ocp_solver.acados_ocp.dims.N
    t = np.arange(0.0, (N_sim + 1) * dt, dt)

    for i in range(N_sim):
        ocp_solver.solve_for_x0(x[-1])
        u.append(ocp_solver.get(0, "u"))
        x.append(ocp_solver.get(1, "x"))

    X = np.vstack(x)
    U = np.vstack(u)

    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].plot(t, X)
    axes[0].legend(["c_A", "c_B"])
    axes[0].set_ylabel("Concentration [mol/L]")
    axes[0].grid()
    axes[1].plot(t[:-1], U)
    axes[1].set_ylabel("Flow rate [L/min]")
    axes[1].set_xlabel("time [min]")
    axes[1].legend(["q"])
    axes[1].grid()
    plt.show()


if __name__ == "__main__":
    param = {"k_r": 0.4, "c_Af": 1.0, "c_Bf": 0.0, "V_R": 10.0}

    test_cstr_closed_loop_phase(param)
    # test_cstr_closed_loop_time(param)
