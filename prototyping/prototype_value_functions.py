import sys

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosOcpConstraints, AcadosSim
import numpy as np
import scipy.linalg

from casadi import SX, vertcat, sin, cos, Function, jacobian

import os
import matplotlib.pyplot as plt
import numpy as np
from acados_template import latexify_plot


def plot_pendulum(
    shooting_nodes,
    u_max,
    U,
    X_true,
    X_est=None,
    Y_measured=None,
    latexify=False,
    plt_show=True,
    X_true_label=None,
):
    """
    Params:
        shooting_nodes: time values of the discretization
        u_max: maximum absolute value of u
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_true: arrray with shape (N_sim, nx)
        X_est: arrray with shape (N_sim-N_mhe, nx)
        Y_measured: array with shape (N_sim, ny)
        latexify: latex style plots
    """

    if latexify:
        latexify_plot()

    WITH_ESTIMATION = X_est is not None and Y_measured is not None

    N_sim = X_true.shape[0]
    nx = X_true.shape[1]

    Tf = shooting_nodes[N_sim - 1]
    t = shooting_nodes

    Ts = t[1] - t[0]
    if WITH_ESTIMATION:
        N_mhe = N_sim - X_est.shape[0]
        t_mhe = np.linspace(N_mhe * Ts, Tf, N_sim - N_mhe)

    plt.subplot(nx + 1, 1, 1)
    (line,) = plt.step(t, np.append([U[0]], U))
    if X_true_label is not None:
        line.set_label(X_true_label)
    else:
        line.set_color("r")
    plt.title("closed-loop simulation")
    plt.ylabel("$u$")
    plt.xlabel("$t$")
    plt.hlines(u_max, t[0], t[-1], linestyles="dashed", alpha=0.7)
    plt.hlines(-u_max, t[0], t[-1], linestyles="dashed", alpha=0.7)
    plt.ylim([-1.2 * u_max, 1.2 * u_max])
    plt.grid()

    states_lables = ["$x$", r"$\theta$", "$v$", r"$\dot{\theta}$"]

    for i in range(nx):
        plt.subplot(nx + 1, 1, i + 2)
        (line,) = plt.plot(t, X_true[:, i], label="true")
        if X_true_label is not None:
            line.set_label(X_true_label)

        if WITH_ESTIMATION:
            plt.plot(t_mhe, X_est[:, i], "--", label="estimated")
            plt.plot(t, Y_measured[:, i], "x", label="measured")

        plt.ylabel(states_lables[i])
        plt.xlabel("$t$")
        plt.grid()
        plt.legend(loc=1)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

    # avoid plotting when running on Travis
    if os.environ.get("ACADOS_ON_CI") is None and plt_show:
        plt.show()


def export_pendulum_ode_model() -> AcadosModel:
    model_name = "pendulum"

    # constants
    if False:
        m_cart = 1.0  # mass of the cart [kg]
    else:
        m_cart = SX.sym("m_cart")
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

    # algebraic variables
    # z = None

    # parameters
    p = m_cart

    # dynamics
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    denominator = m_cart + m - m * cos_theta * cos_theta
    f_expl = vertcat(
        v1,
        dtheta,
        (-m * l * sin_theta * dtheta * dtheta + m * g * cos_theta * sin_theta + F) / denominator,
        (-m * l * cos_theta * sin_theta * dtheta * dtheta + F * cos_theta + (m_cart + m) * g * sin_theta)
        / (l * denominator),
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


# def Lagrange(solver: AcadosOcpSolver) -> np.float:
#     cost = solver.get_cost()

#     f = solver.acados_ocp.model.f_expl_expr


def rename_key_in_dict(d: dict, old_key: str, new_key: str):
    d[new_key] = d.pop(old_key)

    return d


def rename_item_in_list(lst: list, old_item: str, new_item: str):
    if old_item in lst:
        index_old = lst.index(old_item)
        lst[index_old] = new_item

    return lst


class ConstraintDimension(object):
    """
    Class to store dimensions of constraints
    """

    order: list = [
        "lbu",
        "lbx",
        "lg",
        "lh",
        "lphi",
        "ubu",
        "ubx",
        "ug",
        "uh",
        "uphi",
        "lsbu",
        "lsbx",
        "lsg",
        "lsh",
        "lsphi",
        "usbu",
        "usbx",
        "usg",
        "ush",
        "usphi",
    ]

    idx_at_stage: list

    def __init__(self, constraints: AcadosOcpConstraints, N: int = 20):
        super().__init__()

        replacements = {
            0: [("lbx", "lbx_0"), ("ubx", "ubx_0")],
            N: [
                ("lbx", "lbx_e"),
                ("ubx", "ubx_e"),
                ("lg", "lg_e"),
                ("ug", "ug_e"),
                ("lh", "lh_e"),
                ("uh", "uh_e"),
                ("lphi", "lphi_e"),
                ("uphi", "uphi_e"),
                ("lsbx", "lsbx_e"),
                ("usbx", "usbx_e"),
                ("lsg", "lsg_e"),
                ("usg", "usg_e"),
                ("lsh", "lsh_e"),
                ("ush", "ush_e"),
                ("lsphi", "lsphi_e"),
                ("usphi", "usphi_e"),
            ],
        }

        idx_at_stage = [dict.fromkeys(self.order, 0) for _ in range(N + 1)]

        for stage, keys in replacements.items():
            for old_key, new_key in keys:
                idx_at_stage[stage] = rename_key_in_dict(idx_at_stage[stage], old_key, new_key)

        # Loop over all constraints and count the number of constraints of each type. Store the indices in a dict.
        for stage, idx in enumerate(idx_at_stage):
            print("stage = ", stage)
            _start = 0
            _end = 0
            for attr in dir(constraints):
                if idx.keys().__contains__(attr):
                    _end += len(getattr(constraints, attr))
                    idx[attr] = slice(_start, _end)
                    _start = _end

        self.idx_at_stage = idx_at_stage

        # for stage, keys in replacements.items():
        #     for old_key, new_key in keys:
        #         idx_at_stage[stage] = rename_key_in_dict(idx_at_stage[stage], new_key, old_key)

    def get_idx_at_stage(self, stage: int, field: str) -> slice:
        """
        Get the indices of the constraints of the given type at the given stage.

        Parameters:
            stage: stage index
            field: constraint type

        Returns:
            indices: slice object
        """
        return self.idx_at_stage[stage][field]

    def extract_multiplier_at_stage(self, stage: int, field: str, lam: np.ndarray) -> np.ndarray:
        """
        Extract the multipliers at the given stage from the vector of multipliers.

        Parameters:
            stage: stage index
            field: constraint type
            lam: vector of multipliers

        Returns:
            lam: vector of multipliers at the given stage and of the given type
        """
        return lam[self.get_idx_at_stage(stage, field)]


class LagrangeFunction(object):
    """
    Lagrange function for the OCP
    """

    # f_theta: Function
    # ocp: AcadosOcp

    lam_extractor: ConstraintDimension
    integrator: AcadosSimSolver
    fun_f: Function  # Discrete dynamics
    fun_df_dp: Function  # Derivative of discrete dynamics with respect to parameters
    fun_l: Function  # Stage cost
    fun_m: Function  # Terminal cost
    fun_dl_dp: Function  # Derivative of stage cost with respect to parameters
    fun_dm_dp: Function  # Derivative of terminal cost with respect to parameters

    def __init__(self, acados_ocp_solver: AcadosOcpSolver, sim_solver: AcadosSimSolver):
        super().__init__()

        # self.ocp = ocp_solver.acados_ocp
        self.integrator = sim_solver

        self.lam_extractor = ConstraintDimension(
            acados_ocp_solver.acados_ocp.constraints, acados_ocp_solver.acados_ocp.dims.N
        )

        x = acados_ocp_solver.acados_ocp.model.x
        u = acados_ocp_solver.acados_ocp.model.u
        p = acados_ocp_solver.acados_ocp.model.p
        f = acados_ocp_solver.acados_ocp.model.disc_dyn_expr

        self.fun_f = Function("f", [x, u, p], [f], ["x", "u", "p"], ["f"])
        self.fun_df_dp = Function("df_dp", [x, u, p], [jacobian(f, p)], ["x", "u", "p"], ["df_dp"])

        # Build the cost function
        if acados_ocp_solver.acados_ocp.cost.cost_type == "LINEAR_LS":
            """
            In case of LINEAR_LS:
            stage cost is
            :math:`l(x,u,z) = || V_x \, x + V_u \, u + V_z \, z - y_\\text{ref}||^2_W`,
            terminal cost is
            :math:`m(x) = || V^e_x \, x - y_\\text{ref}^e||^2_{W^e}`
            """

            W = acados_ocp_solver.acados_ocp.cost.W
            W_e = acados_ocp_solver.acados_ocp.cost.W_e
            Vx = acados_ocp_solver.acados_ocp.cost.Vx
            Vu = acados_ocp_solver.acados_ocp.cost.Vu
            Vx_e = acados_ocp_solver.acados_ocp.cost.Vx_e
            yref = acados_ocp_solver.acados_ocp.cost.yref
            yref_e = acados_ocp_solver.acados_ocp.cost.yref_e

            l = (Vx @ x + Vu @ u - yref).T @ W @ (Vx @ x + Vu @ u - yref)
            m = (Vx_e @ x - yref_e).T @ W_e @ (Vx_e @ x - yref_e)

            self.fun_l = Function("l", [x, u, p], [l], ["x", "u", "p"], ["l"])
            self.fun_m = Function("m", [x, p], [m], ["x", "p"], ["m"])

            self.fun_dl_dp = Function("dl_dp", [x, u, p], [jacobian(l, p)], ["x", "u", "p"], ["dl_dp"])
            self.fun_dm_dp = Function("dm_dp", [x, p], [jacobian(m, p)], ["x", "p"], ["dm_dp"])

    def __call__(self, acados_ocp_solver: AcadosOcpSolver, p: np.ndarray) -> np.float32:
        """
        Evaluate the Lagrange function at the current solution of the OCP.
        """

        x = np.vstack([acados_ocp_solver.get(stage, "x") for stage in range(acados_ocp_solver.acados_ocp.dims.N + 1)])
        u = np.vstack([acados_ocp_solver.get(stage, "u") for stage in range(acados_ocp_solver.acados_ocp.dims.N)])

        chi = np.vstack([acados_ocp_solver.get(stage, "pi") for stage in range(acados_ocp_solver.acados_ocp.dims.N)])

        # lam = np.vstack(
        #     [acados_ocp_solver.get(stage, "lam") for stage in range(1, acados_ocp_solver.acados_ocp.dims.N)]
        # )

        self.integrator.set("p", p)

        # acados_ocp_solver.set(0, "lbx", 0 * x[0, :])

        # Assumes undiscounted cost function

        nu = acados_ocp_solver.acados_ocp.dims.nu
        nx = acados_ocp_solver.acados_ocp.dims.nx

        cd = ConstraintDimension(acados_ocp_solver.acados_ocp.constraints)

        res = 0.0

        lam = acados_ocp_solver.get(0, "lam")

        constraints = acados_ocp_solver.acados_ocp.constraints

        # Initial condition equality constraint
        stage = 0
        res += cd.extract_multiplier_at_stage(stage, "lbx_0", acados_ocp_solver.get(stage, "lam")) @ (
            getattr(constraints, "lbx_0") - acados_ocp_solver.get(stage, "x")
        )

        res += cd.extract_multiplier_at_stage(stage, "ubx_0", acados_ocp_solver.get(stage, "lam")) @ (
            getattr(constraints, "ubx_0") - acados_ocp_solver.get(stage, "x")
        )

        # State inequality constraints at stage k
        for stage in range(1, acados_ocp_solver.acados_ocp.dims.N):
            res += cd.extract_multiplier_at_stage(stage, "lbx", acados_ocp_solver.get(stage, "lam")) @ (
                getattr(constraints, "lbx") - acados_ocp_solver.get(stage, "u")
            )

            res += cd.extract_multiplier_at_stage(stage, "ubx", acados_ocp_solver.get(stage, "lam")) @ (
                getattr(constraints, "ubx") - acados_ocp_solver.get(stage, "u")
            )

        # Terminal state inequality constraints
        stage = acados_ocp_solver.acados_ocp.dims.N
        if getattr(constraints, "lbx_e").size > 0:
            res += cd.extract_multiplier_at_stage(stage, "lbx_e", acados_ocp_solver.get(stage, "lam")) @ (
                getattr(constraints, "lbx_e") - acados_ocp_solver.get(stage, "x")
            )

        if getattr(constraints, "ubx_e").size > 0:
            res += cd.extract_multiplier_at_stage(stage, "ubx_e", acados_ocp_solver.get(stage, "lam")) @ (
                getattr(constraints, "ubx_e") - acados_ocp_solver.get(stage, "x")
            )

        # Contol inequality constraints
        for stage in range(0, acados_ocp_solver.acados_ocp.dims.N):
            res += cd.extract_multiplier_at_stage(stage, "lbu", acados_ocp_solver.get(stage, "lam")) @ (
                getattr(constraints, "lbu") - acados_ocp_solver.get(stage, "u")
            )

            res += cd.extract_multiplier_at_stage(stage, "ubu", acados_ocp_solver.get(stage, "lam")) @ (
                getattr(constraints, "ubu") - acados_ocp_solver.get(stage, "u")
            )

        res += acados_ocp_solver.get_cost()

        # Dynamic equality constraint
        for stage in range(acados_ocp_solver.acados_ocp.dims.N - 1):
            res += acados_ocp_solver.get(stage + 1, "pi") @ (
                self.integrator.simulate(x=x[stage, :], u=u[stage, :], p=p) - x[stage + 1, :]
            )

        # TODO: Reconstrunct the cost function from the acados_ocp_solver object and accumlate the stage costs.
        # Only needed for the discounted cost function.
        gamma = 1.0  # Discount factor. Placeholder for now.
        # Accumulated stage cost
        # for stage in range(acados_ocp_solver.acados_ocp.dims.N):
        #     res += (
        #         acados_ocp_solver.get(stage, "lam") @ (acados_ocp_solver.get(stage, "cost") - acados_ocp_solver.get_cost())
        #     )

        return res

    def eval_df_dp(self, x: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient of the integrator with respect to the parameters at the current solution of the OCP.

        Parameters:
            x: state
            u: control
            p: parameters

        Returns:
            df_dp: gradient of the integrator with respect to the parameters
        """
        return self.fun_df_dp(x=x, u=u, p=p)["df_dp"].full().flatten()

    def eval_dL_dp(self, acados_ocp_solver: AcadosOcpSolver, p: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient of the Lagrange function with respect to the parameters at the current solution of the OCP.

        Parameters:
            acados_ocp_solver: acados OCP solver object with the current solution
            p: parameters

        Returns:
            dL_dp: gradient of the Lagrange function with respect to the parameters
        """

        res = 0
        # Dynamic equality constraint
        for stage in range(acados_ocp_solver.acados_ocp.dims.N - 1):
            res += acados_ocp_solver.get(stage + 1, "pi") @ self.eval_df_dp(
                acados_ocp_solver.get(stage, "x"), acados_ocp_solver.get(stage, "u"), p
            )

        return res


def ERK4(f, x, u, p, h):
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


# create ocp object to formulate the OCP
ocp = AcadosOcp()

# set model
model = export_pendulum_ode_model()

Tf = 1.0
N = 20
dT = Tf / N

ode = Function("ode", [model.x, model.u, model.p], [model.f_expl_expr])
# set up RK4
# k1 = ode(model.x, model.u, model.p)
# k2 = ode(model.x + dT / 2 * k1, model.u, model.p)
# k3 = ode(model.x + dT / 2 * k2, model.u, model.p)
# k4 = ode(model.x + dT * k3, model.u, model.p)
# xf = model.x + dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

model.disc_dyn_expr = ERK4(ode, model.x, model.u, model.p, dT)
print("built RK4 for pendulum model with dT = ", dT)
# print(xf)

###
ocp.model = model

nx = model.x.size()[0]
nu = model.u.size()[0]
ny = nx + nu
ny_e = nx

# set dimensions
ocp.dims.N = N

# set cost module
ocp.cost.cost_type = "LINEAR_LS"
ocp.cost.cost_type_e = "LINEAR_LS"

Q = 2 * np.diag([1e3, 1e3, 1e-2, 1e-2])
R = 2 * np.diag([1e-1])
# R = 2*np.diag([1e0])

ocp.cost.W = scipy.linalg.block_diag(Q, R)

ocp.cost.W_e = Q

ocp.cost.Vx = np.zeros((ny, nx))
ocp.cost.Vx[:nx, :nx] = np.eye(nx)

Vu = np.zeros((ny, nu))
Vu[4, 0] = 1.0
ocp.cost.Vu = Vu

ocp.cost.Vx_e = np.eye(nx)

# ocp.cost.yref  = np.zeros((ny, ))
# ocp.cost.yref_e = np.zeros((ny_e, ))
ocp.cost.yref = np.array((0.0, np.pi, 0.0, 0.0, 0.0))
ocp.cost.yref_e = np.array((0.0, np.pi, 0.0, 0.0))

# set constraints
Fmax = 80
x0 = np.array([0.5, 0.0, 0.0, 0.0])
ocp.constraints.lbu = np.array([-Fmax])
ocp.constraints.ubu = np.array([+Fmax])
ocp.constraints.x0 = x0
ocp.constraints.idxbu = np.array([0])

# ocp.constraints.lbx = np.array([-2.0, -2 * np.pi, -10.0, -10.0])
# ocp.constraints.ubx = np.array([2.0, 2 * np.pi, 10.0, 10.0])
# ocp.constraints.idxbx = np.array([0, 1, 2, 3])

ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
ocp.solver_options.integrator_type = "DISCRETE"
ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI
# ocp.solver_options.sim_method_num_steps = 2

ocp.solver_options.qp_solver_cond_N = N

ocp.solver_options.qp_solver_iter_max = 200

# set prediction horizon
ocp.solver_options.tf = Tf

ocp.parameter_values = np.array([1.0])

acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + model.name + ".json")

sim = AcadosSim()
sim.model = export_pendulum_ode_model()
sim.dims = ocp.dims
sim.parameter_values = ocp.parameter_values
sim.solver_options.integrator_type = "ERK"
sim.solver_options.num_stages = 4
sim.solver_options.T = dT
acados_integrator = AcadosSimSolver(sim, json_file="acados_sim" + model.name + ".json")

Nsim = 100
simX = np.ndarray((Nsim + 1, nx))
simU = np.ndarray((Nsim, nu))
simX[0, :] = x0

if True:
    acados_ocp_solver.set(0, "lbx", simX[0, :])
    acados_ocp_solver.set(0, "ubx", simX[0, :])

    status = acados_ocp_solver.solve()

    ############################################

    if status != 0:
        print(simX[0, :])
        acados_ocp_solver.print_statistics()
        raise Exception("acados acados_ocp_solver returned status {} in closed loop {}. Exiting.".format(status, 0))

    simU[0, :] = acados_ocp_solver.get(0, "u")

    x = simX[0, :]
    u = simU[0, :]

    # Get residuals from acados solver at stage 0
    residuals = acados_ocp_solver.get_residuals()

    res_stat, res_eq, res_ineq, res_comp = residuals[0], residuals[1], residuals[2], residuals[3]

    lagrange_function = LagrangeFunction(acados_ocp_solver, acados_integrator)

    fun_f = Function(
        "f",
        [
            acados_ocp_solver.acados_ocp.model.x,
            acados_ocp_solver.acados_ocp.model.u,
            acados_ocp_solver.acados_ocp.model.p,
        ],
        [acados_ocp_solver.acados_ocp.model.disc_dyn_expr],
    )

    fun_df_dp = Function(
        "df_dp",
        [
            acados_ocp_solver.acados_ocp.model.x,
            acados_ocp_solver.acados_ocp.model.u,
            acados_ocp_solver.acados_ocp.model.p,
        ],
        [jacobian(acados_ocp_solver.acados_ocp.model.disc_dyn_expr, acados_ocp_solver.acados_ocp.model.p)],
    )

    p_test = np.arange(0.5, 1.5, 0.01)
    f_test = np.array([lagrange_function.integrator.simulate(x=x, u=u, p=p) for p in p_test])

    df_dp_test = np.array([lagrange_function.eval_df_dp(x=x, u=u, p=p) for p in p_test])

    # Compute df_dp via central difference of f_test
    df_dp_test_fd = np.zeros_like(df_dp_test)
    for i in range(df_dp_test.shape[1]):
        df_dp_test_fd[:, i] = np.gradient(f_test[:, i], p_test)

    fig, ax = plt.subplots(nrows=f_test.shape[1], ncols=2, sharex=True)

    for i in range(f_test.shape[1]):
        ax[i, 0].plot(p_test, f_test[:, i])
        ax[i, 1].plot(p_test, df_dp_test[:, i])
        ax[i, 1].plot(p_test, df_dp_test_fd[:, i], "--")
        ax[i, 1].legend(["algorithmic differentiation", "finite difference"])
        ax[i, 0].set_ylabel("f[{}]".format(i))
        ax[i, 1].set_ylabel("df_dp[{}]".format(i))

    ax[-1, 0].set_xlabel("p")
    ax[-1, 1].set_xlabel("p")

    for ax_k in ax.reshape(-1):
        ax_k.grid(True)

    #

    integrator = lagrange_function.integrator

    x = acados_ocp_solver.get(0, "x")
    u = acados_ocp_solver.get(0, "u")
    p = np.array([1.0])

    L_theta = lagrange_function(acados_ocp_solver, p)

    print("L_theta = ", L_theta)

    # Compute dL_dp
    L_test = np.array([lagrange_function(acados_ocp_solver, p) for p in p_test])

    dL_dp_test = np.array([lagrange_function.eval_dL_dp(acados_ocp_solver, p) for p in p_test])

    # Compute dL_dp via central difference of L_test
    dL_dp_test_fd = np.gradient(L_test, p_test)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].plot(p_test, L_test)
    ax[1].plot(p_test, dL_dp_test)
    ax[1].plot(p_test, dL_dp_test_fd, "--")
    ax[1].legend(["algorithmic differentiation", "finite difference"])
    ax[0].set_ylabel("L")
    ax[1].set_ylabel("dL_dp")
    ax[1].set_xlabel("p")
    ax[0].grid(True)
    ax[1].grid(True)

    plt.show()

    exit(0)


# closed loop
for i in range(Nsim):
    # solve ocp
    acados_ocp_solver.set(0, "lbx", simX[i, :])
    acados_ocp_solver.set(0, "ubx", simX[i, :])

    status = acados_ocp_solver.solve()

    if status != 0:
        print(simX[i, :])
        acados_ocp_solver.print_statistics()
        raise Exception("acados acados_ocp_solver returned status {} in closed loop {}. Exiting.".format(status, i))

    simU[i, :] = acados_ocp_solver.get(0, "u")

    if False:
        # get sensitivity

        sens_u = np.ndarray((nu, nx))
        sens_x = np.ndarray((nx, nx))
        for index in range(nx):
            acados_ocp_solver.eval_param_sens(index)
            sens_u[:, index] = acados_ocp_solver.get(0, "sens_u")
            sens_x[:, index] = acados_ocp_solver.get(0, "sens_x")

    # simulate system
    acados_integrator.set("x", simX[i, :])
    acados_integrator.set("u", simU[i, :])

    status = acados_integrator.solve()
    if status != 0:
        raise Exception("acados integrator returned status {}. Exiting.".format(status))

    # update state
    simX[i + 1, :] = acados_integrator.get("x")

# plot results
plot_pendulum(np.linspace(0, Tf / N * Nsim, Nsim + 1), Fmax, simU, simX)
