import sys
from typing import Any, Union

from acados_template import (
    AcadosOcp,
    AcadosOcpSolver,
    AcadosSimSolver,
    AcadosModel,
    AcadosOcpConstraints,
    AcadosSim,
)
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
        (-m * l * sin_theta * dtheta * dtheta + m * g * cos_theta * sin_theta + F)
        / denominator,
        (
            -m * l * cos_theta * sin_theta * dtheta * dtheta
            + F * cos_theta
            + (m_cart + m) * g * sin_theta
        )
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


ACADOS_MULTIPLIER_ORDER = [
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


class LagrangeMultiplierExtractor(object):
    """
    Class to store dimensions of constraints
    """

    order: list = ACADOS_MULTIPLIER_ORDER

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
                idx_at_stage[stage] = rename_key_in_dict(
                    idx_at_stage[stage], old_key, new_key
                )

        # Loop over all constraints and count the number of constraints of each type. Store the indices in a dict.
        for stage, idx in enumerate(idx_at_stage):
            _start = 0
            _end = 0
            for attr in dir(constraints):
                if idx.keys().__contains__(attr):
                    _end += len(getattr(constraints, attr))
                    idx[attr] = slice(_start, _end)
                    _start = _end

        self.idx_at_stage = idx_at_stage

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

    def __call__(self, stage: int, field: str, lam: np.ndarray) -> np.ndarray:
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


def build_cost_functions(
    ocp: AcadosOcp,
) -> tuple[Function, Function, Function, Function]:
    """
    Build the cost functions for the OCP.

    Parameters:
        acados_ocp: acados OCP object

    Returns:
        fun_l: stage cost function
        fun_m: terminal cost function
        fun_dl_dp: derivative of stage cost function with respect to parameters
        fun_dm_dp: derivative of terminal cost function with respect to parameters
    """

    W = ocp.cost.W
    W_e = ocp.cost.W_e

    yref = ocp.cost.yref
    yref_e = ocp.cost.yref_e

    x = ocp.model.x
    u = ocp.model.u
    p = ocp.model.p

    if ocp.cost.cost_type == "LINEAR_LS":
        """
        In case of LINEAR_LS:
        stage cost is
        :math:`l(x,u,z) = || V_x \, x + V_u \, u + V_z \, z - y_\\text{ref}||^2_W`,
        terminal cost is
        :math:`m(x) = || V^e_x \, x - y_\\text{ref}^e||^2_{W^e}`
        """

        cost_y_expr = ocp.cost.Vx @ ocp.model.x + ocp.cost.Vu @ ocp.model.u
        cost_y_expr_e = ocp.cost.Vx_e @ ocp.model.x

    elif ocp.cost.cost_type == "NONLINEAR_LS":
        cost_y_expr = ocp.model.cost_y_expr
        cost_y_expr_e = ocp.model.cost_y_expr_e

    else:
        raise NotImplementedError(
            "Only LINEAR_LS and NONLINEAR_LS cost types are supported at the moment."
        )

    l = (cost_y_expr - yref).T @ W @ (cost_y_expr - yref)
    m = (cost_y_expr_e - yref_e).T @ W_e @ (cost_y_expr_e - yref_e)

    fun_l = Function("l", [x, u, p], [l], ["x", "u", "p"], ["l"])
    fun_m = Function("m", [x, p], [m], ["x", "p"], ["m"])
    fun_dl_dp = Function(
        "dl_dp", [x, u, p], [jacobian(l, p)], ["x", "u", "p"], ["dl_dp"]
    )
    fun_dm_dp = Function("dm_dp", [x, p], [jacobian(m, p)], ["x", "p"], ["dm_dp"])

    return fun_l, fun_m, fun_dl_dp, fun_dm_dp


def build_discrete_dynamics_functions(ocp: AcadosOcp) -> tuple[Function, Function]:
    """
    Build the discrete dynamics functions for the OCP.

    Parameters:
        acados_ocp: acados OCP object

    Returns:
        fun_f: discrete dynamics function
        fun_df_dp: derivative of discrete dynamics function with respect to parameters
    """

    x = ocp.model.x
    u = ocp.model.u
    p = ocp.model.p
    f = ocp.model.disc_dyn_expr

    # Build the discrete dynamics function
    if (
        ocp.solver_options.integrator_type == "ERK"
        and ocp.solver_options.sim_method_num_stages == 4
    ):
        f = ERK4(f, x, u, p)
    elif ocp.solver_options.integrator_type == "DISCRETE":
        f = ocp.model.disc_dyn_expr
    else:
        raise NotImplementedError(
            "Only ERK4 and DISCRETE integrator types are supported at the moment."
        )

    fun_f = Function("f", [x, u, p], [f], ["x", "u", "p"], ["xf"])
    fun_df_dp = Function(
        "df_dp", [x, u, p], [jacobian(f, p)], ["x", "u", "p"], ["dxf_dp"]
    )

    return fun_f, fun_df_dp


def build_constraint_functions(
    ocp: AcadosOcp,
) -> tuple[Function, Function, Function, Function]:
    """
    Build the constraint functions for the OCP.

    Parameters:
        acados_ocp: acados OCP object

    Returns:
        fun_g: stage constraint function
        fun_dg_dp: derivative of stage constraint function with respect to parameters
        fun_g_e: terminal constraint function
        fun_dg_dp_e: derivative of terminal constraint function with respect to parameters
    """
    pass


class LagrangeFunction(object):
    """
    Lagrange function for the OCP
    """

    # f_theta: Function
    # ocp: AcadosOcp

    lam_extractor: LagrangeMultiplierExtractor
    fun_f: Function  # Discrete dynamics
    fun_df_dp: Function  # Derivative of discrete dynamics with respect to parameters
    fun_l: Function  # Stage cost
    fun_m: Function  # Terminal cost
    fun_dl_dp: Function  # Derivative of stage cost with respect to parameters
    fun_dm_dp: Function  # Derivative of terminal cost with respect to parameters

    def __init__(self, acados_ocp_solver: AcadosOcpSolver):
        super().__init__()

        self.lam_extractor = LagrangeMultiplierExtractor(
            acados_ocp_solver.acados_ocp.constraints,
            acados_ocp_solver.acados_ocp.dims.N,
        )

        self.fun_f, self.fun_df_dp = build_discrete_dynamics_functions(
            acados_ocp_solver.acados_ocp
        )
        self.fun_l, self.fun_m, self.fun_dl_dp, self.fun_dm_dp = build_cost_functions(
            acados_ocp_solver.acados_ocp
        )
        _ = build_constraint_functions(acados_ocp_solver.acados_ocp)

    def __call__(self, acados_ocp_solver: AcadosOcpSolver, p: np.ndarray) -> np.float32:
        """
        Evaluate the Lagrange function at the current solution of the OCP.
        """

        res = 0.0

        constraints = acados_ocp_solver.acados_ocp.constraints

        # Initial condition equality constraint
        stage = 0
        res += self.lam_extractor(
            stage, "lbx_0", acados_ocp_solver.get(stage, "lam")
        ) @ (getattr(constraints, "lbx_0") - acados_ocp_solver.get(stage, "x"))

        res += self.lam_extractor(
            stage, "ubx_0", acados_ocp_solver.get(stage, "lam")
        ) @ (getattr(constraints, "ubx_0") - acados_ocp_solver.get(stage, "x"))

        # Inequality constraints at stage k
        if getattr(constraints, "lbx").size > 0:
            for stage in range(1, acados_ocp_solver.acados_ocp.dims.N):
                res += self.lam_extractor(
                    stage, "lbx", acados_ocp_solver.get(stage, "lam")
                ) @ (getattr(constraints, "lbx") - acados_ocp_solver.get(stage, "x"))

        if getattr(constraints, "ubx").size > 0:
            for stage in range(1, acados_ocp_solver.acados_ocp.dims.N):
                res += self.lam_extractor(
                    stage, "ubx", acados_ocp_solver.get(stage, "lam")
                ) @ (getattr(constraints, "ubx") - acados_ocp_solver.get(stage, "x"))

        if getattr(constraints, "lbu").size > 0:
            for stage in range(1, acados_ocp_solver.acados_ocp.dims.N):
                res += self.lam_extractor(
                    stage, "lbu", acados_ocp_solver.get(stage, "lam")
                ) @ (getattr(constraints, "lbu") - acados_ocp_solver.get(stage, "u"))

        if getattr(constraints, "ubu").size > 0:
            for stage in range(1, acados_ocp_solver.acados_ocp.dims.N):
                res += self.lam_extractor(
                    stage, "ubu", acados_ocp_solver.get(stage, "lam")
                ) @ (getattr(constraints, "ubu") - acados_ocp_solver.get(stage, "u"))

        # Terminal state inequality constraints
        stage = acados_ocp_solver.acados_ocp.dims.N
        if getattr(constraints, "lbx_e").size > 0:
            res += self.lam_extractor(
                stage, "lbx_e", acados_ocp_solver.get(stage, "lam")
            ) @ (getattr(constraints, "lbx_e") - acados_ocp_solver.get(stage, "x"))

        if getattr(constraints, "ubx_e").size > 0:
            res += self.lam_extractor(
                stage, "ubx_e", acados_ocp_solver.get(stage, "lam")
            ) @ (getattr(constraints, "ubx_e") - acados_ocp_solver.get(stage, "x"))

        res += acados_ocp_solver.get_cost()

        # Dynamic equality constraint
        for stage in range(acados_ocp_solver.acados_ocp.dims.N - 1):
            res += acados_ocp_solver.get(stage, "pi") @ (
                self.eval_f(
                    x=acados_ocp_solver.get(stage, "x"),
                    u=acados_ocp_solver.get(stage, "u"),
                    p=p,
                )
                - acados_ocp_solver.get(stage + 1, "x")
            )

        return res

    def eval_f(self, x: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Evaluate the integrator at the current solution of the OCP.

        Parameters:
            x: state
            u: control
            p: parameters

        Returns:
            xf: integrated state
        """
        return self.fun_f(x=x, u=u, p=p)["xf"].full().flatten()

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
        return self.fun_df_dp(x=x, u=u, p=p)["dxf_dp"].full().flatten()

    def eval_dL_dp(
        self, acados_ocp_solver: AcadosOcpSolver, p: np.ndarray
    ) -> np.ndarray:
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
            x = acados_ocp_solver.get(stage, "x")
            u = acados_ocp_solver.get(stage, "u")
            pi = acados_ocp_solver.get(stage, "pi")
            res += pi @ self.eval_df_dp(x=x, u=u, p=p)

        return res


def ERK4(
    f: Union[SX, Function],
    x: Union[SX, np.ndarray],
    u: Union[SX, np.ndarray],
    p: Union[SX, np.ndarray],
    h: float,
) -> Union[SX, np.ndarray]:
    """
    Explicit Runge-Kutta 4 integrator

    TODO: Works for numeric values as well as for symbolic values. Type hinting is a bit misleading.

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


def export_acados_ocp_solver() -> AcadosOcpSolver:
    """
    Define the acados OCP solver object.

    Parameters:
        ocp: acados OCP object

    Returns:
        acados_ocp_solver: acados OCP solver object
    """

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_pendulum_ode_model()

    Tf = 1.0
    N = 20
    dT = Tf / N

    ode = Function("ode", [model.x, model.u, model.p], [model.f_expl_expr])
    model.disc_dyn_expr = ERK4(ode, model.x, model.u, model.p, dT)
    print("built ERK4 for pendulum model with dT = ", dT)

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

    ocp.cost.W = scipy.linalg.block_diag(Q, R)

    ocp.cost.W_e = Q

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[4, 0] = 1.0
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

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

    ocp.solver_options.qp_solver_cond_N = N

    ocp.solver_options.qp_solver_iter_max = 200

    # set prediction horizon
    ocp.solver_options.tf = Tf

    ocp.parameter_values = np.array([1.0])

    acados_ocp_solver = AcadosOcpSolver(
        ocp, json_file="acados_ocp_" + model.name + ".json"
    )

    acados_ocp_solver.set(0, "lbx", x0)
    acados_ocp_solver.set(0, "ubx", x0)
    return acados_ocp_solver


def print_attributes(obj: Any, name: str = ""):
    """
    Print all attributes of an object.

    Parameters:
        obj: object
        name: name of the object
    """
    print("Attributes of ", name)
    for attr in dir(obj):
        print(attr, getattr(obj, attr))


def test_dynamics_equality_constraint_term(
    acados_ocp_solver: AcadosOcpSolver, p_test: np.ndarray = np.arange(0.5, 1.5, 0.01)
) -> int:
    x0 = np.array([0.5, 0.0, 0.0, 0.0])
    f, df_dp = build_discrete_dynamics_functions(acados_ocp_solver.acados_ocp)
    eval_f = lambda x, u, p: f(x=x, u=u, p=p)["xf"].full().flatten()
    eval_df_dp = lambda x, u, p: df_dp(x=x, u=u, p=p)["dxf_dp"].full().flatten()

    N = acados_ocp_solver.acados_ocp.dims.N

    u0 = acados_ocp_solver.solve_for_x0(x0)

    res = {"f": np.zeros((p_test.size,)), "df": np.zeros((p_test.size,))}
    for i, p in enumerate(p_test):
        for stage in range(N - 1):
            acados_ocp_solver.set(stage, "p", p)

        acados_ocp_solver.solve()

        acados_ocp_solver.print_statistics()

        for stage in range(N - 1):
            x = acados_ocp_solver.get(stage, "x")
            u = acados_ocp_solver.get(stage, "u")
            pi = acados_ocp_solver.get(stage, "pi")
            # x = x0
            # u = u0
            # pi = np.ones_like(x0)

            res["f"][i] += pi @ (
                eval_f(x=x, u=u, p=p) - acados_ocp_solver.get(stage + 1, "x")
            )
            res["df"][i] += pi @ eval_df_dp(x=x, u=u, p=p)

        res["df_cd"] = np.gradient(res["f"], p_test)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].plot(p_test, res["f"])
    ax[1].plot(p_test, res["df"])
    ax[1].plot(p_test, res["df_cd"], "--")
    # plt.show()


def test_f_vs_df_dp(
    acados_ocp_solver: AcadosOcpSolver, p_test: np.ndarray, plot: bool = False
) -> int:
    _x0 = np.array([0.5, 0.0, 0.0, 0.0])
    _u0 = acados_ocp_solver.solve_for_x0(_x0)
    _f, _df_dp = build_discrete_dynamics_functions(acados_ocp_solver.acados_ocp)
    _eval_f = lambda x, u, p: _f(x=x, u=u, p=p)["xf"].full().flatten()
    _eval_df_dp = lambda x, u, p: _df_dp(x=x, u=u, p=p)["dxf_dp"].full().flatten()

    _f_test = np.array([_eval_f(x=_x0, u=_u0, p=p) for p in p_test])
    _df_dp_test = np.array([_eval_df_dp(x=_x0, u=_u0, p=p) for p in p_test])

    # Compute df_dp via central difference of f_test
    _df_dp_test_fd = np.zeros_like(_df_dp_test)
    for i in range(_df_dp_test.shape[1]):
        _df_dp_test_fd[:, i] = np.gradient(_f_test[:, i], p_test)

    if plot:
        f_test_sum = np.sum(_f_test, axis=1)
        df_dp_test_sum = np.sum(_df_dp_test, axis=1)
        df_dp_test_fd_sum = np.sum(_df_dp_test_fd, axis=1)

        _, ax = plt.subplots(nrows=_f_test.shape[1] + 1, ncols=2, sharex=True)

        for i in range(_f_test.shape[1]):
            ax[i, 0].plot(p_test, _f_test[:, i])
            ax[i, 0].legend(["algorithmic differentiation", "finite difference"])
            ax[i, 1].plot(p_test, _df_dp_test[:, i])
            ax[i, 1].plot(p_test, _df_dp_test_fd[:, i], "--")
            ax[i, 1].legend(
                ["algorithmic differentiation", "np.grad", "central difference"]
            )
            ax[i, 0].set_ylabel("f[{}]".format(i))
            ax[i, 1].set_ylabel("df_dp[{}]".format(i))

        ax[-1, 0].plot(p_test, f_test_sum)
        ax[-1, 1].plot(p_test, df_dp_test_sum)
        ax[-1, 1].plot(p_test, df_dp_test_fd_sum, "--")
        ax[-1, 0].set_ylabel("f_sum")
        ax[-1, 1].set_ylabel("df_dp_sum")
        ax[-1, 1].legend(
            ["algorithmic differentiation", "np.grad", "central difference"]
        )
        ax[-1, 0].set_xlabel("p")

        for ax_k in ax.reshape(-1):
            ax_k.grid(True)

        plt.show()

    return int(
        not np.allclose(_df_dp_test[1:-1], _df_dp_test_fd[1:-1], rtol=1e-2, atol=1e-2)
    )


def test_dL_dp(acados_ocp_solver: AcadosOcpSolver, p_test: np.ndarray, plot=False):
    lagrange_function = LagrangeFunction(acados_ocp_solver)

    L = np.zeros(p_test.shape[0])
    dL_dp = np.zeros(p_test.shape[0])

    for i, p_i in enumerate(p_test):
        for stage in range(acados_ocp_solver.acados_ocp.dims.N):
            acados_ocp_solver.set(stage, "p", p_i)

        status = acados_ocp_solver.solve()

        if status != 0:
            raise Exception(
                f"acados acados_ocp_solver returned status {status} Exiting."
            )

        L[i] = lagrange_function(acados_ocp_solver, p_i)
        dL_dp[i] = lagrange_function.eval_dL_dp(acados_ocp_solver, p_i)

    dL_dp_grad = np.gradient(L, p_test[1] - p_test[0])

    dp = p_test[1] - p_test[0]

    L_reconstructed = np.cumsum(dL_dp) * dp + L[0]
    constant = L[0] - L_reconstructed[0]
    L_reconstructed += constant

    L_reconstructed_np_grad = np.cumsum(dL_dp_grad) * dp + L[0]
    constant = L[0] - L_reconstructed_np_grad[0]
    L_reconstructed_np_grad += constant

    dL_dp_cd = (L[2:] - L[:-2]) / (p_test[2:] - p_test[:-2])

    if plot:
        _, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax[0].plot(p_test, L)
        ax[0].plot(p_test, L_reconstructed, "--")
        ax[0].plot(p_test, L_reconstructed_np_grad, "-.")
        ax[1].legend(["L", "L integrate dL_dp", "L_integrate np.grad"])
        ax[1].plot(p_test, dL_dp)
        ax[1].plot(p_test, dL_dp_grad, "--")
        ax[1].plot(p_test[1:-1], dL_dp_cd, "-.")
        ax[1].legend(["algorithmic differentiation", "np.grad", "central difference"])
        ax[0].set_ylabel("L")
        ax[1].set_ylabel("dL_dp")
        ax[1].set_xlabel("p")
        ax[0].grid(True)
        ax[1].grid(True)

        plt.show()

    return int(not np.allclose(dL_dp, dL_dp_grad, rtol=1e-2, atol=1e-2))


def state_value_function(
    acados_ocp_solver: AcadosOcpSolver, state: np.ndarray
) -> float:
    """
    Evaluate the state-value function at the given state.

    Parameters:
        acados_ocp_solver: acados OCP solver object
        state: state

    Returns:
        float: state value
    """

    acados_ocp_solver.set(0, "x", state)
    acados_ocp_solver.constraints_set(0, "lbx", state)
    acados_ocp_solver.constraints_set(0, "ubx", state)

    status = acados_ocp_solver.solve()

    if status != 0:
        raise Exception(f"acados acados_ocp_solver returned status {status} Exiting.")

    return acados_ocp_solver.get_cost()


def test_dV_dp(
    acados_ocp_solver: AcadosOcpSolver,
    state: np.ndarray = np.array([0.5, 0.0, 0.0, 0.0]),
    p_test: np.ndarray = np.arange(0.5, 1.5, 0.001),
    plot=False,
):
    lagrange_function = LagrangeFunction(acados_ocp_solver)

    V = np.zeros(p_test.shape[0])
    dV_dp = np.zeros(p_test.shape[0])
    for i, p_i in enumerate(p_test):
        acados_ocp_solver = update_parameters(
            acados_ocp_solver=acados_ocp_solver, p=p_i
        )

        # status = acados_ocp_solver.solve()

        # if status != 0:
        #     raise Exception(f"acados acados_ocp_solver returned status {status} Exiting.")

        # V[i] = acados_ocp_solver.get_cost()
        V[i] = state_value_function(acados_ocp_solver=acados_ocp_solver, state=state)
        dV_dp[i] = lagrange_function.eval_dL_dp(acados_ocp_solver, p_i)

    dV_dp_grad = np.gradient(V, p_test)

    dp = p_test[1] - p_test[0]

    V_reconstructed = np.cumsum(dV_dp) * dp + V[0]
    constant = V[0] - V_reconstructed[0]
    V_reconstructed += constant

    if plot:
        _, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax[0].plot(p_test, V)
        ax[0].plot(p_test, V_reconstructed, "--")
        ax[1].plot(p_test, dV_dp)
        ax[1].plot(p_test, dV_dp_grad, "--")
        ax[1].legend(["algorithmic differentiation", "np.grad"])
        ax[0].set_ylabel("V")
        ax[1].set_ylabel("dV_dp")
        ax[1].set_xlabel("p")
        ax[0].grid(True)
        ax[1].grid(True)

        plt.show()

    return int(not np.allclose(dV_dp, dV_dp_grad, rtol=1e-2, atol=1e-2))


def state_action_value_function(
    acados_ocp_solver: AcadosOcpSolver,
    state: np.ndarray,
    action: np.ndarray,
    parameter: np.ndarray,
) -> float:
    """
    Evaluate the action-value function at the given state and action.

    Parameters:
        acados_ocp_solver: acados OCP solver object
        state: state
        a: action

    Returns:
        float: action-value function value
    """

    acados_ocp_solver.set(0, "x", state)
    acados_ocp_solver.set(0, "u", action)
    acados_ocp_solver.constraints_set(0, "lbx", state)
    acados_ocp_solver.constraints_set(0, "ubx", state)
    acados_ocp_solver.constraints_set(0, "lbu", action)
    acados_ocp_solver.constraints_set(0, "ubu", action)

    status = acados_ocp_solver.solve()

    if status != 0:
        raise Exception(f"acados acados_ocp_solver returned status {status} Exiting.")

    return acados_ocp_solver.get_cost()


def test_dQ_dp(
    acados_ocp_solver: AcadosOcpSolver,
    state: np.ndarray = np.array([0.5, 0.0, 0.0, 0.0]),
    action: np.ndarray = np.array([20.0]),
    p_test: np.ndarray = np.arange(0.5, 1.5, 0.001),
    plot: bool = False,
) -> int:
    """
    Test the derivative of the action-value function with respect to the parameters.

    Parameters:
        acados_ocp_solver: acados OCP solver object
        action: action value
        p_test: parameter values to test
        plot: plot the results

    Returns:
        int: 0 if test passed, 1 otherwise
    """

    # Define the Lagrange function
    lagrange_function = LagrangeFunction(acados_ocp_solver)

    action_value = state_action_value_function(
        acados_ocp_solver=acados_ocp_solver,
        state=state,
        action=action,
        parameter=np.array([1.0]),
    )

    # Evaluate the derivative of the action-value function with respect to the parameters

    Q = np.zeros(p_test.shape[0])
    dQ_dp = np.zeros(p_test.shape[0])
    for i, p_i in enumerate(p_test):
        acados_ocp_solver = update_parameters(
            acados_ocp_solver=acados_ocp_solver, p=p_i
        )

        Q[i] = state_action_value_function(
            acados_ocp_solver=acados_ocp_solver,
            state=state,
            action=action,
            parameter=p_i,
        )
        dQ_dp[i] = lagrange_function.eval_dL_dp(acados_ocp_solver, p_i)

    dQ_dp_grad = np.gradient(Q, p_test)

    dp = p_test[1] - p_test[0]

    Q_reconstructed = np.cumsum(dQ_dp) * dp + Q[0]
    constant = Q[0] - Q_reconstructed[0]
    Q_reconstructed += constant

    if plot:
        _, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax[0].plot(p_test, Q)
        ax[0].plot(p_test, Q_reconstructed, "--")
        ax[1].plot(p_test, dQ_dp)
        ax[1].plot(p_test, dQ_dp_grad, "--")
        ax[1].legend(["algorithmic differentiation", "np.grad"])
        ax[0].set_ylabel("Q")
        ax[1].set_ylabel("dQ_dp")
        ax[1].set_xlabel("p")
        ax[0].grid(True)
        ax[1].grid(True)

        plt.show()

    return int(not np.allclose(dQ_dp, dQ_dp_grad, rtol=1e-2, atol=1e-2))


def update_parameters(
    acados_ocp_solver: AcadosOcpSolver, p: np.ndarray
) -> AcadosOcpSolver:
    """
    Update the parameters of the OCP solver.

    Parameters:
        acados_ocp_solver: acados OCP solver object
        p: parameters
    """
    for stage in range(acados_ocp_solver.acados_ocp.dims.N):
        acados_ocp_solver.set(stage, "p", p)

    return acados_ocp_solver


def policy(acados_ocp_solver: AcadosOcpSolver, state: np.ndarray):
    """
    Evaluate the policy at the given state.

    Parameters:
        acados_ocp_solver: acados OCP solver object
        state: state

    Returns:
        float: action
    """

    acados_ocp_solver.constraints_set(0, "lbx", state)
    acados_ocp_solver.constraints_set(0, "ubx", state)
    acados_ocp_solver.set(0, "x", state)

    status = acados_ocp_solver.solve()

    if status != 0:
        raise Exception(f"acados acados_ocp_solver returned status {status} Exiting.")

    return acados_ocp_solver.get(0, "u")


def test_dpi_dp(
    acados_ocp_solver: AcadosOcpSolver, p_test: np.ndarray, plot=False
) -> int:
    """
    Test the sensitivity of the policy with respect to the parameters.

    Parameters:
        acados_ocp_solver: acados OCP solver object
        p_test: parameter values to test
        plot: plot the results

    Returns:
        int: 0 if test passed, 1 otherwise
    """

    return 1


def stack_primary_decision_variables(
    w: np.ndarray, acados_ocp_solver: AcadosOcpSolver
) -> np.ndarray:
    """
    Stack the primary decision variables of the OCP solver.

    Parameters:
        acados_ocp_solver: acados OCP solver object
        w: preallocated array to store the decision variables

    Returns:
        primary_decision_variables: primary decision variables
    """

    for stage in range(acados_ocp_solver.acados_ocp.dims.N + 1):
        w[
            stage
            * acados_ocp_solver.acados_ocp.dims.nx : (stage + 1)
            * acados_ocp_solver.acados_ocp.dims.nx
        ] = acados_ocp_solver.get(stage, "x")

    for stage in range(acados_ocp_solver.acados_ocp.dims.N):
        w[
            stage
            * acados_ocp_solver.acados_ocp.dims.nu : (stage + 1)
            * acados_ocp_solver.acados_ocp.dims.nu
        ] = acados_ocp_solver.get(stage, "u")

    return w


def stack_primary_decision_variables_slow(
    acados_ocp_solver: AcadosOcpSolver,
) -> np.ndarray:
    """
    Stack the primary decision variables of the OCP solver.

    Parameters:
        acados_ocp_solver: acados OCP solver object

    Returns:
        primary_decision_variables: primary decision variables
    """
    w = []
    for stage in range(acados_ocp_solver.acados_ocp.dims.N):
        w += acados_ocp_solver.get(stage, "x").tolist()
        w += acados_ocp_solver.get(stage, "u").tolist()

    w += acados_ocp_solver.get(acados_ocp_solver.acados_ocp.dims.N, "x").tolist()

    return np.array(w)


def allocate_primary_decision_variables(
    acados_ocp_solver: AcadosOcpSolver,
) -> np.ndarray:
    """
    Allocate the primary decision variables of the OCP solver.

    Parameters:
        acados_ocp_solver: acados OCP solver object

    Returns:
        primary_decision_variables: primary decision variables
    """
    w = np.zeros(
        (acados_ocp_solver.acados_ocp.dims.N + 1) * acados_ocp_solver.acados_ocp.dims.nx
        + acados_ocp_solver.acados_ocp.dims.N * acados_ocp_solver.acados_ocp.dims.nu
    )
    return w


import timeit


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


def test_stack_primary_decision_variables(acados_ocp_solver: AcadosOcpSolver) -> None:
    """
    Test the speed of the stack_primary_decision_variables function.

    Parameters:
        acados_ocp_solver: acados OCP solver object

    # TODO: execution_time for stack_pimary_decision_variables_slow is actually faster than for stack_primary_decision_variables
    """
    w = allocate_primary_decision_variables(acados_ocp_solver)

    wrapped = wrapper(stack_primary_decision_variables, w, acados_ocp_solver)
    execution_time = timeit.timeit(wrapped, number=1000)
    print(
        f"stack_primary_decision_variables took {execution_time/1000} seconds on average to run."
    )

    wrapped = wrapper(stack_primary_decision_variables_slow, acados_ocp_solver)
    execution_time = timeit.timeit(wrapped, number=1000)
    print(
        f"stack_primary_decision_variables_slow took {execution_time/1000} seconds on average to run."
    )


if __name__ == "__main__":
    """
    Test the sensitivites value functions with respect a parameter change in the pendulum mass.
    """
    acados_ocp_solver = export_acados_ocp_solver()
    p_test = np.arange(0.5, 1.5, 0.001)

    # w = allocate_primary_decision_variables(acados_ocp_solver)
    # w = stack_primary_decision_variables(w, acados_ocp_solver)

    tests = dict()
    # tests["test_f_vs_df_dp"] = test_f_vs_df_dp(acados_ocp_solver=acados_ocp_solver, p_test=p_test)
    tests["test_dL_dp"] = test_dL_dp(
        acados_ocp_solver=acados_ocp_solver, p_test=p_test, plot=True
    )
    tests["test_dV_dp"] = test_dV_dp(
        acados_ocp_solver=acados_ocp_solver, p_test=p_test, plot=True
    )
    tests["test_dQ_dp"] = test_dQ_dp(
        acados_ocp_solver=acados_ocp_solver, p_test=p_test, plot=True
    )

    # TODO:
    # tests["test_dpi_dp"] = test_dpi_dp(acados_ocp_solver=acados_ocp_solver, p_test=p_test, plot=True)

    print("Tests: ", tests)
