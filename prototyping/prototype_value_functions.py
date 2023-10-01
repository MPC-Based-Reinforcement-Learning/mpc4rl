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

        idx_at_stage = [dict.fromkeys(self.order, 0) for _ in range(N)]

        for stage, keys in replacements.items():
            for old_key, new_key in keys:
                idx_at_stage[stage] = rename_key_in_dict(idx_at_stage[stage], old_key, new_key)

        # Loop over all constraints and count the number of constraints of each type. Store the indices in a dict.
        for idx in idx_at_stage:
            _start = 0
            _end = 0
            for attr in dir(constraints):
                if idx.keys().__contains__(attr):
                    _end += len(getattr(constraints, attr))
                    idx[attr] = slice(_start, _end)
                    _start = _end

        for stage, keys in replacements.items():
            for old_key, new_key in keys:
                idx_at_stage[stage] = rename_key_in_dict(idx_at_stage[stage], new_key, old_key)


class LagrangeFunction(object):
    """
    Lagrange function for the OCP
    """

    f_theta: Function
    ocp: AcadosOcp

    def __init__(self, ocp_solver: AcadosOcpSolver, sim_solver: AcadosSimSolver):
        super().__init__()

        ocp = ocp_solver.acados_ocp
        self.integrator = sim_solver

        self.constraint_dimension = ConstraintDimension(ocp_solver.acados_ocp.constraints)

        # x = ocp.model.x
        # u = ocp.model.u
        # p = ocp.model.p
        # f_expl = ocp.model.f_expl_expr

        self.f_theta = Function("f", [ocp.model.x, ocp.model.u, ocp.model.p], [ocp.model.f_expl_expr])

    def eval_f_theta(self, x: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Evaluate the ODE at the given state, control and parameter values.
        TODO: Use the autogenerated c function instead of the casadi function.
        """
        return self.f_theta(x, u, p).full().flatten()

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

        # Initial condition state equality constraint (chi_0 @ (x_0 - s))
        # TODO: Check when this can go wrong
        res += acados_ocp_solver.get(0, "lam")[nu : nx + nu] @ (
            acados_ocp_solver.acados_ocp.constraints.lbx_0 - acados_ocp_solver.get(0, "x")
        )

        res += acados_ocp_solver.get(0, "lam")[nx + nu : 2 * nx + nu] @ (
            acados_ocp_solver.acados_ocp.constraints.ubx_0 - acados_ocp_solver.get(0, "x")
        )

        # Initial condition control equality constraint would be relevant for Q(s,a) but not for V(s)
        # res += acados_ocp_solver.get(0, "lam")[nx + nu : 2 * nx + nu] @ (
        #     acados_ocp_solver.acados_ocp.constraints.ubx_0 - acados_ocp_solver.get(0, "x")
        # )

        for stage in range(acados_ocp_solver.acados_ocp.dims.N - 1):
            self.integrator.set("x", x[stage, :])
            self.integrator.set("u", u[stage, :])
            self.integrator.solve()

            # Dynamic equality constraint
            # res += chi[stage + 1, :] @ (self.integrator.get("x") - x[stage + 1, :])
            res += acados_ocp_solver.get(stage + 1, "pi") @ (self.integrator.get("x") - x[stage + 1, :])

            # Lower control inequality constraint
            res += acados_ocp_solver.get(stage, "lbu") - self.ocp.constraints.Jbu @ u[stage, :]

        res = acados_ocp_solver.get_cost()

        return res


# create ocp object to formulate the OCP
ocp = AcadosOcp()

# set model
model = export_pendulum_ode_model()

Tf = 1.0
N = 20
dT = Tf / N

ode = Function("ode", [model.x, model.u, model.p], [model.f_expl_expr])
# set up RK4
k1 = ode(model.x, model.u, model.p)
k2 = ode(model.x + dT / 2 * k1, model.u, model.p)
k3 = ode(model.x + dT / 2 * k2, model.u, model.p)
k4 = ode(model.x + dT * k3, model.u, model.p)
xf = model.x + dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

model.disc_dyn_expr = xf
print("built RK4 for pendulum model with dT = ", dT)
print(xf)

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

if False:
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

    x_sym = acados_ocp_solver.acados_ocp.model.x
    u_sym = acados_ocp_solver.acados_ocp.model.u
    p_sym = acados_ocp_solver.acados_ocp.model.p
    f_sym = acados_ocp_solver.acados_ocp.model.f_expl_expr
    n_p = acados_ocp_solver.acados_ocp.dims.np

    # Get number of constraints
    n_h = acados_ocp_solver.acados_ocp.dims.nh

    # Get lagrangian multipliers
    pi = acados_ocp_solver.get(0, "pi")
    lam = acados_ocp_solver.get(0, "lam")
    t = acados_ocp_solver.get(0, "t")

    # Model term
    df_dp = jacobian(f_sym, p_sym)

    # Cost term
    Vx = acados_ocp_solver.acados_ocp.cost.Vx
    Vu = acados_ocp_solver.acados_ocp.cost.Vu
    Vx_e = acados_ocp_solver.acados_ocp.cost.Vx_e
    yref = acados_ocp_solver.acados_ocp.cost.yref
    yref_e = acados_ocp_solver.acados_ocp.cost.yref_e
    W = acados_ocp_solver.acados_ocp.cost.W
    W_e = acados_ocp_solver.acados_ocp.cost.W_e

    # Get cost at stage
    cost = acados_ocp_solver.get_cost()

    # Get residuals from acados solver at stage 0
    res = acados_ocp_solver.get_residuals()

    acados_ocp_solver.print_statistics()

    lagrange_function = LagrangeFunction(acados_ocp_solver, acados_integrator)

    x = acados_ocp_solver.get(0, "x")
    u = acados_ocp_solver.get(0, "u")
    p = np.array([1.0])

    # Test
    x0 = np.array([0.5, 0.0, 0.0, 0.0])
    u0 = np.array([0.0])
    print("f_theta(x,u,p) = ", lagrange_function.eval_f_theta(x0, u0, p))

    L_theta = lagrange_function(acados_ocp_solver, p)

    print("L_theta = ", L_theta)

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
