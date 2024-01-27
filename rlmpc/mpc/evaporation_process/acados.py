from typing import Union
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
import casadi as cs


from rlmpc.mpc.common.mpc import MPC

from rlmpc.mpc.nlp import NLP, build_nlp

from rlmpc.gym.evaporation_process.environment import compute_algebraic_variables


def get_parameter_labels(p: cs.SX.sym) -> list[str]:
    return p.str().strip("[]").split(", ")


def erk4_step(
    f: cs.Function,
    x: Union[cs.SX, cs.MX, np.ndarray],
    u: Union[cs.SX, cs.MX, np.ndarray],
    h: float,
) -> Union[cs.SX, cs.MX, np.ndarray]:
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
    k1 = f(x, u)
    k2 = f(x + h / 2 * k1, u)
    k3 = f(x + h / 2 * k2, u)
    k4 = f(x + h * k3, u)
    xf = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return xf


def erk4_discrete_dynamics(ocp: AcadosOcp):
    dt = ocp.solver_options.tf / ocp.dims.N / ocp.solver_options.sim_method_num_stages
    f = cs.Function("f", [ocp.model.x, ocp.model.u], [ocp.model.f_expl_expr])
    u = ocp.model.u
    x = ocp.model.x
    for _ in range(ocp.solver_options.sim_method_num_stages):
        k1 = f(x, u)
        k2 = f(x + dt / 2 * k1, u)
        k3 = f(x + dt / 2 * k2, u)
        k4 = f(x + dt * k3, u)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x


class AcadosMPC(MPC):
    """docstring for MPC."""

    nlp: NLP

    def __init__(self, model_param: dict, cost_param: dict):
        super(AcadosMPC, self).__init__()

        self.ocp_solver = setup_acados_ocp_solver(model_param=model_param, cost_param=cost_param)

        self.nlp = build_nlp(self.ocp_solver.acados_ocp)


def quadratic_function(H: cs.SX.sym, h: cs.SX.sym, c: cs.SX.sym, x: cs.SX.sym) -> cs.SX.sym:
    return 0.5 * cs.mtimes([x.T, H, x]) + cs.mtimes([h.T, x]) + c


def setup_acados_ocp_solver(model_param: dict, cost_param: dict[dict[np.ndarray]]) -> AcadosOcpSolver:
    ocp = AcadosOcp()
    ocp.dims.nx = 2
    ocp.dims.nu = 2
    ocp.dims.N = 20

    ocp.solver_options.tf = 10.0
    ocp.solver_options.Tsim = ocp.solver_options.tf / ocp.dims.N
    # ocp.solver_options.shooting_nodes = np.array([ocp.solver_options.tf / ocp.dims.N] * (ocp.dims.N + 1))

    ocp.model.name = "evaporation_process"

    ocp.model.x = cs.vertcat(*[cs.SX.sym("X_2"), cs.SX.sym("P_2")])
    ocp.model.xdot = cs.vertcat(*[cs.SX.sym("X_2_dot"), cs.SX.sym("P_2_dot")])
    ocp.model.u = cs.vertcat(*[cs.SX.sym("P_100"), cs.SX.sym("F_200")])

    # p_vals = list(param.values())
    # p_keys = list(param.keys())

    # ocp.model.p = cs.vertcat(*[cs.SX.sym(key) for key in param.keys()])
    # p = {key: cs.SX.sym(key) for key in param.keys()}

    H = {
        "lam": cs.SX.sym("H_lam", ocp.dims.nx, ocp.dims.nx),
        "l": cs.SX.sym("H_l", ocp.dims.nx + ocp.dims.nu, ocp.dims.nx + ocp.dims.nu),
        "Vf": cs.SX.sym("H_Vf", ocp.dims.nx, ocp.dims.nx),
    }
    h = {
        "lam": cs.SX.sym("h_lam", ocp.dims.nx),
        "l": cs.SX.sym("h_l", ocp.dims.nx + ocp.dims.nu),
        "Vf": cs.SX.sym("h_Vf", ocp.dims.nx),
    }

    c = {
        "lam": cs.SX.sym("c_lam"),
        "l": cs.SX.sym("c_l"),
        "Vf": cs.SX.sym("c_Vf"),
        "f": cs.SX.sym("c_f"),
    }

    for key in ["lam", "l", "Vf"]:
        print(H[key].shape)
        print(cost_param["H"][key].shape)
    for key in h.keys():
        print(key)
        print(h[key].shape)
        print(cost_param["h"][key].shape)

    xb = {"x_l": cs.SX.sym("x_l", ocp.dims.nx), "x_u": cs.SX.sym("x_u", ocp.dims.nx)}

    # nlp_param = {"H": H, "h": h, "c": c, "xb": xb}

    ocp.model.p = cs.vertcat(*[cs.reshape(param[key], -1, 1) for param in [H, h, c, xb] for key in param.keys()])

    ocp.parameter_values = cs.vertcat(
        *[
            cs.reshape(param[key], -1, 1)
            for param in [cost_param["H"], cost_param["h"], cost_param["c"], cost_param["xb"]]
            for key in param.keys()
        ]
    ).full()

    if True:
        ocp.cost.cost_type_0 = "EXTERNAL"
        ocp.model.cost_expr_ext_cost_0 = quadratic_function(H["lam"], h["lam"], c["lam"], ocp.model.x) + quadratic_function(
            H["l"], h["l"], c["l"], cs.vertcat(ocp.model.x, ocp.model.u)
        )
        ocp.cost.cost_type = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = quadratic_function(H["l"], h["l"], c["l"], cs.vertcat(ocp.model.x, ocp.model.u))

        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost_e = quadratic_function(H["Vf"], h["Vf"], c["Vf"], ocp.model.x)

        # Constant intervals for all shooting nodes

    algebraic_variables = compute_algebraic_variables(ocp.model.x, ocp.model.u, model_param)

    ocp.model.f_expl_expr = cs.vertcat(
        (model_param["F_1"] * model_param["X_1"] - algebraic_variables["F_5"] * ocp.model.x[0]) / model_param["M"],
        (algebraic_variables["F_4"] - algebraic_variables["F_5"]) / model_param["C"],
    )

    ocp.model.disc_dyn_expr = erk4_discrete_dynamics(ocp) + c["f"]

    # ocp.model.f_impl_expr = ocp.model.xdot - ocp.model.f_expl_expr

    # f_expl = cs.Function("f", [ocp.model.x, ocp.model.u], [compute_f_expl(ocp.model.x, ocp.model.u, model_param)])
    # ocp.model.disc_dyn_expr = erk4_step(f_expl, ocp.model.x, ocp.model.u, ocp.solver_options.shooting_nodes[0])

    # ocp.constraints.idxsbx = np.array([0, 1])
    # ocp.cost.zl = np.array([1e2, 1e2])
    # ocp.cost.zu = np.array([1e2, 1e2])
    # ocp.cost.Zl = np.diag([0, 0])
    # ocp.cost.Zu = np.diag([0, 0])

    # Initial conditions are handled through h
    ocp.constraints.idxbx_0 = np.array([0, 1])
    ocp.constraints.lbx_0 = np.array([50.0, 60.0])
    ocp.constraints.ubx_0 = np.array([50.0, 60.0])

    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = np.array([100.0, 100.0])
    ocp.constraints.ubu = np.array([400.0, 400.0])

    if False:
        print("")
        ocp.constraints.idxbx = np.array([0, 1])
        ocp.constraints.lbx = np.array([25.0, 40.0])
        ocp.constraints.ubx = np.array([100.0, 80.0])
        ocp.constraints.idxsbx = np.array([0, 1])
        ocp.cost.zl = np.array([1, 1])
        ocp.cost.zu = np.array([1, 1])
        ocp.cost.Zl = np.diag([0, 0])
        ocp.cost.Zu = np.diag([0, 0])
    else:
        if True:
            ocp.model.con_h_expr = cs.vertcat(xb["x_l"] - ocp.model.x, ocp.model.x - xb["x_u"])
            # ocp.model.con_h_expr = cs.vertcat(xb["x_l"] - ocp.model.x)
            # ocp.model.con_h_expr = cs.vertcat(ocp.model.x - xb["x_u"])
            # ocp.model.con_h_expr_e = ocp.model.con_h_expr

            ocp.constraints.lh = np.array([-1e6] * ocp.model.con_h_expr.shape[0])
            ocp.constraints.uh = np.array([0] * ocp.model.con_h_expr.shape[0])
            # ocp.constraints.lh_e = ocp.constraints.lh
            # ocp.constraints.uh_e = ocp.constraints.uh

            ocp.dims.nh = ocp.model.con_h_expr.shape[0]

            ocp.constraints.idxsh = np.arange(ocp.dims.nh)
            # ocp.constraints.idxsh_e = np.arange(ocp.dims.nh_e)
            ocp.cost.zl = np.ones(ocp.dims.nh)
            ocp.cost.Zl = np.diag([0] * ocp.dims.nh)
            # ocp.cost.zl_e = np.ones(ocp.dims.nh_e)
            # ocp.cost.Zl_e = np.diag([0] * ocp.dims.nh_e)
            ocp.cost.zu = np.ones(ocp.dims.nh)
            ocp.cost.Zu = np.diag([0] * ocp.dims.nh)
            # ocp.cost.zu_e = np.ones(ocp.dims.nh_e)
            # ocp.cost.Zu_e = np.diag([0] * ocp.dims.nh_e)

            ocp.dims.nsh = ocp.dims.nh
            # ocp.dims.nsh_e = ocp.dims.nh_e
            ocp.dims.ns = ocp.dims.nsh

    if False:
        ocp.model.con_h_expr = cs.vertcat(xb["x_l"] - ocp.model.x, ocp.model.x - xb["x_u"])
        ocp.model.con_h_expr_e = ocp.model.con_h_expr

        ocp.dims.nh = ocp.model.con_h_expr.shape[0]
        ocp.dims.nh_e = ocp.model.con_h_expr_e.shape[0]

        ocp.constraints.lh = np.array([-1e14] * ocp.dims.nh)
        ocp.constraints.uh = np.array([0] * ocp.dims.nh)
        ocp.constraints.lh_e = np.array([-1e14] * ocp.dims.nh_e)
        ocp.constraints.uh_e = np.array([0] * ocp.dims.nh_e)

        ocp.constraints.idxsh = np.arange(ocp.dims.nh)
        ocp.constraints.idxsh_e = np.arange(ocp.dims.nh_e)
        ocp.cost.zl = np.ones(ocp.dims.nh)
        ocp.cost.Zl = np.diag([0] * ocp.dims.nh)
        ocp.cost.zl_e = np.ones(ocp.dims.nh_e)
        ocp.cost.Zl_e = np.diag([0] * ocp.dims.nh_e)
        ocp.cost.zu = np.ones(ocp.dims.nh)
        ocp.cost.Zu = np.diag([0] * ocp.dims.nh)
        ocp.cost.zu_e = np.ones(ocp.dims.nh_e)
        ocp.cost.Zu_e = np.diag([0] * ocp.dims.nh_e)

        ocp.dims.nsh = ocp.dims.nh
        ocp.dims.nsh_e = ocp.dims.nh_e
        ocp.dims.ns = ocp.dims.nsh

    # ocp.constraints.idxbx = np.array([0, 1])
    # ocp.constraints.lbx = np.array([25.0, 40.0])
    # ocp.constraints.ubx = np.array([100.0, 80.0])

    # ocp.constraints.idxsbx = np.array([0, 1])
    # ocp.cost.zl = np.array([1, 1])
    # ocp.cost.zu = np.array([1, 1])
    # ocp.cost.Zl = np.diag([0, 0])
    # ocp.cost.Zu = np.diag([0, 0])

    # ocp.constraints.idxbu = np.array([0])
    # ocp.constraints.lbu = np.array([-1.0])
    # ocp.constraints.ubu = np.array([+1.0])

    # ocp.solver_options.integrator_type = "DISCRETE"

    # self.__qp_solver        = 'PARTIAL_CONDENSING_HPIPM'  # qp solver to be used in the NLP solver
    # ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    # ocp.solver_options.nlp_solver_max_iter = 1000

    ocp_solver = AcadosOcpSolver(ocp)

    # print(ocp_solver.acados_ocp.)

    for stage in range(ocp.dims.N + 1):
        ocp_solver.set(stage, "p", ocp.parameter_values)

    return ocp_solver
