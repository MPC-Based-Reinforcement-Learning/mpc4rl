from typing import Union
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
import casadi as cs


from rlmpc.mpc.common.mpc import MPC

from rlmpc.mpc.nlp import NLP, build_nlp

from rlmpc.gym.evaporation_process.environment import compute_data


def get_value(
    field: str, x: Union[cs.SX.sym, np.ndarray], u: Union[cs.SX.sym, np.ndarray], model_param: Union[cs.SX.sym, np.ndarray]
) -> Union[cs.SX.sym, np.ndarray]:
    if field in model_param.keys():
        return model_param[field]
    elif field == "X_2":
        return x[0]
    elif field == "P_2":
        return x[1]
    elif field == "P_100":
        return u[0]
    elif field == "F_200":
        return u[1]


def compute_economic_cost(x: cs.SX.sym, u: cs.SX.sym, model_param: dict) -> cs.SX.sym:
    algebraic_variables = compute_data(x, u, model_param)

    F_2 = algebraic_variables["F_2"]
    F_3 = model_param["F_3"]
    F_100 = algebraic_variables["F_100"]
    F_200 = u[1]
    s = u[2]

    return 10.09 * (F_2 + F_3) + 600.0 * F_100 + 0.6 * F_200 + 1.0e3 * s


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

    def __init__(
        self,
        model_param: dict = None,
        cost_param: dict = None,
        x0: np.ndarray = np.array([25, 49.743]),
        u0: np.ndarray = np.array([191.713, 215.888, 0.0]),
        gamma: float = 1.0,
        H: np.ndarray = None,
    ):
        super(AcadosMPC, self).__init__()

        self.gamma = gamma

        self.ocp_solver = setup_acados_ocp_solver(model_param=model_param, cost_param=cost_param, gamma=gamma)

        # The evaporation process model requires a non-zero initial guess
        for stage in range(self.ocp_solver.acados_ocp.dims.N + 1):
            self.ocp_solver.set(stage, "x", x0)
        for stage in range(self.ocp_solver.acados_ocp.dims.N):
            self.ocp_solver.set(stage, "u", u0)

        self.nlp = build_nlp(self.ocp_solver.acados_ocp, gamma=gamma, parameterize_tracking_cost=True)

        self.u0 = u0

    def get_action(self, x0: np.ndarray) -> np.ndarray:
        action = super().get_action(x0)

        if self.ocp_solver.status != 0:
            raise RuntimeError(f"Solver failed with status {self.ocp_solver.status}. Exiting.")
            exit(0)

        # self.update_nlp()

        return action

    def reset(self):
        # super().reset(x0)

        x_ss = np.array([25, 49.743])
        u_ss = np.array([191.713, 215.888, 0.0])
        # The evaporation process model requires a non-zero initial guess
        for stage in range(self.ocp_solver.acados_ocp.dims.N + 1):
            self.ocp_solver.set(stage, "x", x_ss)
        for stage in range(self.ocp_solver.acados_ocp.dims.N):
            self.ocp_solver.set(stage, "u", u_ss)


def setup_acados_ocp_solver(model_param: dict, cost_param: dict[dict[np.ndarray]], gamma: float = 0.99) -> AcadosOcpSolver:
    ocp = AcadosOcp()
    ocp.dims.nx = 2
    ocp.dims.nu = 3
    ocp.dims.N = 100

    dT = 1.0

    ocp.solver_options.tf = float(ocp.dims.N * dT)
    ocp.solver_options.Tsim = ocp.solver_options.tf / ocp.dims.N

    ocp.model.name = "evaporation_process"

    ocp.model.x = cs.vertcat(*[cs.SX.sym("X_2"), cs.SX.sym("P_2")])
    ocp.model.xdot = cs.vertcat(*[cs.SX.sym("X_2_dot"), cs.SX.sym("P_2_dot")])
    ocp.model.u = cs.vertcat(*[cs.SX.sym("P_100"), cs.SX.sym("F_200"), cs.SX.sym("s")])

    x_ss = np.array([25, 49.743])
    u_ss = np.array([191.713, 215.888, 0.0])

    w = cs.vertcat(*[ocp.model.x, ocp.model.u])
    w_ss = cs.vertcat(*[x_ss, u_ss])

    # xb = {"x_l": cs.SX.sym("x_l", ocp.dims.nx), "x_u": cs.SX.sym("x_u", ocp.dims.nx)}

    x_ss = np.array([25, 49.743])
    u_ss = np.array([191.713, 215.888, 0.0])

    # ocp.cost.cost_type_0 = "NONLINEAR_LS"
    ocp.cost.cost_type = "NONLINEAR_LS"
    # ocp.cost.cost_type_e = "NONLINEAR_LS"
    # ocp.model.cost_y_expr_0 = w
    ocp.model.cost_y_expr = w
    # ocp.model.cost_y_expr_e = ocp.model.x
    # ocp.cost.yref_0 = w_ss.full().flatten()
    ocp.cost.yref = w_ss.full().flatten()
    # ocp.cost.yref_e = x_ss
    # ocp.cost.W_0 = cost_param["H"]["l"]
    ocp.cost.W = cost_param["H"]["l"]
    # ocp.cost.W_e = cost_param["H"]["Vf"]

    # ocp.dims.ny_0 = 5
    ocp.dims.ny = 5
    # ocp.dims.ny_e = 2

    algebraic_variables = compute_data(ocp.model.x, ocp.model.u, model_param)

    X_2 = ocp.model.x[0]
    ocp.model.f_expl_expr = cs.vertcat(
        (model_param["F_1"] * model_param["X_1"] - algebraic_variables["F_2"] * X_2) / model_param["M"],
        (algebraic_variables["F_4"] - algebraic_variables["F_5"]) / model_param["C"],
    )

    ocp.model.disc_dyn_expr = erk4_discrete_dynamics(ocp)  # + cost_param["c"]["f"]
    ocp.model.f_impl_expr = ocp.model.f_expl_expr - ocp.model.xdot

    ocp.constraints.idxbx_0 = np.array([0, 1])
    ocp.constraints.lbx_0 = np.array([50.0, 60.0])
    ocp.constraints.ubx_0 = np.array([50.0, 60.0])

    ocp.constraints.idxbu = np.array([0, 1, 2])
    ocp.constraints.lbu = np.array([100.0, 100.0, 0.0])
    ocp.constraints.ubu = np.array([400.0, 400.0, 10.0])

    # # ocp.model.con_h_expr = cs.vertcat(xb["x_l"] - ocp.model.x - ocp.model.u[2])
    ocp.model.con_h_expr = cs.vertcat(25.0 - ocp.model.x - ocp.model.u[2])
    ocp.constraints.lh = np.array([-1e3] * ocp.model.con_h_expr.shape[0])
    ocp.constraints.uh = np.array([0] * ocp.model.con_h_expr.shape[0])
    ocp.dims.nh = ocp.model.con_h_expr.shape[0]

    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"

    ocp_solver = AcadosOcpSolver(ocp)

    # Adjust weights for discounting
    for stage in range(1, ocp_solver.acados_ocp.dims.N):
        ocp_solver.cost_set(stage, "W", gamma**stage * ocp_solver.acados_ocp.cost.W)

    if len(ocp_solver.acados_ocp.cost.W_e) > 0:
        ocp_solver.cost_set(
            ocp_solver.acados_ocp.dims.N, "W", gamma**ocp_solver.acados_ocp.dims.N * ocp_solver.acados_ocp.cost.W_e
        )

    return ocp_solver
