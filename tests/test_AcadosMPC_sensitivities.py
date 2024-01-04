from external.stable_baselines3.stable_baselines3.td3 import TD3
from rlmpc.td3.policies import MPCTD3Policy

import gymnasium as gym
from rlmpc.common.utils import read_config

from rlmpc.gym.continuous_cartpole.environment import (
    ContinuousCartPoleBalanceEnv,
    ContinuousCartPoleSwingUpEnv,
)

from casadi.tools import struct_symMX, struct_MX, struct_symSX, struct_SX, entry

import scipy

import casadi as cs

from typing import Union

from rlmpc.mpc.cartpole.acados import AcadosMPC, Config

import gymnasium as gym
import numpy as np

from rlmpc.mpc.cartpole.common import Config

import matplotlib.pyplot as plt

from acados_template import (
    AcadosOcp,
    AcadosOcpSolver,
    AcadosSimSolver,
    AcadosModel,
    AcadosOcpConstraints,
    AcadosSim,
)

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


def rename_key_in_dict(d: dict, old_key: str, new_key: str):
    d[new_key] = d.pop(old_key)

    return d


def rename_item_in_list(lst: list, old_item: str, new_item: str):
    if old_item in lst:
        index_old = lst.index(old_item)
        lst[index_old] = new_item

    return lst


def build_cost_functions(
    ocp: AcadosOcp,
) -> tuple[cs.Function, cs.Function, cs.Function, cs.Function]:
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
        raise NotImplementedError("Only LINEAR_LS and NONLINEAR_LS cost types are supported at the moment.")

    l = (cost_y_expr - yref).T @ W @ (cost_y_expr - yref)
    m = (cost_y_expr_e - yref_e).T @ W_e @ (cost_y_expr_e - yref_e)

    fun_l = cs.Function("l", [x, u, p], [l], ["x", "u", "p"], ["l"])
    fun_m = cs.Function("m", [x, p], [m], ["x", "p"], ["m"])
    fun_dl_dp = cs.Function("dl_dp", [x, u, p], [cs.jacobian(l, p)], ["x", "u", "p"], ["dl_dp"])
    fun_dm_dp = cs.Function("dm_dp", [x, p], [cs.jacobian(m, p)], ["x", "p"], ["dm_dp"])

    return fun_l, fun_m, fun_dl_dp, fun_dm_dp


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
                idx_at_stage[stage] = rename_key_in_dict(idx_at_stage[stage], old_key, new_key)

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


def ERK4(
    f: Union[cs.SX, cs.Function],
    x: Union[cs.SX, np.ndarray],
    u: Union[cs.SX, np.ndarray],
    p: Union[cs.SX, np.ndarray],
    h: float,
) -> Union[cs.SX, np.ndarray]:
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


def build_discrete_dynamics_functions(ocp: AcadosOcp) -> tuple[cs.Function, cs.Function]:
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
    # f = ocp.model.disc_dyn_expr
    h = ocp.solver_options.tf / ocp.dims.N

    # Build the discrete dynamics function
    # TODO: Assumes sim_method_num_stages == 4 for all stages
    if ocp.solver_options.integrator_type == "ERK" and ocp.solver_options.sim_method_num_stages[0] == 4:
        f_cont = cs.Function("f", [x, u, p], [ocp.model.f_expl_expr])
        f_disc = ERK4(f_cont, x, u, p, h)
    elif ocp.solver_options.integrator_type == "DISCRETE":
        f_disc = ocp.model.disc_dyn_expr
    else:
        raise NotImplementedError("Only ERK4 and DISCRETE integrator types are supported at the moment.")

    fun_f = cs.Function("f", [x, u, p], [f_disc], ["x", "u", "p"], ["xf"])
    fun_df_dp = cs.Function("df_dp", [x, u, p], [cs.jacobian(f_disc, p)], ["x", "u", "p"], ["dxf_dp"])

    return fun_f, fun_df_dp


def build_constraint_functions(
    ocp: AcadosOcp,
) -> tuple[cs.Function, cs.Function, cs.Function, cs.Function]:
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


def get_labels_from_SX(SX: cs.SX) -> list[str]:
    """
    Get the labels of the entries of a SX object.

    Parameters:
        SX: SX object

    Returns:
        labels: list of labels
    """
    return SX.str().strip("[]").split(", ")


class LagrangeFunction(object):
    """
    Lagrange function for the OCP
    """

    # f_theta: Function
    # ocp: AcadosOcp

    lam_extractor: LagrangeMultiplierExtractor
    fun_f: cs.Function  # Discrete dynamics
    fun_df_dp: cs.Function  # Derivative of discrete dynamics with respect to parameters
    fun_l: cs.Function  # Stage cost
    fun_m: cs.Function  # Terminal cost
    fun_dl_dp: cs.Function  # Derivative of stage cost with respect to parameters
    fun_dm_dp: cs.Function  # Derivative of terminal cost with respect to parameters

    def __init__(self, acados_ocp_solver: AcadosOcpSolver):
        super().__init__()

        self.lam_extractor = LagrangeMultiplierExtractor(
            acados_ocp_solver.acados_ocp.constraints,
            acados_ocp_solver.acados_ocp.dims.N,
        )

        self.fun_f, self.fun_df_dp = build_discrete_dynamics_functions(acados_ocp_solver.acados_ocp)
        self.fun_l, self.fun_m, self.fun_dl_dp, self.fun_dm_dp = build_cost_functions(acados_ocp_solver.acados_ocp)
        _ = build_constraint_functions(acados_ocp_solver.acados_ocp)

        # Build symbolic functions for the Lagrange function and its derivatives

        x = acados_ocp_solver.acados_ocp.model.x
        u = acados_ocp_solver.acados_ocp.model.u
        p = acados_ocp_solver.acados_ocp.model.p

        ocp = acados_ocp_solver.acados_ocp

        labels = dict.fromkeys(["x", "u", "p"])
        labels["x"] = get_labels_from_SX(x)
        labels["u"] = get_labels_from_SX(u)
        labels["p"] = get_labels_from_SX(p)

        states = struct_symSX([tuple([entry(label) for label in labels["x"]])])

        # X = struct_symSX([tuple([entry(label) for label in labels["x"]])])

        entries = {"w": []}

        entries["w"].append(
            entry("x", repeat=ocp.dims.N, struct=struct_symSX([tuple([entry(label) for label in labels["x"]])]))
        )

        entries["w"].append(
            entry("u", repeat=ocp.dims.N - 1, struct=struct_symSX([tuple([entry(label) for label in labels["u"]])]))
        )

        # TODO: If slacks are used, add them to the structure

        w = struct_symSX([tuple(entries["w"])])

        # U = struct_SX(entry("u", repeat=ocp.dims.N - 1, struct=struct_symSX([tuple([entry(label) for label in labels["u"]])])))
        # entries["w"].append(
        #     entry("u", repeat=ocp.dims.N - 1, struct=struct_symSX([tuple([entry(label) for label in labels["u"]])]))
        # )

        print("hallo")

    def __call__(self, acados_ocp_solver: AcadosOcpSolver, p: np.ndarray) -> np.float32:
        """
        Evaluate the Lagrange function at the current solution of the OCP.
        """

        res = 0.0

        constraints = acados_ocp_solver.acados_ocp.constraints

        # Initial condition equality constraint
        stage = 0
        res += self.lam_extractor(stage, "lbx_0", acados_ocp_solver.get(stage, "lam")) @ (
            getattr(constraints, "lbx_0") - acados_ocp_solver.get(stage, "x")
        )

        res += self.lam_extractor(stage, "ubx_0", acados_ocp_solver.get(stage, "lam")) @ (
            acados_ocp_solver.get(stage, "x") - getattr(constraints, "ubx_0")
        )

        # Inequality constraints at stage k
        if getattr(constraints, "lbx").size > 0:
            for stage in range(1, acados_ocp_solver.acados_ocp.dims.N):
                res += self.lam_extractor(stage, "lbx", acados_ocp_solver.get(stage, "lam")) @ (
                    getattr(constraints, "lbx") - acados_ocp_solver.get(stage, "x")
                )

        if getattr(constraints, "ubx").size > 0:
            for stage in range(1, acados_ocp_solver.acados_ocp.dims.N):
                res += self.lam_extractor(stage, "ubx", acados_ocp_solver.get(stage, "lam")) @ (
                    acados_ocp_solver.get(stage, "x") - getattr(constraints, "ubx")
                )

        if getattr(constraints, "lbu").size > 0:
            for stage in range(1, acados_ocp_solver.acados_ocp.dims.N):
                res += self.lam_extractor(stage, "lbu", acados_ocp_solver.get(stage, "lam")) @ (
                    getattr(constraints, "lbu") - acados_ocp_solver.get(stage, "u")
                )

        if getattr(constraints, "ubu").size > 0:
            for stage in range(1, acados_ocp_solver.acados_ocp.dims.N):
                res += self.lam_extractor(stage, "ubu", acados_ocp_solver.get(stage, "lam")) @ (
                    acados_ocp_solver.get(stage, "u") - getattr(constraints, "ubu")
                )

        # Terminal state inequality constraints
        stage = acados_ocp_solver.acados_ocp.dims.N
        if getattr(constraints, "lbx_e").size > 0:
            res += self.lam_extractor(stage, "lbx_e", acados_ocp_solver.get(stage, "lam")) @ (
                getattr(constraints, "lbx_e") - acados_ocp_solver.get(stage, "x")
            )

        if getattr(constraints, "ubx_e").size > 0:
            res += self.lam_extractor(stage, "ubx_e", acados_ocp_solver.get(stage, "lam")) @ (
                acados_ocp_solver.get(stage, "x") - getattr(constraints, "ubx_e")
            )

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
            x = acados_ocp_solver.get(stage, "x")
            u = acados_ocp_solver.get(stage, "u")
            pi = acados_ocp_solver.get(stage, "pi")
            res += pi @ self.eval_df_dp(x=x, u=u, p=p)

        return res

    def eval_dL_dw(self, acados_ocp_solver: AcadosOcpSolver, p: np.ndarray) -> np.ndarray:
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


def test_dL_dp(acados_ocp_solver: AcadosOcpSolver, p_test: np.ndarray, plot=False):
    lagrange_function = LagrangeFunction(acados_ocp_solver)

    L = np.zeros(p_test.shape[0])
    dL_dp = np.zeros(p_test.shape[0])

    for i, p_i in enumerate(p_test):
        for stage in range(acados_ocp_solver.acados_ocp.dims.N):
            acados_ocp_solver.set(stage, "p", p_i)

        status = acados_ocp_solver.solve()

        if status != 0:
            raise Exception(f"acados acados_ocp_solver returned status {status} Exiting.")

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


if __name__ == "__main__":
    config = read_config("config/test_AcadosMPC_sensitivities.yaml")

    mpc = AcadosMPC(config=Config.from_dict(config["mpc"]), build=True)

    ocp_solver = mpc.ocp_solver

    p_test = np.linspace(0.5, 1.5, 200)

    test = test_dL_dp(acados_ocp_solver=ocp_solver, p_test=p_test, plot=True)
