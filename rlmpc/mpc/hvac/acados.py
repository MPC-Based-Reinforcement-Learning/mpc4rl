import os
from acados_template.acados_ocp_solver import ocp_generate_external_functions
import numpy as np
from acados_template import (
    AcadosOcp,
    AcadosOcpSolver,
    AcadosOcpConstraints,
    AcadosOcpCost,
    AcadosOcpDims,
    AcadosModel,
    AcadosOcpOptions,
)

import casadi as cs

from rlmpc.common.mpc import MPC

import matplotlib.pyplot as plt

from rlmpc.mpc.cartpole.common import Config, ModelParams, define_parameter_values

from rlmpc.common.integrator import ERK4

from rlmpc.mpc.nlp import LagrangeMultiplierMap, NLP, update_nlp, find_nlp_entry_expr_dependencies, build_nlp

from rlmpc.gym.house.environment import build_A, build_B, update_A, update_B, ode

import scipy


def define_ocp_solver(param: list) -> AcadosOcpSolver:
    ocp = AcadosOcp()

    nx = param[1].shape[0]
    nu = param[2].shape[0]
    x = cs.SX.sym("T", nx)
    u = cs.SX.sym("u", nu)
    xdot = cs.SX.sym("xdot", nx)

    resistance = cs.SX.sym("R", nx, nx)
    capacity = cs.SX.sym("C", nx - 1)
    efficiency = cs.SX.sym("eta", nu)

    p_sym = cs.vertcat(
        cs.reshape(capacity, (nx - 1, 1)), cs.reshape(resistance, (nx * nx, 1)), cs.reshape(efficiency, (nu, 1))
    )

    A = cs.SX.zeros(nx, nx)
    A = update_A(A, [capacity, resistance, efficiency])
    B = cs.SX.zeros(nx, nu)
    B = update_B(B, [capacity, resistance, efficiency])

    z = None

    f_expl = cs.mtimes(A, x) + cs.mtimes(B, u)

    f_impl = xdot - f_expl
    # f_disc =

    ocp.model.f_impl_expr = f_impl
    ocp.model.f_expl_expr = f_expl
    ocp.model.x = x
    ocp.model.xdot = xdot
    ocp.model.cost_y_expr = cs.vertcat(x[:-1], u)
    ocp.model.cost_y_expr_e = x[:-1]
    ocp.model.p = p_sym
    ocp.model.u = u
    ocp.model.z = z
    ocp.model.name = "house"

    ocp.dims.N = 100
    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.dims.np = len(param[0]) + len(param[1]) * len(param[1]) + len(param[2])
    ocp.dims.ny_0 = nx - 1 + nu
    ocp.dims.ny = nx - 1 + nu
    ocp.dims.ny_e = nx - 1

    ocp.constraints.idxbx = np.array([0, 1, 2, 3])
    ocp.constraints.lbx = np.array([15, 15, 15, 15])
    ocp.constraints.ubx = np.array([25, 25, 25, 25])

    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4])
    ocp.constraints.lbx_0 = np.array([20, 25, 15, 22, 10])
    ocp.constraints.ubx_0 = np.array([20, 25, 15, 22, 10])

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    Q = np.diag([1e-2, 1e-2, 1e-2, 1e-2])
    R = np.diag([1e0, 1e0])
    Qe = Q

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_0 = ocp.cost.W
    ocp.cost.W_e = Qe
    ocp.cost.yref = np.array([20, 20, 20, 20, 0, 0])
    ocp.cost.yref_e = np.array([20, 20, 20, 20])

    ocp.solver_options.tf = 1e3
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"

    ocp.parameter_values = np.concatenate((param[0].reshape(-1), param[1].reshape(-1), param[2].reshape(-1)))

    return AcadosOcpSolver(ocp, json_file="hvac_ocp.json")


class AcadosMPC(MPC):
    """docstring for CartpoleMPC."""

    _parameters: np.ndarray
    ocp_solver: AcadosOcpSolver
    nlp: NLP
    idx: dict
    muliplier_map: LagrangeMultiplierMap

    def __init__(self, config: dict, build: bool = True):
        super().__init__()

        param = config["params"]

        self.ocp_solver = define_ocp_solver(param)

        ocp = self.ocp_solver.acados_ocp

        self.ocp = ocp

        if False:
            ocp_generate_external_functions(ocp, ocp.model)

            nlp = build_nlp(ocp=self.ocp)

            self.nlp = nlp

            self.muliplier_map = LagrangeMultiplierMap(ocp)

            nlp.L.sym = nlp.cost.sym + cs.dot(nlp.pi.sym, nlp.g.sym) + cs.dot(nlp.lam.sym, nlp.h.sym)
            nlp.L.fun = cs.Function(
                "L",
                [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym, nlp.dT.sym],
                [nlp.L.sym],
                ["w", "lbw", "ubw", "pi", "lam", "p", "dT"],
                ["L"],
            )

            nlp.dL_dw.sym = cs.jacobian(nlp.L.sym, nlp.w.sym)

            arg_list, name_list = find_nlp_entry_expr_dependencies(nlp, "dL_dw", ["w", "lbw", "ubw", "pi", "lam", "p", "dT"])

            nlp.dL_dw.fun = cs.Function(
                "dL_dw",
                arg_list,
                [nlp.dL_dw.sym],
                name_list,
                ["dL_dw"],
            )

            # Check if nlp.dL_dw.sym is function of nlp.w.sym

            nlp.dL_dp.sym = cs.jacobian(nlp.L.sym, nlp.p.sym)

            arg_list, name_list = find_nlp_entry_expr_dependencies(nlp, "dL_dp", ["w", "lbw", "ubw", "pi", "lam", "p", "dT"])

            nlp.dL_dp.fun = cs.Function(
                "dL_dp",
                arg_list,
                [nlp.dL_dp.sym],
                name_list,
                ["dL_dp"],
            )

            nlp.R.sym = cs.vertcat(cs.transpose(nlp.dL_dw.sym), nlp.g.sym, nlp.lam.sym * nlp.h.sym)

            arg_list, name_list = find_nlp_entry_expr_dependencies(nlp, "R", ["w", "lbw", "ubw", "pi", "lam", "p", "dT"])

            nlp.R.fun = cs.Function(
                "R",
                arg_list,
                [nlp.R.sym],
                name_list,
                ["R"],
            )

            z = cs.vertcat(nlp.w.sym, nlp.pi.sym, nlp.lam.sym)

            # Generate sensitivity of the KKT matrix with respect to primal-dual variables
            nlp.dR_dz.sym = cs.jacobian(nlp.R.sym, z)
            arg_list, name_list = find_nlp_entry_expr_dependencies(nlp, "dR_dz", ["w", "lbw", "ubw", "pi", "lam", "p", "dT"])
            nlp.dR_dz.fun = cs.Function(
                "dR_dz",
                arg_list,
                [nlp.dR_dz.sym],
                name_list,
                ["dR_dz"],
            )

            nlp.dR_dp.sym = cs.jacobian(nlp.R.sym, nlp.p.sym)
            arg_list, name_list = find_nlp_entry_expr_dependencies(nlp, "dR_dp", ["w", "lbw", "ubw", "pi", "lam", "p", "dT"])
            nlp.dR_dp.fun = cs.Function(
                "dR_dp",
                arg_list,
                [nlp.dR_dp.sym],
                name_list,
                ["dR_dp"],
            )

            self.nlp = nlp

        if False:
            # Check path to config.meta.json file. Create the directory if it does not exist.
            if not os.path.exists(os.path.dirname(config["meta"]["json_file"])):
                os.makedirs(os.path.dirname(config["meta"]["json_file"]))

            # TODO: Add config entries for json file and c_generated_code folder, and build, generate flags
            if build:
                self.ocp_solver = AcadosOcpSolver(ocp, json_file=config["meta"]["json_file"])
            else:
                # Assumes json file and c_generated_code folder already exists
                self.ocp_solver = AcadosOcpSolver(ocp, json_file=config["meta"]["json_file"], build=False, generate=False)

            # Set nlp constraints

            # Initial stage

            # lbu_0 = np.ones((self.ocp_solver.acados_ocp.dims.nu,)) * self.ocp_solver.acados_ocp.constraints.lbu[0]
            self.nlp.set(0, "lbu", self.ocp_solver.acados_ocp.constraints.lbu)
            self.nlp.set(0, "lbx", self.ocp_solver.acados_ocp.constraints.lbx_0)
            self.nlp.set(0, "ubu", self.ocp_solver.acados_ocp.constraints.ubu)
            self.nlp.set(0, "ubx", self.ocp_solver.acados_ocp.constraints.ubx_0)

            # Middle stages
            for stage in range(1, self.ocp_solver.acados_ocp.dims.N):
                self.nlp.set(stage, "lbx", self.ocp_solver.acados_ocp.constraints.lbx)
                self.nlp.set(stage, "ubx", self.ocp_solver.acados_ocp.constraints.ubx)
                self.nlp.set(stage, "lbu", self.ocp_solver.acados_ocp.constraints.lbu)
                self.nlp.set(stage, "ubu", self.ocp_solver.acados_ocp.constraints.ubu)

            # Final stage
            stage = self.ocp_solver.acados_ocp.dims.N
            self.nlp.set(stage, "lbx", self.ocp_solver.acados_ocp.constraints.lbx_e)
            self.nlp.set(stage, "ubx", self.ocp_solver.acados_ocp.constraints.ubx_e)

            self._parameters = ocp.parameter_values

    def reset(self, x0: np.ndarray):
        self.ocp_solver.reset()

        for stage in range(self.ocp_solver.acados_ocp.dims.N + 1):
            # self.ocp_solver.set(stage, "x", self.ocp_solver.acados_ocp.constraints.lbx_0)
            self.ocp_solver.set(stage, "x", x0)

    def set(self, stage, field, value):
        self.ocp_solver.set(stage, field, value)

        if field == "p":
            self.nlp.vars.val["p"] = value

    def get(self, stage, field):
        return self.ocp_solver.get(stage, field)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low = self.ocp.constraints.lbu
        high = self.ocp.constraints.ubu

        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low = self.ocp.constraints.lbu
        high = self.ocp.constraints.ubu

        return 0.5 * (high - low) * (action + 1.0) + low

    # def update(self) -> int:
    #     status = self.ocp_solver.solve()

    #     self.nlp = update_nlp(self.nlp, self.ocp_solver, self.muliplier_map)

    #     test_nlp_sanity(self.nlp)

    #     return status

    def q_update(self, x0: np.ndarray, u0: np.ndarray) -> int:
        """
        Update the solution of the OCP solver.

        Args:
            x0: Initial state.

        Returns:
            status: Status of the solver.
        """
        # Set initial state
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        self.nlp.vars.val["lbx_0"] = x0
        self.nlp.vars.val["ubx_0"] = x0

        # Set initial action (needed for state-action value)
        self.ocp_solver.set(0, "u", u0)
        # self.ocp_solver.set(0, "lbu", u0)
        # self.ocp_solver.set(0, "ubu", u0)
        self.ocp_solver.constraints_set(0, "lbu", u0)
        self.ocp_solver.constraints_set(0, "ubu", u0)

        self.nlp.set(0, "lbu", u0)
        self.nlp.set(0, "ubu", u0)

        # Solve the optimization problem

        status = self.ocp_solver.solve()

        # self.ocp_solver.set(0, "lbu", self.ocp_solver.acados_ocp.constraints.lbu)
        self.ocp_solver.constraints_set(0, "lbu", self.ocp_solver.acados_ocp.constraints.lbu)
        self.ocp_solver.constraints_set(0, "ubu", self.ocp_solver.acados_ocp.constraints.ubu)

        self.nlp = update_nlp(self.nlp, self.ocp_solver, self.muliplier_map)

        self.nlp.set(0, "lbu", self.ocp_solver.acados_ocp.constraints.lbu)
        self.nlp.set(0, "ubu", self.ocp_solver.acados_ocp.constraints.ubu)

        # self.nlp.set(0, "lbu", u0)
        # self.nlp.set(0, "ubu", u0)

        # test_nlp_sanity(self.nlp)

        return status

    def assert_nlp_kkt_conditions(self) -> None:
        """
        Assert that the NLP is sane.
        """
        # Check if the NLP is sane

    def update(self, x0: np.ndarray) -> int:
        """
        Update the solution of the OCP solver.

        Args:
            x0: Initial state.

        Returns:
            status: Status of the solver.
        """
        # Set initial state
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        self.nlp.set(0, "lbx", x0)
        self.nlp.set(0, "ubx", x0)

        # self.nlp.lbw.val["lbx", 0] = x0
        # self.nlp.ubw.val["ubx", 0] = x0

        # Set initial action (needed for state-action value)
        # u0 = np.zeros((self.ocp.dims.nu,))
        # self.ocp_solver.set(0, "u", u0)
        # self.ocp_solver.set(0, "lbu", u0)
        # self.ocp_solver.set(0, "ubu", u0)

        # Solve the optimization problem
        status = self.ocp_solver.solve()

        self.update_nlp()

        # test_nlp_sanity(self.nlp)

        return status

    def update_nlp(self) -> None:
        """
        Update the NLP with the solution of the OCP solver.
        """
        self.nlp = update_nlp(self.nlp, self.ocp_solver, self.muliplier_map)

    def set_p(self, p: np.ndarray) -> None:
        """
        Set the value of the parameters.

        Args:
            p: Parameters.
        """
        # self._parameters = theta
        # self.ocp_solver.set(0, "p", theta)
        # self.nlp.p.val = theta

        for stage in range(self.ocp_solver.acados_ocp.dims.N + 1):
            self.set(stage, "p", p)

        self.nlp.set_p(p)

        # self.update_nlp()

    def get_p(self) -> np.ndarray:
        # return self.nlp.p.val
        return self.nlp.get_p()

    def get_parameters(self) -> np.ndarray:
        return self._parameters

    def get_dL_dp(self) -> np.ndarray:
        """
        Get the value of the sensitivity of the Lagrangian with respect to the parameters.

        Returns:
            dL_dp: Sensitivity of the Lagrangian with respect to the parameters.
        """
        return self.nlp.dL_dp.val.full().flatten()

    def get_L(self) -> float:
        """
        Get the value of the Lagrangian.

        Returns:
            L: Lagrangian.
        """
        return self.nlp.L.val

    def get_V(self) -> float:
        """
        Get the value of the value function.

        Assumes OCP is solved for state.

        Returns:
            V: Value function.
        """
        return self.ocp_solver.get_cost()

    def get_pi(self) -> np.ndarray:
        """
        Get the value of the policy.

        Assumes OCP is solved for state.
        """
        return self.ocp_solver.get(0, "u")

    def get_dpi_dp(self) -> np.ndarray:
        """
        Get the value of the sensitivity of the policy with respect to the parameters.

        Assumes OCP is solved for state and parameters.
        """

        dR_dz = self.nlp.dR_dz.val.full()
        dR_dp = self.nlp.dR_dp.val.full()

        # Find the constraints that are active
        lam_non_active_constraints = np.where(self.nlp.lam.val.full() < 1e-6)[0]
        # h_non_active_constraints = np.where(self.nlp.h.val.full() < -1e-10)[0]
        # non_active_constraints = np.where(self.nlp.h.val.full() < -1e-6)[0]

        # Add len(w) to the indices of the non-active constraints
        x = self.nlp.x.fun(self.nlp.vars.val)
        u = self.nlp.u.fun(self.nlp.vars.val)
        pi = self.nlp.pi.val.cat

        # non_active_constraints += self.nlp.w.val.cat.full().shape[0]

        # # Add len(pi) to the indices of the non-active constraints
        # non_active_constraints += self.nlp.pi.val.cat.full().shape[0]

        idx = x.shape[0] + u.shape[0] + pi.shape[0] + lam_non_active_constraints

        # Remove the non-active constraints from dR_dz
        dR_dz = np.delete(dR_dz, idx, axis=0)
        dR_dz = np.delete(dR_dz, idx, axis=1)

        # Remove the non-active constraints from dR_dp
        dR_dp = np.delete(dR_dp, idx, axis=0)

        dz_dp = np.linalg.solve(dR_dz, -dR_dp)

        dpi_dp = dz_dp[: self.ocp_solver.acados_ocp.dims.nu, :]

        return dpi_dp

    def get_dV_dp(self) -> float:
        """
        Get the value of the sensitivity of the value function with respect to the parameters.

        Assumes OCP is solved for state and parameters.

        Returns:
            dV_dp: Sensitivity of the value function with respect to the parameters.
        """
        return self.get_dL_dp()

    def get_Q(self) -> float:
        """
        Get the value of the state-action value function.

        Assumes OCP is solved for state and action.

        Returns:
            Q: State-action value function.
        """
        return self.ocp_solver.get_cost()

    def get_dQ_dp(self) -> float:
        """
        Get the value of the sensitivity of the state-action value function with respect to the parameters.

        Assumes OCP is solved for state, action and parameters.

        Returns:
            dQ_dp: Sensitivity of the state-action value function with respect to the parameters.
        """
        return self.get_dL_dp()

    def get_action(self, x0: np.ndarray) -> np.ndarray:
        """
        Update the solution of the OCP solver.

        Args:
            x0: Initial state.

        Returns:
            u: Optimal control action.
        """
        # Set initial state
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        # self.nlp.set(0, "lbx", x0)
        # self.nlp.set(0, "ubx", x0)

        # Solve the optimization problem
        self.status = self.ocp_solver.solve()

        # Get solution
        action = self.ocp_solver.get(0, "u")

        # Scale to [-1, 1] for gym
        # action = self.scale_action(action)

        return action

    def get_predicted_state_trajectory(self) -> np.ndarray:
        """
        Get the predicted state trajectory.

        Returns:
            x: Predicted state trajectory.
        """
        x = np.zeros((self.ocp.dims.N + 1, self.ocp.dims.nx))

        for i in range(self.ocp.dims.N + 1):
            x[i, :] = self.ocp_solver.get(i, "x")

        return x

    def get_predicted_control_trajectory(self) -> np.ndarray:
        """
        Get the predicted control trajectory.

        Returns:
            u: Predicted control trajectory.
        """
        u = np.zeros((self.ocp.dims.N, self.ocp.dims.nu))

        for i in range(self.ocp.dims.N):
            u[i, :] = self.ocp_solver.get(i, "u")

        return u

    def plot_prediction(self) -> None:
        """
        Plot the predicted trajectory.
        """

        x = self.get_predicted_state_trajectory()
        u = self.get_predicted_control_trajectory()

        _, ax = plt.subplots(self.ocp.dims.nx + self.ocp.dims.nu, 1, figsize=(10, 7))

        for i in range(self.ocp.dims.nx):
            ax[i].plot(x[:, i], "-o")
            ax[i].grid(True)
            ax[i].set_ylabel(f"x_{i}")

        # Make a stairs plot for u
        ax[self.ocp.dims.nx].step(np.arange(0, u.shape[0]), u.flatten(), where="post")
        ax[self.ocp.dims.nx].grid(True)
        ax[self.ocp.dims.nx].set_ylabel("u")

        # Draw bounds
        # lbx = self.ocp.constraints.lbx
        # ubx = self.ocp.constraints.ubx
        # lbu = self.ocp.constraints.lbu
        # ubu = self.ocp.constraints.ubu

        # for i in range(self.ocp.dims.nx):
        #     ax[i].plot(np.ones_like(x[:, i]) * lbx[i], "--", color="grey")
        #     ax[i].plot(np.ones_like(x[:, i]) * ubx[i], "--", color="gray")

        # ax[self.ocp.dims.nx].plot(np.ones_like(u) * lbu, "--", color="gray")
        # ax[self.ocp.dims.nx].plot(np.ones_like(u) * ubu, "--", color="gray")

        plt.show()

    # def check_kkt_conditions_at_solution(self) -> None:
    #     """
    #     Check the KKT conditions at the solution of the OCP solver.
    #     """
    #     self.nlp.

    def print_header(self) -> None:
        """
        Print the header for the data table.
        """
        print("{:>8} {:>8} {:>8} {:>8} {:>8}".format("x", "x_dot", "theta", "theta_dot", "u"))

    def print_data(self, x: np.ndarray, u: np.ndarray) -> None:
        """
        Print the data table.

        Args:
            x: State.
            u: Control.
        """
        print("{:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f}".format(x[0], x[1], x[2], x[3], u))
