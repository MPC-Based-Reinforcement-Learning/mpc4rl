import numpy as np
from abc import ABC
from acados_template import AcadosOcp, AcadosOcpSolver
from rlmpc.mpc.common.nlp import NLP, update_nlp, get_state_labels, get_input_labels, get_parameter_labels

# from rlmpc.mpc.common.acados import set_discount_factor
from ctypes import POINTER, c_double, c_int, c_void_p, cast


class MPC(ABC):
    """
    MPC abstract base class.
    """

    ocp: AcadosOcp
    # nlp: NLP
    ocp_solver: AcadosOcpSolver
    ocp_sensitivity_solver: AcadosOcpSolver

    def __init__(self, gamma: float = 1.0, **kwargs):
        super().__init__()

        self.discount_factor = gamma

        self.nlp_timing = {}

    def get_parameters(self) -> np.ndarray:
        return self.get_p()

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

        # Solve the optimization problem
        self.status = self.ocp_solver.solve()

        # Get solution
        action = self.ocp_solver.get(0, "u")

        # Scale to [-1, 1] for gym
        # action = self.scale_action(action)

        return action

    def solve(self, store_iterate: bool = True, iterate_file: str = "iterate.json") -> int:
        self.ocp_solver.solve()
        self.ocp_solver.store_iterate(filename=iterate_file, overwrite=True, verbose=False)
        # self.ocp_sensitivity_solver.load_iterate(filename=iterate_file, verbose=False)
        # self.ocp_sensitivity_solver.solve()

        # # _, sens_u_ = self.ocp_sensitivity_solver.eval_solution_sensitivity(0, "params_global")
        # dVdp = self.ocp_sensitivity_solver.eval_and_get_optimal_value_gradient(with_respect_to = "params_global")

    def q_update(self, x0: np.ndarray, u0: np.ndarray, p: np.ndarray = None) -> int:
        """
        Update the solution of the OCP solver.

        Args:
            x0: Initial state.

        Returns:
            status: Status of the solver.
        """

        # Set initial action (needed for state-action value)
        self.ocp_solver.set(0, "u", u0)

        self.ocp_solver.constraints_set(0, "lbu", u0)
        self.ocp_solver.constraints_set(0, "ubu", u0)

        optimal_value, optimal_value_gradient = self.v_update(x0, p)

        # Change bounds back to original
        self.ocp_solver.constraints_set(0, "lbu", self.ocp_solver.acados_ocp.constraints.lbu)
        self.ocp_solver.constraints_set(0, "ubu", self.ocp_solver.acados_ocp.constraints.ubu)

        return optimal_value, optimal_value_gradient

    def v_update(self, x0: np.ndarray, p: np.ndarray = None) -> int:
        if p is not None:
            self.ocp_solver.acados_ocp.parameter_values = p
            for stage in range(self.ocp_solver.acados_ocp.dims.N + 1):
                self.ocp_solver.set(stage, "p", p)

        # Set initial state
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        # Solve the optimization problem
        status = self.ocp_solver.solve()

        optimal_value = self.ocp_solver.get_cost()

        if status != 0:
            raise RuntimeError(f"Solver failed with status {status}. Exiting.")

        optimal_value_gradient = self.ocp_solver.eval_and_get_optimal_value_gradient(with_respect_to="params_global")

        return optimal_value, optimal_value_gradient

    def pi_update(self, x0: np.ndarray, p: np.ndarray = None) -> int:
        if p is not None:
            self.ocp_solver.acados_ocp.parameter_values = p
            for stage in range(self.ocp_solver.acados_ocp.dims.N + 1):
                self.ocp_solver.set(stage, "p", p)
                self.ocp_sensitivity_solver.set(stage, "p", p)

        # Set initial state
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        # Solve the optimization problem
        pi = self.ocp_solver.solve_for_x0(x0)

        self.ocp_solver.store_iterate(filename="iterate.json", overwrite=True, verbose=False)
        self.ocp_sensitivity_solver.load_iterate(filename="iterate.json", verbose=False)
        self.ocp_sensitivity_solver.solve_for_x0(x0, fail_on_nonzero_status=False, print_stats_on_failure=False)

        # Calculate the policy gradient
        _, dpidp = self.ocp_sensitivity_solver.eval_solution_sensitivity(0, "params_global")

        return pi, dpidp

    def update_nlp(self) -> None:
        """
        Update the NLP with the solution of the OCP solver.
        """
        self.nlp, self.nlp_timing = update_nlp(self.nlp, self.ocp_solver)

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
        # TODO: Implement this
        pass

    def set_p(self, p: np.ndarray) -> None:
        """
        Set the value of the parameters.

        Args:
            p: Parameters.
        """
        self.ocp_solver.acados_ocp.parameter_values = p
        for stage in range(self.ocp_solver.acados_ocp.dims.N + 1):
            self.ocp_solver.set(stage, "p", p)
            self.ocp_sensitivity_solver.set(stage, "p", p)

    def get_p(self) -> np.ndarray:
        """
        Get the value of the parameters for the nlp.
        """
        return self.ocp_solver.acados_ocp.parameter_values

    def get_parameter_values(self) -> np.ndarray:
        """
        Get the value of the parameters for the nlp.
        """
        return self.get_p()

    def get_parameter_labels(self) -> list:
        return get_parameter_labels(self.ocp_solver.acados_ocp)

    def get_state_labels(self) -> list:
        return get_state_labels(self.ocp_solver.acados_ocp)

    def get_input_labels(self) -> list:
        return get_input_labels(self.ocp_solver.acados_ocp)

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

        # Solve the optimization problem
        status = self.ocp_solver.solve()

        if status != 0:
            raise RuntimeError(f"Solver failed update with status {status}. Exiting.")

        # test_nlp_sanity(self.nlp)

        return status

    def reset(self, x0: np.ndarray):
        self.ocp_solver.reset()

        set_discount_factor(self.ocp_solver, self.discount_factor)
        set_discount_factor(self.ocp_sensitivity_solver, self.discount_factor)

        for stage in range(self.ocp_solver.acados_ocp.dims.N + 1):
            # self.ocp_solver.set(stage, "x", self.ocp_solver.acados_ocp.constraints.lbx_0)
            self.ocp_solver.set(stage, "x", x0)
            self.ocp_sensitivity_solver.set(stage, "x", x0)

    def set(self, stage, field, value):
        if field == "p":
            # TODO: Implement this
            pass

            # p_temp = self.nlp.p.sym(value)
            # This allows to set parameters in the NLP but not in the OCP solver. This can be a problem.
            # for key in p_temp.keys():
            #     self.nlp.set_parameter(key, p_temp[key])

            # if self.ocp_solver.acados_ocp.dims.np > 0:
            #     self.ocp_solver.set(stage, field, p_temp["model"].full().flatten())

            # if self.nlp.vars.val["p", "W_0"].shape[0] > 0:
            #     p_temp = self.nlp.vars.sym(0)
            #     self.ocp_solver.cost_set
            #     # self.nlp.set(stage, field, value)
            #     W_0 = self.nlp.vars.val["p", "W_0"].full()
            #     print("Not implemented")

        else:
            self.ocp_solver.set(stage, field, value)

    def set_parameter(self, value_, api="new"):
        p_temp = self.nlp.p.sym(value_)

        if "W_0" in p_temp.keys():
            self.nlp.set_parameter("W_0", p_temp["W_0"])
            self.ocp_solver.cost_set(0, "W", self.nlp.get_parameter("W_0").full(), api=api)

        if "W" in p_temp.keys():
            self.nlp.set_parameter("W", p_temp["W"])
            for stage in range(1, self.ocp_solver.acados_ocp.dims.N):
                self.ocp_solver.cost_set(stage, "W", self.nlp.get_parameter("W").full(), api=api)

        if "yref_0" in p_temp.keys():
            self.nlp.set_parameter("yref_0", p_temp["yref_0"])
            self.ocp_solver.cost_set(0, "yref", self.nlp.get_parameter("yref_0").full().flatten(), api=api)

        if "yref" in p_temp.keys():
            self.nlp.set_parameter("yref", p_temp["yref"])
            for stage in range(1, self.ocp_solver.acados_ocp.dims.N):
                self.ocp_solver.cost_set(stage, "yref", self.nlp.get_parameter("yref").full().flatten(), api=api)

        if self.ocp_solver.acados_ocp.dims.np > 0:
            self.nlp.set_parameter("model", p_temp["model"])
            for stage in range(self.ocp_solver.acados_ocp.dims.N + 1):
                self.ocp_solver.set(stage, "p", p_temp["model"].full().flatten())

    def set_discount_factor(self, discount_factor_: float) -> None:
        """
        Set the discount factor.

        NB: This overwrites the scaling of the cost function (control interval length by default).

        Args:
            gamma: Discount factor.

        """

        print(f"Setting discount factor to {discount_factor_}")

        self.discount_factor = discount_factor_

        for stage in range(0, self.ocp_solver.acados_ocp.dims.N + 1):
            self.ocp_solver.cost_set(stage, "scaling", discount_factor_**stage)
            self.ocp_sensitivity_solver.cost_set(stage, "scaling", discount_factor_**stage)

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
        low = self.ocp_solver.acados_ocp.constraints.lbu
        high = self.ocp_solver.acados_ocp.constraints.ubu

        return 0.5 * (high - low) * (action + 1.0) + low

    def get_dL_dp(self) -> np.ndarray:
        """
        Get the value of the sensitivity of the Lagrangian with respect to the parameters.

        Returns:
            dL_dp: Sensitivity of the Lagrangian with respect to the parameters.
        """
        # TODO: Implement this or remove. Only need get_dV_dp and get_dQ_dp.

    def get_L(self) -> float:
        """
        Get the value of the Lagrangian.

        Returns:
            L: Lagrangian.
        """
        # TODO: Implement this

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

    def compute_dpi_dp_finite_differences(self, p: np.ndarray, idx: int = None, delta: float = 1e-4) -> np.ndarray:
        """
        Compute the sensitivity of the policy with respect to the parameters using finite differences.

        Assumes OCP is solved for state and parameters.
        """

        # TODO: Implement this using its own ocp_sensitivity_solver class

        pi0 = self.get_pi().copy()

        p0 = self.nlp.p.val.cat.full().flatten()
        pplus = p0.copy()

        nu = self.ocp_solver.acados_ocp.dims.nu
        nparam = p0.shape[0]

        dpi_dp = np.zeros((nu, nparam))

        # Check if idx is None
        if idx is None:
            for i in range(nparam):
                pplus[i] += delta

                self.set_p(pplus)

                self.update(self.ocp_solver.acados_ocp.constraints.lbx_0)

                piplus = self.get_pi()

                dpi_dp[:, i] = (piplus - pi0) / (delta)

                pplus[i] = p0[i]

            return dpi_dp

        # for i in range(nparam):
        pplus[idx] += delta

        self.set_p(pplus)

        self.update(self.ocp_solver.acados_ocp.constraints.lbx_0)

        piplus = self.get_pi()

        dpi_dp[:, idx] = (piplus - pi0) / (delta)

        pplus[idx] = p0[idx]

        return dpi_dp

    def get_dpi_dp(self, finite_differences: bool = False, idx: int = 0) -> np.ndarray:
        """
        Get the value of the sensitivity of the policy with respect to the parameters.

        Assumes OCP is solved for state and parameters.
        """

        # TODO: Implement this using ocp_sensitivity_solver
