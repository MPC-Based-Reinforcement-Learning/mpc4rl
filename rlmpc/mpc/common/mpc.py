import numpy as np
from abc import ABC
from acados_template import AcadosOcp, AcadosOcpSolver
from rlmpc.mpc.nlp import NLP, update_nlp, get_state_labels, get_input_labels, get_parameter_labels


class MPC(ABC):
    """
    MPC abstract base class.
    """

    ocp: AcadosOcp
    nlp: NLP
    ocp_solver: AcadosOcpSolver

    def __init__(self):
        super().__init__()

    # def get_parameters(self) -> np.ndarray:
    #     """
    #     Get the parameters of the MPC.

    #     :return: the parameters
    #     """
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

        # self.nlp.set(0, "lbx", x0)
        # self.nlp.set(0, "ubx", x0)

        # Solve the optimization problem
        self.status = self.ocp_solver.solve()

        # Get solution
        action = self.ocp_solver.get(0, "u")

        # Scale to [-1, 1] for gym
        # action = self.scale_action(action)

        return action

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

        # TODO: Implement this through nlp.set
        self.nlp.vars.val["lbx_0"] = x0
        self.nlp.vars.val["ubx_0"] = x0

        # Set initial action (needed for state-action value)
        self.ocp_solver.set(0, "u", u0)
        self.ocp_solver.constraints_set(0, "lbu", u0)
        self.ocp_solver.constraints_set(0, "ubu", u0)

        self.nlp.set(0, "lbu", u0)
        self.nlp.set(0, "ubu", u0)

        # Solve the optimization problem
        status = self.ocp_solver.solve()

        if status != 0:
            raise RuntimeError(f"Solver failed with status {status}. Exiting.")
            exit(0)

        self.ocp_solver.constraints_set(0, "lbu", self.ocp_solver.acados_ocp.constraints.lbu)
        self.ocp_solver.constraints_set(0, "ubu", self.ocp_solver.acados_ocp.constraints.ubu)

        self.nlp = update_nlp(self.nlp, self.ocp_solver)

        # Change bounds back to original
        self.nlp.set(0, "lbu", self.ocp_solver.acados_ocp.constraints.lbu)
        self.nlp.set(0, "ubu", self.ocp_solver.acados_ocp.constraints.ubu)

        # test_nlp_sanity(self.nlp)

        return status

    def update_nlp(self) -> None:
        """
        Update the NLP with the solution of the OCP solver.
        """
        self.nlp = update_nlp(self.nlp, self.ocp_solver)

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

        # self.nlp.p.val = p
        self.nlp.vars.val["p"] = p

        # self.update_nlp()

    def get_p(self) -> np.ndarray:
        return self.nlp.vars.val["p"].full().flatten()

    def get_parameter_values(self) -> np.ndarray:
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

        if status != 0:
            raise RuntimeError(f"Solver failed with status {status}. Exiting.")
            exit(0)

        self.update_nlp()

        # test_nlp_sanity(self.nlp)

        return status

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
        low = self.ocp_solver.acados_ocp.constraints.lbu
        high = self.ocp_solver.acados_ocp.constraints.ubu

        return 0.5 * (high - low) * (action + 1.0) + low

    def get_dL_dp(self) -> np.ndarray:
        """
        Get the value of the sensitivity of the Lagrangian with respect to the parameters.

        Returns:
            dL_dp: Sensitivity of the Lagrangian with respect to the parameters.
        """
        return self.nlp.dL_dp.val.full()

    def get_L(self) -> float:
        """
        Get the value of the Lagrangian.

        Returns:
            L: Lagrangian.
        """
        return float(self.nlp.L.val)

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
