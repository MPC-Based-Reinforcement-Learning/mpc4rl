"""
    Agent class.
"""

import numpy as np
import matplotlib.pyplot as plt

from acados_template import AcadosOcp, AcadosOcpSolver

import src.agent.function_approximators.mpc.pendulum.pendulum as pendulum

from src.agent.function_approximators.function_approximator import FunctionApproximator


class MPC(FunctionApproximator):
    """
    MPC class.

    Args:
        name: str
        param: dict

    Attributes:
        ocp: AcadosOcp
        acados_solver: AcadosOcpSolver
    """

    def __init__(self, name: str, param: dict, **kwargs):
        super(MPC, self).__init__(name, param)

        # create ocp object to formulate the OCP
        self.ocp = AcadosOcp("/usr/local")

        if name == "pendulum_mpc":
            export_ocp = pendulum.export_ocp
            export_model = pendulum.export_model
            export_cost = pendulum.export_cost
            export_dims = pendulum.export_dims
            export_constraints = pendulum.export_constraints
            export_solver_options = pendulum.export_solver_options
            initialize_solver = pendulum.initialize_solver
        else:
            raise NotImplementedError(f"MPC {name} not implemented.")

        # set model
        self.ocp.model = export_model(param["model"])

        # set dimensions
        self.ocp.dims = export_dims(param["dims"])

        # # set cost
        self.ocp.cost = export_cost(self.ocp.dims, param["cost"])

        # # set constraints
        self.ocp.constraints = export_constraints(param["constraints"])

        # set options
        self.ocp.solver_options = export_solver_options(
            self.ocp.dims, param["solver_options"]
        )

        self.acados_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")

        status = initialize_solver(
            self.acados_solver, self.ocp, param["initial_values"]
        )

        if status != 0:
            raise Exception(
                "acados_ocp_solver_create() returned status {}. Exiting.".format(status)
            )

    def update(self, x0: np.ndarray) -> int:
        """
        Update the OCP solution.

        Args:
            x0: np.ndarray

        Returns:
            status: int
        """

        self.acados_solver.set(0, "x", x0)
        self.acados_solver.constraints_set(0, "lbx", x0)
        self.acados_solver.constraints_set(0, "ubx", x0)

        status = self.acados_solver.solve()

        return status

    def get_action(self, x: np.ndarray) -> np.ndarray:
        """
        Return the action.

        Args:
            x0: np.ndarray

        Returns:
            u: np.ndarray
        """

        self.update(x0=x)

        return self.acados_solver.get(0, "u")

    def plot_prediction(self):
        x_pred = np.ndarray((self.ocp.dims.N + 1, self.ocp.dims.nx))
        u_pred = np.zeros((self.ocp.dims.N, self.ocp.dims.nu))

        for stage in range(self.ocp.dims.N + 1):
            x_pred[stage, :] = self.acados_solver.get(stage, "x")
        for stage in range(self.ocp.dims.N):
            u_pred[stage, :] = self.acados_solver.get(stage, "u")

        _, ax = plt.subplots(5, 1)

        ax[0].stairs(x_pred[:, 0])
        ax[1].stairs(x_pred[:, 1])
        ax[2].stairs(x_pred[:, 2])
        ax[3].stairs(x_pred[:, 3])
        ax[4].stairs(u_pred[:, 0])

        plt.show()

    def evaluate_policy(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the policy.

        Args:
            x: np.ndarray

        Returns:
            u: np.ndarray

        """
        self.update(x0=x)

        return self.acados_solver.get(0, "u")

    def V(self, x: np.ndarray) -> float:
        """
        Evaluate the value function.

        Args:
            x: np.ndarray
            u: np.ndarray

        NB: Solver needs to be updated with x before calling this function.
        """
        return self.acados_solver.get_cost()

    def Q(self, x: np.ndarray, u: np.ndarray) -> float:
        """
        Evaluate the action-value function.
        """
        raise NotImplementedError("MPC action-value function not implemented.")

    def dQdp(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient of the cost function with respect to the parameters.

        Args:
            x: np.ndarray
            u: np.ndarray

        Returns:
            dQdp: np.ndarray

        """
        raise NotImplementedError("MPC action-value function gradient not implemented.")

    def dpidp(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient of the policy with respect to the parameters.

        Args:
            x: np.ndarray

        Returns:
            dpidp: np.ndarray

        """
        raise NotImplementedError("MPC policy gradient not implemented.")
