from typing import Iterator
import numpy as np
from abc import ABC, abstractmethod
from acados_template import AcadosOcp, AcadosOcpSolver
import casadi as cs


class MPC(ABC):
    """
    MPC abstract base class.
    """

    ocp: AcadosOcp
    ocp_solver: AcadosOcpSolver

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Get the action from the MPC.

        :param observation: the input observation
        :return: the action
        """

    @abstractmethod
    def get_parameters(self) -> np.ndarray:
        """
        Get the parameters of the MPC.

        :return: the parameters
        """
