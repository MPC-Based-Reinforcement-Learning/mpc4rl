import numpy as np
from src.agent.function_approximators.function_approximator import FunctionApproximator
from src.agent.function_approximators.mpc.mpc import MPC


def get_function_approximator(name: str, param: dict, **kwargs) -> FunctionApproximator:
    """
    Get the function approximator.

    Args:
        name: str
        param: dict

    Returns:
        function_approximator: FunctionApproximator
    """

    if name == "pendulum_mpc":
        return MPC(name=name, param=param)
    else:
        raise NotImplementedError("Function approximator not implemented.")


class Agent(object):
    """docstring for Agent."""

    def __init__(self, name: str, param: dict, **kwargs):
        super(Agent, self).__init__()

        # self.function_approximator = FunctionApproximator(name=name, param=param['function_approximator'])
        self.function_approximator = get_function_approximator(
            name=name, param=param["function_approximator"]["param"]
        )

        print("Agent: ", self.function_approximator)

    def evaluate_policy(self, x: np.ndarray) -> np.ndarray:
        return self.function_approximator.evaluate_policy(x=x)
