from abc import abstractmethod
import numpy as np

class FunctionApproximator(object):
    """docstring for Function_approximator."""
    def __init__(self, name: str, param: dict, **kwargs):
        super(FunctionApproximator, self).__init__()

    @abstractmethod
    def evaluate_policy(self, x: np.ndarray) -> np.ndarray:
        # self.function_object.evaluate_policy(x=x)
        raise NotImplementedError
    
    @abstractmethod
    def evaluate_value_function(self, x: np.ndarray) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def evaluate_action_value_function(self, x: np.ndarray, u: np.ndarray) -> float:
        raise NotImplementedError

    