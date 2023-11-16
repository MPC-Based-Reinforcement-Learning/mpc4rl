import casadi as cs
import numpy as np
from typing import Union


def ERK4(
    f: Union[cs.SX, cs.MX],
    x: Union[cs.SX, cs.MX, np.ndarray],
    u: Union[cs.SX, cs.MX, np.ndarray],
    p: Union[cs.SX, cs.MX, np.ndarray],
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
    k1 = f(x, u, p)
    k2 = f(x + h / 2 * k1, u, p)
    k3 = f(x + h / 2 * k2, u, p)
    k4 = f(x + h * k3, u, p)
    xf = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return xf
