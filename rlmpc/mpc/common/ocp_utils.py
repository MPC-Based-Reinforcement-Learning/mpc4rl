import casadi as ca
from casadi.tools import struct_symSX


def export_discrete_erk4_integrator_step(
    f_expl: ca.SX, x: ca.SX, u: ca.SX, p: struct_symSX, h: float, n_stages: int = 1
) -> ca.SX:
    """Define ERK4 integrator for continuous dynamics."""
    dt = h
    ode = ca.Function("f", [x, u, p], [f_expl])

    k1 = ode(x, u, p)
    k2 = ode(x + dt / 2 * k1, u, p)
    k3 = ode(x + dt / 2 * k2, u, p)
    k4 = ode(x + dt * k3, u, p)

    xnext = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return xnext
