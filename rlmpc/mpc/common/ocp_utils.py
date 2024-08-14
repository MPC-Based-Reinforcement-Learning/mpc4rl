import casadi as ca
from casadi.tools import struct_symSX


def export_discrete_erk4_integrator_step(
    f_expl: ca.SX, x: ca.SX, u: ca.SX, p: struct_symSX, h: float, n_stages: int = 2
) -> ca.SX:
    """Define ERK4 integrator for continuous dynamics."""
    dt = h / n_stages
    ode = ca.Function("f", [x, u, p], [f_expl])
    xnext = x
    for _ in range(n_stages):
        k1 = ode(xnext, u, p)
        k2 = ode(xnext + dt / 2 * k1, u, p)
        k3 = ode(xnext + dt / 2 * k2, u, p)
        k4 = ode(xnext + dt * k3, u, p)
        xnext = xnext + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return xnext
