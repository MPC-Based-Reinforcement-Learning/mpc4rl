"""
    Pendulum ode model exported from CasADi.
    Modified from acados github repo.

    Author: Dirk Reinhardt
"""

import casadi as cs
from acados_template import AcadosModel

def export_pendulum_ode_model(param: dict = {
    'M': 1., # mass of the cart [kg]
    'm': 0.1, # mass of the ball [kg]
    'g': 9.81, # gravity constant [m/s^2]
    'l': 0.8 # length of the rod [m]
    }) -> AcadosModel:

    model_name = 'pendulum_ode'

    # constants
    M = param['M']
    m = param['m']
    g = param['g']
    l = param['l']

    # set up states & controls
    x1      = cs.SX.sym('x1')
    theta   = cs.SX.sym('theta')
    v1      = cs.SX.sym('v1')
    dtheta  = cs.SX.sym('dtheta')

    x = cs.vertcat(x1, theta, v1, dtheta)

    F = cs.SX.sym('F')
    u = cs.vertcat(F)

    # xdot
    x1_dot      = cs.SX.sym('x1_dot')
    theta_dot   = cs.SX.sym('theta_dot')
    v1_dot      = cs.SX.sym('v1_dot')
    dtheta_dot  = cs.SX.sym('dtheta_dot')

    xdot = cs.vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot)

    # algebraic variables
    # z = None

    # parameters
    p = []

    # dynamics
    cos_theta = cs.cos(theta)
    sin_theta = cs.sin(theta)
    denominator = M + m - m*cos_theta*cos_theta
    f_expl = cs.vertcat(v1,
                     dtheta,
                     (-m*l*sin_theta*dtheta*dtheta + m*g*cos_theta*sin_theta+F)/denominator,
                     (-m*l*cos_theta*sin_theta*dtheta*dtheta + F*cos_theta+(M+m)*g*sin_theta)/(l*denominator)
                     )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.name = model_name

    return model