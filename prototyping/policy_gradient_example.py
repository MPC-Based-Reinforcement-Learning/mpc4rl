from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import SX, vertcat, cos, sin
import numpy as np
import scipy
import matplotlib.pyplot as plt


def export_parameter_augmented_pendulum_ode_model() -> AcadosModel:
    """
    Augment the normal state vector with cart mass M.

    Return:
        AcadosModel: model of the augmented state cart
        nparam: number of states that are actually parameters
    """
    model_name = "parameter_augmented_pendulum_ode"

    # constants
    # M = 1.0  # mass of the cart [kg]
    M = SX.sym("M")  # mass of the cart [kg]
    m = 0.1  # mass of the ball [kg]
    g = 9.81  # gravity constant [m/s^2]
    l = 0.8  # length of the rod [m]

    nparam = 1

    # set up states & controls
    p = SX.sym("p")
    theta = SX.sym("theta")
    v = SX.sym("v")
    omega = SX.sym("omega")

    x = vertcat(p, theta, v, omega, M)

    F = SX.sym("F")
    u = vertcat(F)

    # xdot
    p_dot = SX.sym("p_dot")
    theta_dot = SX.sym("theta_dot")
    v_dot = SX.sym("v_dot")
    omega_dot = SX.sym("omega_dot")
    M_dot = SX.sym("M_dot")

    xdot = vertcat(p_dot, theta_dot, v_dot, omega_dot, M_dot)

    # algebraic variables
    # z = None

    # parameters
    param = []

    # dynamics
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    denominator = M + m - m * cos_theta * cos_theta
    f_expl = vertcat(
        v,
        omega,
        (-m * l * sin_theta * omega * omega + m * g * cos_theta * sin_theta + F) / denominator,
        (-m * l * cos_theta * sin_theta * omega * omega + F * cos_theta + (M + m) * g * sin_theta) / (l * denominator),
        0.0,
    )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = param
    model.name = model_name

    return model, nparam


def export_parameter_augmented_ocp(
    x0=np.array([0.0, np.pi / 6, 0.0, 0.0, 1.0]), N_horizon=50, T_horizon=2.0, Fmax=80.0
) -> AcadosOcp:
    """
    OCP with augmented state vector (p, theta, v, omega, M).
    """
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    ocp.model, nparam = export_parameter_augmented_pendulum_ode_model()

    # set dimensions
    ocp.dims.N = N_horizon
    nu = ocp.model.u.size()[0]
    nx = ocp.model.x.size()[0]

    # set cost
    Q_mat = 2 * np.diag([1e3, 1e3, 1e-2, 1e-2])
    R_mat = 2 * np.diag([1e-1])

    # We use the NONLINEAR_LS cost type and GAUSS_NEWTON Hessian approximation - One can also use the external cost module to specify generic cost.
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat

    ocp.model.cost_y_expr = vertcat(ocp.model.x[:-nparam], ocp.model.u)
    ocp.model.cost_y_expr_e = ocp.model.x[:-nparam]
    ocp.cost.yref = np.zeros((nx - nparam + nu,))
    ocp.cost.yref_e = np.zeros((nx - nparam,))

    # set constraints
    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.x0 = x0

    # set options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = "IRK"  # "DISCRETE"
    # ocp.solver_options.print_level = 1
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI, SQP
    ocp.solver_options.nlp_solver_max_iter = 400
    # ocp.solver_options.levenberg_marquardt = 1e-4

    # set prediction horizon
    ocp.solver_options.tf = T_horizon

    return ocp, nparam


if __name__ == "__main__":
    """
    Evaluate policy and calculate its gradient for the pendulum on a cart with an augmented state formulation for
    varying M.
    """

    p_nominal = 1.0
    delta_p = 0.01
    p_test = np.arange(p_nominal - 0.5, p_nominal + 0.5, delta_p)
    x0 = np.array([0.0, np.pi / 2, 0.0, 0.0, p_nominal])

    ocp, nparam = export_parameter_augmented_ocp(x0=x0)

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="parameter_augmented_acados_ocp.json")

    x0_augmented = [np.array(x0[:-1].tolist() + [p]) for p in p_test]

    dpi_dp = {
        "sens_u": np.zeros(p_test.shape[0]),
        "np.grad": np.zeros(p_test.shape[0]),
        "cd": np.zeros(p_test.shape[0]),
    }

    pi = np.zeros(p_test.shape[0])
    for i, x in enumerate(x0_augmented):
        # Evaluate the policy
        pi[i] = acados_ocp_solver.solve_for_x0(x)[0]

        # Calculate the policy gradient
        acados_ocp_solver.eval_param_sens(acados_ocp_solver.acados_ocp.dims.nx - nparam)
        dpi_dp["sens_u"][i] = acados_ocp_solver.get(0, "sens_u")[0]

    # Compare to numerical gradients
    dpi_dp["np.grad"] = np.gradient(pi, delta_p)
    dpi_dp["cd"] = (pi[2:] - pi[:-2]) / (p_test[2:] - p_test[:-2])

    pi_reconstructed = dict.fromkeys(dpi_dp.keys())

    for key in pi_reconstructed.keys():
        pi_reconstructed[key] = np.cumsum(dpi_dp[key]) * delta_p + pi[0]
        pi_reconstructed[key] += pi[0] - pi_reconstructed[key][0]

    _, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].plot(p_test, pi)
    # ax[0].plot(p_test, pi_reconstructed, "--")
    ax[0].plot(p_test, pi_reconstructed["np.grad"], "--")
    ax[0].plot(p_test[1:-1], pi_reconstructed["cd"], "-.")
    ax[1].legend(["pi", "pi integrate np.grad", "pi integrate central difference"])
    ax[1].plot(p_test, dpi_dp["sens_u"])
    ax[1].plot(p_test, dpi_dp["np.grad"], "--")
    ax[1].plot(p_test[1:-1], dpi_dp["cd"], "-.")
    ax[1].legend(["sens_u", "np.grad", "central difference"])
    ax[0].set_ylabel("p")
    ax[1].set_ylabel("dpi_dp")
    ax[1].set_xlabel("p")
    ax[0].grid(True)
    ax[1].grid(True)

    plt.show()
