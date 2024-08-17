import os
import numpy as np
from rlmpc.mpc.pendulum_on_cart.acados import AcadosMPC
import matplotlib.pyplot as plt
# from rlmpc.mpc.chain_mass.ocp_utils import find_idx_for_labels

from common import (
    run_test_v_update_for_varying_parameters,
    run_test_q_update_for_varying_parameters,
    run_test_pi_update_for_varying_parameters,
    set_up_test_parameters,
)


kwargs = {
    "ocp_solver": {"json_file": "acados_ocp.json", "generate": True, "build": True},
    "ocp_sensitivity_solver": {"json_file": "acados_ocp_sensitivity.json", "generate": True, "build": True},
}


def build_mpc_args(generate_code: bool = False, build_code: bool = False, json_file_prefix: str = "acados_ocp_pole_on_a_cart"):
    kwargs = {
        "ocp_solver": {"json_file": f"{json_file_prefix}.json"},
        "ocp_sensitivity_solver": {"json_file": f"{json_file_prefix}_sensitivity.json"},
    }

    for key in kwargs.keys():
        if os.path.isfile(kwargs[key]["json_file"]):
            kwargs[key]["generate"] = generate_code
            kwargs[key]["build"] = build_code
        else:
            kwargs[key]["generate"] = True
            kwargs[key]["build"] = True

    return kwargs


def build_mpc_params() -> dict[np.ndarray, float]:
    """
    Define discrete double integrator matrices.
    """

    param = {
        "M": 1.0,  # Mass of the cart
        "m": 0.1,  # Mass of the ball
        "g": 9.81,  # Gravity constant
        "l": 0.8,  # Length of the rod
        "Q": 2 * np.diag([1e3, 1e3, 1e-2, 1e-2]),  # State cost matrix
        "R": 2 * np.diag([1e-2]),  # Control cost matrix
        "model_name": "pendulum_on_cart",
        "Tf": 1.0,  # Prediction horizon
        "N": 20,  # Number of control intervals
    }

    # param["Ts"] = param["Tf"] / param["N"]

    return param


def set_up_mpc(
    generate_code: bool = False, build_code: bool = False, json_file_prefix: str = "acados_ocp_pendulum_on_cart"
) -> AcadosMPC:
    kwargs = build_mpc_args(generate_code, build_code, json_file_prefix)
    params = build_mpc_params()
    mpc = AcadosMPC(param=params, **kwargs)

    # mpc.ocp_solver.solve()

    # x0 = mpc.ocp_solver.get(0, "x")
    # u0 = mpc.ocp_solver.get(0, "u")
    # p0 = mpc.ocp_solver.acados_ocp.parameter_values

    return mpc


def test_mpc_initializes():
    # mpc = AcadosMPC(param, discount_factor=0.99, **kwargs)
    mpc = set_up_mpc()
    assert mpc is not None
    assert mpc.ocp_solver is not None
    assert mpc.ocp_solver.acados_ocp is not None
    assert mpc.ocp_solver.acados_ocp.model is not None
    assert mpc.ocp_solver.acados_ocp.dims is not None
    assert mpc.ocp_solver.acados_ocp.cost is not None


def test_set_p_get_p():
    """
    Test if the set_p and get_p methods work correctly.
    """

    mpc = set_up_mpc()

    p = mpc.get_p()

    p += np.random.randn(p.shape[0])

    mpc.set_p(p)

    assert np.allclose(mpc.get_p(), p)


# def set_up_test_parameters(mpc: AcadosMPC, np_test: int = 10) -> np.ndarray:
#     parameter_values = mpc.ocp_solver.acados_ocp.parameter_values

#     test_param = np.repeat(parameter_values, np_test).reshape(len(parameter_values), -1)

#     # Vary parameter along one dimension of p_label
#     p_idx = find_idx_for_labels(mpc.ocp_solver.acados_ocp.model.p, "M")[0]
#     test_param[p_idx, :] = np.linspace(0.5 * parameter_values[p_idx], 1.5 * parameter_values[p_idx], np_test).flatten()

#     return test_param


def test_v_update(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_pendulum_on_cart",
    varying_param_label="M",
    x0=np.array([0.0, np.pi, 0.0, 0.0]),
    np_test: int = 100,
    plot: bool = False,
):
    mpc = set_up_mpc(generate_code, build_code, json_file_prefix)

    test_param = set_up_test_parameters(mpc, np_test, varying_param_label=varying_param_label)

    absolute_difference = run_test_v_update_for_varying_parameters(mpc, x0, test_param, plot)

    assert np.median(absolute_difference) <= 1e-1


def test_q_update(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_pendulum_on_cart",
    varying_param_label="M",
    x0=np.array([0.0, np.pi, 0.0, 0.0]),
    u0=0.0,
    np_test: int = 100,
    plot: bool = False,
):
    mpc = set_up_mpc(generate_code, build_code, json_file_prefix)

    test_param = set_up_test_parameters(mpc, np_test, varying_param_label=varying_param_label)

    absolute_difference = run_test_q_update_for_varying_parameters(mpc, x0, u0, test_param, plot)

    assert np.median(absolute_difference) <= 1e-1


def test_pi_update(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_pendulum_on_cart",
    varying_param_label="M",
    x0=np.array([0.0, np.pi, 0.0, 0.0]),
    np_test: int = 100,
    plot: bool = False,
):
    mpc = set_up_mpc(generate_code, build_code, json_file_prefix)

    # test_param = set_up_test_parameters(mpc, np_test)
    test_param = set_up_test_parameters(mpc, np_test, varying_param_label=varying_param_label)

    absolute_difference = run_test_pi_update_for_varying_parameters(mpc, x0, test_param, plot)

    assert np.median(absolute_difference) <= 1e-1


def test_closed_loop(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_pendulum_on_cart",
    x0: np.ndarray = np.array([0.0, np.pi, 0.0, 0.0]),
    n_sim: int = 50,
    plot: bool = False,
):
    mpc = set_up_mpc(generate_code, build_code, json_file_prefix)

    x = [x0]
    u = []

    mpc.ocp_solver.constraints_set(0, "lbx", x0)

    for _ in range(n_sim):
        u.append(mpc.ocp_solver.solve_for_x0(x[-1]))
        x.append(mpc.ocp_solver.get(1, "x"))
        assert mpc.ocp_solver.get_status() == 0

    x = np.array(x)
    u = np.array(u)

    if plot:
        plt.figure()
        plt.subplot(5, 1, 1)
        plt.step(np.arange(n_sim + 1), x[:, 0], label="x_0")
        plt.subplot(5, 1, 2)
        plt.step(np.arange(n_sim + 1), x[:, 1], label="x_1")
        plt.subplot(5, 1, 3)
        plt.step(np.arange(n_sim + 1), x[:, 2], label="x_2")
        plt.subplot(5, 1, 4)
        plt.step(np.arange(n_sim + 1), x[:, 3], label="x_3")
        plt.subplot(5, 1, 5)
        plt.step(np.arange(n_sim), u, label="u")
        plt.legend()
        plt.show()

    for i in range(x.shape[1]):
        assert np.median(x[-10:, i]) <= 1e-1
    assert np.median(u[-10:]) <= 1e-1


def main():
    # test_set_p_get_p()
    # test_v_update(plot=True, np_test=100, varying_param_label="M")
    # test_closed_loop(plot=True)
    # test_q_update(plot=True, np_test=100)
    # test_pi_update(plot=True, np_test=100)
    # test_mpc_initializes()
    # set_up_mpc(generate_code=False, build_code=False)

    assert True


if __name__ == "__main__":
    main()
