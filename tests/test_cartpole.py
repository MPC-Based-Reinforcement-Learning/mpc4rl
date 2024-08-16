import os
import numpy as np
from rlmpc.mpc.cartpole.acados import AcadosMPC
from rlmpc.mpc.chain_mass.ocp_utils import find_idx_for_labels

from test_chain_mass import (
    run_test_v_update_for_varying_parameters,
    run_test_q_update_for_varying_parameters,
    run_test_pi_update_for_varying_parameters,
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
        "model_name": "pendulum_on_a_cart",
        "Ts": 0.1,  # Sampling time
    }

    return param


def set_up_mpc(
    generate_code: bool = False, build_code: bool = False, json_file_prefix: str = "acados_ocp_pendulum_on_a_cart"
) -> tuple[AcadosMPC, np.ndarray, np.ndarray, np.ndarray]:
    kwargs = build_mpc_args(generate_code, build_code, json_file_prefix)
    params = build_mpc_params()
    mpc = AcadosMPC(param=params, **kwargs)
    mpc.ocp_solver.solve()

    x0 = mpc.ocp_solver.get(0, "x")
    u0 = mpc.ocp_solver.get(0, "u")
    p0 = mpc.ocp_solver.acados_ocp.parameter_values

    return mpc, x0, u0, p0


def test_mpc_initializes():
    # mpc = AcadosMPC(param, discount_factor=0.99, **kwargs)
    mpc, _, _, _ = set_up_mpc()
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

    mpc, _, _, _ = set_up_mpc()

    p = mpc.get_p()

    p += np.random.randn(p.shape[0])

    mpc.set_p(p)

    assert np.allclose(mpc.get_p(), p)


def set_up_test_parameters(mpc: AcadosMPC, np_test: int = 10) -> np.ndarray:
    parameter_values = mpc.ocp_solver.acados_ocp.parameter_values

    test_param = np.repeat(parameter_values, np_test).reshape(len(parameter_values), -1)

    # Vary parameter along one dimension of p_label
    p_idx = find_idx_for_labels(mpc.ocp_solver.acados_ocp.model.p, "M")[0]
    test_param[p_idx, :] = np.linspace(0.5 * parameter_values[p_idx], 1.5 * parameter_values[p_idx], np_test).flatten()

    return test_param


def test_v_update(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_pendulum_on_a_cart",
    np_test: int = 100,
    plot: bool = False,
):
    mpc, x0, _, _ = set_up_mpc(generate_code, build_code, json_file_prefix)

    test_param = set_up_test_parameters(mpc, np_test)

    absolute_difference = run_test_v_update_for_varying_parameters(mpc, x0, test_param, plot)

    assert np.median(absolute_difference) <= 1e-1


def test_q_update(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_pendulum_on_a_cart",
    np_test: int = 100,
    plot: bool = False,
):
    mpc, x0, u0, _ = set_up_mpc(generate_code, build_code, json_file_prefix)

    test_param = set_up_test_parameters(mpc, np_test)

    absolute_difference = run_test_q_update_for_varying_parameters(mpc, x0, u0, test_param, plot)

    assert np.median(absolute_difference) <= 1e-1


def test_pi_update(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_pendulum_on_a_cart",
    np_test: int = 100,
    plot: bool = False,
):
    mpc, x0, _, _ = set_up_mpc(generate_code, build_code, json_file_prefix)

    test_param = set_up_test_parameters(mpc, np_test)

    absolute_difference = run_test_pi_update_for_varying_parameters(mpc, x0, test_param, plot)

    assert np.median(absolute_difference) <= 1e-1


def test_get_param_returns_correct_shape():
    # TODO: Implement this test
    pass


def test_cost_scaling():
    # TODO: Implement this test. Can use a two-dimensional point mass system with a lower 2-norm constraint away from zero.
    pass


def main():
    # test_set_p_get_p()
    # test_v_update(plot=True, np_test=100)
    # test_q_update(plot=True, np_test=100)
    # test_pi_update(plot=True, np_test=100)
    # test_mpc_initializes()
    # set_up_mpc(generate_code=True, build_code=True)

    assert True


if __name__ == "__main__":
    main()
