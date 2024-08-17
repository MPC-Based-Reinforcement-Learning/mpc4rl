import os
import numpy as np
from rlmpc.mpc.linear_system.acados import AcadosMPC


from tests.common import (
    run_test_v_update_for_varying_parameters,
    run_test_q_update_for_varying_parameters,
    run_test_pi_update_for_varying_parameters,
    set_up_test_parameters,
)


# from common import set_up_test_parameters


param = {
    "A": np.array([[1.0, 0.25], [0.0, 1.0]]),
    # "B": np.array([[0.03125], [0.25]]),
    "B": np.array([[0.03125], [0.25]]),
    "Q": np.identity(2),
    "R": 1e-2 * np.identity(1),
    "b": np.array([[0.0], [0.0]]),
    "f": np.array([[0.0], [0.0], [0.0]]),
    "V_0": np.array([1e-3]),
}

kwargs = {
    "ocp_solver": {"json_file": "acados_ocp.json", "generate": True, "build": True},
    "ocp_sensitivity_solver": {"json_file": "acados_ocp_sensitivity.json", "generate": True, "build": True},
}


def build_mpc_args(generate_code: bool = False, build_code: bool = False, json_file_prefix: str = "acados_ocp_linear_system"):
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


def build_mpc_params() -> dict[np.ndarray]:
    """
    Define discrete double integrator matrices.
    """

    # A = np.array([[1.0, 0.25], [0.0, 1.0]])
    # B = np.array([[0.0], [0.25]])
    # b = np.array([[0.0], [0.0]])

    # Q = np.identity(2)
    # R = np.identity(1)

    # f = np.array([[0.0], [0.0], [0.0]])

    # V_0 = np.array([0])

    # return {"A": A, "B": B, "Q": Q, "R": R, "b": b, "f": f, "V_0": V_0}

    param = {
        "A": np.array([[1.0, 0.25], [0.0, 1.0]]),
        # "B": np.array([[0.03125], [0.25]]),
        "B": np.array([[0.03125], [0.25]]),
        "Q": np.identity(2),
        "R": np.identity(1),
        "b": np.array([[0.0], [0.0]]),
        "f": np.array([[0.0], [0.0], [0.0]]),
        "V_0": np.array([1e-3]),
    }

    return param


def set_up_mpc(
    generate_code: bool = False, build_code: bool = False, json_file_prefix: str = "acados_ocp_linear_system"
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


def test_get_param_does_not_give_None():
    # Use the existing ocp solver
    assert AcadosMPC(param, discount_factor=0.99, **kwargs).get_p() is not None


def test_set_p_get_p():
    """
    Test if the set_p and get_p methods work correctly.
    """

    mpc = set_up_mpc()

    p = mpc.get_p()

    p += np.random.randn(p.shape[0], p.shape[1])

    mpc.set_p(p)

    assert np.allclose(mpc.get_p(), p)


def test_v_update(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_linear_system",
    x0: np.ndarray = np.array([0.1, 0.1]),
    varying_param_label: str = "A_0",
    np_test: int = 10,
    plot: bool = False,
):
    mpc = set_up_mpc(generate_code, build_code, json_file_prefix)

    mpc.ocp_solver.solve()

    test_param = set_up_test_parameters(mpc, np_test, varying_param_label=varying_param_label)

    absolute_difference = run_test_v_update_for_varying_parameters(mpc, x0, test_param, plot)

    assert np.median(absolute_difference) <= 1e-1


def test_q_update(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_linear_system",
    x0: np.ndarray = np.array([0.1, 0.1]),
    u0: np.ndarray = np.array([0.0]),
    varying_param_label: str = "A_0",
    np_test: int = 10,
    plot: bool = False,
):
    mpc = set_up_mpc(generate_code, build_code, json_file_prefix)

    u0 = mpc.ocp_solver.solve_for_x0(x0)

    # mpc.ocp_solver.solve()
    # x0 = mpc.ocp_solver.get(0, "x")
    # u0 = mpc.ocp_solver.get(0, "u")

    test_param = set_up_test_parameters(mpc, np_test, varying_param_label=varying_param_label)

    absolute_difference = run_test_q_update_for_varying_parameters(mpc, x0, u0, test_param, plot)

    assert np.median(absolute_difference) <= 1e-1


def test_pi_update(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_linear_system",
    x0: np.ndarray = np.array([0.1, 0.1]),
    varying_param_label: str = "A_0",
    np_test: int = 10,
    plot: bool = False,
):
    # x0 = np.array([1.0, 1.0])
    mpc = set_up_mpc(generate_code, build_code, json_file_prefix)

    # mpc.ocp_solver.solve()
    # x0 = mpc.ocp_solver.get(0, "x")

    test_param = set_up_test_parameters(mpc, np_test, varying_param_label=varying_param_label)

    absolute_difference = run_test_pi_update_for_varying_parameters(mpc, x0, test_param, plot)

    assert np.median(absolute_difference) <= 1e-1


def test_get_param_returns_correct_shape():
    # TODO: Implement this test
    pass


def test_cost_scaling():
    # TODO: Implement this test. Can use a two-dimensional point mass system with a lower 2-norm constraint away from zero.
    pass


def test_use_existing_ocp_solver():
    # Generate and build the ocp solver

    _ = AcadosMPC(
        param,
        discount_factor=0.99,
        **{
            "ocp_solver": {"json_file": "acados_ocp.json", "generate": True, "build": True},
            "ocp_sensitivity_solver": {"json_file": "acados_ocp_sensitivity.json", "generate": True, "build": True},
        },
    )

    # Use the existing ocp solver
    _ = AcadosMPC(param, discount_factor=0.99, **kwargs)

    assert True


def main():
    test_v_update(plot=True, np_test=100, varying_param_label="A_2")
    # test_q_update(plot=True, np_test=100, varying_param_label="A_3")
    # test_pi_update(plot=True, np_test=100)
    # mpc = set_up_mpc()

    # u0 = mpc.ocp_solver.solve_for_x0(np.array([1.0, 1.0]))

    # print(u0)
    # print(mpc.ocp_solver.get_cost())


if __name__ == "__main__":
    main()
