from rlmpc.mpc.chain_mass.acados import AcadosMPC
from rlmpc.mpc.chain_mass.ocp_utils import get_chain_params, define_x0
# from test_linear_system import set_up_test_parameters

import numpy as np
import os

from rlmpc.mpc.common.testing import (
    run_test_pi_update_for_varying_parameters,
    run_test_q_update_for_varying_parameters,
    run_test_v_update_for_varying_parameters,
    set_up_test_parameters,
)


def test_AcadosMPC_initializes(
    generate_code: bool = False,
    build_code: bool = False,
    n_mass: int = 3,
    json_file_prefix: str = "acados_ocp_chain_mass_ds",
):
    kwargs = build_mpc_args(generate_code, build_code, n_mass, json_file_prefix)
    param = build_mpc_params(n_mass)
    _ = AcadosMPC(param=param, **kwargs)
    assert True


def test_get_p(
    generate_code: bool = False,
    build_code: bool = False,
    n_mass: int = 3,
    json_file_prefix: str = "acados_ocp_chain_mass_ds",
):
    kwargs = build_mpc_args(generate_code, build_code, n_mass, json_file_prefix)
    param = build_mpc_params(n_mass)
    mpc = AcadosMPC(param=param, **kwargs)
    p = mpc.get_p()
    assert p is not None


def test_set_p_get_p(
    generate_code: bool = False,
    build_code: bool = False,
    n_mass: int = 3,
    json_file_prefix: str = "acados_ocp_chain_mass_ds",
):
    """
    Test if the set_p and get_p methods work correctly.
    """

    param = build_mpc_params(n_mass)
    kwargs = build_mpc_args(generate_code, build_code, n_mass, json_file_prefix)

    mpc = AcadosMPC(param=param, **kwargs)

    p = mpc.get_p()

    p += np.random.randn(len(p))

    mpc.set_p(p)

    assert np.allclose(mpc.get_p(), p)


def set_up_mpc(
    generate_code: bool = False, build_code: bool = False, n_mass: int = 3, json_file_prefix: str = "acados_ocp_chain_mass_ds"
):
    kwargs = build_mpc_args(generate_code, build_code, n_mass, json_file_prefix)
    param = build_mpc_params(n_mass)
    mpc = AcadosMPC(param=param, **kwargs)
    # mpc.ocp_solver.solve()
    # u0 = mpc.ocp_solver.get(0, "u")

    # _, x0 = define_x0(param, mpc.ocp_solver.acados_ocp)
    # return mpc, x0, u0, mpc.ocp_solver.acados_ocp.parameter_values
    return mpc


def test_q_update(
    generate_code: bool = False,
    build_code: bool = False,
    n_mass: int = 3,
    json_file_prefix: str = "acados_ocp_chain_mass_ds",
    varying_param_label="C_1_0",
    np_test: int = 10,
    plot: bool = False,
):
    # kwargs = build_mpc_args(generate_code, build_code, n_mass, json_file_prefix)

    # mpc, x0, u0, p0 = set_up_mpc(generate_code, build_code, n_mass, json_file_prefix)
    mpc = set_up_mpc(generate_code, build_code, n_mass, json_file_prefix)

    x0 = define_x0(chain_params_=build_mpc_params(n_mass), ocp=mpc.ocp_solver.acados_ocp)

    u0 = mpc.ocp_solver.solve_for_x0(x0)

    # test_param = set_up_test_parameters(n_mass, np_test, p0)
    test_param = set_up_test_parameters(mpc=mpc, np_test=np_test, varying_param_label=varying_param_label)

    absolute_difference = run_test_q_update_for_varying_parameters(mpc, x0, u0, test_param, plot)

    assert np.median(absolute_difference) <= 1e-1


def test_v_update(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_chain_mass_ds",
    n_mass: int = 3,
    varying_param_labels: list = [],
    np_test: int = 10,
    plot: bool = False,
):
    # mpc = set_up_mpc(generate_code, build_code, n_mass, json_file_prefix)

    mpc = set_up_mpc(generate_code=False, build_code=False, n_mass=n_mass, json_file_prefix="acados_ocp_chain_mass_ds")

    if varying_param_labels == []:
        varying_param_labels = get_non_zero_parameter_labels(mpc)

    # x0 = define_x0(chain_params_=build_mpc_params(n_mass=n_mass), ocp=mpc.ocp_solver.acados_ocp)
    x0 = define_x0(chain_params_=build_mpc_params(n_mass=n_mass), ocp=mpc.ocp_solver.acados_ocp)

    for label in varying_param_labels:
        print(f"Testing for value gradients for varying parameter: {label}")
        test_param = set_up_test_parameters(mpc=mpc, np_test=np_test, varying_param_label=label)

        absolute_difference = run_test_v_update_for_varying_parameters(mpc, x0, test_param, plot)

    # assert np.median(absolute_difference) <= 1e-1
    assert True


def test_pi_update(
    generate_code: bool = False,
    build_code: bool = False,
    n_mass: int = 3,
    json_file_prefix: str = "acados_ocp_chain_mass_ds",
    varying_param_label="C_1_0",
    np_test: int = 10,
    plot: bool = False,
):
    mpc = set_up_mpc(generate_code, build_code, n_mass, json_file_prefix)

    test_param = set_up_test_parameters(mpc=mpc, np_test=np_test, varying_param_label=varying_param_label)

    x0 = define_x0(chain_params_=build_mpc_params(n_mass=n_mass), ocp=mpc.ocp_solver.acados_ocp)

    absolute_difference = run_test_pi_update_for_varying_parameters(mpc, x0, test_param, plot)

    assert np.median(absolute_difference) <= 1e-1


# def set_up_test_parameters(n_mass, np_test, p0):
#     parameter_values = p0

#     test_param = np.repeat(parameter_values, np_test).reshape(len(parameter_values), -1)
#     # Vary parameter along one dimension of p_label
#     p_label = f"C_{n_mass - 2}_0"
#     p_sym = define_param_struct_symSX(n_mass, disturbance=True)
#     p_idx = find_idx_for_labels(p_sym.cat, p_label)[0]
#     test_param[p_idx, :] = np.linspace(0.5 * parameter_values[p_idx], 1.5 * parameter_values[p_idx], np_test)
#     return test_param


def build_mpc_args(generate_code, build_code, n_mass, json_file_prefix):
    kwargs = {
        "ocp_solver": {"json_file": f"{json_file_prefix}_{n_mass}.json"},
        "ocp_sensitivity_solver": {"json_file": f"{json_file_prefix}_{n_mass}_sensitivity.json"},
    }

    for key in kwargs.keys():
        if os.path.isfile(kwargs[key]["json_file"]):
            kwargs[key]["generate"] = generate_code
            kwargs[key]["build"] = build_code
        else:
            kwargs[key]["generate"] = True
            kwargs[key]["build"] = True

    return kwargs


def build_mpc_params(n_mass: int = 3) -> dict:
    param = get_chain_params()
    param["n_mass"] = n_mass

    return param


def get_non_zero_parameter_labels(mpc: AcadosMPC):
    return [
        mpc.ocp_solver.acados_ocp.model.p[i].name() for i in np.where(mpc.ocp_solver.acados_ocp.parameter_values != 0.0)[0]
    ]


if __name__ == "__main__":
    # mpc = set_up_mpc(generate_code=False, build_code=False, n_mass=3, json_file_prefix="acados_ocp_chain_mass_ds")
    # from rlmpc.mpc.chain_mass.ocp_utils import export_chain_mass_model
    # model = export_chain_mass_model(n_mass=3)
    # print(model.p.cat)
    # test_AcadosMPC_initializes()
    # test_pi_update(generate_code=False, build_code=False)

    # for param in ["m_0", "m_1", "D_0_0"]:

    test_v_update(plot=True)
