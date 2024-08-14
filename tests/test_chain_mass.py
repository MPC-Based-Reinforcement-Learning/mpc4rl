from rlmpc.mpc.chain_mass.acados import AcadosMPC
from rlmpc.mpc.chain_mass.ocp_utils import get_chain_params, define_x0, find_idx_for_labels, define_param_struct_symSX
import numpy as np
import os
import matplotlib.pyplot as plt


def test_AcadosMPC_initializes(
    generate_code: bool = False,
    build_code: bool = False,
    n_mass: int = 3,
    json_file_prefix: str = "acados_ocp_chain_mass_ds",
):
    kwargs, param = build_mpc_args(generate_code, build_code, n_mass, json_file_prefix)
    _ = AcadosMPC(param=param, **kwargs)
    assert True


def test_get_p(
    generate_code: bool = False,
    build_code: bool = False,
    n_mass: int = 3,
    json_file_prefix: str = "acados_ocp_chain_mass_ds",
):
    kwargs, param = build_mpc_args(generate_code, build_code, n_mass, json_file_prefix)
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

    kwargs, param = build_mpc_args(generate_code, build_code, n_mass, json_file_prefix)

    mpc = AcadosMPC(param=param, **kwargs)

    p = mpc.get_p()

    p += np.random.randn(len(p))

    mpc.set_p(p)

    assert np.allclose(mpc.get_p(), p)


def set_up_mpc(
    generate_code: bool = False, build_code: bool = False, n_mass: int = 3, json_file_prefix: str = "acados_ocp_chain_mass_ds"
):
    kwargs, param = build_mpc_args(generate_code, build_code, n_mass, json_file_prefix)
    mpc = AcadosMPC(param=param, **kwargs)
    mpc.ocp_solver.solve()
    u0 = mpc.ocp_solver.get(0, "u")

    _, x0 = define_x0(param, mpc.ocp_solver.acados_ocp)
    return mpc, x0, u0, mpc.ocp_solver.acados_ocp.parameter_values


def test_q_update(
    generate_code: bool = False,
    build_code: bool = False,
    n_mass: int = 3,
    json_file_prefix: str = "acados_ocp_chain_mass_ds",
    np_test: int = 10,
    plot: bool = False,
):
    mpc, x0, u0, p0 = set_up_mpc(generate_code, build_code, n_mass, json_file_prefix)

    test_param = set_up_test_parameters(n_mass, np_test, p0)

    absolute_difference = run_test_q_update_for_varying_parameters(mpc, x0, u0, test_param, plot)

    assert np.median(absolute_difference) <= 1e-1


def run_test_q_update_for_varying_parameters(mpc: AcadosMPC, x0, u0, test_param, plot: bool = False):
    # Evaluate q and dqdp using acados
    value = []
    value_gradient = []
    for i in range(test_param.shape[1]):
        q_i, dqdp_i = mpc.q_update(x0=x0, u0=u0, p=test_param[:, i])
        value.append(q_i)
        value_gradient.append(dqdp_i)
    value = np.array(value)
    value_gradient = np.array(value_gradient)

    # Evaluate value and value_gradient using finite differences and compare
    absolute_difference = compare_acados_value_gradients_to_finite_differences(test_param, value, value_gradient, plot=plot)

    return absolute_difference


def test_v_update(
    generate_code: bool = False,
    build_code: bool = False,
    n_mass: int = 3,
    json_file_prefix: str = "acados_ocp_chain_mass_ds",
    np_test: int = 10,
    plot: bool = False,
):
    mpc, x0, _, p0 = set_up_mpc(generate_code, build_code, n_mass, json_file_prefix)

    test_param = set_up_test_parameters(n_mass, np_test, p0)

    absolute_difference = run_test_v_update_for_varying_parameters(mpc, x0, test_param, plot)

    assert np.median(absolute_difference) <= 1e-1


def run_test_v_update_for_varying_parameters(mpc: AcadosMPC, x0, test_param, plot: bool = False):
    np_test = test_param.shape[1]

    parameter_index, _ = find_param_index_and_increment(test_param)

    # Evaluate value and value_gradient using acados
    value = []
    value_gradient = []
    for i in range(np_test):
        v_i, dvdp_i = mpc.v_update(x0=x0, p=test_param[:, i])
        value.append(v_i)
        value_gradient.append(dvdp_i)
    value = np.array(value)
    value_gradient = np.array(value_gradient)

    # Evaluate v and dvdp using finite differences and compare
    absolute_difference = compare_acados_value_gradients_to_finite_differences(test_param, value, value_gradient, plot=plot)

    return absolute_difference


def test_pi_update(
    generate_code: bool = False,
    build_code: bool = False,
    n_mass: int = 3,
    json_file_prefix: str = "acados_ocp_chain_mass_ds",
    np_test: int = 10,
    plot: bool = False,
):
    mpc, x0, _, p0 = set_up_mpc(generate_code, build_code, n_mass, json_file_prefix)

    test_param = set_up_test_parameters(n_mass, np_test, p0)

    absolute_difference = run_test_pi_update_for_varying_parameters(mpc, x0, test_param, plot)

    assert np.median(absolute_difference) <= 1e-1


def find_param_index_and_increment(test_param):
    parameter_increment = test_param[:, 1] - test_param[:, 0]

    # Find index of nonzero element in parameter_increment
    parameter_index = np.where(parameter_increment != 0)[0][0]

    return parameter_index, parameter_increment


def run_test_pi_update_for_varying_parameters(mpc: AcadosMPC, x0, test_param, plot: bool = False):
    # Evaluate v and dvdp using acados
    np_test = test_param.shape[1]

    parameter_index, parameter_increment = find_param_index_and_increment(test_param)

    policy = []
    policy_gradient = []
    for i in range(np_test):
        pi_i, dpidp_i = mpc.pi_update(x0=x0, p=test_param[:, i])
        policy.append(pi_i)
        policy_gradient.append(dpidp_i)

    policy = np.array(policy)
    policy_gradient = np.array(policy_gradient)

    policy_gradient_acados = policy_gradient[:, :, parameter_index]

    # Evaluate pi and dpidp using finite differences and compare
    # Assumes a constant parameter increment
    dp = parameter_increment[parameter_index]
    policy_gradient_finite_differences = np.gradient(policy, dp, axis=0)

    absolute_difference = np.abs(policy_gradient_finite_differences - policy_gradient_acados)

    if plot:
        reconstructed_policy = np.cumsum(policy_gradient_acados * dp, axis=0)
        reconstructed_policy += policy[0] - reconstructed_policy[0]

        # Avoid division by zero when policy_gradient_acados is zero
        relative_difference = np.zeros_like(absolute_difference)
        for i in range(np_test):
            for j in range(mpc.ocp_solver.acados_ocp.dims.nu):
                if np.abs(policy_gradient_acados[i, j]) > 1e-10:
                    relative_difference[i, j] = absolute_difference[i, j] / np.abs(policy_gradient_acados[i, j])

        if mpc.ocp_solver.acados_ocp.dims.nu == 1:
            fig, ax = plt.subplots(4, mpc.ocp_solver.acados_ocp.dims.nu, figsize=(10, 20))
            for i in range(mpc.ocp_solver.acados_ocp.dims.nu):
                ax[0].plot(policy, label="policy")
                ax[0].plot(reconstructed_policy, label="reconstructed policy")
                ax[1].plot(policy_gradient_acados, label="policy gradient via acados")
                ax[1].plot(policy_gradient_finite_differences, label="policy gradient via finite differences")
                ax[2].plot(absolute_difference, label="absolute difference")
                ax[3].plot(relative_difference, label="relative difference")
                for j in range(4):
                    ax[j].legend()
                    ax[j].grid()
            plt.legend()
            plt.show()
        else:
            fig, ax = plt.subplots(4, mpc.ocp_solver.acados_ocp.dims.nu, figsize=(10, 20))
            for i in range(mpc.ocp_solver.acados_ocp.dims.nu):
                ax[0, i].plot(policy[:, i], label="policy")
                ax[0, i].plot(reconstructed_policy[:, i], label="reconstructed policy")
                ax[1, i].plot(policy_gradient_acados[:, i], label="policy gradient via acados")
                ax[1, i].plot(policy_gradient_finite_differences[:, i], label="policy gradient via finite differences")
                ax[2, i].plot(absolute_difference[:, i], label="absolute difference")
                ax[3, i].plot(relative_difference[:, i], label="relative difference")
                for j in range(4):
                    ax[j, i].legend()
                    ax[j, i].grid()
            plt.legend()
            plt.show()

    return absolute_difference


def compare_acados_value_gradients_to_finite_differences(test_param, values, value_gradient_acados, plot: bool = True):
    # Assumes a constant parameter increment
    parameter_index, parameter_increment = find_param_index_and_increment(test_param)

    value_gradient_finite_differences = np.gradient(values, parameter_increment[parameter_index])

    absolute_difference = np.abs(
        np.gradient(values, parameter_increment[parameter_index]) - value_gradient_acados[:, parameter_index]
    )

    if plot:
        reconstructed_values = np.cumsum(value_gradient_acados @ parameter_increment)
        reconstructed_values += values[0] - reconstructed_values[0]

        relative_difference = absolute_difference / np.abs(value_gradient_acados[:, parameter_index])

        plt.figure()
        plt.subplot(4, 1, 1)
        plt.plot(values, label="original values")
        plt.plot(reconstructed_values, label="reconstructed values")
        plt.subplot(4, 1, 2)
        plt.plot(value_gradient_finite_differences, label="value gradient via finite differences")
        plt.plot(value_gradient_acados[:, parameter_index], label="value gradient via acados")
        plt.subplot(4, 1, 3)
        plt.plot(absolute_difference, label="absolute difference")
        plt.subplot(4, 1, 4)
        plt.plot(relative_difference, label="relative difference")
        plt.legend()
        plt.show()

    return absolute_difference


def set_up_test_parameters(n_mass, np_test, p0):
    parameter_values = p0

    test_param = np.repeat(parameter_values, np_test).reshape(len(parameter_values), -1)
    # Vary parameter along one dimension of p_label
    p_label = f"C_{n_mass - 2}_0"
    p_sym = define_param_struct_symSX(n_mass, disturbance=True)
    p_idx = find_idx_for_labels(p_sym.cat, p_label)[0]
    test_param[p_idx, :] = np.linspace(0.5 * parameter_values[p_idx], 1.5 * parameter_values[p_idx], np_test)
    return test_param


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

    param = get_chain_params()
    param["n_mass"] = n_mass
    return kwargs, param


if __name__ == "__main__":
    # test_AcadosMPC_initializes()
    # test_pi_update(generate_code=False, build_code=False)
    test_q_update(generate_code=False, build_code=False, plot=True)
