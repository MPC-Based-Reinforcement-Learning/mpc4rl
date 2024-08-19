import matplotlib.pyplot as plt
import numpy as np
from rlmpc.mpc.common.mpc_acados_sensitivities import MPC as AcadosMPC
from rlmpc.mpc.chain_mass.ocp_utils import find_idx_for_labels


def set_up_test_parameters(
    mpc: AcadosMPC, np_test: int = 10, scale_low: float = 0.9, scale_high: float = 1.1, varying_param_label="A_0"
) -> np.ndarray:
    parameter_values = mpc.ocp_solver.acados_ocp.parameter_values

    test_param = np.repeat(parameter_values, np_test).reshape(len(parameter_values), -1)

    # Vary parameter along one dimension of p_label
    p_idx = find_idx_for_labels(mpc.ocp_solver.acados_ocp.model.p, varying_param_label)[0]
    test_param[p_idx, :] = np.linspace(
        scale_low * parameter_values[p_idx], scale_high * parameter_values[p_idx], np_test
    ).flatten()

    return test_param


def find_param_index_and_increment(test_param):
    parameter_increment = test_param[:, 1] - test_param[:, 0]

    # Find index of nonzero element in parameter_increment
    parameter_index = np.where(parameter_increment != 0)[0][0]

    return parameter_index, parameter_increment


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
        plt.ylabel("value")
        plt.grid()
        plt.legend()
        plt.subplot(4, 1, 2)
        plt.plot(value_gradient_finite_differences, label="value gradient via finite differences")
        plt.plot(value_gradient_acados[:, parameter_index], label="value gradient via acados")
        plt.ylabel("value gradient")
        plt.grid()
        plt.legend()
        plt.subplot(4, 1, 3)
        plt.plot(absolute_difference)
        plt.ylabel("absolute difference")
        plt.grid()
        plt.subplot(4, 1, 4)
        plt.plot(relative_difference)
        plt.ylabel("relative difference")
        plt.grid()
        plt.show()

    return absolute_difference


def run_test_v_update_for_varying_parameters(mpc: AcadosMPC, x0, test_param, plot: bool = False):
    np_test = test_param.shape[1]

    # parameter_index, _ = find_param_index_and_increment(test_param)

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


def run_test_q_update_for_varying_parameters(
    mpc: AcadosMPC, x0: np.ndarray, u0: np.ndarray, test_param: np.ndarray, plot: bool = False
):
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
                ax[0].plot(reconstructed_policy, label="reconstructed policy from policy gradients")
                ax[0].set_ylabel("policy")
                ax[0].legend()
                ax[1].plot(policy_gradient_acados, label="policy gradient via acados")
                ax[1].plot(policy_gradient_finite_differences, label="policy gradient via finite differences")
                ax[1].set_ylabel("policy gradient")
                ax[1].legend()
                ax[2].plot(absolute_difference, label="absolute difference")
                ax[2].set_ylabel("absolute difference")
                ax[3].plot(relative_difference, label="relative difference")
                ax[3].set_ylabel("relative difference")
                for j in range(4):
                    ax[j].grid()
            plt.show()
        else:
            fig, ax = plt.subplots(4, mpc.ocp_solver.acados_ocp.dims.nu, figsize=(10, 20))
            for i in range(mpc.ocp_solver.acados_ocp.dims.nu):
                ax[0, i].plot(policy[:, i], label="policy")
                ax[0, i].plot(reconstructed_policy[:, i], label="reconstructed policy from policy gradients")
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


if __name__ == "__main__":
    pass
