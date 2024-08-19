import numpy as np
import matplotlib.pyplot as plt

import os

from rlmpc.common.utils import get_root_path
from rlmpc.mpc.chain_mass.ocp_utils import (
    get_chain_params,
    find_idx_for_labels,
    define_param_struct_symSX,
    define_x0,
    define_M,
)
from rlmpc.mpc.chain_mass.acados import AcadosMPC


def plot_results(p_label, p_var, sens_u, u_opt):
    u_opt = np.vstack(u_opt)
    sens_u = np.vstack(sens_u)

    delta_p = p_var[1] - p_var[0]

    # Compare to numerical gradients
    sens_u_fd = np.gradient(u_opt, p_var, axis=0)
    u_opt_reconstructed_fd = np.cumsum(sens_u_fd, axis=0) * delta_p + u_opt[0, :]
    u_opt_reconstructed_fd += u_opt[0, :] - u_opt_reconstructed_fd[0, :]

    u_opt_reconstructed_acados = np.cumsum(sens_u, axis=0) * delta_p + u_opt[0, :]
    u_opt_reconstructed_acados += u_opt[0, :] - u_opt_reconstructed_acados[0, :]

    plt.figure(figsize=(7, 10))
    for col in range(3):
        plt.subplot(4, 1, col + 1)
        plt.plot(p_var, u_opt[:, col], label=f"$u^*_{col}$")
        plt.plot(p_var, u_opt_reconstructed_fd[:, col], label=f"$u^*_{col}$, reconstructed with fd gradients", linestyle="--")
        plt.plot(
            p_var, u_opt_reconstructed_acados[:, col], label=f"$u^*_{col}$, reconstructed with acados gradients", linestyle=":"
        )
        plt.ylabel(f"$u^*_{col}$")
        plt.grid(True)
        plt.legend()
        plt.xlim(p_var[0], p_var[-1])

    for col in range(3):
        plt.subplot(4, 1, 4)
        plt.plot(p_var, np.abs(sens_u[:, col] - sens_u_fd[:, col]), label=f"$u^*_{col}$", linestyle="--")

    plt.ylabel("abs difference")
    plt.grid(True)
    plt.legend()
    plt.yscale("log")
    plt.xlabel(p_label)
    plt.xlim(p_var[0], p_var[-1])


def main_acados(
    chain_params_: dict = get_chain_params(),
    generate_code: bool = True,
    build_code: bool = True,
    np_test: int = 100,
    plot: bool = True,
) -> None:
    kwargs = {
        "ocp_solver": {
            "json_file": f"acados_ocp_chain_mass_ds_{chain_params_['n_mass']}.json",
            "generate": generate_code,
            "build": build_code,
        },
        "ocp_sensitivity_solver": {
            "json_file": f"acados_ocp_chain_mass_ds_{chain_params_['n_mass']}_sensitivity.json",
            "generate": generate_code,
            "build": build_code,
        },
    }

    mpc = AcadosMPC(param=chain_params_, **kwargs)

    parameter_values = mpc.ocp_solver.acados_ocp.parameter_values

    x0 = define_x0(chain_params_, mpc.ocp_solver.acados_ocp)

    M = define_M(chain_params_)

    p_label = f"C_{M}_0"
    # p_label = "Q_0"

    # p_sym = define_param_struct_symSX(chain_params_["n_mass"], disturbance=True)

    p_idx = find_idx_for_labels(define_param_struct_symSX(chain_params_["n_mass"], disturbance=True).cat, p_label)[0]

    p_var = np.linspace(0.5 * parameter_values[p_idx], 1.5 * parameter_values[p_idx], np_test)

    sens_u = []
    u_opt = []

    for i in range(np_test):
        parameter_values[p_idx] = p_var[i]

        for stage in range(mpc.ocp_solver.acados_ocp.dims.N + 1):
            mpc.ocp_solver.set(stage, "p", parameter_values)
            mpc.ocp_sensitivity_solver.set(stage, "p", parameter_values)

        u_opt.append(mpc.ocp_solver.solve_for_x0(x0))
        print(f"ocp_solver status {mpc.ocp_solver.status}")

        mpc.ocp_solver.store_iterate(filename="iterate.json", overwrite=True, verbose=False)
        mpc.ocp_sensitivity_solver.load_iterate(filename="iterate.json", verbose=False)
        mpc.ocp_sensitivity_solver.solve_for_x0(x0, fail_on_nonzero_status=False, print_stats_on_failure=False)

        print(f"sensitivity_solver status {mpc.ocp_sensitivity_solver.status}")

        # Calculate the policy gradient
        _, sens_u_ = mpc.ocp_sensitivity_solver.eval_solution_sensitivity(0, "params_global")

        mpc.ocp_sensitivity_solver.eval_and_get_optimal_value_gradient("params_global")

        sens_u.append(sens_u_[:, p_idx])

    if plot:
        plot_results(p_label, p_var, sens_u, u_opt)


def main_nlp(chain_params_=get_chain_params(), np_test: int = 100, plot: bool = True):
    discount_factor = 1.0

    fig_path = os.path.join(get_root_path(), "scripts", "figures")
    os.makedirs(fig_path, exist_ok=True)

    mpc = AcadosMPC(chain_params_, discount_factor)

    x0 = define_x0(chain_params_, mpc.ocp_solver.acados_ocp)
    M = define_M(chain_params_)

    p_label = f"C_{M}_0"
    p_idx = find_idx_for_labels(
        define_param_struct_symSX(chain_params_["n_mass"], disturbance=True).cat,
        p_label,
    )[0]
    p_nom = mpc.nlp.p.val.cat.full().flatten()

    p_test = []
    p_var = np.linspace(0.5 * p_nom[p_idx], 1.5 * p_nom[p_idx], np_test)
    for i in range(np_test):
        p_test.append(p_nom.copy())
        p_test[-1][p_idx] = p_var[i]

    u_opt = []
    sens_u = []
    for i in range(np_test):
        print(f"Test {i}/{np_test}")

        mpc.set_p(p_test[i])

        _ = mpc.update(x0)

        mpc.update_nlp()

        u_opt.append(mpc.get_pi())

        sens_u.append(mpc.get_dpi_dp()[:, p_idx].flatten())

    u_opt = np.vstack(u_opt)
    sens_u = np.vstack(sens_u)

    if plot:
        plot_results(p_label, p_var, sens_u, u_opt)


if __name__ == "__main__":
    params = get_chain_params()

    for n_mass in [3]:
        params["n_mass"] = n_mass
        # main_nlp(chain_params_=params, np_test=20)
        main_acados(chain_params_=params, np_test=20, generate_code=False)
        plt.show()
