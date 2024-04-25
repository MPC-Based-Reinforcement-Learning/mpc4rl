import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os

from rlmpc.common.utils import get_root_path
from rlmpc.mpc.chain_mass.ocp_utils import get_chain_params, find_idx_for_labels, define_param_struct_symSX
from rlmpc.mpc.chain_mass.acados import AcadosMPC


def main(params_=get_chain_params()):
    discount_factor = 1.0

    fig_path = os.path.join(get_root_path(), "scripts", "figures")
    os.makedirs(fig_path, exist_ok=True)

    mpc = AcadosMPC(params_, discount_factor)

    x0 = mpc.ocp_solver.acados_ocp.constraints.lbx_0

    p_label = f"m_{params_['n_mass']-3}"
    p_idx = find_idx_for_labels(
        define_param_struct_symSX(params_["n_mass"], disturbance=True).cat,
        p_label,
    )[0]
    p_nom = mpc.nlp.p.val.cat.full().flatten()

    timings = {key: [] for key in ["solve_ocp_solver", "dL_dp", "lin_params", "solve_params"]}

    np_test = 100
    p_test = []
    p_var = np.linspace(0.5 * p_nom[p_idx], 1.5 * p_nom[p_idx], np_test)
    for i in range(np_test):
        p_test.append(p_nom.copy())
        p_test[-1][p_idx] = p_var[i]

    pi = []
    dpi_dp = []
    for i in range(np_test):
        print(f"Test {i}/{np_test}")

        mpc.set_p(p_test[i])

        _ = mpc.update(x0)

        timings["solve_ocp_solver"].append(mpc.ocp_solver.get_stats("time_tot"))

        mpc.update_nlp()

        for key in ["dL_dp", "lin_params", "solve_params"]:
            timings[key].append(mpc.nlp_timing[key])

        pi.append(mpc.get_pi())

        dpi_dp.append(mpc.get_dpi_dp()[:, p_idx].flatten())

    # Make a pandas dataframe with the timings
    timings_df = pd.DataFrame(timings)

    # Save the dataframe to a csv file
    timings_df.to_csv(f"{fig_path}/chain_mass_{params_['n_mass']}_timings_splinalg.csv")

    pi = np.vstack(pi)
    dpi_dp = np.vstack(dpi_dp)

    # Compare to numerical gradients
    delta_p = p_var[1] - p_var[0]

    np_grad = np.gradient(pi, p_var, axis=0)
    pi_reconstructed_np_grad = np.cumsum(np_grad, axis=0) * delta_p + pi[0, :]
    pi_reconstructed_np_grad += pi[0, :] - pi_reconstructed_np_grad[0, :]

    pi_reconstructed_acados = np.cumsum(dpi_dp, axis=0) * delta_p + pi[0, :]
    pi_reconstructed_acados += pi[0, :] - pi_reconstructed_acados[0, :]

    plt.figure()
    for col in range(3):
        plt.subplot(4, 1, col + 1)
        plt.plot(p_var, pi[:, col], label=f"pi_{col}")
        plt.plot(p_var, pi_reconstructed_np_grad[:, col], label=f"pi_reconstructed_np_grad_{col}", linestyle="--")
        plt.plot(p_var, pi_reconstructed_acados[:, col], label=f"pi_reconstructed_acados_{col}", linestyle=":")
        plt.ylabel(f"pi_{col}")
        plt.grid(True)
        plt.legend()

    for col in range(3):
        plt.subplot(4, 1, 4)
        plt.plot(p_var, np.abs(dpi_dp[:, col] - np_grad[:, col]), label=f"pi_{col}", linestyle="--")

    plt.ylabel("abs difference")
    plt.grid(True)
    plt.legend()
    plt.yscale("log")
    plt.xlabel("p")

    # Save the figure
    plt.savefig(os.path.join(fig_path, f"chain_mass_{params_['n_mass']}_sensitivity"))
    plt.show()


if __name__ == "__main__":
    params = get_chain_params()
    for n_mass in [3, 4, 5, 6]:
        params["n_mass"] = n_mass
        main(params_=params)
