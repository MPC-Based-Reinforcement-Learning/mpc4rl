from rlmpc.common.utils import read_config
from rlmpc.mpc.cartpole.acados import AcadosMPC
from rlmpc.common.utils import get_root_path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def compute_state_action_value_approximation(
    mpc: AcadosMPC,
    p_test: list,
    update_function,
    update_function_args: dict,
    value_function,
    value_function_sensitivity,
    plot=True,
):
    Q = {"true": np.zeros(len(p_test)), "approx": np.zeros(len(p_test))}

    # Function handlers

    # Initial update and value calculation
    for stage in range(mpc.ocp_solver.acados_ocp.dims.N + 1):
        mpc.set(stage, "p", p_test[0])

    update_function(**update_function_args)

    Q["true"][0] = value_function()
    Q["approx"][0] = Q["true"][0]

    # Loop through the rest of p_test
    for i in range(1, len(p_test)):
        for stage in range(mpc.ocp_solver.acados_ocp.dims.N + 1):
            mpc.set(stage, "p", p_test[i])

        update_function(**update_function_args)

        Q["true"][i] = value_function()
        Q["approx"][i] = Q["approx"][i - 1] + np.dot(value_function_sensitivity(), p_test[i] - p_test[i - 1])

    return Q["approx"], Q["true"]


def initialize_results_dict(p_test: list[np.ndarray]) -> dict[dict]:
    # results = {key: {"true": np.zeros(len(p_test)), "approx": np.zeros(len(p_test))} for key in ["Q", "V", "pi"]}
    # results["dQ_dp"] = {"true": np.zeros((len(p_test), len(p_test[0]))), "approx": np.zeros((len(p_test), len(p_test[0])))}
    # results["dV_dp"] = {"true": np.zeros((len(p_test), len(p_test[0]))), "approx": np.zeros((len(p_test), len(p_test[0])))}

    # # Note: the following is only for the cartpole example where u is scalar
    # results["dpi_dp"] = {"true": np.zeros((len(p_test), len(p_test[0]))), "approx": np.zeros((len(p_test), len(p_test[0])))}

    results = {
        "V": {"true": np.zeros(len(p_test)), "approx": np.zeros(len(p_test))},
        "Q": {"true": np.zeros(len(p_test)), "approx": np.zeros(len(p_test))},
        "pi": {"true": np.zeros(len(p_test)), "approx": np.zeros(len(p_test))},
        "dV_dp": {"true": np.zeros((len(p_test), len(p_test[0])))},
        "dQ_dp": {"true": np.zeros((len(p_test), len(p_test[0])))},
        "dpi_dp": {"true": np.zeros((len(p_test), len(p_test[0])))},
    }

    return results


def get_test_params(p_min: np.ndarray, p_max: np.ndarray, n_test_params: int = 50):
    return [p_min + (p_max - p_min) * i / n_test_params for i in range(n_test_params)]


def plot_results(results: dict[dict], fig, axes) -> None:
    axes[0].plot(results["V"]["true"], label="true")
    axes[0].set_title("V")
    axes[0].legend()
    axes[1].plot(results["Q"]["true"], label="true")
    axes[1].set_title("Q")
    axes[2].plot(results["pi"]["true"], label="true")
    axes[2].set_title("pi")
    axes[2].legend()

    for ax in axes:
        ax.grid(True)

    return fig, axes


def compute_value_functions_and_sensitivities(results: dict[dict], mpc: AcadosMPC, p_test: list[np.ndarray]) -> dict[dict]:
    x0 = np.array([0.0, 0.0, np.pi / 2, 0.0])
    u0 = np.array([-30.0])

    for i in tqdm(range(0, len(p_test)), desc="Compute value functions and sensitivities"):
        for stage in range(mpc.ocp_solver.acados_ocp.dims.N + 1):
            mpc.set(stage, "p", p_test[i])

        mpc.q_update(x0=x0, u0=u0)
        results["Q"]["true"][i] = mpc.get_Q()
        results["dQ_dp"]["true"][i, :] = mpc.get_dQ_dp()

        mpc.update(x0=x0)
        results["V"]["true"][i] = mpc.get_V()
        results["dV_dp"]["true"][i, :] = mpc.get_dV_dp()

        results["pi"]["true"][i] = mpc.get_pi()
        results["dpi_dp"]["true"][i] = mpc.get_dpi_dp()

    return results


def compute_approximations(results: dict[dict]) -> dict[dict]:
    for key in ["V", "Q", "pi"]:
        results[key]["approx"][0] = results[key]["true"][0]
        for i in range(1, len(results[key]["true"])):
            results[key]["approx"][i] = results[key]["approx"][i - 1] + np.dot(
                results[f"d{key}_dp"]["true"][i, :], p_test[i] - p_test[i - 1]
            )

    return results


if __name__ == "__main__":
    config = read_config(f"{get_root_path()}/config/cartpole.yaml")

    mpc = AcadosMPC(config=config["mpc"], build=True)

    N_TEST_PARAMS = [100]

    results = dict.fromkeys(N_TEST_PARAMS)

    # approximation =

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)

    p_test = get_test_params(
        p_min=1.0 * mpc.ocp_solver.acados_ocp.parameter_values,
        p_max=3.0 * mpc.ocp_solver.acados_ocp.parameter_values,
        n_test_params=100,
    )

    # Store results in a pandas dataframe
    df = pd.DataFrame(index=range(len(p_test)), columns=[f"{key}_approx::{s}" for s in [3, 2, 1] for key in ["V", "Q", "pi"]])

    df["p_test"] = p_test

    results = initialize_results_dict(p_test=p_test)

    results = compute_value_functions_and_sensitivities(results=results, mpc=mpc, p_test=p_test)

    df["V_true"] = results["V"]["true"]
    df["Q_true"] = results["Q"]["true"]
    df["pi_true"] = results["pi"]["true"]
    df["dV_dp"] = results["dV_dp"]["true"].tolist()
    df["dQ_dp"] = results["dQ_dp"]["true"].tolist()
    df["dpi_dp"] = results["dpi_dp"]["true"].tolist()

    # Compute approximations
    for key in ["V", "Q", "pi"]:
        results[key]["approx"] = dict()

    for s in [3, 2, 1]:
        for i_key, key in enumerate(["V", "Q", "pi"]):
            p = p_test[::s]

            gradient = results[f"d{key}_dp"]["true"][::s, :]

            approximation = np.zeros(len(p))
            approximation[0] = results[key]["true"][::s][0]

            for i in range(1, len(p)):
                approximation[i] = approximation[i - 1] + np.dot(gradient[i, :], p[i] - p[i - 1])

            results[key]["approx"][s] = approximation

            idx = np.linspace(0, 1, len(p))

            if s == 1:
                axes[i_key].plot(idx, results[key]["true"], label="true", color="k")
            axes[i_key].plot(idx, results[key]["approx"][s], linestyle="--", label=f"{key} ::{s}")

    axes[0].set_ylabel("V")
    axes[1].set_ylabel("Q")
    axes[2].set_ylabel("pi")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    # Save figure
    plt.savefig("cartpole_mpc_sensitivities.pdf")

    plt.show()
