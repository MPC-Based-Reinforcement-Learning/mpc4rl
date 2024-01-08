from rlmpc.common.utils import read_config
from rlmpc.mpc.cartpole.acados import AcadosMPC
from rlmpc.common.utils import get_root_path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


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
    results = {key: {"true": np.zeros(len(p_test)), "approx": np.zeros(len(p_test))} for key in ["Q", "V", "pi"]}
    results["dQ_dp"] = {"true": np.zeros((len(p_test), len(p_test[0]))), "approx": np.zeros((len(p_test), len(p_test[0])))}
    results["dV_dp"] = {"true": np.zeros((len(p_test), len(p_test[0]))), "approx": np.zeros((len(p_test), len(p_test[0])))}

    # Note: the following is only for the cartpole example where u is scalar
    results["dpi_dp"] = {"true": np.zeros((len(p_test), len(p_test[0]))), "approx": np.zeros((len(p_test), len(p_test[0])))}

    return results


def get_test_params(p_min: np.ndarray, p_max: np.ndarray, n_test_params: int = 50):
    return [p_min + (p_max - p_min) * i / n_test_params for i in range(n_test_params)]


def plot_results(results: dict[dict]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].plot(results["V"]["true"], label="true")
    axes[0].plot(results["V"]["approx"], linestyle="--", label="approx")
    axes[0].set_title("V")
    axes[0].legend()
    axes[1].plot(results["Q"]["true"], label="true")
    axes[1].plot(results["Q"]["approx"], linestyle="--", label="approx")
    axes[1].set_title("Q")
    axes[1].legend()

    for ax in axes:
        ax.grid(True)

    plt.show()


def compute_value_functions_and_sensitivities(mpc: AcadosMPC, p_test: list[np.ndarray]) -> dict[dict]:
    results = initialize_results_dict(p_test=p_test)

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

    # Approximate value functions and sensitivities with the computed sensitivities
    results["V"]["approx"][0] = results["V"]["true"][0]
    results["Q"]["approx"][0] = results["Q"]["true"][0]

    for i in range(1, len(p_test)):
        results["V"]["approx"][i] = results["V"]["approx"][i - 1] + np.dot(
            results["dV_dp"]["true"][i, :], p_test[i] - p_test[i - 1]
        )
        results["Q"]["approx"][i] = results["Q"]["approx"][i - 1] + np.dot(
            results["dQ_dp"]["true"][i, :], p_test[i] - p_test[i - 1]
        )

    return results


if __name__ == "__main__":
    config = read_config(f"{get_root_path()}/config/cartpole.yaml")

    mpc = AcadosMPC(config=config["mpc"], build=True)

    p_test = get_test_params(
        p_min=1.0 * mpc.ocp_solver.acados_ocp.parameter_values,
        p_max=5.0 * mpc.ocp_solver.acados_ocp.parameter_values,
        n_test_params=100,
    )

    results = compute_value_functions_and_sensitivities(mpc=mpc, p_test=p_test)

    plot_results(results=results)
