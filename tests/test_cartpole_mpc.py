from rlmpc.common.utils import read_config
from rlmpc.mpc.cartpole.acados import AcadosMPC
import numpy as np
import gymnasium as gym


from rlmpc.gym.continuous_cartpole.environment import ContinuousCartPoleSwingUpEnv  # noqa: F401

import os
import matplotlib.pyplot as plt


def create_mpc(config: dict) -> AcadosMPC:
    mpc = AcadosMPC(config=config, build=True)

    return mpc


def create_environment(config: dict) -> gym.Env:
    env = gym.make(
        config["environment"]["id"],
        render_mode=config["environment"]["render_mode"],
        min_action=-1.0,
        max_action=1.0,
        force_mag=config["environment"]["force_mag"],
    )

    return env


def compute_sensitivities(mpc: AcadosMPC, mpc_fun, mpc_sensitivity_fun, x0: np.ndarray, p_test: list, plot=True):
    val = {"true": np.zeros(len(p_test)), "approx": np.zeros(len(p_test))}

    for i, p_i in enumerate(p_test):
        for stage in range(mpc.ocp_solver.acados_ocp.dims.N + 1):
            mpc.set(stage, "p", p_i)

        mpc.update(x0=x0)

        val["true"][i] = mpc_fun()

        if i == 0:
            val["approx"][i] = val["true"][i]
        else:
            dp = p_test[i] - p_test[i - 1]
            val["approx"][i] = val["approx"][i - 1] + np.dot(mpc_sensitivity_fun(), dp)

    return val["approx"], val["true"]


# def compute_state_action_value_approximation(mpc: AcadosMPC, x0: np.ndarray, u0: np.ndarray, p_test: list, plot=True):
#     Q = {"true": np.zeros(len(p_test)), "approx": np.zeros(len(p_test))}


#     update_function = mpc.q_update
#     update_function_args = {"x0": x0, "u0": u0}
#     value_function = mpc.get_Q
#     value_function_sensitivity = mpc.get_dQ_dp


#     for stage in range(mpc.ocp_solver.acados_ocp.dims.N + 1):
#         mpc.set(stage, "p", p_test[0])

#     mpc.q_update(x0=x0, u0=u0)

#     Q["true"][0] = mpc.get_Q()
#     Q["approx"][0] = Q["true"][0]

#     for i in range(1, len(p_test)):
#         for stage in range(mpc.ocp_solver.acados_ocp.dims.N + 1):
#             mpc.set(stage, "p", p_test[i])

#         mpc.q_update(x0=x0, u0=u0)

#         Q["true"][i] = mpc.get_Q()
#         Q["approx"][i] = Q["approx"][i - 1] + np.dot(mpc.get_dQ_dp(), p_test[i] - p_test[i - 1])

#     return Q["approx"], Q["true"]


# def compute_state_action_value_approximation(mpc: AcadosMPC, x0: np.ndarray, u0: np.ndarray, p_test: list, plot=True):
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


def test_read_config():
    # Get the path to this file
    path = os.path.dirname(os.path.realpath(__file__))

    # Read the config file
    _ = read_config(os.path.join(path, "../config/test_AcadosMPC.yaml"))


def test_cartpole_mpc_closed_loop():
    path = os.path.dirname(os.path.realpath(__file__))
    config = read_config(os.path.join(path, "../config/test_AcadosMPC.yaml"))

    mpc = create_mpc(config=config["mpc"])

    env = create_environment(config=config)

    nstep = 500

    states = np.zeros((nstep, env.observation_space.shape[0]))
    actions = np.zeros((nstep, env.action_space.shape[0]))
    costs = np.zeros(nstep)

    states[0, :] = env.reset()[0]
    done = False
    i = 0
    while not done and i < nstep - 1:
        actions[i, :] = mpc.get_action(states[i, :])

        states[i + 1, :], costs[i], done, _, info = env.step(actions[i, :].flatten().astype(np.float32))

        i += 1

    if done:
        print("Swing-up successful!")
        status = 0
    else:
        print("Swing-up failed ...")
        status = 1

    assert status == 0


def get_test_params(p_min: np.ndarray, p_max: np.ndarray, n_test_params: int = 50):
    p_test = [p_min + (p_max - p_min) * i / n_test_params for i in range(n_test_params)]

    return p_test


def test_cartpole_mpc_sensitivities(plot: bool = False):
    path = os.path.dirname(os.path.realpath(__file__))
    config = read_config(os.path.join(path, "../config/test_AcadosMPC.yaml"))

    mpc = create_mpc(config=config["mpc"])

    x0 = np.array([0.0, 0.0, np.pi / 2, 0.0])

    p_nom = np.array([1.0, 0.1, 0.5, 9.8])

    p_min = 0.9 * p_nom
    p_max = 1.1 * p_nom

    n_test_params = 50
    p_test = [p_min + (p_max - p_min) * i / n_test_params for i in range(n_test_params)]

    L_approx, L_true = compute_sensitivities(mpc, mpc.get_L, mpc.get_dL_dp, x0, p_test)
    V_approx, V_true = compute_sensitivities(mpc, mpc.get_V, mpc.get_dV_dp, x0, p_test)

    if plot:
        _, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax[0].plot(L_true)
        ax[0].plot(L_approx, "--")

        plt.show()

    # V_approx, V_true = compute_sensitivities(mpc, mpc.get_dV_dp, x0, p_test)
    # Q_approx, Q_true = compute_sensitivities(mpc, mpc.get_dQ_dp, x0, p_test)

    assert np.allclose(L_approx, L_true, atol=1e0)
    assert np.allclose(V_approx, V_true, atol=1e0)
    # assert np.allclose(Q_approx, Q_true, atol=1e-2)


def test_mpc_state_action_value_sensitivity(plot: bool = False):
    path = os.path.dirname(os.path.realpath(__file__))
    config = read_config(os.path.join(path, "../config/test_AcadosMPC.yaml"))

    mpc = create_mpc(config=config["mpc"])

    p_min = 0.9 * np.array([1.0, 0.1, 0.5, 9.8])
    p_max = 1.1 * np.array([1.0, 0.1, 0.5, 9.8])

    Q_approx, Q_true = compute_state_action_value_approximation(
        mpc=mpc,
        p_test=get_test_params(p_min, p_max, n_test_params=50),
        update_function=mpc.q_update,
        update_function_args={"x0": np.array([0.0, 0.0, np.pi / 2, 0.0]), "u0": np.array([1.0])},
        value_function=mpc.get_Q,
        value_function_sensitivity=mpc.get_dQ_dp,
    )

    if plot:
        _, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax[0].plot(Q_true)
        ax[0].plot(Q_approx, "--")

        plt.show()

    assert np.allclose(Q_approx, Q_true, atol=1e0)


if __name__ == "__main__":
    # test_cartpole_mpc_sensitivities(plot=True)

    test_mpc_state_action_value_sensitivity(plot=True)
