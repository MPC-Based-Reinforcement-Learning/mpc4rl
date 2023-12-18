from external.stable_baselines3.stable_baselines3.td3 import TD3
from rlmpc.td3.policies import MPCTD3Policy

import gymnasium as gym
from rlmpc.common.utils import read_config

from rlmpc.gym.continuous_cartpole.environment import (
    ContinuousCartPoleBalanceEnv,
    ContinuousCartPoleSwingUpEnv,
)

from rlmpc.mpc.cartpole.casadi import CasadiMPC, Config

import gymnasium as gym
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import (
    NormalActionNoise,
)

import matplotlib.pyplot as plt

from tqdm import tqdm


def test_dL_dp(mpc: CasadiMPC, p_test: np.ndarray, x0: np.ndarray, plot_: bool = False):
    # Repeat dp ocp.dims.N times (one for each stage)
    # TODO: This is not general. Works only for scalar parameters
    dp = np.repeat(p_test[1] - p_test[0], mpc.q_ocp_solver.ocp.dims.N)

    #### Test dL_dp

    L = {"true": np.zeros(p_test.shape[0]), "approx": np.zeros(p_test.shape[0])}
    V = {"true": np.zeros(p_test.shape[0]), "approx": np.zeros(p_test.shape[0])}
    Q = {"true": np.zeros(p_test.shape[0]), "approx": np.zeros(p_test.shape[0])}
    pi = {"true": np.zeros(p_test.shape[0]), "approx": np.zeros(p_test.shape[0])}

    a0 = 0.1

    for i, p_i in tqdm(enumerate(p_test), total=len(p_test)):
        for stage_ in range(mpc.v_ocp_solver.ocp.dims.N):
            mpc.q_ocp_solver.set(stage_=stage_, field_="p", value_=p_i)
            mpc.q_ocp_solver.set(stage_=stage_, field_="x", value_=x0)
            mpc.v_ocp_solver.set(stage_=stage_, field_="p", value_=p_i)
            mpc.v_ocp_solver.set(stage_=stage_, field_="x", value_=x0)

        # TODO: check status
        # if status != 0:

        V["true"][i] = mpc.v_ocp_solver.compute_state_value_function_value(state=x0)

        L["true"][i] = mpc.v_ocp_solver.compute_lagrange_function_value()

        # Q["true"][i] = mpc.q_ocp_solver.compute_state_action_value_function_value(state=x0, action=a0)

        # pi["true"][i] = mpc.get_action(x0=x0, solve=False)

        if i == 0:
            # L["approx"][i] = mpc.v_ocp_solver.compute_lagrange_function_value()
            # V["approx"][i] = mpc.v_ocp_solver.compute_state_value_function_value(state=x0)
            # Q["approx"][i] = mpc.q_ocp_solver.compute_state_action_value_function_value(state=x0, action=a0)
            # pi["approx"][i] = mpc.get_action(x0=x0, solve=False)
            L["approx"][i] = L["true"][i]
            V["approx"][i] = V["true"][i]
            # Q["approx"][i] = Q["true"][i]
            pi["approx"][i] = pi["true"][i]
        else:
            L["approx"][i] = L["approx"][i - 1] + np.dot(
                mpc.v_ocp_solver.compute_lagrange_function_parametric_sensitivity(), dp
            )
            V["approx"][i] = V["approx"][i - 1] + np.dot(
                mpc.v_ocp_solver.compute_state_value_function_parametric_sensitivity(), dp
            )
            # Q["approx"][i] = Q["approx"][i - 1] + np.dot(
            #     mpc.q_ocp_solver.compute_state_action_value_function_parametric_sensitivity(), dp
            # )
            pi["approx"][i] = pi["approx"][i - 1] + np.dot(mpc.v_ocp_solver.compute_policy_parametric_sensitivity(), dp)

    if plot_:
        _, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
        ax[0].plot(p_test, L["true"])
        ax[0].plot(p_test, L["approx"], "--")
        ax[0].set_ylabel("L_V")
        ax[0].grid(True)

        ax[1].plot(p_test, V["true"])
        ax[1].plot(p_test, V["approx"], "--")
        ax[1].set_ylabel("V")
        ax[1].grid(True)

        ax[2].plot(p_test, Q["true"])
        ax[2].plot(p_test, Q["approx"], "--")
        ax[2].set_ylabel("Q")
        ax[2].grid(True)

        ax[-1].set_xlabel("p")

        #### Test dV_dp
        plt.show()

    return np.allclose(L["true"], L["approx"], atol=1e-2)


if __name__ == "__main__":
    config = read_config("config/test_td3_CasadiMPC.yaml")

    mpc = CasadiMPC(config=Config.from_dict(config["mpc"]), build=False)

    p_test = np.linspace(0.9, 1.1, 100)

    x0 = np.array([0.0, 0.0, np.deg2rad(90), 0.0])

    print(f"Test dL_dP: {test_dL_dp(mpc, p_test, x0, plot_=True)}")
