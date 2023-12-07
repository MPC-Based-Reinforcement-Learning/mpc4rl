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


def test_dL_dp(mpc: CasadiMPC, p_test: np.ndarray, x0: np.ndarray = np.array([0.0, 0.0, np.pi / 2, 0.0]), plot=False):
    L = np.zeros(p_test.shape[0])
    dL_dp = np.zeros(p_test.shape[0])

    for i, p_i in enumerate(p_test):
        for stage_ in range(mpc.ocp_solver.ocp.dims.N):
            mpc.ocp_solver.set(stage_=stage_, field_="p", value_=p_i)

        _ = mpc.get_action(x0=x0)

        # if status != 0:
        #     raise Exception(f"acados acados_ocp_solver returned status {status} Exiting.")

        L[i] = mpc.compute_lagrange_function_value()
        # dL_dp[i] = lagrange_function.eval_dL_dp(acados_ocp_solver, p_i)

    dL_dp_grad = np.gradient(L, p_test[1] - p_test[0])

    dp = p_test[1] - p_test[0]

    L_reconstructed = np.cumsum(dL_dp) * dp + L[0]
    constant = L[0] - L_reconstructed[0]
    L_reconstructed += constant

    L_reconstructed_np_grad = np.cumsum(dL_dp_grad) * dp + L[0]
    constant = L[0] - L_reconstructed_np_grad[0]
    L_reconstructed_np_grad += constant

    dL_dp_cd = (L[2:] - L[:-2]) / (p_test[2:] - p_test[:-2])

    if plot:
        _, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax[0].plot(p_test, L)
        ax[0].plot(p_test, L_reconstructed, "--")
        ax[0].plot(p_test, L_reconstructed_np_grad, "-.")
        ax[1].legend(["L", "L integrate dL_dp", "L_integrate np.grad"])
        ax[1].plot(p_test, dL_dp)
        ax[1].plot(p_test, dL_dp_grad, "--")
        ax[1].plot(p_test[1:-1], dL_dp_cd, "-.")
        ax[1].legend(["algorithmic differentiation", "np.grad", "central difference"])
        ax[0].set_ylabel("L")
        ax[1].set_ylabel("dL_dp")
        ax[1].set_xlabel("p")
        ax[0].grid(True)
        ax[1].grid(True)

        plt.show()

    return int(not np.allclose(dL_dp, dL_dp_grad, rtol=1e-2, atol=1e-2))


if __name__ == "__main__":
    config = read_config("config/test_td3_CasadiMPC.yaml")

    mpc = CasadiMPC(config=Config.from_dict(config["mpc"]))

    p_test = np.linspace(0.5, 1.1, 100)

    x0 = np.array([0.0, 0.0, np.deg2rad(90), 0.0])

    L = np.zeros(p_test.shape[0])
    L_test = np.zeros(p_test.shape[0])
    dL_dp = np.zeros((p_test.shape[0], mpc.ocp_solver.ocp.dims.N))

    # Repeat dp ocp.dims.N times (one for each stage)
    dp = p_test[1] - p_test[0]
    dp = np.repeat(dp, mpc.ocp_solver.ocp.dims.N)

    for i, p_i in enumerate(p_test):
        for stage_ in range(mpc.ocp_solver.ocp.dims.N):
            mpc.ocp_solver.constraints_set(0, "lbx", x0)
            mpc.ocp_solver.constraints_set(0, "ubx", x0)
            mpc.ocp_solver.set(stage_=stage_, field_="p", value_=p_i)
            mpc.ocp_solver.set(stage_=stage_, field_="x", value_=x0)

        mpc.ocp_solver.solve()

        # TODO: check status
        # if status != 0:

        L[i] = mpc.ocp_solver.compute_lagrange_function_value()

        dL_dp[i, :] = mpc.ocp_solver.compute_lagrange_function_parametric_sensitivity()
        if i == 0:
            L_test[i] = mpc.ocp_solver.compute_lagrange_function_value()
        else:
            L_test[i] = L_test[i - 1] + np.dot(dL_dp[i, :], dp)

    _, ax = plt.subplots()
    ax.plot(p_test, L)
    ax.plot(p_test, L_test, "--")
    ax.set_ylabel("L")
    ax.grid(True)

    plt.show()
