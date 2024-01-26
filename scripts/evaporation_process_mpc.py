import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3.common.buffers import ReplayBuffer
from rlmpc.gym.evaporation_process.environment import EvaporationProcessEnv  # noqa: F401
from rlmpc.mpc.evaporation_process.acados import AcadosMPC


def main():
    env = gym.make("EvaporationProcessEnv-v0")

    # p_keys = ["H_lam", "h_lam", "c_lam", "H_Vf", "h_Vf", "c_Vf", "H_l", "h_l", "c_l", "c_f", "x_l", "x_u"]

    cost_param = {
        "H": {"lam": np.diag([1.0, 1.0]), "l": np.diag([1.0, 1.0, 1.0, 1.0]), "Vf": np.diag([1.0, 1.0])},
        "h": {
            "lam": np.array([1.0, 1.0]).reshape(-1, 1),
            "l": np.array([1.0, 1.0, 1.0, 1.0]).reshape(-1, 1),
            "Vf": np.array([1.0, 1.0]).reshape(-1, 1),
        },
        "c": {"lam": 1.0, "l": 1.0, "Vf": 1.0, "f": 0.0},
        "xb": {"x_l": np.array([25.0, 40.0]), "x_u": np.array([100.0, 80.0])},
    }

    model_param = {
        "a": 0.5616,
        "b": 0.3126,
        "c": 48.43,
        "d": 0.507,
        "e": 55.0,
        "f": 0.1538,
        "g": 55.0,
        "h": 0.16,
        "M": 20.0,
        "C": 4.0,
        "U_A2": 6.84,
        "C_p": 0.07,
        "lam": 38.5,
        "lam_s": 36.6,
        "F_1": 10.0,
        "X_1": 0.05,
        "F_3": 50.0,
        "T_1": 40.0,
        "T_200": 25.0,
    }

    mpc = AcadosMPC(model_param=model_param, cost_param=cost_param)

    obs, _ = env.reset()
    x0 = obs
    u0 = np.array([250.0, 250.0])

    for stage in range(mpc.ocp_solver.acados_ocp.dims.N + 1):
        mpc.ocp_solver.set(stage, "x", x0)
    for stage in range(mpc.ocp_solver.acados_ocp.dims.N):
        mpc.ocp_solver.set(stage, "u", u0)

    replay_buffer = ReplayBuffer(1000, env.observation_space, env.action_space, handle_timeout_termination=False)

    Sl = []
    Su = []
    for _ in range(replay_buffer.buffer_size):
        action = mpc.ocp_solver.solve_for_x0(obs)

        Sl.append(mpc.ocp_solver.get(0, "sl"))
        Su.append(mpc.ocp_solver.get(0, "su"))
        # action = mpc.get_action(obs)
        next_obs, reward, done, _, info = env.step(action.astype(np.float32))
        replay_buffer.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, infos=info)
        obs = next_obs

    Sl = np.vstack(Sl)
    Su = np.vstack(Su)
    X = np.vstack([replay_buffer.observations[i].reshape(-1) for i in range(replay_buffer.size())])
    U = np.vstack([replay_buffer.actions[i].reshape(-1) for i in range(replay_buffer.size())])

    bounds_kwargs = {"linestyle": "--", "color": "r"}

    h = np.hstack([cost_param["xb"]["x_l"] - X, X - cost_param["xb"]["x_u"]])

    plt.figure(1)
    for i in range(2):
        plt.subplot(3, 1, i + 1)
        plt.grid()
        plt.plot(X[:, i], label=f"x_{i}")
        plt.plot(cost_param["xb"]["x_l"][i] * np.ones_like(X[:, i]) - Su[:, i], **bounds_kwargs, label="lower constraint")
        plt.legend()
    plt.subplot(3, 1, 3)
    plt.grid()
    plt.plot(U, label="u")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
