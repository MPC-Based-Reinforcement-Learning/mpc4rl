import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3.common.buffers import ReplayBuffer
from rlmpc.gym.evaporation_process.environment import EvaporationProcessEnv  # noqa: F401
from rlmpc.mpc.evaporation_process.acados import AcadosMPC


def main():
    env = gym.make("EvaporationProcessEnv-v0")

    # H_t = np.array(
    #     [
    #         [6.96 * 1e0, -7.42 * 1e-1, 1.54 * 1e-1, -9.55 * 1e-4, 0.0],
    #         [-7.42 * 1e-1, 1.23 * 1e-1, -1.62 * 1e-2, 6.86 * 1e-5, 0.0],
    #         [1.54 * 1e-1, -1.62 * 1e-2, 7.93 * 1e-3, -2.10 * 1e-5, 0.0],
    #         [-9.55 * 1e-4, 6.86 * 1e-5, -2.10 * 1e-5, 4.53 * 1e-3, 0.0],
    #         [0.0, 0.0, 0.0, 0.0, 1.49 * 1e1],
    #     ]
    # )

    H = np.array(
        [
            [1.50256940e-02, -9.07789284e-03, 3.90839146e-04, -4.31561484e-05, 0.0],
            [-9.07789284e-03, 1.09796920e-02, -2.12424161e-04, -4.87237154e-03, 0.0],
            [3.90839146e-04, -2.12424161e-04, 9.27472116e-05, -2.38869483e-05, 0.0],
            [-4.31561484e-05, -4.87237154e-03, -2.38869483e-05, 4.54066990e-03, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.49 * 1e1],
        ]
    )

    # cost_param = {
    #     "H": {"lam": H[:2, :2], "l": H,
    # }

    # q = np.array([60.29, 0.0, 0.0, 0.0])

    # p_keys = ["H_lam", "h_lam", "c_lam", "H_Vf", "h_Vf", "c_Vf", "H_l", "h_l", "c_l", "c_f", "x_l", "x_u"]

    cost_param = {
        "H": {"lam": np.diag([1.0, 1.0]), "l": np.diag([1.0, 1.0, 1.0, 1.0]), "Vf": np.diag([1.0, 1.0])},
        "h": {
            "lam": np.array([1.0, 1.0]).reshape(-1, 1),
            "l": np.array([1.0, 1.0, 1.0, 1.0]).reshape(-1, 1),
            "Vf": np.array([1.0, 1.0]).reshape(-1, 1),
        },
        # "c": {"lam": 1.0, "l": 1.0, "Vf": 1.0, "f": 0.0},
        "c": {"lam": 1.0, "l": 1.0, "Vf": 1.0, "f": 0.0},
        "xb": {"x_l": np.array([25.0, 40.0]), "x_u": np.array([100.0, 80.0])},
    }

    model_param = env.param

    # mpc = AcadosMPC(model_param=model_param, cost_param=cost_param)
    mpc = AcadosMPC(model_param=model_param, H=H)

    obs, _ = env.reset()
    x0 = obs
    u0 = np.array([250.0, 250.0, 0.0])

    # x0 = np.array([25, 49.743], dtype=np.float32)
    # u0 = np.array([191.713, 215.888, 0.0])

    for stage in range(mpc.ocp_solver.acados_ocp.dims.N + 1):
        mpc.ocp_solver.set(stage, "x", x0)
    for stage in range(mpc.ocp_solver.acados_ocp.dims.N):
        mpc.ocp_solver.set(stage, "u", u0)

    replay_buffer = ReplayBuffer(1000, env.observation_space, env.action_space, handle_timeout_termination=False)

    Sl = []
    Su = []
    for i in range(replay_buffer.buffer_size):
        # print(f"{i}: {obs} , X_1 = {env.data['X_1']}")
        action = mpc.get_action(obs)[:2]
        # action = u0

        # mpc.update_nlp()

        # Sl.append(mpc.ocp_solver.get(0, "sl"))
        # Su.append(mpc.ocp_solver.get(0, "su"))
        # action = mpc.get_action(obs)
        next_obs, reward, done, _, info = env.step(action.astype(np.float32))
        replay_buffer.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, infos=info)
        obs = next_obs

        print(f"{i}: {obs} , X_1 = {env.data['X_1']}")

    # Sl = np.vstack(Sl)
    # Su = np.vstack(Su)
    X = np.vstack([replay_buffer.observations[i].reshape(-1) for i in range(replay_buffer.size())])
    U = np.vstack([replay_buffer.actions[i].reshape(-1) for i in range(replay_buffer.size())])

    # bounds_kwargs = {"linestyle": "--", "color": "r"}

    # h = np.hstack([cost_param["xb"]["x_l"] - X, X - cost_param["xb"]["x_u"]])

    plt.figure(1)
    for i in range(2):
        plt.subplot(3, 1, i + 1)
        plt.grid()
        plt.plot(X[:, i], label=f"x_{i}")
        # plt.plot(cost_param["xb"]["x_l"][i] * np.ones_like(X[:, i]) - Su[:, i], **bounds_kwargs, label="lower constraint")
        plt.legend()
    plt.subplot(3, 1, 3)
    plt.grid()
    plt.plot(U, label="u")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
