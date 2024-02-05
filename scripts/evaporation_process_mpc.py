import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3.common.buffers import ReplayBuffer
from rlmpc.gym.evaporation_process.environment import EvaporationProcessEnv  # noqa: F401
from rlmpc.mpc.evaporation_process.acados import AcadosMPC


def cost_param_from_H(H: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    return {
        "H": {"lam": H[:2, :2], "l": H, "Vf": H[:2, :2]},
        "h": {
            "lam": np.zeros((2, 1)).reshape(-1, 1),
            "l": np.zeros((5, 1)).reshape(-1, 1),
            "Vf": np.zeros((2, 1)).reshape(-1, 1),
        },
        "c": {"lam": 0.0, "l": 0.0, "Vf": 0.0, "f": 0.0},
        "xb": {"x_l": np.array([25.0, 40.0]), "x_u": np.array([100.0, 80.0])},
    }


H_tuned = np.array(
    [
        [1.50256940e-02, -9.07789284e-03, 3.90839146e-04, -4.31561484e-05, 0.0],
        [-9.07789284e-03, 1.09796920e-02, -2.12424161e-04, -4.87237154e-03, 0.0],
        [3.90839146e-04, -2.12424161e-04, 9.27472116e-05, -2.38869483e-05, 0.0],
        [-4.31561484e-05, -4.87237154e-03, -2.38869483e-05, 4.54066990e-03, 0.0],
        # [0.0, 0.0, 0.0, 0.0, 1.49 * 1e1],
        [0.0, 0.0, 0.0, 0.0, 1.0 * 1e0],
    ]
)

H_nominal = np.diag([10.0, 10.0, 0.1, 0.1, 0.1])


def main():
    env = gym.make("EvaporationProcessEnv-v0")
    model_param = env.param

    H = H_tuned
    H[0, 0] = 0.9 * H[0, 0]

    cost_param = cost_param_from_H(H)

    gamma = 0.95

    mpc = AcadosMPC(model_param=model_param, cost_param=cost_param, gamma=gamma)
    # mpc = AcadosMPC(model_param=model_param, H=H)

    obs, _ = env.reset()
    x0 = obs
    u0 = np.array([250.0, 250.0, 0.0])

    # x0 = np.array([25, 49.743], dtype=np.float32)
    x0 = np.array([30, 60.0], dtype=np.float32)

    # env.set_state(x0)
    # obs = x0
    # u0 = np.array([191.713, 215.888, 0.0])

    for stage in range(mpc.ocp_solver.acados_ocp.dims.N + 1):
        mpc.ocp_solver.set(stage, "x", x0)
        mpc.nlp.set(stage, "x", x0)
    for stage in range(mpc.ocp_solver.acados_ocp.dims.N):
        mpc.ocp_solver.set(stage, "u", u0)
        mpc.nlp.set(stage, "u", u0)

    if False:
        W = mpc.nlp.get_parameter("W")
        W[0, 0] = 1.0
        mpc.nlp.set_parameter("W", W)

        W_0 = mpc.nlp.get_parameter("W_0")
        W_0[0, 0] = 1.0
        mpc.nlp.set_parameter("W_0", W_0)

        mpc.set_parameter(mpc.nlp.p.val.cat.full().flatten())

    mpc.update(x0)

    cost_diff = mpc.nlp.cost.val.full()[0][0] - mpc.ocp_solver.get_cost()
    print(f"nlp_cost - ocp_cost: {cost_diff}")

    exit(0)

    # action = mpc.get_action(x0)

    # mpc.update_nlp()

    # exit(0)

    replay_buffer = ReplayBuffer(100, env.observation_space, env.action_space, handle_timeout_termination=False)

    for i in range(replay_buffer.buffer_size):
        action = mpc.get_action(obs)

        mpc.update_nlp()

        next_obs, reward, done, _, info = env.step(action.astype(np.float32))
        replay_buffer.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, infos=info)
        obs = next_obs

    X = np.vstack([replay_buffer.observations[i].reshape(-1) for i in range(replay_buffer.size())])
    U = np.vstack([replay_buffer.actions[i].reshape(-1) for i in range(replay_buffer.size())])
    L = np.vstack([replay_buffer.rewards[i].reshape(-1) for i in range(replay_buffer.size())])

    # bounds_kwargs = {"linestyle": "--", "color": "r"}

    # h = np.hstack([cost_param["xb"]["x_l"] - X, X - cost_param["xb"]["x_u"]])

    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.grid()
    plt.plot(X)
    # plt.plot(cost_param["xb"]["x_l"][i] * np.ones_like(X[:, i]) - Su[:, i], **bounds_kwargs, label="lower constraint")
    plt.subplot(3, 1, 2)
    plt.grid()
    plt.plot(U, label="u")
    plt.subplot(3, 1, 3)
    plt.grid()
    plt.plot(L)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
