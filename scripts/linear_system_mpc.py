import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


from rlmpc.gym.linear_system.environment import LinearSystemEnv  # noqa: F401
from rlmpc.mpc.linear_system.acados import AcadosMPC
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import tqdm


def test_acados_ocp_solver(param: dict) -> None:
    ocp_solver = AcadosMPC(param).ocp_solver

    x0 = np.array([[0.5], [0.5]])

    for stage in range(ocp_solver.acados_ocp.dims.N + 1):
        ocp_solver.set(stage, "p", ocp_solver.acados_ocp.parameter_values)

    ocp_solver.solve_for_x0(x0)

    X = np.vstack([ocp_solver.get(stage, "x") for stage in range(ocp_solver.acados_ocp.dims.N + 1)])
    U = np.vstack([ocp_solver.get(stage, "u") for stage in range(ocp_solver.acados_ocp.dims.N)])

    plt.figure(1)
    plt.subplot(211)
    plt.grid()
    plt.plot(X)
    plt.subplot(212)
    plt.stairs(edges=np.arange(ocp_solver.acados_ocp.dims.N + 1), values=U[:, 0])
    plt.grid()
    plt.show()


def test_acados_mpc(param):
    env = gym.make("LinearSystemEnv-v0", lb_noise=-0.1, ub_noise=0.1)

    mpc = AcadosMPC(param)

    replay_buffer = ReplayBuffer(
        buffer_size=int(1e2),
        observation_space=env.observation_space,
        action_space=env.action_space,
        handle_timeout_termination=False,
    )

    obs, _ = env.reset()
    for _ in tqdm(range(replay_buffer.buffer_size), desc="Filling replay buffer"):
        action = mpc.get_action(obs)
        mpc.update_nlp()
        next_obs, reward, done, _, info = env.step(action.astype(np.float32))
        replay_buffer.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, infos=info)
        obs = next_obs

    plot_test_acados_mpc(replay_buffer, mpc)


def plot_test_acados_mpc(replay_buffer: ReplayBuffer, mpc: AcadosMPC):
    X = np.vstack([replay_buffer.observations[i].reshape(-1) for i in range(replay_buffer.size())])
    U = np.vstack([replay_buffer.actions[i].reshape(-1) for i in range(replay_buffer.size())])

    bounds_kwargs = {"linestyle": "-", "color": "r"}

    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.grid()
    plt.plot(X[:, 0])
    plt.plot(mpc.ocp_solver.acados_ocp.constraints.lbx[0] * np.ones_like(X[:, 0]), **bounds_kwargs)
    plt.plot(mpc.ocp_solver.acados_ocp.constraints.ubx[0] * np.ones_like(X[:, 0]), **bounds_kwargs)
    plt.ylabel("s_1")
    plt.subplot(3, 1, 2)
    plt.grid()
    plt.plot(X[:, 1])
    plt.plot(mpc.ocp_solver.acados_ocp.constraints.lbx[1] * np.ones_like(X[:, 1]), **bounds_kwargs)
    plt.plot(mpc.ocp_solver.acados_ocp.constraints.ubx[1] * np.ones_like(X[:, 1]), **bounds_kwargs)
    plt.ylabel("s_2")
    plt.subplot(313)
    plt.plot(U)
    plt.plot(mpc.ocp_solver.acados_ocp.constraints.lbu[0] * np.ones_like(U), **bounds_kwargs)
    plt.plot(mpc.ocp_solver.acados_ocp.constraints.ubu[0] * np.ones_like(U), **bounds_kwargs)
    plt.ylabel("a")
    plt.xlabel("t")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    param = {
        "A": np.array([[1.0, 0.25], [0.0, 1.0]]),
        "B": np.array([[0.03125], [0.25]]),
        "Q": np.identity(2),
        "R": np.identity(1),
        "b": np.array([[0.0], [0.0]]),
        "gamma": 0.9,
        "f": np.array([[0.0], [0.0], [0.0]]),
        "V_0": np.array([1e-3]),
    }

    test_acados_mpc(param)
