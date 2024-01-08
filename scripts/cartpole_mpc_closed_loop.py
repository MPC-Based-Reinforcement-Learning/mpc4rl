from rlmpc.common.utils import read_config
from rlmpc.mpc.cartpole.acados import AcadosMPC
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from rlmpc.gym.continuous_cartpole.environment import ContinuousCartPoleSwingUpEnv  # noqa: F401

from rlmpc.common.utils import get_root_path


def create_mpc(config: dict) -> AcadosMPC:
    mpc = AcadosMPC(config=config, build=False)

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


def plot_results(states: np.ndarray, actions: np.ndarray, costs: np.ndarray):
    fig, axes = plt.subplots(nrows=6, ncols=1, sharex=True)
    axes[0].plot(states[:, 0])
    axes[1].plot(states[:, 1])
    axes[2].plot(states[:, 2])
    axes[3].plot(states[:, 3])
    axes[4].plot(actions)
    axes[5].plot(costs)

    axes[0].set_ylabel("x")
    axes[1].set_ylabel("v")
    axes[2].set_ylabel("theta")
    axes[3].set_ylabel("omega")
    axes[4].set_ylabel("u")
    axes[5].set_ylabel("cost")
    axes[5].set_xlabel("t")

    for ax in axes:
        ax.grid()

    plt.show()


PLOT = True


if __name__ == "__main__":
    print("Running test_acados_mpc_closed_loop.py ...")

    config = read_config(f"{get_root_path()}/config/cartpole_mpc.yaml")

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

    # Truncate arrays
    states = states[: i + 1, :]
    actions = actions[: i + 1, :]
    costs = costs[: i + 1]

    if done:
        print("Swing-up successful!")
        status = 0
    else:
        print("Swing-up failed ...")
        status = 1

    if PLOT:
        plot_results(states=states, actions=actions, costs=costs)
