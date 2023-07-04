from src.agent.agent import Agent
from src.utils.utils import read_config
from src.environment.environment import Environment
from src.agent.function_approximators.mpc.mpc import MPC
import numpy as np
import matplotlib.pyplot as plt


def plot_simulation(t_sim: np.ndarray, x_sim: np.ndarray, u_sim: np.ndarray):
    fig, ax = plt.subplots(5, 1, sharex=True)

    ax[0].plot(t_sim, x_sim[:, 0], label="x")
    ax[0].set_ylabel("x [m]")
    ax[0].grid(True)

    ax[1].plot(t_sim, x_sim[:, 1], label="theta")
    ax[1].set_ylabel("theta [rad]")
    ax[1].grid(True)

    ax[2].plot(t_sim, x_sim[:, 2], label="v")
    ax[2].set_ylabel("v [m/s]")
    ax[2].grid(True)

    ax[3].plot(t_sim, x_sim[:, 3], label="dtheta")
    ax[3].set_ylabel("dtheta [rad/s]")
    ax[3].grid(True)

    ax[4].plot(t_sim, u_sim[:, 0], label="F")
    ax[4].set_ylabel("F [N]")
    ax[4].set_xlabel("t [s]")
    ax[4].grid(True)

    return fig, ax


def main_open_loop_simulation(config: dict) -> None:
    config = read_config("src/config/pendulum.yaml")

    environment = Environment(
        name="pendulum",
        param=config["environment"],
        simulation_settings=config["simulation"],
        acados_settings=config["acados_settings"],
    )

    x0 = config["simulation"]["x0"]

    u_sim = np.zeros(
        (
            int(config["simulation"]["T"] / config["simulation"]["dt"]),
            environment.sim.dims.nu,
        )
    )

    t_sim, x_sim, u_sim = environment.simulate(
        x0=x0, u=u_sim, sim_param=config["simulation"]
    )

    _, _ = plot_simulation(t_sim=t_sim, x_sim=x_sim, u_sim=u_sim)

    plt.show()


def main_closed_loop_simulation(config: dict) -> None:
    environment = Environment(
        name="pendulum",
        param=config["environment"],
        simulation_settings=config["simulation"],
        acados_settings=config["acados_settings"],
    )

    x0 = config["simulation"]["x0"]

    agent = Agent(name="pendulum_mpc", param=config["agent"])

    x_sim = np.zeros((int(config["simulation"]["T"] / config["simulation"]["dt"]), 4))
    u_sim = np.zeros((int(config["simulation"]["T"] / config["simulation"]["dt"]), 1))

    t_sim = np.arange(
        config["simulation"]["t0"],
        config["simulation"]["T"],
        config["simulation"]["dt"],
    )

    # Run 10 simulations from random initial conditions
    for _ in range(1):
        x0 = np.array(
            [
                np.random.normal(0.0, 0.1, size=1)[0],  # Initial position
                np.random.normal(np.pi, 0.1, size=1)[0],  # Initial angle
                0.0,  # Initial velocity
                0.0,
            ]
        )  # Initial angular velocity

        x_sim[0, :] = x0

        for k, _ in enumerate(t_sim[:-1]):
            u_sim[k, :] = agent.evaluate_policy(x=x_sim[k, :])

            x_sim[k + 1, :] = environment.step(x=x_sim[k, :], u=u_sim[k, :])

        plot_simulation(t_sim=t_sim, x_sim=x_sim, u_sim=u_sim)

        plt.show()


if __name__ == "__main__":
    config = read_config("src/config/pendulum.yaml")

    main_closed_loop_simulation(config=config)
