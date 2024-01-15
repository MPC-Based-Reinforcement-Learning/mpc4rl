import numpy as np

import scipy.integrate as integrate
import matplotlib.pyplot as plt

import gymnasium as gym

from rlmpc.gym.house.environment import build_A, build_B, ode

from rlmpc.gym.house.environment import HouseEnv  # noqa: F401

from stable_baselines3.common.buffers import ReplayBuffer

from rlmpc.mpc.hvac.acados import AcadosMPC, define_ocp_solver


def erk4_integrator_step(f, x, u, p, h) -> np.ndarray:
    """
    Explicit Runge-Kutta 4 integrator


    Parameters:
        f: function to integrate
        x: state
        u: control
        p: parameters
        h: step size

        Returns:
            xf: integrated state
    """
    k1 = f(x, u, p)
    k2 = f(x + h / 2 * k1, u, p)
    k3 = f(x + h / 2 * k2, u, p)
    k4 = f(x + h * k3, u, p)
    xf = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return xf


class House(object):
    """docstring for House."""

    state: np.ndarray
    input: np.ndarray
    R: np.ndarray
    C: np.ndarray
    A: np.ndarray
    B: np.ndarray
    P: np.ndarray
    params: dict[np.ndarray]

    def __init__(self):
        super(House, self).__init__()

        self.state = np.array([20, 25, 15, 22, 10])

        self.nx = 5
        self.nu = 2
        self.nd = 1

        self.p = [sample_capacity(self.nx), sample_resistance(self.nx), sample_heatpump_efficiency(self.nu)]

        self.np = len(self.p[0]) + len(self.p[1]) * len(self.p[1]) + len(self.p[2])

        self.A = build_A(self.p)
        self.B = build_B(self.p)

    def get_A(self):
        return self.A

    def get_B(self):
        return self.B

    def get_x(self):
        return self.state

    def get_p(self, as_list: bool = True):
        return np.concatenate((self.p[0].reshape(-1), self.p[1].reshape(-1), self.p[2].reshape(-1)))


# def get_A_from_p(p:np.ndarray, nx:int) -> np.ndarray:
#     C, R = get_C_R_from_p(p, nx)


def test_house():
    house = House()

    A = house.get_A()

    # Compute eigenvalues of A
    eigenvalues, eigenvectors = np.linalg.eig(A)

    assert np.all(eigenvalues < 1e-10), "System is not passively stable"

    x0 = np.array([20, 25, 15, 22, 10])

    integrator = integrate.RK45(lambda t, x: np.matmul(A, x), y0=x0, t0=0, t_bound=5e3, max_step=5)

    t = []
    x = []

    while integrator.status == "running":
        t.append(integrator.t)
        x.append(integrator.y)
        integrator.step()

        print(integrator.status)

    x = np.vstack(x)
    t = np.vstack(t)

    figure, axes = plt.subplots()
    for i in range(4):
        axes.plot(t / 3600.0, x[:, i], label=f"T{i}")

    axes.set_xlabel("Time [h]")
    axes.set_ylabel("Temperature [deg C]")
    axes.legend()
    axes.grid()
    plt.show()


def test_house_2():
    nx = 5
    nu = 2

    p = [sample_capacity(nx), sample_resistance(nx), sample_heatpump_efficiency(nu)]

    u = np.array([3, 5])
    x0 = np.array([20, 25, 15, 22, 10])

    h = 10.0
    t = np.arange(0.0, 9e3, h)
    N = t.shape[0]

    x = np.zeros((N, x0.shape[0]))
    x[0, :] = x0

    for i in range(N - 1):
        x[i + 1] = erk4_integrator_step(ode, x[i, :], u, p, h)

    x = np.vstack(x)
    t = np.vstack(t)

    figure, axes = plt.subplots()
    for i in range(4):
        axes.plot(t / 3600.0, x[:, i], label=f"T{i}")

    axes.set_xlabel("Time [h]")
    axes.set_ylabel("Temperature [deg C]")
    axes.legend()
    axes.grid()
    plt.show()


def sample_capacity(n_capacity: int) -> np.ndarray:
    capacity = np.random.uniform(0, 1, (n_capacity))

    return capacity


def sample_resistance(nx: int) -> np.ndarray:
    resistance = np.random.uniform(2e3, 3e3, (nx, nx))

    # Make matrix symmetric
    for i in range(nx - 1):
        for j in range(nx - 1):
            resistance[j, i] = resistance[i, j]

    # Set diagonal to zero
    for i in range(nx):
        resistance[i, i] = 0

    indoor_resistance = resistance[:-1, :-1]

    assert np.all(indoor_resistance == indoor_resistance.T), "Resistance matrix is not symmetric"

    # Outdoor insulation
    resistance[:-1, -1] = np.random.uniform(10e3, 20e3, (nx - 1))

    resistance = resistance

    return resistance


def sample_heatpump_efficiency(nu: int) -> np.ndarray:
    eta = np.random.uniform(1e-4, 5e-4, (nu))

    return eta


def test_ocp_prediction():
    house = House()
    ocp_solver = define_ocp_solver(house)

    x0 = np.array([20, 25, 15, 22, 10])

    for stage in range(ocp_solver.acados_ocp.dims.N):
        ocp_solver.set(stage, "x", x0)
        ocp_solver.set(stage, "x", x0)

    ocp_solver.constraints_set(0, "lbx", x0)
    ocp_solver.constraints_set(0, "ubx", x0)

    status = ocp_solver.solve()

    print(status)

    ocp_solver.print_statistics()

    x_traj = np.zeros((ocp_solver.acados_ocp.dims.N + 1, ocp_solver.acados_ocp.dims.nx))
    u_traj = np.zeros((ocp_solver.acados_ocp.dims.N, ocp_solver.acados_ocp.dims.nu))
    for stage in range(ocp_solver.acados_ocp.dims.N + 1):
        x_traj[stage, :] = ocp_solver.get(stage, "x")
    for stage in range(ocp_solver.acados_ocp.dims.N):
        u_traj[stage, :] = ocp_solver.get(stage, "u")

    figure, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].plot(x_traj)
    axes[1].plot(u_traj)

    plt.show()


if __name__ == "__main__":
    # test_ocp_prediction()

    p = [sample_capacity(4), sample_resistance(5), sample_heatpump_efficiency(2)]

    config = {"params": p}

    mpc = AcadosMPC(config)

    env = gym.make(
        "House-v0",
        p=p,
        min_power=0.0,
        max_power=100.0,
    )

    # ocp_solver = define_ocp_solver(param=p)

    buffer_size = 5000

    replay_buffer = ReplayBuffer(
        buffer_size=buffer_size,
        observation_space=env.observation_space,
        action_space=env.action_space,
        handle_timeout_termination=False,
    )

    replay_buffer.reset()

    obs, _ = env.reset()

    for _ in range(buffer_size):
        action = mpc.get_action(obs)
        next_obs, reward, done, _, info = env.step(np.array(action, dtype=np.float32))

        replay_buffer.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, infos=info)

        obs = next_obs

    X = np.vstack(replay_buffer.observations)
    U = np.vstack(replay_buffer.actions)

    figure, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].plot(X)
    axes[1].plot(U)
    plt.show()
