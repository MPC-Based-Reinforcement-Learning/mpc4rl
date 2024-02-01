import numpy as np
from rlmpc.mpc.linear_system.acados import AcadosMPC
from tqdm import tqdm
import matplotlib.pyplot as plt


def setup_p_test(p: np.ndarray, i_param: int = 0, n_test_params: int = 50) -> list[np.ndarray]:
    p_min = p.copy()
    p_max = p.copy()

    p_min[i_param] = 0.9 * p[i_param]
    p_max[i_param] = 1.1 * p[i_param]

    return [p_min + (p_max - p_min) * i / n_test_params for i in range(n_test_params)]


def test_acados_ocp_nlp(
    mpc: AcadosMPC, x0: np.ndarray = np.array([[0.2], [0.2]]), u0: np.ndarray = np.array([-0.5]), plot: bool = False
) -> None:
    param = mpc.get_parameters()
    # for i_param in tqdm(range(mpc.ocp_solver.acados_ocp.dims.np), desc="Testing dV_dp for non-zero parameters"):
    for i_param in tqdm(range(len(param)), desc="Testing dV_dp for non-zero parameters"):
        p_test = setup_p_test(mpc.get_parameters(), i_param, n_test_params=50)

        # Only test for parameters that are not equal (otherwise nominal parameter should be zero)
        if p_test[0][i_param] == p_test[1][i_param]:
            continue

        V = []
        dV_dp = []
        for i in range(len(p_test)):
            p_i = p_test[i]
            for stage in range(mpc.ocp_solver.acados_ocp.dims.N + 1):
                mpc.set(stage, "p", p_i)

            mpc.update(x0)

            V.append(mpc.get_V())

            dV_dp.append(mpc.get_dV_dp())

        V = np.array(V)

        dV_dp_true = np.gradient(V, p_test[1][i_param] - p_test[0][i_param])[1:-1]

        dV_dp = np.vstack(dV_dp[1:-1])[:, i_param]

        np.allclose(dV_dp_true, dV_dp, atol=1e-6)

        # print(f"i_param: {i_param}")
        # print(f"dV_dp_true - dV_dp: {dV_dp_true - dV_dp}")

        # if False:
        if plot:
            plt.figure(1)
            plt.plot(dV_dp_true, label="finite difference")
            plt.plot(dV_dp, label="AD")
            plt.ylabel(f"dV_dp[{i_param}]")
            plt.xlabel("param variation [-]")
            plt.legend()
            plt.grid()
            plt.show()

    for i_param in tqdm(range(mpc.ocp_solver.acados_ocp.dims.np), desc="Testing dQ_dp for non-zero parameters"):
        p_test = setup_p_test(mpc.get_parameters(), i_param, n_test_params=50)

        # Only test for parameters that are not equal (otherwise nominal parameter should be zero)
        if p_test[0][i_param] == p_test[1][i_param]:
            continue

        Q = []
        dQ_dp = []
        for i in range(len(p_test)):
            p_i = p_test[i]
            for stage in range(mpc.ocp_solver.acados_ocp.dims.N + 1):
                mpc.set(stage, "p", p_i)

            mpc.q_update(x0, u0)

            Q.append(mpc.get_Q())

            dQ_dp.append(mpc.get_dQ_dp())

        Q = np.array(Q)

        dQ_dp_true = np.gradient(Q, p_test[1][i_param] - p_test[0][i_param])[1:-1]

        dQ_dp = np.vstack(dQ_dp[1:-1])[:, i_param]

        np.allclose(dQ_dp_true, dQ_dp, atol=1e-6)

        # print(f"i_param: {i_param}")
        # print(f"dQ_dp_true - dQ_dp: {dQ_dp_true - dQ_dp}")

        if plot:
            plt.figure(1)
            plt.plot(dQ_dp_true, label="finite difference")
            plt.plot(dQ_dp, label="AD")
            plt.ylabel(f"dQ_dp[{i_param}]")
            plt.xlabel("param variation [-]")
            plt.legend()
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

    mpc = AcadosMPC(param)
    test_acados_ocp_nlp(mpc, plot=False)
