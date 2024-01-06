from rlmpc.common.utils import read_config
from rlmpc.mpc.cartpole.acados import AcadosMPC
from rlmpc.mpc.cartpole.common import Config
import numpy as np
import matplotlib.pyplot as plt


def test_dL_dp(mpc: AcadosMPC, x0: np.ndarray, p_test: list, plot=True):
    L = {"true": np.zeros(len(p_test)), "approx": np.zeros(len(p_test))}

    for i, p_i in enumerate(p_test):
        for stage in range(mpc.ocp_solver.acados_ocp.dims.N + 1):
            mpc.set(stage, "p", p_i)

        mpc.update(x0=x0)

        L["true"][i] = mpc.get_L()

        if i == 0:
            L["approx"][i] = L["true"][i]
        else:
            dp = np.repeat(p_test[i] - p_test[i - 1], mpc.ocp_solver.acados_ocp.dims.N + 1).reshape(-1, 1)
            L["approx"][i] = L["approx"][i - 1] + np.dot(mpc.get_dL_dp(), dp)[0][0]

    if plot:
        _, ax = plt.subplots()
        ax.plot(p_test, L["true"])
        ax.plot(p_test, L["approx"], "--")

        plt.show()


def test_dV_dp(mpc: AcadosMPC, x0: np.ndarray, p_test: list, plot=True):
    V = {"true": np.zeros(len(p_test)), "approx": np.zeros(len(p_test))}

    for i, p_i in enumerate(p_test):
        for stage in range(mpc.ocp_solver.acados_ocp.dims.N + 1):
            mpc.set(stage, "p", p_i)

        mpc.update(x0=x0)

        V["true"][i] = mpc.get_V()

        if i == 0:
            V["approx"][i] = V["true"][i]
        else:
            dp = np.repeat(p_test[i] - p_test[i - 1], mpc.ocp_solver.acados_ocp.dims.N + 1).reshape(-1, 1)
            V["approx"][i] = V["approx"][i - 1] + np.dot(mpc.get_dV_dp(), dp)[0][0]

    if plot:
        _, ax = plt.subplots()
        ax.plot(p_test, V["true"])
        ax.plot(p_test, V["approx"], "--")

        plt.show()


def test_dQ_dp(mpc: AcadosMPC, x0: np.ndarray, p_test: list, plot=True):
    Q = {"true": np.zeros(len(p_test)), "approx": np.zeros(len(p_test))}

    u0 = np.array([1.0])

    for i, p_i in enumerate(p_test):
        for stage in range(mpc.ocp_solver.acados_ocp.dims.N + 1):
            mpc.set(stage, "p", p_i)

        mpc.update(x0=x0)

        mpc.q_update(x0=x0, u0=u0)

        Q["true"][i] = mpc.get_Q()

        if i == 0:
            Q["approx"][i] = Q["true"][i]
        else:
            # dp = np.repeat(p_test[i] - p_test[i - 1], mpc.ocp_solver.acados_ocp.dims.N + 1).reshape(-1, 1)
            dp = p_test[i] - p_test[i - 1]
            # dQ_dp["true"][i] = mpc.get_dQ_dp()
            Q["approx"][i] = Q["approx"][i - 1] + np.dot(mpc.get_dQ_dp(), dp)

    if plot:
        _, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax[0].plot(Q["true"])
        ax[0].plot(Q["approx"], "--")

        plt.show()


if __name__ == "__main__":
    config = read_config("config/test_AcadosMPC.yaml")

    mpc = AcadosMPC(config=Config.from_dict(config["mpc"]), build=True)

    x0 = np.array([0.0, 0.0, np.pi / 2, 1.0])

    p_nom = mpc.nlp.p.val
    p_min = 0.5 * mpc.nlp.p.val
    p_max = 1.5 * mpc.nlp.p.val

    n_test_params = 200
    p_test = [p_min + (p_max - p_min) * i / n_test_params for i in range(n_test_params)]

    # test_dL_dp(mpc, x0, p_test)
    # test_dV_dp(mpc, x0, p_test)
    test_dQ_dp(mpc, x0, p_test)

    exit()

    for stage in range(mpc.ocp_solver.acados_ocp.dims.N + 1):
        mpc.ocp_solver.set(stage, "p", 1.0)

    status = mpc.update(x0=x0)

    mpc.plot_prediction()

    # mpc.nlp = update_nlp(nlp=mpc.nlp, ocp_solver=mpc.ocp_solver, multiplier_map=mpc.muliplier_map)

    nlp = mpc.nlp

    # TODO: Need a generic way of passing the parameters to the functions through the arg_list

    # Make a spy plot of dR_dz
    # plt.spy(nlp.dR_dz.val)
    # plt.show()

    # Check if dR_dz contains rows with only zeros
    print(not any(np.all(nlp.dR_dz.val == 0, axis=0)))
    print(not any(np.all(nlp.dR_dz.val == 0, axis=1)))

    # Compute the rank of dR_dz
    print(np.linalg.matrix_rank(nlp.dR_dz.val))

    # Check if nlp.dR_dz.val is a singular matrix
    if np.linalg.det(nlp.dR_dz.val) == 0:
        print("dR_dz is singular")

    # Get idx of lam that are less than 1e-6. Corresponds to constraint not being active
    idx = np.where(nlp.lam.val.cat < 1e-6)[0]

    # Get idx of h that are less than -1e-6. Corresponds to constraint not being active
    # idx = np.where(nlp.h.val < -1e-6)[0]

    # Add length of nlp.w.val and nlp.pi.val to idx
    idx += nlp.w.val.cat.shape[0] + nlp.pi.val.cat.shape[0]

    # Remove idx rows and colums from nlp.dR_dz.val
    nlp.dR_dz.val = np.delete(nlp.dR_dz.val, idx, axis=0)
    nlp.dR_dz.val = np.delete(nlp.dR_dz.val, idx, axis=1)

    # Remove idx rows from nlp.dR_dp.val
    nlp.dR_dp.val = np.delete(nlp.dR_dp.val, idx, axis=0)

    if np.linalg.det(nlp.dR_dz.val) == 0:
        print("active-set dR_dz is singular")

    # Compute the gradient of the policy
    nx = mpc.ocp.dims.nx
    nu = mpc.ocp.dims.nu
    dpi_dp = np.linalg.solve(nlp.dR_dz.val, -nlp.dR_dp.val)[nx : nx + nu, :]

    acados_cost = mpc.ocp_solver.get_cost()

    nlp_cost = nlp.cost.fun(w=nlp.w.val, dT=nlp.dT.val)["cost"]

    assert abs(acados_cost - nlp_cost) < 1e-2

    # Load cs.Function from path
    # fun = cs.Function.load(path)

    # R = mpc.nlp.R.fun(
    #     w=mpc.nlp.w.val, lbw=mpc.nlp.lbw.val, ubw=mpc.nlp.ubw.val, pi=mpc.nlp.pi.val, lam=mpc.nlp.lam.val, p=mpc.nlp.p.val
    # )

    # print("norm(R)", np.linalg.norm(R))

    print("norm dot(pi, g)", np.linalg.norm(mpc.nlp.pi.val * mpc.nlp.g.val))
    print("norm dot(lam, h)", np.linalg.norm(mpc.nlp.lam.val * mpc.nlp.h.val))

    print("hallo")
