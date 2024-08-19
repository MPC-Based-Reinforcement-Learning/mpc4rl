import os
import numpy as np
import matplotlib.pyplot as plt
from rlmpc.mpc.pendulum_on_cart.acados import AcadosMPC as MPC

from rlmpc.mpc.common.testing import (
    run_test_v_update_for_varying_parameters,
    run_test_q_update_for_varying_parameters,
    run_test_pi_update_for_varying_parameters,
    set_up_test_parameters,
)


def build_mpc_args(generate_code: bool = False, build_code: bool = False, json_file_prefix: str = "acados_ocp_pole_on_a_cart"):
    kwargs = {
        "ocp_solver": {"json_file": f"{json_file_prefix}.json"},
        "ocp_sensitivity_solver": {"json_file": f"{json_file_prefix}_sensitivity.json"},
    }

    for key in kwargs.keys():
        if os.path.isfile(kwargs[key]["json_file"]):
            kwargs[key]["generate"] = generate_code
            kwargs[key]["build"] = build_code
        else:
            kwargs[key]["generate"] = True
            kwargs[key]["build"] = True

    return kwargs


def main(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_pendulum_on_cart",
    x0: np.ndarray = np.array([0.0, np.pi, 0.0, 0.0]),
    n_sim: int = 50,
    plot: bool = False,
):
    param = {
        "M": 1.0,  # Mass of the cart
        "m": 0.1,  # Mass of the ball
        "g": 9.81,  # Gravity constant
        "l": 0.8,  # Length of the rod
        "Q": 2 * np.diag([1e3, 1e3, 1e-2, 1e-2]),  # State cost matrix
        "R": 2 * np.diag([1e-2]),  # Control cost matrix
        "model_name": "pendulum_on_cart",
        "Tf": 1.0,  # Prediction horizon
        "N": 20,  # Number of control intervals
    }

    mpc = MPC(
        param=param, **build_mpc_args(generate_code=generate_code, build_code=build_code, json_file_prefix=json_file_prefix)
    )

    x = [x0]

    # Closed loop over mpc predictions
    res = {"x": [], "pi": [], "v": [], "q": [], "dpidp": [], "dvdp": [], "dqdp": []}
    for _ in range(n_sim):
        pi, dpidp = mpc.pi_update(x[-1])
        v, dvdp = mpc.v_update(x[-1])
        q, dqdp = mpc.q_update(x[-1], pi)

        res["x"].append(x[-1])
        res["pi"].append(pi)
        res["v"].append(v)
        res["q"].append(q)
        res["dpidp"].append(dpidp)
        res["dvdp"].append(dvdp)
        res["dqdp"].append(dqdp)

        assert mpc.ocp_solver.get_status() == 0
        x.append(mpc.ocp_solver.get(1, "x"))

    for key, val in res.items():
        res[key] = np.array(val)

    res["dpidp"] = np.squeeze(res["dpidp"])

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.step(np.arange(n_sim), res["x"][:, 0], label="x")
    plt.subplot(3, 1, 2)
    plt.step(np.arange(n_sim), res["x"][:, 1], label="theta")
    plt.subplot(3, 1, 3)
    plt.step(np.arange(n_sim), res["pi"], label="u")
    plt.legend()
    plt.show()

    for param_label in ["Q_5", "M"]:
        # NB: varying cost parameters (e.g. Q_5) seems gives incorrectly zero value gradients
        mpc = MPC(param=param, **build_mpc_args(generate_code=False, build_code=False, json_file_prefix=json_file_prefix))
        test_param = set_up_test_parameters(mpc, np_test=100, varying_param_label=param_label, scale_low=0.9, scale_high=1.1)
        _ = run_test_v_update_for_varying_parameters(mpc=mpc, x0=x0, test_param=test_param, plot=True)
        _ = run_test_q_update_for_varying_parameters(mpc=mpc, x0=x0, u0=np.array([0]), test_param=test_param, plot=True)
        _ = run_test_pi_update_for_varying_parameters(
            mpc=mpc, x0=np.array([0.0, np.pi / 2, 0.0, 0.0]), test_param=test_param, plot=True
        )


if __name__ == "__main__":
    main(generate_code=False, build_code=False)
