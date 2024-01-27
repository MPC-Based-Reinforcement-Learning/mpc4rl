import numpy as np
from rlmpc.gym.evaporation_process.environment import EvaporationProcessEnv  # noqa: F401
from rlmpc.mpc.evaporation_process.acados import AcadosMPC
from scripts.linear_system_mpc_nlp import test_acados_ocp_nlp


def main():
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

    x0 = np.array([50.0, 60.0])

    u0 = mpc.get_action(x0)

    mpc.update_nlp()

    test_acados_ocp_nlp(mpc, x0, u0, plot=True)


if __name__ == "__main__":
    main()
