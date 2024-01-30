import numpy as np
from rlmpc.gym.evaporation_process.environment import EvaporationProcessEnv, PARAM  # noqa: F401
from rlmpc.mpc.evaporation_process.acados import AcadosMPC
from scripts.linear_system_mpc_nlp import test_acados_ocp_nlp
from scripts.evaporation_process_mpc import H_tuned, cost_param_from_H


def main():
    cost_param = cost_param_from_H(H_tuned)

    model_param = PARAM

    mpc = AcadosMPC(model_param=model_param, cost_param=cost_param)

    x0 = np.array([50.0, 60.0])

    u0 = mpc.get_action(x0)

    mpc.update_nlp()

    test_acados_ocp_nlp(mpc, x0, u0, plot=True)


if __name__ == "__main__":
    main()
