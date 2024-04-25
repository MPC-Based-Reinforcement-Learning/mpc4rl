import numpy as np
import time
import pandas as pd
import os
from rlmpc.common.utils import get_root_path
from rlmpc.mpc.chain_mass.ocp_utils import get_chain_params, find_idx_for_labels, define_param_struct_symSX
from rlmpc.mpc.chain_mass.acados import AcadosMPC


def main(params_=get_chain_params()):
    mpc = AcadosMPC(params_, discount_factor=1.0)

    x0 = mpc.ocp_solver.acados_ocp.constraints.lbx_0

    p_test = []
    np_test = 100

    p_label = f"m_{params_['n_mass']-3}"
    p_idx = find_idx_for_labels(
        define_param_struct_symSX(params_["n_mass"], disturbance=True).cat,
        p_label,
    )[0]

    p_nom = mpc.nlp.p.val.cat.full().flatten()

    p_var = np.linspace(0.5 * p_nom[p_idx], 1.5 * p_nom[p_idx], np_test)

    timings = []

    pi = []
    dpi_dp = []
    for i in range(np_test):
        p_test.append(p_nom.copy())
        p_test[-1][p_idx] = p_var[i]

    for i in range(np_test):
        print(f"Test {i+1}/{np_test}")

        mpc.set_p(p_test[i])

        _ = mpc.update(x0)

        pi.append(mpc.get_pi())

        start_time = time.time()
        dpi_dp.append(mpc.get_dpi_dp(finite_differences=True, idx=None)[:, p_idx].flatten())
        timings.append(time.time() - start_time)

    # Make a pandas dataframe with the timings
    timings_df = pd.DataFrame(timings)

    # Save the dataframe to a csv file
    fig_path = os.path.join(get_root_path(), "scripts", "figures")
    os.makedirs(fig_path, exist_ok=True)
    timings_df.to_csv(f"{fig_path}/chain_mass_{params_['n_mass']}_timings_finitedifferences.csv")


if __name__ == "__main__":
    params = get_chain_params()
    for n_mass in [3, 4, 5, 6]:
        params["n_mass"] = n_mass
        main(params_=params)
