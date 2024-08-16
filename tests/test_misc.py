import numpy as np
from rlmpc.mpc.pendulum_on_cart.acados import AcadosMPC
from test_pendulum_on_cart import build_mpc_params, build_mpc_args
from rlmpc.mpc.pendulum_on_cart.ocp_utils import cost_expr_ext_cost_0, cost_expr_ext_cost, cost_expr_ext_cost_e


def check_default_scaling(mpc: AcadosMPC, params: dict) -> bool:
    """
    Check the default scaling of the cost function. Initial stage and middle stages scaled by dt.
    """

    mpc.ocp_solver.solve()
    ocp_solver = mpc.ocp_solver

    dt = params["Tf"] / params["N"]

    cost = dt * cost_expr_ext_cost_0(ocp_solver.get(0, "x"), ocp_solver.get(0, "u"), params["Q"], params["R"])

    for stage in range(1, ocp_solver.acados_ocp.dims.N):
        cost += dt * cost_expr_ext_cost(ocp_solver.get(stage, "x"), ocp_solver.get(stage, "u"), params["Q"], params["R"])

    cost += cost_expr_ext_cost_e(ocp_solver.get(ocp_solver.acados_ocp.dims.N, "x"), params["Q"])

    assert np.abs(ocp_solver.get_cost() - cost) < 1e-6


def check_discount_factor_scaling(mpc: AcadosMPC, params: dict, gamma: float) -> bool:
    """
    Check the scaling of the cost function with the discount factor.
    """

    mpc.ocp_solver.solve()
    x0 = mpc.ocp_solver.get(0, "x")
    ocp_solver = mpc.ocp_solver

    # Set discount_factor
    mpc.set_discount_factor(gamma)
    mpc.ocp_solver.solve_for_x0(x0)

    cost = gamma**0 * cost_expr_ext_cost_0(ocp_solver.get(0, "x"), ocp_solver.get(0, "u"), params["Q"], params["R"])

    for stage in range(1, ocp_solver.acados_ocp.dims.N):
        cost += gamma**stage * cost_expr_ext_cost(
            ocp_solver.get(stage, "x"), ocp_solver.get(stage, "u"), params["Q"], params["R"]
        )

    cost += gamma**ocp_solver.acados_ocp.dims.N * cost_expr_ext_cost_e(
        ocp_solver.get(ocp_solver.acados_ocp.dims.N, "x"), params["Q"]
    )

    assert np.abs(ocp_solver.get_cost() - cost) < 1e-6


def test_set_discount_factor(
    generate_code: bool = False, build_code: bool = False, json_file_prefix: str = "acados_ocp_pendulum_on_cart"
):
    params = build_mpc_params()
    kwargs = build_mpc_args(generate_code, build_code, json_file_prefix)
    mpc = AcadosMPC(param=params, **kwargs)

    check_default_scaling(mpc, params)

    for gamma in np.arange(0.9, 1.0, 0.02):
        check_discount_factor_scaling(mpc, params, gamma=gamma)

    assert True


if __name__ == "__main__":
    test_set_discount_factor()
