from acados_template import AcadosOcpSolver
from rlmpc.mpc.common.mpc_acados_sensitivities import MPC
from .ocp_utils import export_parametric_ocp


class AcadosMPC(MPC):
    """docstring for MPC."""

    def __init__(self, param, discount_factor=0.99, **kwargs):
        super(AcadosMPC, self).__init__()

        ocp_solver_kwargs = kwargs["ocp_solver"] if "ocp_solver" in kwargs else {}

        self.ocp_solver = setup_acados_ocp_solver(param, **ocp_solver_kwargs)

        ocp_sensitivity_solver_kwargs = kwargs["ocp_sensitivity_solver"] if "ocp_sensitivity_solver" in kwargs else {}

        self.ocp_sensitivity_solver = setup_ocp_sensitivity_solver(param, **ocp_sensitivity_solver_kwargs)


def setup_acados_ocp_solver(param: dict, **kwargs) -> AcadosOcpSolver:
    ocp = export_parametric_ocp(param=param, qp_solver_ric_alg=1, integrator_type="DISCRETE")

    ocp.solver_options.with_value_sens_wrt_params = True

    ocp_solver = AcadosOcpSolver(ocp, **kwargs)

    status = ocp_solver.solve()

    if status != 0:
        raise ValueError(f"Initial solve failed with status {status}")

    return ocp_solver


def setup_ocp_sensitivity_solver(param: dict, **kwargs) -> AcadosOcpSolver:
    ocp = export_parametric_ocp(param=param, qp_solver_ric_alg=0, integrator_type="DISCRETE")

    ocp.code_export_directory = ocp.code_export_directory + "_sensitivity"

    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.nlp_solver_step_length = 0.0
    ocp.solver_options.nlp_solver_max_iter = 1
    ocp.solver_options.qp_solver_iter_max = 200
    ocp.solver_options.tol = 1e-10
    ocp.solver_options.qp_solver_ric_alg = 0
    ocp.solver_options.qp_solver_cond_N = ocp.dims.N
    ocp.solver_options.with_solution_sens_wrt_params = True
    ocp.solver_options.with_value_sens_wrt_params = True

    ocp_sensitivity_solver = AcadosOcpSolver(ocp, **kwargs)

    # set_discount_factor(ocp_sensitivity_solver, discount_factor=discount_factor)

    return ocp_sensitivity_solver
