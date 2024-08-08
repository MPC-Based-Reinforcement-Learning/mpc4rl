# import os
# import numpy as np
import copy
from acados_template import AcadosOcpSolver
# import casadi as cs

from rlmpc.mpc.common.mpc_acados_sensitivities import MPC
from rlmpc.mpc.common.nlp import NLP, build_nlp

from .ocp_utils import (
    # get_chain_params,
    # find_idx_for_labels,
    # export_discrete_erk4_integrator_step,
    # export_chain_mass_model,
    export_parametric_ocp,
    # define_param_struct_symSX,
)


class AcadosMPC(MPC):
    """docstring for MPC."""

    nlp: NLP

    def __init__(self, param, discount_factor=0.99, **kwargs):
        super(AcadosMPC, self).__init__()

        ocp_solver_kwargs = kwargs["ocp_solver"] if "ocp_solver" in kwargs else {}

        self.ocp_solver = setup_acados_ocp_solver(param, **ocp_solver_kwargs)

        ocp_sensitivity_solver_kwargs = kwargs["ocp_sensitivity_solver"] if "ocp_sensitivity_solver" in kwargs else {}

        self.ocp_sensitivity_solver = setup_ocp_sensitivity_solver(param, **ocp_sensitivity_solver_kwargs)

        # self.nlp = build_nlp(self.ocp_solver.acados_ocp, gamma=discount_factor)

        # self.set_discount_factor(discount_factor)


def setup_acados_ocp_solver(param: dict, **kwargs) -> AcadosOcpSolver:
    # ocp, _ = export_parametric_ocp(chain_params_=param, integrator_type="DISCRETE")

    ocp, _ = export_parametric_ocp(chain_params_=param, qp_solver_ric_alg=0, integrator_type="DISCRETE")

    # ocp.solver_options.sim_method_num_stages = 4
    # ocp.solver_options.sim_method_num_steps = 1
    # ocp_json_file = "acados_ocp_nlp_" + ocp.model.name + ".json"

    # kwargs["json_file"] = ocp_json_file
    # kwargs["generate"] = False
    # kwargs["build"] = False

    # TODO: Check if we can also use the json_file to generate the ocp
    ocp_solver = AcadosOcpSolver(ocp, **kwargs)

    status = ocp_solver.solve()

    if status != 0:
        raise ValueError(f"Initial solve failed with status {status}")

    return ocp_solver


# def setup_ocp_sensitivity_solver(ocp_solver: AcadosOcpSolver, discount_factor: float = 0.99, **kwargs) -> AcadosOcpSolver:
def setup_ocp_sensitivity_solver(param: dict, **kwargs) -> AcadosOcpSolver:
    ocp, _ = export_parametric_ocp(chain_params_=param, qp_solver_ric_alg=0, integrator_type="DISCRETE")

    # ocp_json_file = "acados_ocp_nlp_" + ocp.model.name + "sensitivity.json"

    # kwargs["json_file"] = ocp_json_file
    # kwargs["generate"] = True
    # kwargs["build"] = True

    ocp.code_export_directory = ocp.code_export_directory + "_sensitivity"

    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.nlp_solver_step_length = 0.0
    ocp.solver_options.nlp_solver_max_iter = 1
    ocp.solver_options.qp_solver_iter_max = 200
    ocp.solver_options.tol = 1e-10
    ocp.solver_options.qp_solver_ric_alg = 0
    ocp.solver_options.qp_solver_cond_N = ocp.dims.N
    ocp.solver_options.with_solution_sens_wrt_params = True

    ocp_sensitivity_solver = AcadosOcpSolver(ocp, **kwargs)

    # set_discount_factor(ocp_sensitivity_solver, discount_factor=discount_factor)

    return ocp_sensitivity_solver
