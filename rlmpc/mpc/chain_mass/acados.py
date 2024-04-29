# import os
# import numpy as np
from acados_template import AcadosOcpSolver
# import casadi as cs

from rlmpc.mpc.common.mpc import MPC
from rlmpc.mpc.nlp import NLP, build_nlp

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

    def __init__(self, param, discount_factor=0.99):
        super(AcadosMPC, self).__init__()

        self.ocp_solver = setup_acados_ocp_solver(param)

        self.nlp = build_nlp(self.ocp_solver.acados_ocp, gamma=discount_factor)

        # self.set_discount_factor(discount_factor)


def setup_acados_ocp_solver(param: dict) -> AcadosOcpSolver:
    ocp, _ = export_parametric_ocp(chain_params_=param, integrator_type="DISCRETE")

    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1

    ocp_json_file = "acados_ocp_" + ocp.model.name + ".json"

    return AcadosOcpSolver(ocp, json_file=ocp_json_file, build=True, generate=True)

