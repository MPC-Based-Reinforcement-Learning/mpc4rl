import os
from acados_template.acados_ocp_solver import ocp_generate_external_functions
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosOcpConstraints, AcadosOcpCost, AcadosOcpDims, AcadosModel

from typing import Union

import casadi as cs


from rlmpc.common.mpc import MPC

from rlmpc.mpc.cartpole.common import find_nlp_entry_expr_dependencies

import matplotlib.pyplot as plt

from rlmpc.mpc.cartpole.common import (
    CasadiNLP,
    Config,
    define_dimensions,
    define_cost,
    define_constraints,
    define_parameter_values,
    build_nlp,
)


def define_acados_model(ocp: AcadosOcp, config: Config) -> AcadosModel:
    # Check if ocp is an instance of AcadosOcp
    if not isinstance(ocp, AcadosOcp):
        raise TypeError("ocp must be an instance of AcadosOcp")

    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise TypeError("config must be an instance of Config")

    # try:
    #     model = define_model_expressions(config)
    # except Exception as e:
    #     # Handle or re-raise exception from define_constraints
    #     raise RuntimeError("Error in define_acados_model: " + str(e))

    # for key, val in model.items():
    #     # Check if the attribute exists in ocp.constraints
    #     if not hasattr(ocp.model, key):
    #         raise AttributeError(f"Attribute {key} does not exist in ocp.model")

    #     # Set the attribute, assuming the value is correct
    #     # TODO: Add validation for the value here
    #     setattr(ocp.model, key, val)

    name = config.model_name

    # set up states & controls
    s = cs.SX.sym("x")
    s_dot = cs.SX.sym("x_dot")
    theta = cs.SX.sym("theta")
    theta_dot = cs.SX.sym("theta_dot")

    x = cs.vertcat(s, s_dot, theta, theta_dot)

    F = cs.SX.sym("F")
    u = cs.vertcat(F)

    x_dot = cs.SX.sym("xdot", 4, 1)

    # algebraic variables
    z = None

    # parameters
    p_sym = []

    # Set up parameters to nominal values
    p = {key: param["value"] for key, param in config.model_params.to_dict().items()}

    parameter_values = []
    # Set up parameters to symbolic variables if not fixed
    for key, param in config.model_params.to_dict().items():
        if not param["fixed"]:
            p_sym += [cs.SX.sym(key)]
            p[key] = p_sym[-1]
            parameter_values += [param["value"]]

    p_sym = cs.vertcat(*p_sym)
    parameter_values = np.array(parameter_values)

    # Define model dynamics
    cos_theta = cs.cos(theta)
    sin_theta = cs.sin(theta)
    temp = (u + p["m"] * theta_dot**2 * sin_theta) / (p["m"] + p["M"])

    theta_ddot = (p["g"] * sin_theta - cos_theta * temp) / (p["l"] * (4.0 / 3.0 - p["m"] * cos_theta**2 / (p["m"] + p["M"])))

    f_expl = cs.vertcat(
        s_dot,
        temp - p["m"] * theta_ddot * cos_theta / (p["m"] + p["M"]),  # x_ddot
        theta_dot,
        (p["g"] * sin_theta - cos_theta * temp) / (p["l"] * (4.0 / 3.0 - p["m"] * cos_theta**2 / (p["m"] + p["M"]))),
    )

    f_impl = x_dot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = x_dot
    model.p = p_sym
    model.u = u
    model.z = z
    model.name = name

    return model


def define_acados_dims(ocp: AcadosOcp, config: Config) -> AcadosOcpDims:
    # Check if ocp is an instance of AcadosOcp
    if not isinstance(ocp, AcadosOcp):
        raise TypeError("ocp must be an instance of AcadosOcp")

    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise TypeError("config must be an instance of Config")

    try:
        dims = define_dimensions(config)
    except Exception as e:
        # Handle or re-raise exception from define_constraints
        raise RuntimeError("Error in define_acados_dims: " + str(e))

    for key, val in dims.items():
        # Check if the attribute exists in ocp.constraints
        if not hasattr(ocp.dims, key):
            raise AttributeError(f"Attribute {key} does not exist in ocp.dims")

        # Set the attribute, assuming the value is correct
        # TODO: Add validation for the value here
        setattr(ocp.dims, key, val)

    ocp.dims.np = ocp.model.p.size()[0]

    # TODO: Add other slack variable dimensions
    ocp.dims.nsbx = ocp.constraints.idxsbx.shape[0]
    ocp.dims.nsbu = ocp.constraints.idxsbu.shape[0]
    ocp.dims.ns = ocp.dims.nsbx + ocp.dims.nsbu

    ocp.dims.nsbx_e = ocp.constraints.idxsbx_e.shape[0]
    ocp.dims.ns_e = ocp.dims.nsbx_e

    ocp.dims.nbx_e = ocp.constraints.idxbx_e.shape[0]

    return ocp.dims


def define_acados_cost(ocp: AcadosOcp, config: Config) -> AcadosOcpCost:
    # Check if ocp is an instance of AcadosOcp
    if not isinstance(ocp, AcadosOcp):
        raise TypeError("ocp must be an instance of AcadosOcp")

    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise TypeError("config must be an instance of Config")

    try:
        cost = define_cost(config)
    except Exception as e:
        # Handle or re-raise exception from define_constraints
        raise RuntimeError("Error in define_acados_cost: " + str(e))

    for key, val in cost.items():
        # Check if the attribute exists in ocp.constraints
        if not hasattr(ocp.cost, key):
            raise AttributeError(f"Attribute {key} does not exist in ocp.cost")

        # Set the attribute, assuming the value is correct
        # TODO: Add validation for the value here
        setattr(ocp.cost, key, val)

    return ocp.cost


def define_acados_constraints(ocp: AcadosOcp, config: Config) -> AcadosOcpConstraints:
    # Check if ocp is an instance of AcadosOcp
    if not isinstance(ocp, AcadosOcp):
        raise TypeError("ocp must be an instance of AcadosOcp")

    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise TypeError("config must be an instance of Config")

    try:
        constraints = define_constraints(config)
    except Exception as e:
        # Handle or re-raise exception from define_constraints
        raise RuntimeError("Error in define_constraints: " + str(e))

    for key, val in constraints.items():
        # Check if the attribute exists in ocp.constraints
        if not hasattr(ocp.constraints, key):
            raise AttributeError(f"Attribute {key} does not exist in ocp.constraints")

        # Set the attribute, assuming the value is correct
        # TODO: Add validation for the value here
        setattr(ocp.constraints, key, val)

    return ocp.constraints


def ERK4(
    f: Union[cs.SX, cs.Function],
    x: Union[cs.SX, np.ndarray],
    u: Union[cs.SX, np.ndarray],
    p: Union[cs.SX, np.ndarray],
    h: float,
) -> Union[cs.SX, np.ndarray]:
    """
    Explicit Runge-Kutta 4 integrator

    TODO: Works for numeric values as well as for symbolic values. Type hinting is a bit misleading.

    Parameters:
        f: function to integrate
        x: state
        u: control
        p: parameters
        h: step size

        Returns:
            xf: integrated state
    """
    k1 = f(x, u, p)
    k2 = f(x + h / 2 * k1, u, p)
    k3 = f(x + h / 2 * k2, u, p)
    k4 = f(x + h * k3, u, p)
    xf = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return xf


ACADOS_MULTIPLIER_ORDER = [
    "lbu",
    "lbx",
    "lg",
    "lh",
    "lphi",
    "ubu",
    "ubx",
    "ug",
    "uh",
    "uphi",
    "lsbu",
    "lsbx",
    "lsg",
    "lsh",
    "lsphi",
    "usbu",
    "usbx",
    "usg",
    "ush",
    "usphi",
]


def rename_key_in_dict(d: dict, old_key: str, new_key: str):
    d[new_key] = d.pop(old_key)
    return d


def rename_item_in_list(lst: list, old_item: str, new_item: str):
    if old_item in lst:
        index_old = lst.index(old_item)
        lst[index_old] = new_item

    return lst


class LagrangeMultiplierMap(object):
    """
    Class to store dimensions of constraints
    """

    order: list = ACADOS_MULTIPLIER_ORDER

    idx_at_stage: list

    def __init__(self, constraints: AcadosOcpConstraints, N: int = 20):
        super().__init__()

        replacements = {
            0: [("lbx", "lbx_0"), ("ubx", "ubx_0")],
            N: [
                ("lbx", "lbx_e"),
                ("ubx", "ubx_e"),
                ("lg", "lg_e"),
                ("ug", "ug_e"),
                ("lh", "lh_e"),
                ("uh", "uh_e"),
                ("lphi", "lphi_e"),
                ("uphi", "uphi_e"),
                ("lsbx", "lsbx_e"),
                ("usbx", "usbx_e"),
                ("lsg", "lsg_e"),
                ("usg", "usg_e"),
                ("lsh", "lsh_e"),
                ("ush", "ush_e"),
                ("lsphi", "lsphi_e"),
                ("usphi", "usphi_e"),
            ],
        }

        idx_at_stage = [dict.fromkeys(self.order, 0) for _ in range(N + 1)]

        # Remove lbu, ubu from idx_at_stage at stage N
        idx_at_stage[N].pop("lbu")
        idx_at_stage[N].pop("ubu")

        if False:
            for stage, keys in replacements.items():
                for old_key, new_key in keys:
                    idx_at_stage[stage] = rename_key_in_dict(idx_at_stage[stage], old_key, new_key)

        # Loop over all constraints and count the number of constraints of each type. Store the indices in a dict.
        for stage, idx in enumerate(idx_at_stage):
            _start = 0
            _end = 0
            for attr in dir(constraints):
                if idx.keys().__contains__(attr):
                    _end += len(getattr(constraints, attr))
                    idx[attr] = slice(_start, _end)
                    _start = _end

        self.idx_at_stage = idx_at_stage

    def get_idx_at_stage(self, stage: int, field: str) -> slice:
        """
        Get the indices of the constraints of the given type at the given stage.

        Parameters:
            stage: stage index
            field: constraint type

        Returns:
            indices: slice object
        """
        return self.idx_at_stage[stage][field]

    def __call__(self, stage: int, field: str, lam: np.ndarray) -> np.ndarray:
        """
        Extract the multipliers at the given stage from the vector of multipliers.

        Parameters:
            stage: stage index
            field: constraint type
            lam: vector of multipliers

        Returns:
            lam: vector of multipliers at the given stage and of the given type
        """
        return lam[self.get_idx_at_stage(stage, field)]


def update_nlp_w(nlp: CasadiNLP, ocp_solver: AcadosOcpSolver) -> CasadiNLP:
    """
    Update the primal variables.

    Args:
        nlp: NLP to update.
        ocp_solver: OCP solver to get the solution from.

    Returns:
        nlp: Updated NLP.
    """

    for stage in range(ocp_solver.acados_ocp.dims.N):
        nlp.w.val["x", stage] = ocp_solver.get(stage, "x")
        nlp.w.val["u", stage] = ocp_solver.get(stage, "u")

    stage = ocp_solver.acados_ocp.dims.N

    nlp.w.val["x", stage] = ocp_solver.get(stage, "x")

    return nlp


def update_nlp_h(nlp: CasadiNLP):
    """
    Update the inequality constraints.

    Args:
        nlp: NLP to update.
        ocp_solver: OCP solver to get the solution from.

    Returns:
        h: Updated inequality constraints.
    """

    return nlp.h.fun(w=nlp.w.val, lbw=nlp.lbw.val, ubw=nlp.ubw.val)["h"]


def update_nlp_g(nlp: CasadiNLP):
    """
    Update the equality constraints.

    Args:
        nlp: NLP to update.
        ocp_solver: OCP solver to get the solution from.

    Returns:
        g: Updated equality constraints.
    """

    return nlp.g.fun(w=nlp.w.val, p=nlp.p.val)["g"]


def update_nlp_pi(nlp: CasadiNLP, ocp_solver: AcadosOcpSolver) -> CasadiNLP:
    """
    Update the multipliers associated with the equality constraints.

    Args:
        nlp: NLP to update.
        ocp_solver: OCP solver to get the solution from.

    Returns:
        nlp: Updated NLP.
    """

    for stage in range(ocp_solver.acados_ocp.dims.N):
        nlp.pi.val["pi", stage] = ocp_solver.get(stage, "pi")

    return nlp


def update_nlp_lam(nlp: CasadiNLP, ocp_solver: AcadosOcpSolver, multiplier_map: LagrangeMultiplierMap) -> CasadiNLP:
    """
    Update the multipliers associated with the inequality constraints.

    Args:
        nlp: NLP to update.
        ocp_solver: OCP solver to get the solution from.
        multiplier_map: Map of multipliers.

    Returns:
        nlp: Updated NLP.
    """

    for stage in range(ocp_solver.acados_ocp.dims.N):
        nlp.lam.val["lbx", stage] = multiplier_map(stage, "lbx", ocp_solver.get(stage, "lam"))
        nlp.lam.val["ubx", stage] = multiplier_map(stage, "ubx", ocp_solver.get(stage, "lam"))
        nlp.lam.val["lbu", stage] = multiplier_map(stage, "lbu", ocp_solver.get(stage, "lam"))
        nlp.lam.val["ubu", stage] = multiplier_map(stage, "ubu", ocp_solver.get(stage, "lam"))

    stage = ocp_solver.acados_ocp.dims.N

    nlp.lam.val["lbx", stage] = multiplier_map(stage, "lbx", ocp_solver.get(stage, "lam"))
    nlp.lam.val["ubx", stage] = multiplier_map(stage, "ubx", ocp_solver.get(stage, "lam"))

    return nlp


def update_nlp_R(nlp: CasadiNLP):
    """
    Update the KKT matrix R of the NLP.

    Args:
        nlp: NLP to update.

    Returns:
        nlp: Updated NLP.
    """

    return nlp.R.fun(
        w=nlp.w.val, lbw=nlp.lbw.val, ubw=nlp.ubw.val, pi=nlp.pi.val, lam=nlp.lam.val, p=nlp.p.val, dT=nlp.dT.val
    )["R"]


def update_nlp_L(nlp: CasadiNLP):
    """
    Update the Lagrangian of the NLP.

    Args:
        nlp: NLP to update.

    Returns:
        nlp: Updated NLP.
    """

    return nlp.L.fun(
        w=nlp.w.val, lbw=nlp.lbw.val, ubw=nlp.ubw.val, pi=nlp.pi.val, lam=nlp.lam.val, p=nlp.p.val, dT=nlp.dT.val
    )["L"]


def update_nlp_dL_dw(nlp: CasadiNLP):
    """
    Update the sensitivity of the Lagrangian with respect to the primal variables.

    Args:
        nlp: NLP to update.

    Returns:
        nlp: Updated NLP.
    """
    return nlp.dL_dw.fun(w=nlp.w.val, pi=nlp.pi.val, lam=nlp.lam.val, p=nlp.p.val, dT=nlp.dT.val)["dL_dw"]


def update_nlp_dL_dp(nlp: CasadiNLP):
    """
    Update the sensitivity of the Lagrangian with respect to the parameters.

    Args:
        nlp: NLP to update.

    Returns:
        nlp: Updated NLP.
    """
    return nlp.dL_dp.fun(w=nlp.w.val, pi=nlp.pi.val, p=nlp.p.val)["dL_dp"]


def update_nlp_dR_dz(nlp: CasadiNLP):
    """
    Update the sensitivity of the KKT matrix with respect to the primal-dual variables.

    Args:
        nlp: NLP to update.

    Returns:
        nlp: Updated NLP.
    """
    return nlp.dR_dz.fun(
        w=nlp.w.val, lbw=nlp.lbw.val, ubw=nlp.ubw.val, pi=nlp.pi.val, lam=nlp.lam.val, p=nlp.p.val, dT=nlp.dT.val
    )["dR_dz"]


def update_nlp_dR_dp(nlp: CasadiNLP) -> CasadiNLP:
    """
    Update the sensitivity of the KKT matrix with respect to the parameters.

    Args:
        nlp: NLP to update.

    Returns:
        nlp: Updated NLP.
    """
    return nlp.dR_dp.fun(w=nlp.w.val, pi=nlp.pi.val, p=nlp.p.val)["dR_dp"]


def test_nlp_is_primal_feasible(nlp: CasadiNLP, tol: float = 1e-6) -> bool:
    """
    Check if the primal variables are feasible.
    """
    # TODO: Add message to assert. Detail which constraint is violated.
    assert np.allclose(nlp.g.val, 0.0, atol=tol)
    assert np.all(nlp.h.val < tol)

    return True


def test_nlp_kkt_residual(nlp: CasadiNLP, tol: float = 1e-6) -> bool:
    # KKT residual check
    assert np.allclose(nlp.R.val, 0.0, atol=tol)

    return True


def test_nlp_stationarity(nlp: CasadiNLP, tol: float = 1e-6) -> bool:
    # Stationarity check
    assert np.allclose(nlp.dL_dw.val, 0.0, atol=tol)

    return True


def test_nlp_is_dual_feasible(nlp: CasadiNLP) -> bool:
    # Dual feasibility check
    assert np.all(nlp.lam.val.cat >= 0.0)

    return True


def test_nlp_satisfies_complementarity(nlp: CasadiNLP, tol: float = 1e-6) -> bool:
    # Complementary slackness check
    assert np.allclose(nlp.lam.val * nlp.h.val, 0.0, atol=tol)

    return True


def test_nlp_sanity(nlp: CasadiNLP, tol: float = 1e-6) -> bool:
    """
    Check if the NLP is feasible and satisfies the KKT conditions.
    """
    test_nlp_is_primal_feasible(nlp=nlp, tol=tol)
    test_nlp_kkt_residual(nlp=nlp, tol=tol)
    test_nlp_stationarity(nlp=nlp, tol=tol)
    test_nlp_is_dual_feasible(nlp=nlp)
    test_nlp_satisfies_complementarity(nlp=nlp, tol=tol)

    return True


def update_nlp(nlp: CasadiNLP, ocp_solver: AcadosOcpSolver, multiplier_map: LagrangeMultiplierMap) -> CasadiNLP:
    """
    Update the NLP with the solution of the OCP solver.

    Args:
        nlp: NLP to update.
        ocp_solver: OCP solver to get the solution from.
        multiplier_map: Map of multipliers.

    Returns:
        nlp: Updated NLP.
    """

    nlp = update_nlp_w(nlp=nlp, ocp_solver=ocp_solver)

    nlp = update_nlp_pi(nlp=nlp, ocp_solver=ocp_solver)

    nlp = update_nlp_lam(nlp=nlp, ocp_solver=ocp_solver, multiplier_map=multiplier_map)

    nlp.h.val = update_nlp_h(nlp=nlp)

    nlp.g.val = update_nlp_g(nlp=nlp)

    nlp.R.val = update_nlp_R(nlp)

    nlp.L.val = update_nlp_L(nlp)

    nlp.dL_dw.val = update_nlp_dL_dw(nlp)

    nlp.dL_dp.val = update_nlp_dL_dp(nlp)

    nlp.dR_dz.val = update_nlp_dR_dz(nlp)

    nlp.dR_dp.val = update_nlp_dR_dp(nlp)

    return nlp


class AcadosMPC(MPC):
    """docstring for CartpoleMPC."""

    _parameters: np.ndarray
    ocp_solver: AcadosOcpSolver
    nlp: CasadiNLP
    idx: dict
    muliplier_map: LagrangeMultiplierMap

    def __init__(self, config: Config, build: bool = True):
        super().__init__()

        ocp = AcadosOcp()

        ocp.model = define_acados_model(ocp=ocp, config=config)

        ocp.model.disc_dyn_expr = ERK4(
            cs.Function("ode", [ocp.model.x, ocp.model.u, ocp.model.p], [ocp.model.f_expl_expr]),
            ocp.model.x,
            ocp.model.u,
            ocp.model.p,
            config.ocp_options.tf / config.dimensions.N / config.ocp_options.sim_method_num_stages,
        )

        ocp.parameter_values = define_parameter_values(ocp=ocp, config=config)

        ocp.constraints = define_acados_constraints(ocp=ocp, config=config)

        ocp.dims = define_acados_dims(ocp=ocp, config=config)

        ocp.cost = define_acados_cost(ocp=ocp, config=config)

        ocp.cost.W_0 = ocp.cost.W
        ocp.dims.ny_0 = ocp.dims.ny
        ocp.cost.yref_0 = ocp.cost.yref

        ocp.model.cost_y_expr_0 = cs.vertcat(ocp.model.x, ocp.model.u)
        ocp.model.cost_y_expr = cs.vertcat(ocp.model.x, ocp.model.u)
        ocp.model.cost_y_expr_e = ocp.model.x

        # Build cost function

        ocp.solver_options = config.ocp_options

        ocp.code_export_directory = config.meta.code_export_dir

        self.ocp = ocp

        ocp_generate_external_functions(ocp, ocp.model)

        nlp, self.idx = build_nlp(ocp=self.ocp)

        # TODO: Move the constraints set to the corresponding MPC function. This function should only update the multipliers
        for stage in range(ocp.dims.N):
            # nlp.w.val["x", stage] = ocp_solver.get(stage, "x")
            # nlp.w.val["u", stage] = ocp_solver.get(stage, "u")

            if stage == 0:
                nlp.lbw.val["lbx", stage] = ocp.constraints.lbx_0
                nlp.ubw.val["ubx", stage] = ocp.constraints.ubx_0
            else:
                nlp.lbw.val["lbx", stage] = ocp.constraints.lbx
                nlp.ubw.val["ubx", stage] = ocp.constraints.ubx

            nlp.lbw.val["lbu", stage] = ocp.constraints.lbu
            nlp.ubw.val["ubu", stage] = ocp.constraints.ubu

        self.muliplier_map = LagrangeMultiplierMap(constraints=ocp.constraints, N=ocp.dims.N)

        nlp.L.sym = nlp.cost.sym + cs.dot(nlp.pi.sym, nlp.g.sym) + cs.dot(nlp.lam.sym, nlp.h.sym)
        nlp.L.fun = cs.Function(
            "L",
            [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym, nlp.dT.sym],
            [nlp.L.sym],
            ["w", "lbw", "ubw", "pi", "lam", "p", "dT"],
            ["L"],
        )

        nlp.dL_dw.sym = cs.jacobian(nlp.L.sym, nlp.w.sym)

        arg_list, name_list = find_nlp_entry_expr_dependencies(nlp, "dL_dw", ["w", "lbw", "ubw", "pi", "lam", "p", "dT"])

        nlp.dL_dw.fun = cs.Function(
            "dL_dw",
            arg_list,
            [nlp.dL_dw.sym],
            name_list,
            ["dL_dw"],
        )

        # Check if nlp.dL_dw.sym is function of nlp.w.sym

        nlp.dL_dp.sym = cs.jacobian(nlp.L.sym, nlp.p.sym)

        arg_list, name_list = find_nlp_entry_expr_dependencies(nlp, "dL_dp", ["w", "lbw", "ubw", "pi", "lam", "p", "dT"])

        nlp.dL_dp.fun = cs.Function(
            "dL_dp",
            arg_list,
            [nlp.dL_dp.sym],
            name_list,
            ["dL_dp"],
        )

        nlp.R.sym = cs.vertcat(cs.transpose(nlp.dL_dw.sym), nlp.g.sym, nlp.lam.sym * nlp.h.sym)

        arg_list, name_list = find_nlp_entry_expr_dependencies(nlp, "R", ["w", "lbw", "ubw", "pi", "lam", "p", "dT"])

        nlp.R.fun = cs.Function(
            "R",
            arg_list,
            [nlp.R.sym],
            name_list,
            ["R"],
        )

        z = cs.vertcat(nlp.w.sym, nlp.pi.sym, nlp.lam.sym)

        # Generate sensitivity of the KKT matrix with respect to primal-dual variables
        nlp.dR_dz.sym = cs.jacobian(nlp.R.sym, z)
        arg_list, name_list = find_nlp_entry_expr_dependencies(nlp, "dR_dz", ["w", "lbw", "ubw", "pi", "lam", "p", "dT"])
        nlp.dR_dz.fun = cs.Function(
            "dR_dz",
            arg_list,
            [nlp.dR_dz.sym],
            name_list,
            ["dR_dz"],
        )

        nlp.dR_dp.sym = cs.jacobian(nlp.R.sym, nlp.p.sym)
        arg_list, name_list = find_nlp_entry_expr_dependencies(nlp, "dR_dp", ["w", "lbw", "ubw", "pi", "lam", "p", "dT"])
        nlp.dR_dp.fun = cs.Function(
            "dR_dp",
            arg_list,
            [nlp.dR_dp.sym],
            name_list,
            ["dR_dp"],
        )

        self.nlp = nlp

        # Check path to config.meta.json file. Create the directory if it does not exist.
        if not os.path.exists(os.path.dirname(config.meta.json_file)):
            os.makedirs(os.path.dirname(config.meta.json_file))

        # TODO: Add config entries for json file and c_generated_code folder, and build, generate flags
        if build:
            self.ocp_solver = AcadosOcpSolver(ocp, json_file=config.meta.json_file)
        else:
            # Assumes json file and c_generated_code folder already exists
            self.ocp_solver = AcadosOcpSolver(ocp, json_file=config.meta.json_file, build=False, generate=False)

        self._parameters = ocp.parameter_values

    def set(self, stage, field, value):
        self.ocp_solver.set(stage, field, value)
        self.nlp.set(stage, field, value)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low = self.ocp.constraints.lbu
        high = self.ocp.constraints.ubu

        return 2.0 * ((action - low) / (high - low)) - 1.0

    # def update(self) -> int:
    #     status = self.ocp_solver.solve()

    #     self.nlp = update_nlp(self.nlp, self.ocp_solver, self.muliplier_map)

    #     test_nlp_sanity(self.nlp)

    #     return status

    def q_update(self, x0: np.ndarray, u0: np.ndarray) -> int:
        """
        Update the solution of the OCP solver.

        Args:
            x0: Initial state.

        Returns:
            status: Status of the solver.
        """
        # Set initial state
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        self.nlp.lbw.val["lbx", 0] = x0
        self.nlp.ubw.val["ubx", 0] = x0

        # Set initial action (needed for state-action value)
        self.ocp_solver.set(0, "u", u0)
        self.ocp_solver.set(0, "lbu", u0)
        self.ocp_solver.set(0, "ubu", u0)

        self.nlp.lbw.val["lbu", 0] = u0
        self.nlp.ubw.val["ubu", 0] = u0

        # Solve the optimization problem
        status = self.ocp_solver.solve()

        self.nlp = update_nlp(self.nlp, self.ocp_solver, self.muliplier_map)

        # test_nlp_sanity(self.nlp)

        return status

    def update(self, x0: np.ndarray) -> int:
        """
        Update the solution of the OCP solver.

        Args:
            x0: Initial state.

        Returns:
            status: Status of the solver.
        """
        # Set initial state
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        self.nlp.lbw.val["lbx", 0] = x0
        self.nlp.ubw.val["ubx", 0] = x0

        # Set initial action (needed for state-action value)
        # u0 = np.zeros((self.ocp.dims.nu,))
        # self.ocp_solver.set(0, "u", u0)
        # self.ocp_solver.set(0, "lbu", u0)
        # self.ocp_solver.set(0, "ubu", u0)

        # Solve the optimization problem
        status = self.ocp_solver.solve()

        self.nlp = update_nlp(self.nlp, self.ocp_solver, self.muliplier_map)

        # test_nlp_sanity(self.nlp)

        return status

    def get_dL_dp(self) -> np.ndarray:
        """
        Get the value of the sensitivity of the Lagrangian with respect to the parameters.

        Returns:
            dL_dp: Sensitivity of the Lagrangian with respect to the parameters.
        """
        return self.nlp.dL_dp.val

    def get_L(self) -> float:
        """
        Get the value of the Lagrangian.

        Returns:
            L: Lagrangian.
        """
        return self.nlp.L.val

    def get_V(self) -> float:
        """
        Get the value of the value function.

        Assumes OCP is solved for state.

        Returns:
            V: Value function.
        """
        return self.ocp_solver.get_cost()

    def get_dV_dp(self) -> float:
        """
        Get the value of the sensitivity of the value function with respect to the parameters.

        Assumes OCP is solved for state and parameters.

        Returns:
            dV_dp: Sensitivity of the value function with respect to the parameters.
        """
        return self.get_dL_dp()

    def get_Q(self) -> float:
        """
        Get the value of the state-action value function.

        Assumes OCP is solved for state and action.

        Returns:
            Q: State-action value function.
        """
        return self.ocp_solver.get_cost()

    def get_dQ_dp(self) -> float:
        """
        Get the value of the sensitivity of the state-action value function with respect to the parameters.

        Assumes OCP is solved for state, action and parameters.

        Returns:
            dQ_dp: Sensitivity of the state-action value function with respect to the parameters.
        """
        return self.get_dL_dp()

    def get_action(self, x0: np.ndarray) -> np.ndarray:
        """
        Update the solution of the OCP solver.

        Args:
            x0: Initial state.

        Returns:
            u: Optimal control action.
        """
        # Set initial state
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        self.nlp.lbw.val["lbx", 0] = x0
        self.nlp.ubw.val["ubx", 0] = x0

        # Solve the optimization problem
        self.ocp_solver.solve()

        # Get solution
        action = self.ocp_solver.get(0, "u")

        # Scale to [-1, 1] for gym
        action = self.scale_action(action)

        return action

    def get_parameters(self) -> np.ndarray:
        return self._parameters

    def get_predicted_state_trajectory(self) -> np.ndarray:
        """
        Get the predicted state trajectory.

        Returns:
            x: Predicted state trajectory.
        """
        x = np.zeros((self.ocp.dims.N + 1, self.ocp.dims.nx))

        for i in range(self.ocp.dims.N + 1):
            x[i, :] = self.ocp_solver.get(i, "x")

        return x

    def get_predicted_control_trajectory(self) -> np.ndarray:
        """
        Get the predicted control trajectory.

        Returns:
            u: Predicted control trajectory.
        """
        u = np.zeros((self.ocp.dims.N, self.ocp.dims.nu))

        for i in range(self.ocp.dims.N):
            u[i, :] = self.ocp_solver.get(i, "u")

        return u

    def plot_prediction(self) -> None:
        """
        Plot the predicted trajectory.
        """

        x = self.get_predicted_state_trajectory()
        u = self.get_predicted_control_trajectory()

        _, ax = plt.subplots(self.ocp.dims.nx + self.ocp.dims.nu, 1, figsize=(10, 7))

        for i in range(self.ocp.dims.nx):
            ax[i].plot(x[:, i], "-o")
            ax[i].grid(True)
            ax[i].set_ylabel(f"x_{i}")

        # Make a stairs plot for u
        ax[self.ocp.dims.nx].step(np.arange(0, u.shape[0]), u.flatten(), where="post")
        ax[self.ocp.dims.nx].grid(True)
        ax[self.ocp.dims.nx].set_ylabel("u")

        # Draw bounds
        lbx = self.ocp.constraints.lbx
        ubx = self.ocp.constraints.ubx
        lbu = self.ocp.constraints.lbu
        ubu = self.ocp.constraints.ubu

        for i in range(self.ocp.dims.nx):
            ax[i].plot(np.ones_like(x[:, i]) * lbx[i], "--", color="grey")
            ax[i].plot(np.ones_like(x[:, i]) * ubx[i], "--", color="gray")

        ax[self.ocp.dims.nx].plot(np.ones_like(u) * lbu, "--", color="gray")
        ax[self.ocp.dims.nx].plot(np.ones_like(u) * ubu, "--", color="gray")

        plt.show()

    def print_header(self) -> None:
        """
        Print the header for the data table.
        """
        print("{:>8} {:>8} {:>8} {:>8} {:>8}".format("x", "x_dot", "theta", "theta_dot", "u"))

    def print_data(self, x: np.ndarray, u: np.ndarray) -> None:
        """
        Print the data table.

        Args:
            x: State.
            u: Control.
        """
        print("{:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f}".format(x[0], x[1], x[2], x[3], u))
