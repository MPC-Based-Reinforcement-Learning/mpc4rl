from typing import Union
import numpy as np
import casadi as cs
from casadi.tools import struct_symMX, struct_MX, struct_symSX, struct_SX, entry
from casadi.tools import *

from rlmpc.common.mpc import MPC
from rlmpc.mpc.utils import ERK4

import matplotlib.pyplot as plt

from rlmpc.mpc.cartpole.common import (
    Config,
    define_model_expressions,
    define_dimensions,
    define_cost,
    define_constraints,
    # define_parameters,
)

from rlmpc.mpc.cartpole.acados import (
    define_acados_constraints,
    define_acados_cost,
    define_acados_dims,
    define_acados_model,
)

from acados_template import (
    AcadosOcpOptions,
    AcadosOcpDims,
    AcadosOcp,
    AcadosOcpConstraints,
    AcadosOcpCost,
    AcadosModel,
)


class CasadiNLP:
    """docstring for CasadiNLP."""

    cost: Union[cs.SX, cs.MX]
    w: Union[cs.SX, cs.MX]
    w0: Union[list, np.ndarray]
    lbw: Union[list, np.ndarray]
    ubw: Union[list, np.ndarray]
    g: Union[cs.SX, cs.MX]
    lbg: Union[list, np.ndarray]
    ubg: Union[list, np.ndarray]
    p: Union[cs.SX, cs.MX]
    f_disc: cs.Function
    shooting: struct_symSX

    def __init__(self):
        super().__init__()

        self.cost = None
        self.w = None
        self.w0 = None
        self.lbw = None
        self.ubw = None
        self.g = None
        self.lbg = None
        self.ubg = None
        self.p = None
        self.f_disc = None


class CasadiModel(AcadosModel):
    """docstring for CasadiModel."""

    def __init__(self):
        super().__init__()


class CasadiOcpDims(AcadosOcpDims):
    """docstring for CasadiOcpDims."""

    def __init__(self):
        super().__init__()


class CasadiOcpConstraints(AcadosOcpConstraints):
    """docstring for CasadiOcpConstraints."""

    Jbx: np.ndarray

    def __init__(self):
        super().__init__()


class CasadiOcpCost(AcadosOcpCost):
    """docstring for CasadiOcpCost."""

    def __init__(self):
        super().__init__()


class CasadiOcpOptions(AcadosOcpOptions):
    """docstring for CasadiOcpOptions."""

    def __init__(self):
        super().__init__()


class CasadiOcp(AcadosOcp):
    """docstring for CasadiOcp."""

    model: CasadiModel
    dims: CasadiOcpDims
    constraints: CasadiOcpConstraints
    cost: CasadiOcpCost
    solver_options: CasadiOcpOptions

    def __init__(self):
        super().__init__()

        self.model = CasadiModel()
        self.dims = CasadiOcpDims()
        self.constraints = CasadiOcpConstraints()
        self.cost = CasadiOcpCost()
        self.solver_options = CasadiOcpOptions()


def idx_to_J(shape: tuple, idx: np.ndarray) -> np.ndarray:
    J = np.zeros(shape)
    for k, idx in enumerate(idx):
        J[k, idx] = 1
    return J


def get_Jbx(ocp: CasadiOcp) -> np.ndarray:
    # Jbx = np.zeros((ocp.constraints.idxbx.shape[0], ocp.dims.nx))
    # for k, idx in enumerate(ocp.constraints.idxbx):
    #     Jbx[k, idx] = 1
    # return Jbx
    return idx_to_J((ocp.constraints.idxbx.shape[0], ocp.dims.nx), ocp.constraints.idxbx)


def get_Jbu(ocp: CasadiOcp) -> np.ndarray:
    # Jbu = np.zeros((ocp.constraints.idxbu.shape[0], ocp.dims.nu))
    # for k, idx in enumerate(ocp.constraints.idxbu):
    #     Jbu[k, idx] = 1
    # return Jbu
    return idx_to_J((ocp.constraints.idxbu.shape[0], ocp.dims.nu), ocp.constraints.idxbu)


def get_Jsbx(ocp: CasadiOcp) -> np.ndarray:
    # Jsbx = np.zeros((ocp.constraints.idxsbx.shape[0], ocp.dims.nx))
    # for k, idx in enumerate(ocp.constraints.idxsbx):
    #     Jsbx[k, idx] = 1
    # return Jsbx
    return idx_to_J((ocp.constraints.idxsbx.shape[0], ocp.dims.nx), ocp.constraints.idxsbx)


def get_Jsbu(ocp: CasadiOcp) -> np.ndarray:
    # Jsbu = np.zeros((ocp.constraints.idxsbu.shape[0], ocp.dims.nu))
    # for k, idx in enumerate(ocp.constraints.idxsbu):
    #     Jsbu[k, idx] = 1
    # return Jsbu
    return idx_to_J((ocp.constraints.idxsbu.shape[0], ocp.dims.nu), ocp.constraints.idxsbu)


class CasadiOcpSolver:
    """docstring for CasadiOcp."""

    # _model: dict
    # _cost: cs.Function
    # _constraints: cs.Function

    ocp: CasadiOcp
    nlp: CasadiNLP
    p: np.ndarray
    nlp_solution: dict

    # Use generate and build mehods to implement jit compilation
    @classmethod
    def generate(cls, casadi_ocp: CasadiOcp):
        pass

    @classmethod
    def build(cls, casadi_ocp: CasadiOcp):
        pass

    def __init__(self, _ocp: CasadiOcp):
        super().__init__()

        # self._ocp.model, self._ocp.parameter_values = define_acados_model(
        #     ocp=self._ocp, config=config
        # )

        self.ocp = _ocp

        self.nlp = build_nlp_with_slack(self.ocp)

        # Create an NLP solver
        self.nlp_solver = cs.nlpsol(
            "solver",
            "ipopt",
            {"f": self.nlp.cost, "x": self.nlp.w, "p": self.nlp.p, "g": self.nlp.g},
        )

        if False:
            self.nlp_solution = self.nlp_solver(
                x0=self.nlp.w0,
                p=1.0,
                lbg=self.nlp.lbg,
                ubg=self.nlp.ubg,
                lbx=self.nlp.lbw,
                ubx=self.nlp.ubw,
            )

            x_opt = self.nlp_solution["x"]

            x = x_opt[0 :: (self.ocp.dims.nx + self.ocp.dims.nu)]
            v = x_opt[1 :: (self.ocp.dims.nx + self.ocp.dims.nu)]
            theta = x_opt[2 :: (self.ocp.dims.nx + self.ocp.dims.nu)]
            dtheta = x_opt[3 :: (self.ocp.dims.nx + self.ocp.dims.nu)]
            u = x_opt[4 :: (self.ocp.dims.nx + self.ocp.dims.nu)]

            fig, axes = plt.subplots(5, 1, figsize=(10, 10))
            for k, ax in enumerate(axes):
                ax.plot(x_opt[k :: (self.ocp.dims.nx + self.ocp.dims.nu)].full().flatten())

            plt.show()

        # build_lagrange_function(out, self.ocp)

        self.nlp_solution = None
        self.w_opt = None

    def set(self, stage_, field_, value_):
        """
        Set numerical data inside the solver.

            :param stage: integer corresponding to shooting node
            :param field: string in ['x', 'u', 'pi', 'lam', 't', 'p', 'xdot_guess', 'z_guess']

            .. note:: regarding lam, t: \n
                    the inequalities are internally organized in the following order: \n
                    [ lbu lbx lg lh lphi ubu ubx ug uh uphi; \n
                      lsbu lsbx lsg lsh lsphi usbu usbx usg ush usphi]

            .. note:: pi: multipliers for dynamics equality constraints \n
                      lam: multipliers for inequalities \n
                      t: slack variables corresponding to evaluation of all inequalities (at the solution) \n
                      sl: slack variables of soft lower inequality constraints \n
                      su: slack variables of soft upper inequality constraints \n
        """
        cost_fields = ["y_ref", "yref"]
        constraints_fields = ["lbx", "ubx", "lbu", "ubu"]
        out_fields = ["x", "u", "pi", "lam", "t", "z", "sl", "su"]
        mem_fields = ["xdot_guess", "z_guess"]

        if not isinstance(stage_, int):
            raise Exception("stage should be integer.")
        elif stage_ < 0 or stage_ > self.ocp.dims.N:
            raise Exception(f"stage should be in [0, N], got {stage_}")

        # cast value_ to avoid conversion issues
        if isinstance(value_, (float, int)):
            value_ = np.array([value_])
        value_ = value_.astype(float)

        # treat parameters separately
        if field_ == "p":
            self.p = value_

    def cost_set(self, stage_, field_, value_):
        """
        The following is a modified implementation of parts of acados_template/acados_ocp_solver.py

        Set numerical data in the cost module of the solver.

            :param stage: integer corresponding to shooting node
            :param field: string, e.g. 'yref', 'W', 'ext_cost_num_hess', 'zl', 'zu', 'Zl', 'Zu'
            :param value: of appropriate size
        """

        # cast value_ to avoid conversion issues
        if isinstance(value_, (float, int)):
            value_ = np.array([value_])
        value_ = value_.astype(float)

        if not isinstance(stage_, int):
            raise Exception("stage should be integer.")
        elif stage_ < 0 or stage_ > self.N:
            raise Exception(f"stage should be in [0, N], got {stage_}")

        if field_ == "yref":
            raise Exception("yref is not implemented yet.")
        elif field_ == "W":
            raise Exception("W is not implemented yet.")
        elif field_ == "ext_cost_num_hess":
            raise Exception("ext_cost_num_hess is not implemented yet.")
        elif field_ == "zl":
            raise Exception("zl is not implemented yet.")
        elif field_ == "zu":
            raise Exception("zu is not implemented yet.")
        elif field_ == "Zl":
            raise Exception("Zl is not implemented yet.")
        elif field_ == "Zu":
            raise Exception("Zu is not implemented yet.")

    def constraints_set(self, stage_, field_, value_):
        """
        The following is a modified implementation of parts of acados_template/acados_ocp_solver.py

        Set numerical data in the constraint module of the solver.

            :param stage: integer corresponding to shooting node
            :param field: string in ['lbx', 'ubx', 'lbu', 'ubu', 'lg', 'ug', 'lh', 'uh', 'uphi', 'C', 'D']
            :param value: of appropriate size
        """

        # cast value_ to avoid conversion issues
        if isinstance(value_, (float, int)):
            value_ = np.array([value_])
        value_ = value_.astype(float)

        if not isinstance(stage_, int):
            raise Exception("stage should be integer.")
        elif stage_ < 0 or stage_ > self.ocp.dims.N:
            raise Exception(f"stage should be in [0, N], got {stage_}")

        if field_ == "lbx":
            self.nlp.lbw["X", stage_] = value_
        elif field_ == "ubx":
            self.nlp.ubw["X", stage_] = value_
        elif field_ == "lbu":
            self.nlp.lbw["U", stage_] = value_
        elif field_ == "ubu":
            self.nlp.ubw["U", stage_] = value_
        elif field_ == "lg":
            raise Exception("lg is not implemented yet.")
        elif field_ == "ug":
            raise Exception("ug is not implemented yet.")
        elif field_ == "lh":
            raise Exception("lh is not implemented yet.")
        elif field_ == "uh":
            raise Exception("uh is not implemented yet.")
        elif field_ == "uphi":
            raise Exception("uphi is not implemented yet.")
        elif field_ == "C":
            raise Exception("C is not implemented yet.")
        elif field_ == "D":
            raise Exception("D is not implemented yet.")

    def get(self, stage_: int, field_: str) -> np.ndarray:
        """
        The following is a modified implementation of parts of acados_template/acados_ocp_solver.py

        Get the last solution of the solver:

            :param stage: integer corresponding to shooting node
            :param field: string in ['x', 'u', 'z', 'pi', 'lam', 't', 'sl', 'su',]

            .. note:: regarding lam, t: \n
                    the inequalities are internally organized in the following order: \n
                    [ lbu lbx lg lh lphi ubu ubx ug uh uphi; \n
                      lsbu lsbx lsg lsh lsphi usbu usbx usg ush usphi]

            .. note:: pi: multipliers for dynamics equality constraints \n
                      lam: multipliers for inequalities \n
                      t: slack variables corresponding to evaluation of all inequalities (at the solution) \n
                      sl: slack variables of soft lower inequality constraints \n
                      su: slack variables of soft upper inequality constraints \n

        Get a field of the OCP solver.

        Args:
            stage: Stage index.
            field: Field name.

        Returns:
            Field value.
        """

        out_fields = ["x", "u", "z", "pi", "lam", "t", "sl", "su"]

        all_fields = out_fields

        if field_ not in all_fields:
            raise Exception(
                f"CasadiOcpSolver.get(stage={stage_}, field={field_}): '{field_}' is an invalid argument.\
                    \n Possible values are {all_fields}."
            )

        if not isinstance(stage_, int):
            raise Exception(
                f"CasadiOcpSolver.get(stage={stage_}, field={field_}): stage index must be an integer, got type {type(stage_)}."
            )

        # Check if self.nlp_solution is NoneType
        if self.nlp_solution is None:
            raise Exception(
                f"CasadiOcpSolver.get(stage={stage_}, field={field_}): self.nlp_solution is NoneType.\
                    \n Please run self.solve() first."
            )

        if stage_ < 0 or stage_ > self.ocp.dims.N:
            raise Exception(
                f"AcadosOcpSolver.get(stage={stage_}, field={field_}): stage index must be in [0, {self.ocp.dims.N}], got: {stage_}."
            )

        if stage_ == self.ocp.dims.N and field_ == "pi":
            raise Exception(
                f"AcadosOcpSolver.get(stage={stage_}, field={field_}): field '{field_}' does not exist at final stage {stage_}."
            )

        if field_ == "x":
            return self.w_opt["X", stage_].full()

        if field_ == "u":
            return self.w_opt["U", stage_].full()

        if field_ == "s":
            return self.w_opt["S", stage_].full()

        if field_ == "pi":
            # start_idx = (
            #     stage_ * (self.ocp.dims.nx + self.ocp.dims.nu) + self.ocp.dims.nx
            # )
            # end_idx = start_idx + self.ocp.dims.nx

            # return self.nlp_solution["lam_g"][start_idx:end_idx].full().flatten()
            raise NotImplementedError()

        raise NotImplementedError()

    def solve(self) -> np.ndarray:
        """
        Solve the OCP.

        Args:
            x0: Initial state.
            p: Parameters.

        Returns:
            Solution.
        """

        self.nlp_solution = self.nlp_solver(
            x0=self.nlp.w0,
            lbx=self.nlp.lbw,
            ubx=self.nlp.ubw,
            lbg=self.nlp.lbg,
            ubg=self.nlp.ubg,
            p=self.p,
        )

        self.w_opt = self.nlp.w(self.nlp_solution["x"])

        return self.nlp_solution, self.w_opt


def define_casadi_dims(model: dict, config: Config) -> dict:
    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise TypeError("config must be an instance of Config")

    try:
        dims = define_dimensions(config)
    except Exception as e:
        # Handle or re-raise exception from define_constraints
        raise RuntimeError("Error in define_casadi_dims: " + str(e))

    # for key, val in dims.items():

    # Set the attribute, assuming the value is correct
    # TODO: Add validation for the value here
    # setattr(ocp.dims, key, val)

    return dims


def build_discrete_dynamics_functions(
    config: Config,
) -> tuple((Union[cs.SX, cs.MX], cs.Function, cs.Function)):
    """
    Build the discrete dynamics functions for the OCP.

    Parameters:
        acados_ocp: acados OCP object

    Returns:
        fun_f: discrete dynamics function
        fun_df_dp: derivative of discrete dynamics function with respect to parameters
    """

    model, parameter_values = define_model_expressions(config=config)

    dims = define_dimensions(config=config)
    dims["np"] = parameter_values.shape[0]

    ocp_options = config.ocp_options
    # p = define_parameters(config=config)

    x = model["x"]
    u = model["u"]
    p = model["p"]
    f_expl = model["f_expl_expr"]
    f = cs.Function("f_expl", [x, u, p], [f_expl], ["x", "u", "p"], ["xf"])

    h = ocp_options.tf / dims["N"]

    # TODO: Add support for other integrator types
    # TODO: Use the integrator type from the config file (independent of acados)
    # Next state expression.

    if ocp_options.integrator_type == "ERK" and ocp_options.sim_method_num_stages == 4:
        xf = ERK4(f, x, u, p, h)
    else:
        raise NotImplementedError("Only ERK4 integrator types are supported at the moment.")

    # Integrator function.
    # fun_f = cs.Function("f", [x, u, p], [xf], ["x", "u", "p"], ["xf"])

    # Jacobian of the integrator function with respect to the parameters.
    df_dp = cs.Function("df_dp", [x, u, p], [cs.jacobian(xf, p)])

    return (xf, f, df_dp)


def define_discrete_dynamics_function(ocp: CasadiOcp) -> cs.Function:
    # Step size.
    h = ocp.solver_options.tf / ocp.dims.N / ocp.solver_options.sim_method_num_stages

    x = ocp.model.x
    u = ocp.model.u
    p = ocp.model.p
    f_expl = ocp.model.f_expl_expr

    # Continuous dynamics function.
    f = cs.Function("f", [x, u, p], [f_expl])

    # TODO: Add support for other integrator types
    # Integrate given amount of steps over the interval with Runge-Kutta 4 scheme
    if ocp.solver_options.integrator_type == "ERK":
        for _ in range(ocp.solver_options.sim_method_num_steps):
            k1 = f(x, u, p)
            k2 = f(x + h / 2 * k1, u, p)
            k3 = f(x + h / 2 * k2, u, p)
            k4 = f(x + h * k3, u, p)

            xnext = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return cs.Function("F", [x, u, p], [xnext])
    else:
        raise NotImplementedError("Only ERK integrator types are supported at the moment.")


# def define_stage_cost_function(ocp: CasadiOcp) -> cs.Function:
#     model = ocp.model
#     cost = ocp.cost

#     # TODO: Add support for other cost types
#     # TODO: Add yref as a parameter
#     if cost.cost_type == "LINEAR_LS":
#         Vx = cost.Vx
#         Vu = cost.Vu
#         yref = cost.yref.reshape(-1, 1)
#         # W = cost.W

#         x = model.x
#         u = model.u
#         y = cs.mtimes([Vx, x]) + cs.mtimes([Vu, u])

#         cost = 0
#         cost += cs.mtimes([(y - yref).T, cost.W, (y - yref)])
#         cost +=

#         # stage_cost_function = cs.Function(
#         #     "stage_cost",
#         #     [x, u, sl, su],
#         #     [cs.mtimes([(y - yref).T, W, (y - yref)])],
#         # )

#         return stage_cost_function


def define_stage_cost_function(
    x: cs.SX,
    u: cs.SX,
    sl: cs.SX,
    su: cs.SX,
    yref: cs.SX,
    W: np.ndarray,
    Zl: np.ndarray,
    Zu: np.ndarray,
    zl: np.ndarray,
    zu: np.ndarray,
    cost: CasadiOcpCost,
) -> cs.SX:
    if cost.cost_type == "LINEAR_LS":
        y = cs.mtimes([cost.Vx, x]) + cs.mtimes([cost.Vu, u])

        cost = 0
        cost += cs.mtimes([(y - yref).T, W, (y - yref)])
        cost += cs.mtimes([sl.T, Zl, sl])
        cost += cs.mtimes([su.T, Zu, su])
        cost += cs.mtimes([sl.T, zl])
        cost += cs.mtimes([su.T, zu])

        stage_cost = cs.Function(
            "l",
            # [x, u, sl, su, yref, W, Zl, Zu, zl, zu],
            [x, u, sl, su],
            [cost],
            ["x", "u", "sl", "su"],
            ["out"],
        )

        return stage_cost
    else:
        raise NotImplementedError("Only LINEAR_LS cost types are supported at the moment.")


# def define_terminal_cost_function(ocp: CasadiOcp) -> cs.Function:
#     model = ocp.model
#     cost = ocp.cost

#     # TODO: Add support for other cost types
#     # TODO: Add yref as a parameter
#     if cost.cost_type == "LINEAR_LS":
#         Vx_e = cost.Vx_e
#         yref_e = cost.yref_e.reshape(-1, 1)
#         W_e = cost.W_e

#         x = model.x
#         y_e = cs.mtimes([Vx_e, x])

#         terminal_cost_function = cs.Function(
#             "stage_cost",
#             [x],
#             [cs.mtimes([(y_e - yref_e).T, W_e, (y_e - yref_e)])],
#         )

#         return terminal_cost_function


def define_terminal_cost_function(
    x_e: cs.SX,
    sl_e: cs.SX,
    su_e: cs.SX,
    yref_e: cs.SX,
    W_e: np.ndarray,
    Zl_e: np.ndarray,
    Zu_e: np.ndarray,
    zl_e: np.ndarray,
    zu_e: np.ndarray,
    cost: CasadiOcpCost,
) -> cs.SX:
    if cost.cost_type == "LINEAR_LS":
        y_e = cs.mtimes([cost.Vx_e, x_e])

        cost = 0
        cost += cs.mtimes([(y_e - yref_e).T, W_e, (y_e - yref_e)])
        cost += cs.mtimes([sl_e.T, Zl_e, sl_e])
        cost += cs.mtimes([su_e.T, Zu_e, su_e])
        cost += cs.mtimes([sl_e.T, zl_e])
        cost += cs.mtimes([su_e.T, zu_e])

        terminal_cost = cs.Function(
            "m",
            # [x, u, sl, su, yref, W, Zl, Zu, zl, zu],
            [x_e, sl_e, su_e],
            [cost],
            ["x_e", "sl_e", "su_e"],
            ["out"],
        )

        return terminal_cost
    else:
        raise NotImplementedError("Only LINEAR_LS cost types are supported at the moment.")


def build_nlp_with_slack(ocp: CasadiOcp) -> CasadiNLP:
    phi = define_discrete_dynamics_function(ocp)

    constraints = ocp.constraints

    state_labels = ocp.model.x.str().strip("[]").split(", ")
    control_labels = ocp.model.u.str().strip("[]").split(", ")

    slack_labels = dict()
    slack_labels["Slbx"] = [state_labels[idx] for idx in constraints.idxsbx]
    slack_labels["Subx"] = [state_labels[idx] for idx in constraints.idxsbx]
    # slack_labels["Slbu"] = [control_labels[idx] for idx in constraints.idxsbu]
    # slack_labels["Subu"] = [control_labels[idx] for idx in constraints.idxsbu]

    w = struct_symSX(
        [
            (
                entry(
                    "X",
                    repeat=ocp.dims.N,
                    struct=struct_symSX(state_labels),
                ),
                entry(
                    "U",
                    repeat=ocp.dims.N - 1,
                    struct=struct_symSX(control_labels),
                ),
                # entry("S", repeat=ocp.dims.N, struct=struct_symSX(slack_labels)),
                # entry("S", repeat=ocp.dims.N, struct=S),
                entry(
                    "Slbx",
                    repeat=ocp.dims.N,
                    struct=struct_symSX(slack_labels["Slbx"]),
                ),
                # entry(
                #     "Slbu",
                #     repeat=ocp.dims.N - 1,
                #     struct=struct_symSX(slack_labels["Slbu"]),
                # ),
                # entry("Slh", struct=struct_symSX([])),
                entry(
                    "Subx",
                    repeat=ocp.dims.N,
                    struct=struct_symSX(slack_labels["Subx"]),
                ),
                # entry(
                #     "Subu",
                #     repeat=ocp.dims.N - 1,
                #     struct=struct_symSX(slack_labels["Subu"]),
                # ),
                # entry("Suh", struct=struct_symSX([])),
            )
        ]
    )

    # Index of box constraints
    idxbx = constraints.idxbx
    idxbu = constraints.idxbu

    # Index of soft box constraints
    idxsbx = constraints.idxsbx
    idxsbu = constraints.idxsbu

    # Index of hard box constraints
    idxhbx = [idx for idx in idxbx if idx not in idxsbx]
    idxhbu = [idx for idx in idxbu if idx not in idxsbu]

    # Hard box constraints
    lbx = [constraints.lbx[i] if i in idxhbx else -np.inf for i in range(ocp.dims.nx)]
    lbu = [constraints.lbu[i] if i in idxhbu else -np.inf for i in range(ocp.dims.nu)]
    ubx = [constraints.ubx[i] if i in idxhbx else np.inf for i in range(ocp.dims.nx)]
    ubu = [constraints.ubu[i] if i in idxhbu else np.inf for i in range(ocp.dims.nu)]

    # Soft box constraints
    lsbx = constraints.lbx[constraints.idxsbx]
    lsbu = constraints.lbu[constraints.idxsbu]
    usbx = constraints.ubx[constraints.idxsbx]
    usbu = constraints.ubu[constraints.idxsbu]

    lbw = w(0)
    lbw["X", lambda x: cs.vertcat(*x)] = np.tile(lbx, (1, ocp.dims.N))
    lbw["U", lambda x: cs.vertcat(*x)] = np.tile(lbu, (1, ocp.dims.N - 1))
    lbw["Slbx", lambda x: cs.vertcat(*x)] = np.tile([0 for _ in constraints.idxsbx], (1, ocp.dims.N))
    lbw["Subx", lambda x: cs.vertcat(*x)] = np.tile([0 for _ in constraints.idxsbx], (1, ocp.dims.N))

    ubw = w(0)
    ubw["X", lambda x: cs.vertcat(*x)] = np.tile(ubx, (1, ocp.dims.N))
    ubw["U", lambda x: cs.vertcat(*x)] = np.tile(ubu, (1, ocp.dims.N - 1))
    ubw["Slbx", lambda x: cs.vertcat(*x)] = np.tile([np.inf for _ in constraints.idxsbx], (1, ocp.dims.N))
    ubw["Subx", lambda x: cs.vertcat(*x)] = np.tile([np.inf for _ in constraints.idxsbx], (1, ocp.dims.N))

    x0 = ocp.constraints.lbx_0.tolist()
    u0 = 0

    w0 = w(0)
    w0["X", lambda x: cs.vertcat(*x)] = np.tile(x0, (1, ocp.dims.N))
    w0["U", lambda x: cs.vertcat(*x)] = np.tile(u0, (1, ocp.dims.N - 1))
    w0["Slbx", lambda x: cs.vertcat(*x)] = np.tile([0 for _ in constraints.idxsbx], (1, ocp.dims.N))
    w0["Subx", lambda x: cs.vertcat(*x)] = np.tile([0 for _ in constraints.idxsbx], (1, ocp.dims.N))

    # Parameter vector
    # TODO: Add support for multivariable parameters
    p = ocp.model.p

    # Build other constraints (dynamics, relaxed box constraints, etc.)
    g = []
    lbg = []
    ubg = []

    # Alias for the states, control, and slack variables
    (X, U, Slbx, Subx) = w[...]

    # S = cs.vertcat(Sl, Su)

    for i in range(ocp.dims.N - 1):
        # Add dynamics constraints
        g.append(X[i + 1] - phi(X[i], U[i], p))
        lbg.append([0 for _ in range(ocp.dims.nx)])
        ubg.append([0 for _ in range(ocp.dims.nx)])

        # Add relaxed box constraints for lower bounds
        g.append(lsbx - X[i][idxsbx] - Slbx[i])
        lbg.append([-cs.inf for _ in idxsbx])
        ubg.append([0 for _ in idxsbx])

        # Add relaxed box constraints for upper bounds
        g.append(-usbx + X[i][idxsbx] - Subx[i])
        lbg.append([-cs.inf for _ in idxsbx])
        ubg.append([0 for _ in idxsbx])

    g = cs.vertcat(*g)
    lbg = cs.vertcat(*lbg)
    ubg = cs.vertcat(*ubg)

    Sl = Slbx
    Su = Subx

    stage_cost_function = define_stage_cost_function(
        x=ocp.model.x,
        u=ocp.model.u,
        sl=cs.SX.sym("sl", idxsbx.shape[0]),
        su=cs.SX.sym("su", idxsbx.shape[0]),
        yref=ocp.cost.yref,
        W=ocp.cost.W,
        Zl=ocp.cost.Zl,
        Zu=ocp.cost.Zu,
        zl=ocp.cost.zl,
        zu=ocp.cost.zu,
        cost=ocp.cost,
    )

    terminal_cost_function = define_terminal_cost_function(
        x_e=ocp.model.x,
        sl_e=cs.SX.sym("sl_e", idxsbx.shape[0]),
        su_e=cs.SX.sym("su_e", idxsbx.shape[0]),
        yref_e=ocp.cost.yref_e,
        W_e=ocp.cost.W_e,
        Zl_e=ocp.cost.Zl_e,
        Zu_e=ocp.cost.Zu_e,
        zl_e=ocp.cost.zl_e,
        zu_e=ocp.cost.zu_e,
        cost=ocp.cost,
    )

    cost = 0
    # Build the cost function
    for i in range(ocp.dims.N - 1):
        cost += stage_cost_function(X[i], U[i], Sl[i], Su[i])

    # Add terminal cost
    cost += terminal_cost_function(X[ocp.dims.N - 1], Sl[ocp.dims.N - 1], Su[ocp.dims.N - 1])

    nlp = CasadiNLP()
    nlp.cost = cost
    nlp.w = w
    nlp.w0 = w0
    nlp.lbw = lbw
    nlp.ubw = ubw
    nlp.g = g
    nlp.lbg = lbg
    nlp.ubg = ubg
    nlp.p = p
    nlp.f_disc = phi

    return nlp


def build_nlp(ocp: CasadiOcp) -> CasadiNLP:
    F = define_discrete_dynamics_function(ocp)

    constraints = ocp.constraints

    stage_cost_function = define_stage_cost_function(ocp)

    terminal_cost_function = define_terminal_cost_function(ocp)

    Jbx0 = np.identity(ocp.dims.nx)
    Jbx = get_Jbx(ocp)
    Jbu = get_Jbu(ocp)
    # Jsbx = get_Jsbx(ocp)
    # Jsbu = get_Jsbu(ocp)

    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    cost = 0

    h = []  # Inequality constraints
    g = []  # Equality constraints

    lbg = []
    ubg = []

    lbx = []
    ubx = []

    lbu = []
    ubu = []

    lbh = []
    ubh = []

    xout = []
    uout = []

    x0 = ocp.constraints.lbx_0.tolist()

    # TODO: Add support for multivariable parameters
    p = ocp.model.p

    if isinstance(ocp.model.x, cs.SX):
        sym = cs.SX.sym
    elif isinstance(ocp.model.x, cs.MX):
        sym = cs.MX.sym

    # "Lift" initial conditions
    xk = sym("x0", ocp.dims.nx)
    w += [xk]
    lbw += ocp.constraints.lbx_0.tolist()
    ubw += ocp.constraints.ubx_0.tolist()
    xout += [xk]

    # h += [constraints.lbx_0 - Jbx0 @ xk]
    # h += [Jbx0 @ xk - constraints.ubx_0]

    w0 += x0

    # Formulate the NLP
    for k in range(ocp.dims.N):
        # New NLP variable for the control
        uk = sym("u_" + str(k))
        w += [uk]
        uout += [uk]

        lbw += constraints.lbu.tolist()
        ubw += constraints.ubu.tolist()

        # h += [constraints.lbu - Jbu @ uk]
        # lbg += -np.inf * np.ones((ocp.dims.nu,)).tolist()

        # h += [Jbu @ uk - constraints.ubu]
        # ubg = np.zeros((ocp.dims.nu,)).tolist()

        w0 += [0]

        cost = cost + stage_cost_function(xk, uk)

        # Integrate till the end of the interval
        xk_end = F(xk, uk, p)

        # New NLP variable for state at end of interval
        xk = sym("x_" + str(k + 1), ocp.dims.nx)
        w += [xk]
        xout += [xk]
        lbw += constraints.lbx.tolist()
        ubw += constraints.ubx.tolist()

        # h += [constraints.lbx - Jbx @ xk]
        # lbg += -np.inf * np.ones((ocp.dims.nx,)).tolist()

        # h += [Jbx @ xk - constraints.ubx]
        # ubg += np.zeros((ocp.dims.nx,)).tolist()

        w0 += x0

        # Add equality constraint
        g += [xk_end - xk]
        lbg += np.zeros((ocp.dims.nx,)).tolist()
        ubg += np.zeros((ocp.dims.nx,)).tolist()

    # Add terminal cost
    cost = cost + terminal_cost_function(xk_end)

    trajectories = cs.Function("trajectories", [cs.vertcat(*w)], [cs.vertcat(*xout), cs.vertcat(*uout)])

    xtest, utest = trajectories(cs.vertcat(*w))

    nlp = CasadiNLP()
    nlp.cost = cost
    nlp.w = cs.vertcat(*w)
    nlp.w0 = w0
    nlp.lbw = lbw
    nlp.ubw = ubw
    nlp.g = cs.vertcat(*g)
    nlp.lbg = lbg
    nlp.ubg = ubg
    nlp.p = p
    nlp.f_disc = F

    return nlp


def build_lagrange_function(nlp: CasadiNLP, ocp: CasadiOcp) -> cs.Function:
    pass


def build_kkt_residual_function(ocp: CasadiOcp) -> cs.Function:
    pass


def build_policy_gradient_function(ocp: CasadiOcp) -> cs.Function:
    pass


def build_state_action_value_function(ocp: CasadiOcp) -> cs.Function:
    pass


def build_state_value_function(ocp: CasadiOcp) -> cs.Function:
    pass


class CasadiMPC(MPC):
    """docstring for CartpoleMPC."""

    ocp: CasadiOcp
    ocp_solver: CasadiOcpSolver
    parameter_values: np.ndarray

    parameter_values: np.ndarray

    def __init__(self, config: Config, build: bool = True):
        super().__init__()

        self.ocp = CasadiOcp()

        self.ocp.model, self.ocp.parameter_values = define_acados_model(ocp=self.ocp, config=config)

        self.ocp.dims = define_acados_dims(ocp=self.ocp, config=config)

        self.ocp.cost = define_acados_cost(ocp=self.ocp, config=config)

        self.ocp.constraints = define_acados_constraints(ocp=self.ocp, config=config)

        self.ocp.dims.nsbx = self.ocp.constraints.idxsbx.shape[0]
        self.ocp.dims.nsbu = self.ocp.constraints.idxsbu.shape[0]

        self.ocp.solver_options = config.ocp_options

        self.ocp_solver = CasadiOcpSolver(self.ocp)

        self.parameter_values = self.ocp.parameter_values

        # TODO: At the moment we only support one parameter for all stages. Add support for stage-wise parameters.
        for stage_ in range(self.ocp.dims.N):
            self.ocp_solver.set(stage_, "p", self.ocp.parameter_values)

    def get_parameters(self) -> np.ndarray:
        return self.parameter_values

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

    def get_action(self, x0: np.ndarray) -> np.ndarray:
        """
        Update the solution of the OCP solver.

        Args:
            x0: Initial state.

        Returns:
            action: Scaled optimal control action.
        """

        # Set initial state
        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        # Solve the optimization problem
        self.ocp_solver.solve()

        # Get solution
        action = self.ocp_solver.get(0, "u")

        # Scale to [-1, 1] for gym
        action = self.scale_action(action)

        ####
        # nlp_solver = self.ocp_solver.nlp_solver

        # sol = self.ocp_solver.nlp_solution

        # lam_w = sol["lam_x"]
        # w = sol["x"]
        # lam_g = sol["lam_g"]
        # g = sol["g"]

        return action

    def plot_prediction(self) -> (plt.figure, plt.axes):
        X = np.vstack([self.ocp_solver.get(stage_, "x") for stage_ in range(self.ocp.dims.N)])
        U = np.vstack([self.ocp_solver.get(stage_, "u") for stage_ in range(self.ocp.dims.N - 1)])

        fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(10, 10))
        for k, ax in enumerate(axes[:-1]):
            ax.step(range(self.ocp.dims.N), X[:, k], color="k")
            ax.grid(True)
        axes[-1].step(range(self.ocp.dims.N - 1), U[:, 0], color="k")
        axes[-1].grid(True)

        axes[0].set_ylabel("x")
        axes[1].set_ylabel("v")
        axes[2].set_ylabel("theta")
        axes[3].set_ylabel("omega")

        axes[-1].set_xlabel("Step [-]")

        return (fig, axes)
