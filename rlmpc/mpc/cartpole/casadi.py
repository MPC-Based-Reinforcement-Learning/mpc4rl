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

from casadi.tools.structure3 import CasadiStructureDerivable


class CasadiNLP:
    """docstring for CasadiNLP."""

    cost: Union[cs.SX, cs.MX]
    w: Union[cs.SX, cs.MX]
    w0: Union[list, np.ndarray]
    lbw: Union[list, np.ndarray]
    ubw: Union[list, np.ndarray]
    g_solver: Union[cs.SX, cs.MX]
    lbg_solver: Union[list, np.ndarray]
    ubg_solver: Union[list, np.ndarray]
    p: Union[cs.SX, cs.MX]
    f_disc: cs.Function
    shooting: struct_symSX
    g: Union[cs.SX, cs.MX]  # Dynamics equality constraints
    pi: Union[cs.SX, cs.MX]  # Lange multiplier for dynamics equality constraints
    h: Union[cs.SX, cs.MX]  # Inequality constraints
    lam: Union[cs.SX, cs.MX]  # Lange multiplier for inequality constraints
    h_fun: cs.Function  # Function to build inequality constraints, i.e. h(w)
    g_fun: cs.Function  # Function to build dynamics equality constraints, i.e. g(w, p)

    def __init__(self):
        super().__init__()

        self.cost = None
        self.w = None
        self.w0 = None
        self.lbw = None
        self.ubw = None
        self.g_solver = None
        self.lbg_solver = None
        self.ubg_solver = None
        self.p = None
        self.f_disc = None
        self.shooting = None
        self.g = None
        self.pi = None
        self.h = None
        self.lam = None
        self.h_fun = None
        self.g_fun = None


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

    # Idx to keep track of constraints and multipliers
    idx: dict

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

        self.nlp, self.idx = build_nlp_with_slack(self.ocp)

        nlp = self.nlp

        ############################## Build the NLP sensitivities #####################################
        # Move this part to a separate function alter when it is working

        # Define the Lagrangian
        Lag = nlp.cost + cs.mtimes([nlp.lam.cat.T, nlp.h]) + cs.mtimes([nlp.pi.cat.T, nlp.g])

        # Define the Lagrangian gradient with respect to the decision variables
        dL_dw = cs.jacobian(Lag, nlp.w.cat)

        # Define the Lagrangian gradient with respect to the parameters
        # TODO: Add support for multivariable parameters
        dL_dp = cs.jacobian(Lag, nlp.p)

        # Build KKT matrix
        # TODO: Move etau to solver options
        etau = 10e-8
        R_kkt = cs.vertcat(cs.transpose(dL_dw), nlp.g, nlp.lam.cat * nlp.h + etau)

        # Concatenate primal-dual variables
        z = cs.vertcat(nlp.w.cat, nlp.lam.cat, nlp.pi.cat)

        # Generate sensitivity of the KKT matrix with respect to primal-dual variables
        dR_dz = cs.jacobian(R_kkt, z)

        # Generate sensitivity of the KKT matrix with respect to parameters
        dR_dp = cs.jacobian(R_kkt, nlp.p)

        fun = dict()
        fun["casadi"] = dict()
        fun["casadi"]["R_kkt"] = cs.Function("R_kkt", [nlp.w.cat, nlp.pi.cat, nlp.lam.cat, nlp.p], [R_kkt])
        fun["casadi"]["dL_dw"] = cs.Function("dL_dw", [nlp.w.cat, nlp.pi.cat, nlp.lam.cat, nlp.p], [dL_dw])
        fun["casadi"]["cost"] = cs.Function("cost", [nlp.w.cat], [nlp.cost])
        fun["casadi"]["g"] = cs.Function("g", [nlp.w.cat, nlp.p], [nlp.g], ["w", "p"], ["g"])
        fun["casadi"]["h"] = cs.Function("h", [nlp.w.cat, nlp.p], [nlp.h], ["w", "p"], ["h"])
        fun["casadi"]["dg_dw"] = cs.Function("dg_dw", [nlp.w.cat, nlp.p], [cs.jacobian(nlp.g, nlp.w.cat)])
        fun["casadi"]["dh_dw"] = cs.Function("dh_dw", [nlp.w.cat, nlp.p], [cs.jacobian(nlp.h, nlp.w.cat)])
        # fun["casadi"]["dR_dz"] = cs.Function("dR_dz", [z, nlp.p], [dR_dz])
        # fun["casadi"]["dR_dp"] = cs.Function("dR_dp", [z, nlp.p], [dR_dp])

        build_flag = True
        if build_flag:
            if True:
                # Generate  and compile c-code for the functions
                fun["compiled"] = dict.fromkeys(fun["casadi"].keys())
                build_flag = True
                for key, val in fun["casadi"].items():
                    val.generate(f"{key}.c", {"mex": False})

                    # Compile using os.system
                    if build_flag:
                        flag = os.system(f"gcc -fPIC -shared {key}.c -o {key}.so")

                    fun["compiled"][key] = cs.external(key, f"{key}.so")

                if False:  # Do this outside
                    # Test use of the compiled function

                    # test_z = np.zeros((nlp.w.cat.shape[0] + pi.cat.shape[0] + lam.cat.shape[0], 1))
                    w_test = np.zeros((self.nlp.w.cat.shape[0], 1))
                    lam_test = np.zeros((self.nlp.lam.cat.shape[0], 1))
                    pi_test = np.zeros((self.nlp.pi.cat.shape[0], 1))
                    p_test = np.ones((self.nlp.p.shape[0], 1))

                    test = dict.fromkeys(fun["compiled"].keys())

                    for key in ["R_kkt", "dL_dw"]:
                        test[key] = fun["compiled"][key](w_test, pi_test, lam_test, p_test)
                    for key in ["g", "h", "dg_dw", "dh_dw"]:
                        test[key] = fun["compiled"][key](w_test, p_test)
                    for key in ["cost"]:
                        test[key] = fun["compiled"][key](w_test)

                    # test_res = fun["compiled"]["R_kkt"](w_test, pi_test, lam_test, p_test)

                    # Make a spy plot of dh_dw
                    fig, ax = plt.subplots()
                    ax.spy(test["dh_dw"])

                    # Make a spy plot of dg_dw
                    fig, ax = plt.subplots()
                    ax.spy(test["dg_dw"])

                    plt.show()

                    x_test = np.array([0.0, 0.0, np.pi / 2, 0.0])
                    u_test = np.array([0.0])

        self.fun = fun

        # Create an NLP solver
        self.nlp_solver = cs.nlpsol(
            "solver",
            "ipopt",
            {"f": self.nlp.cost, "x": self.nlp.w, "p": self.nlp.p, "g": self.nlp.g_solver},
            {"ipopt": {"max_iter": 100, "print_level": 0}, "jit": False, "verbose": False, "print_time": False},
        )

        if False:
            self.nlp_solution = self.nlp_solver(
                x0=self.nlp.w0,
                p=1.0,
                lbg=self.nlp.lbg_solver,
                ubg=self.nlp.ubg_solver,
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

    # def compute_KKT(self, w: cs.SX, pi: cs.SX, lam: cs.SX, p: cs.SX) -> cs.SX:
    def compute_KKT(
        self,
        w: Union[np.ndarray, cs.DM],
        pi: Union[np.ndarray, cs.DM],
        lam: Union[np.ndarray, cs.DM],
        p: Union[np.ndarray, cs.DM],
    ) -> Union[np.ndarray, cs.DM]:
        """
        Compute the KKT matrix.

        Args:
            w: Decision variables.
            pi: Multipliers for dynamics equality constraints.
            lam: Multipliers for inequalities.
            p: Parameters.

        Returns:
            KKT matrix.
        """
        pass

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
            self.nlp.lbw["x", stage_] = value_
        elif field_ == "ubx":
            self.nlp.ubw["x", stage_] = value_
        elif field_ == "lbu":
            self.nlp.lbw["u", stage_] = value_
        elif field_ == "ubu":
            self.nlp.ubw["u", stage_] = value_
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
            return self.w_opt["x", stage_].full()

        if field_ == "u":
            return self.w_opt["u", stage_].full()

        if field_ == "sx":
            return self.w_opt["sx", stage_].full()

        if field_ == "pi":
            return self.lam_g_opt["pi", stage_]

        if field_ == "lam":
            # lam_x = self.nlp.w(self.nlp_solution["lam_x"].full())
            # lam_g = self.nlp_solution["lam_g"][stage_].full()

            # return self.nlp_solution["lam_g"][self.idx["g"]["lam"][stage_]].full()
            raise NotImplementedError()

        """
            .. note:: regarding lam, t: \n
                    the inequalities are internally organized in the following order: \n
                    [ lbu lbx lg lh lphi ubu ubx ug uh uphi; \n
                      lsbu lsbx lsg lsh lsphi usbu usbx usg ush usphi]
        """

        raise NotImplementedError()

    def get_multiplier(self, stage_: int, field_: str) -> np.ndarray:
        # Multipliers are positive if upper bound is active and negative if lower bound is active
        # https://groups.google.com/g/casadi-users/c/fcjt-JX5BIc/m/cKJGV9h9BwAJ

        if field_ == "lbu":
            return np.clip(self.lam_x_opt["u", stage_].full(), self.lam_x_opt["u", stage_].full(), 0.0)
        elif field_ == "lbx":
            return np.clip(self.lam_x_opt["x", stage_].full(), self.lam_x_opt["x", stage_].full(), 0.0)
        elif field_ == "ubu":
            return np.clip(self.lam_x_opt["u", stage_].full(), 0.0, self.lam_x_opt["u", stage_].full())
        elif field_ == "ubx":
            return np.clip(self.lam_x_opt["x", stage_].full(), 0.0, self.lam_x_opt["x", stage_].full())
        elif field_ == "lsbx":
            return self.lam_g_opt["lsbx", stage_].full()
        elif field_ == "usbx":
            return self.lam_g_opt["usbx", stage_].full()

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
            lbg=self.nlp.lbg_solver,
            ubg=self.nlp.ubg_solver,
            p=self.p,
        )

        self.w_opt = self.nlp.w(self.nlp_solution["x"])

        self.lam_x_opt = self.nlp.w(self.nlp_solution["lam_x"])
        self.lam_g_opt = self.nlp_solution["lam_g"]

        if False:
            # Repackage lam_g into a dictionary with the same structure as w
            self.lam_g_opt = dict()
            for stage_ in range(self.ocp.dims.N - 1):
                self.lam_g_opt[("pi", stage_)] = self.nlp_solution["lam_g"][self.idx["g"]["pi"][stage_]]
                self.lam_g_opt[("lsbx", stage_)] = self.nlp_solution["lam_g"][self.idx["g"]["lsbx"][stage_]]
                self.lam_g_opt[("usbx", stage_)] = self.nlp_solution["lam_g"][self.idx["g"]["usbx"][stage_]]

        # Initial condition is an equality constraint. Treat separately
        # Collect multipliers first. Do the clipping later.
        # lbu
        lam_opt = []
        for stage_ in range(0, self.ocp.dims.N - 1):
            lam_opt += [self.lam_x_opt["u", stage_]]
            lam_opt += [self.lam_x_opt["x", stage_]]
            lam_opt += [self.lam_x_opt["u", stage_]]
            lam_opt += [self.lam_x_opt["x", stage_]]
            lam_opt += [self.lam_x_opt["sx", stage_, "lbx"]]
            lam_opt += [self.lam_x_opt["sx", stage_, "ubx"]]

        lam_opt = np.concatenate(lam_opt)

        # Eliminate the multipliers that correspond to inf bounds
        # lbu

        pi_opt = self.nlp.pi(
            cs.vertcat(
                *[self.nlp_solution["lam_g"][self.idx["g"]["pi"][stage_]].full() for stage_ in range(self.ocp.dims.N - 1)]
            )
        )

        # g_opt_old = self.nlp.g(
        #     cs.vertcat(*[self.nlp_solution["g"][self.idx["g"]["pi"][stage_]].full() for stage_ in range(self.ocp.dims.N - 1)])
        # )

        # lsbx = [self.nlp_solution["g"][self.idx["g"]["lsbx"][stage_]].full() for stage_ in range(self.ocp.dims.N - 1)]
        # usbx = [self.nlp_solution["g"][self.idx["g"]["usbx"][stage_]].full() for stage_ in range(self.ocp.dims.N - 1)]

        # h_opt = []
        # for stage_ in range(self.ocp.dims.N - 1):
        #     h_opt += []

        g_opt = self.fun["compiled"]["g"](self.w_opt, self.p)
        h_opt = self.fun["compiled"]["h"](self.w_opt, self.p)

        # Build equality constraints from pi. Done
        # pi = [self.nlp_solution["pi"][self.idx["g"]["pi"][stage_]].full() for stage_ in range(self.ocp.dims.N - 1)]
        # g = [self.nlp_solution["g"][self.idx["g"]["pi"][stage_]].full() for stage_ in range(self.ocp.dims.N - 1)]

        # Build inequality constraints from g and lwb, ubw
        # h = []
        # for stage_ in range(self.ocp.dims.N - 1):
        #     # Box constraints
        #     h += [self.nlp_solution["w"][self.idx["g"]["lsbx"][stage_]].full()]

        # Build inequality constraints from g and lwb, ubw

        a = self.get_multiplier(0, "ubx")

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


def build_nlp_with_slack(ocp: CasadiOcp) -> (CasadiNLP, dict, dict):
    """
    Build the NLP for the OCP.

    TODO: Add support for other cost types
    TODO: Adapt to SX/MX depending on the provided model
    TODO: Add support for different parameters at each stage
    TODO: Add support for varying/learning reference trajectories, i.e. set as parameters
    TODO: Add support for varying/learning cost weights, i.e. set as parameters
    TODO: Add support for varying/learning constraints, i.e. set as parameters
    """

    nlp = CasadiNLP()

    nlp.f_disc = define_discrete_dynamics_function(ocp)

    constraints = ocp.constraints

    state_labels = ocp.model.x.str().strip("[]").split(", ")
    input_labels = ocp.model.u.str().strip("[]").split(", ")

    # Index of box constraints
    idxbx = constraints.idxbx
    idxbu = constraints.idxbu

    # Index of soft box constraints
    idxsbx = constraints.idxsbx
    idxsbu = constraints.idxsbu

    # Index of hard box constraints
    idxhbx = [idx for idx in idxbx if idx not in idxsbx]
    idxhbu = [idx for idx in idxbu if idx not in idxsbu]

    decision_entries = []

    # Add states to decision variables
    states = struct_symSX([tuple([entry(label) for label in state_labels])])
    decision_entries += [entry("x", repeat=ocp.dims.N, struct=states)]

    # Add inputs to decision variables
    inputs = struct_symSX([tuple([entry(label) for label in input_labels])])
    decision_entries += [entry("u", repeat=ocp.dims.N - 1, struct=inputs)]

    # Add slack variables for relaxed state box constraints to decision variables
    # TODO: Handle the case when there are no relaxed state box constraints
    # TODO: Add support for relaxed input box constraints
    # TODO: Add support for other types of relaxed constraints

    sx_entries = []
    sx_entries += [entry("lbx", struct=struct_symSX([state_labels[idx] for idx in constraints.idxsbx]))]
    sx_entries += [entry("ubx", struct=struct_symSX([state_labels[idx] for idx in constraints.idxsbx]))]
    sx = struct_symSX([tuple(sx_entries)])
    decision_entries += [entry("sx", repeat=ocp.dims.N, struct=sx)]

    nlp.w = struct_symSX([tuple(decision_entries)])

    # Hard box constraints
    lhbu = [constraints.lbu[i] if i in idxhbu else -np.inf for i in range(ocp.dims.nu)]
    lhbx = [constraints.lbx[i] if i in idxhbx else -np.inf for i in range(ocp.dims.nx)]
    uhbu = [constraints.ubu[i] if i in idxhbu else np.inf for i in range(ocp.dims.nu)]
    uhbx = [constraints.ubx[i] if i in idxhbx else np.inf for i in range(ocp.dims.nx)]

    # Soft box constraints
    lsbx = cs.vertcat(*[constraints.lbx[i] for i in idxsbx])
    lsbu = cs.vertcat(*[constraints.lbu[i] for i in idxsbu])
    usbx = cs.vertcat(*[constraints.ubx[i] for i in idxsbx])
    usbu = cs.vertcat(*[constraints.ubu[i] for i in idxsbu])

    ############ Build the NLP ############

    ############ Hard box constraints ############

    nlp.lbw = nlp.w(0)
    nlp.lbw["x", lambda x: cs.vertcat(*x)] = np.tile(lhbx, (1, ocp.dims.N))
    nlp.lbw["u", lambda x: cs.vertcat(*x)] = np.tile(lhbu, (1, ocp.dims.N - 1))
    for stage_ in range(ocp.dims.N):
        nlp.lbw["sx", stage_, "lbx"] = [0 for _ in constraints.idxsbx]
        nlp.lbw["sx", stage_, "ubx"] = [0 for _ in constraints.idxsbx]

    nlp.ubw = nlp.w(0)
    nlp.ubw["x", lambda x: cs.vertcat(*x)] = np.tile(uhbx, (1, ocp.dims.N))
    nlp.ubw["u", lambda x: cs.vertcat(*x)] = np.tile(uhbu, (1, ocp.dims.N - 1))
    for stage_ in range(ocp.dims.N):
        nlp.ubw["sx", stage_, "lbx"] = [np.inf for _ in constraints.idxsbx]
        nlp.ubw["sx", stage_, "ubx"] = [np.inf for _ in constraints.idxsbx]

    x0 = ocp.constraints.lbx_0.tolist()
    u0 = 0

    ############# Initial guess #############

    nlp.w0 = nlp.w(0)
    nlp.w0["x", lambda x: cs.vertcat(*x)] = np.tile(x0, (1, ocp.dims.N))
    nlp.w0["u", lambda x: cs.vertcat(*x)] = np.tile(u0, (1, ocp.dims.N - 1))
    for stage_ in range(ocp.dims.N):
        nlp.w0["sx", stage_, "lbx"] = [0 for _ in constraints.idxsbx]
        nlp.w0["sx", stage_, "ubx"] = [0 for _ in constraints.idxsbx]

    ############# Parameter vector #############

    # Parameter vector
    # TODO: Add support for multivariable parameters
    nlp.p = ocp.model.p

    # Build other constraints (dynamics, relaxed box constraints, etc.)

    # Alias for the states, control, and slack variables
    (X, U, _) = nlp.w[...]

    # S = cs.vertcat(Sl, Su)

    nlp.g_solver = []
    nlp.lbg_solver = []
    nlp.ubg_solver = []

    h = []

    equality_constraints = []

    idx = dict()
    idx["g"] = dict()
    idx["g"]["pi"] = []
    idx["g"]["lsbx"] = []
    idx["g"]["usbx"] = []

    running_index = 0

    for stage_ in range(ocp.dims.N - 1):
        # Add dynamics constraints
        nlp.g_solver.append(nlp.w["x", stage_ + 1] - nlp.f_disc(nlp.w["x", stage_], nlp.w["u", stage_], nlp.p))
        nlp.lbg_solver.append([0 for _ in range(ocp.dims.nx)])
        nlp.ubg_solver.append([0 for _ in range(ocp.dims.nx)])

        # Add indices for the added elements to g
        idx["g"]["pi"].append([running_index + i for i in range(ocp.dims.nx)])
        running_index = idx["g"]["pi"][-1][-1] + 1

        # Add relaxed box constraints for lower bounds
        nlp.g_solver.append(lsbx - nlp.w["x", stage_][idxsbx] - nlp.w["sx", stage_, "lbx"])
        nlp.lbg_solver.append([-cs.inf for _ in idxsbx])
        nlp.ubg_solver.append([0 for _ in idxsbx])

        # Add indices for the added elements to g
        idx["g"]["lsbx"].append([running_index + i for i in range(idxsbx.shape[0])])
        running_index = idx["g"]["lsbx"][-1][-1] + 1

        # Add relaxed box constraints for upper bounds
        nlp.g_solver.append(-usbx + nlp.w["x", stage_][idxsbx] - nlp.w["sx", stage_, "ubx"])
        nlp.lbg_solver.append([-cs.inf for _ in idxsbx])
        nlp.ubg_solver.append([0 for _ in idxsbx])

        # Add indices for the added elements to g
        idx["g"]["usbx"].append([running_index + i for i in range(idxsbx.shape[0])])
        running_index = idx["g"]["usbx"][-1][-1] + 1

    nlp.g_solver = cs.vertcat(*nlp.g_solver)
    nlp.lbg_solver = cs.vertcat(*nlp.lbg_solver)
    nlp.ubg_solver = cs.vertcat(*nlp.ubg_solver)

    """
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

    keys = [
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

    lam = dict.fromkeys(keys)

    # Build equality constraints
    pi_entries = [entry("pi", repeat=ocp.dims.N - 1, struct=struct_symSX([state_labels[idx] for idx in range(ocp.dims.nx)]))]
    pi = struct_symSX([tuple(pi_entries)])
    nlp.g = pi(cs.vertcat(*[nlp.g_solver[idx["g"]["pi"][stage_]] for stage_ in range(ocp.dims.N - 1)]))

    nlp.pi = pi

    # Build inequality constraint
    lam_entries = []
    h = []

    # Initial condition
    # h += [lhbx_0 - X[0][idxhbx_0]]

    for stage_ in range(ocp.dims.N - 1):
        if any(idxhbu):
            if stage_ == 0:
                lam_entries += [
                    entry("lbu", repeat=ocp.dims.N - 1, struct=struct_symSX([input_labels[idx] for idx in idxhbu]))
                ]
            h += [(cs.SX(lhbu) - U[stage_])[idxhbu]]
        if any(idxhbx):
            if stage_ == 0:
                lam_entries += [
                    entry("lbx", repeat=ocp.dims.N - 1, struct=struct_symSX([state_labels[idx] for idx in idxhbx]))
                ]
            h += [(cs.SX(lhbx) - X[stage_])[idxhbx]]
        if any(idxhbu):
            if stage_ == 0:
                lam_entries += [
                    entry("ubu", repeat=ocp.dims.N - 1, struct=struct_symSX([input_labels[idx] for idx in idxhbu]))
                ]
            h += [(U[stage_] - cs.SX(uhbu))[idxhbu]]
        if any(idxhbx):
            if stage_ == 0:
                lam_entries += [
                    entry("ubx", repeat=ocp.dims.N - 1, struct=struct_symSX([state_labels[idx] for idx in idxhbx]))
                ]
            h += [(X[stage_] - cs.SX(uhbx))[idxhbx]]
        if any(idxsbx):
            if stage_ == 0:
                lam_entries += [
                    entry("lsbx", repeat=ocp.dims.N - 1, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))
                ]
            h += [nlp.g_solver[idx["g"]["lsbx"][stage_]]]
        if any(idxsbx):
            if stage_ == 0:
                lam_entries += [
                    entry("usbx", repeat=ocp.dims.N - 1, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))
                ]
            h += [nlp.g_solver[idx["g"]["usbx"][stage_]]]

    nlp.lam = struct_symSX([tuple(lam_entries)])
    nlp.h = nlp.lam(cs.vertcat(*h))

    # TODO: Possibly move this to separate function where cs.Functions are built and compiled
    # nlp.h_fun = cs.Function("h", [nlp.w], [nlp.h], ["w"], ["h"])
    # nlp.g_fun = cs.Function("g", [nlp.w, nlp.p], [nlp.g], ["w", "p"], ["g"])

    # sl = nlp.w["sx", :, "lbx"]
    # su = nlp.w["sx", :, "ubx"]

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

    nlp.cost = 0
    # Build the cost function
    for stage_ in range(ocp.dims.N - 1):
        nlp.cost += stage_cost_function(
            nlp.w["x", stage_], nlp.w["u", stage_], nlp.w["sx", stage_, "lbx"], nlp.w["sx", stage_, "ubx"]
        )

    # Add terminal cost
    nlp.cost += terminal_cost_function(nlp.w["x", stage_], nlp.w["sx", stage_, "lbx"], nlp.w["sx", stage_, "ubx"])

    # nlp.g = g
    # nlp.h = h

    return nlp, idx


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
    nlp.g_solver = cs.vertcat(*g)
    nlp.lbg_solver = lbg
    nlp.ubg_solver = ubg
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

    def compute_lagrange_function_value(self):
        # self.ocp_solver.get()

        w_ipopt = self.ocp_solver.nlp_solution["x"].full().flatten()
        lam_g_ipopt = self.ocp_solver.nlp_solution["lam_g"].full().flatten()
        g_ipopt = self.ocp_solver.nlp_solution["g"].full().flatten()

        print("hallo")

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

        sol = self.ocp_solver.nlp_solution

        if False:
            J = sol["f"]

            w = sol["x"].full().flatten()

            # # lam_w = sol["lam_x"].full().flatten().tolist()
            # # lam_lbw = np.array([value_ if value_ < 0.0 else 0.0 for value_ in lam_w])
            # # lam_ubw = np.array([value_ if value_ > 0.0 else 0.0 for value_ in lam_w])

            g = sol["g"].full().flatten()

            # lam_g = sol["lam_g"].full().flatten()

            # test = self.ocp_solver.nlp.w(sol["x"].full())

            X = np.hstack([self.ocp_solver.get(stage_, "x") for stage_ in range(self.ocp.dims.N)]).T
            U = np.hstack([self.ocp_solver.get(stage_, "u") for stage_ in range(self.ocp.dims.N - 1)]).T
            pi = np.hstack([self.ocp_solver.get(stage_, "pi") for stage_ in range(self.ocp.dims.N - 1)]).T
            # lam = np.hstack([self.ocp_solver.get(stage_, "lam") for stage_ in range(self.ocp.dims.N)]).T

            lbu = np.hstack([self.ocp_solver.get_multiplier(stage_, "lbu") for stage_ in range(self.ocp.dims.N - 1)]).T
            lbx = np.hstack([self.ocp_solver.get_multiplier(stage_, "lbx") for stage_ in range(self.ocp.dims.N)]).T
            ubu = np.hstack([self.ocp_solver.get_multiplier(stage_, "ubu") for stage_ in range(self.ocp.dims.N - 1)]).T
            ubx = np.hstack([self.ocp_solver.get_multiplier(stage_, "ubx") for stage_ in range(self.ocp.dims.N)]).T
            lsbx = np.hstack([self.ocp_solver.get_multiplier(stage_, "lsbx") for stage_ in range(self.ocp.dims.N - 1)]).T
            usbx = np.hstack([self.ocp_solver.get_multiplier(stage_, "usbx") for stage_ in range(self.ocp.dims.N - 1)]).T

            # # t = np.hstack([self.ocp_solver.get(stage_, "t") for stage_ in range(self.ocp.dims.N)]).T
            # # sl = np.hstack([self.ocp_solver.get(stage_, "sl") for stage_ in range(self.ocp.dims.N)]).T
            # # su = np.hstack([self.ocp_solver.get(stage_, "su") for stage_ in range(self.ocp.dims.N)]).T

        # Build the Lagrangian
        # L = J + lam_g.T @ g + lam_lbw.T @ (w - self.ocp_solver.nlp.lbw) + lam_ubw.T @ (self.ocp_solver.nlp.ubw - w)

        # ubu = [constraints.ubu[i] if i in idxhbu else np.inf for i in range(ocp.dims.nu)]

        # print("hallo")

        ####
        # nlp_solver = self.ocp_solver.nlp_solver

        # sol = self.ocp_solver.nlp_solution

        # lam_w = sol["lam_x"]
        # w = sol["x"]
        # lam_g = sol["lam_g"]
        # g = sol["g"]

        return action

    def plot_prediction(self) -> (plt.figure, plt.axes):
        X = np.hstack([self.ocp_solver.get(stage_, "x") for stage_ in range(self.ocp.dims.N)]).T
        U = np.hstack([self.ocp_solver.get(stage_, "u") for stage_ in range(self.ocp.dims.N - 1)]).T

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