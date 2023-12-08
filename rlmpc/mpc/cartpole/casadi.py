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


class CasadiNLPEntry:
    """docstring for CasadiNLPEntry."""

    sym: Union[cs.SX, cs.MX]
    val: Union[list, np.ndarray]
    fun: cs.Function

    def __init__(self):
        super().__init__()

        self.sym = None
        self.val = None
        self.fun = None
        # self.cat = None
        # self.shape = None
        # self.size = None
        # self.type = None


class CasadiNLP:
    """docstring for CasadiNLP."""

    cost: Union[cs.SX, cs.MX]
    w: CasadiNLPEntry
    lbw: CasadiNLPEntry
    ubw: CasadiNLPEntry
    g_solver: Union[cs.SX, cs.MX]
    lbg_solver: Union[list, np.ndarray]
    ubg_solver: Union[list, np.ndarray]
    p: CasadiNLPEntry
    p_solver: CasadiNLPEntry
    p_val: Union[list, np.ndarray]
    f_disc: cs.Function
    shooting: struct_symSX
    # g: Union[cs.SX, cs.MX]  # Dynamics equality constraints
    g: CasadiNLPEntry  # Dynamics equality constraints
    pi: CasadiNLPEntry  # Lange multiplier for dynamics equality constraints
    h: CasadiNLPEntry  # Inequality constraints
    lam: CasadiNLPEntry  # Lange multiplier for inequality constraints
    idxhbx: list
    idxsbx: list
    idxhbu: list
    idxsbu: list

    def __init__(self):
        super().__init__()

        self.cost = None
        self.w = CasadiNLPEntry()
        self.lbw = CasadiNLPEntry()
        self.ubw = CasadiNLPEntry()
        self.g_solver = None
        self.lbg_solver = None
        self.ubg_solver = None
        self.p_solver = CasadiNLPEntry()
        self.p_val = None
        self.p = CasadiNLPEntry()
        self.f_disc = None
        self.shooting = None
        self.g = CasadiNLPEntry()
        self.pi = CasadiNLPEntry()
        self.h = CasadiNLPEntry()
        self.lam = CasadiNLPEntry()
        self.idxhbx = None
        self.idxsbx = None
        self.idxhbu = None
        self.idxsbu = None


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

    def __init__(self, _ocp: CasadiOcp, build, name: str = "ocp_solver"):
        super().__init__()

        # self._ocp.model, self._ocp.parameter_values = define_acados_model(
        #     ocp=self._ocp, config=config
        # )

        self.ocp = _ocp

        self.nlp, self.idx = build_nlp(self.ocp)

        nlp = self.nlp

        ############################## Build the NLP sensitivities #####################################
        # Move this part to a separate function alter when it is working

        # p = cs.vertcat(*nlp.p_sym["p", :])

        if True:
            # Define the Lagrangian
            # L = nlp.cost + cs.mtimes([nlp.lam.cat.T, nlp.h]) + cs.mtimes([nlp.pi.cat.T, nlp.g])
            L = nlp.cost + cs.dot(nlp.pi.sym, nlp.g.sym) + cs.dot(nlp.lam.sym, nlp.h.sym)

            # a = cs.DM.ones(3, 1)
            # b = cs.DM.ones(3, 1)
            # b[0] = 1
            # b[1] = 2
            # b[2] = 3

            # print(cs.mtimes([a.T, b]))
            # print(cs.dot(a, b))

            # Define the Lagrangian gradient with respect to the decision variables
            dL_dw = cs.jacobian(L, nlp.w.sym)

            # Define the Lagrangian gradient with respect to the parameters
            # TODO: Add support for multivariable parameters
            dL_dp = cs.jacobian(L, nlp.p.sym)

            # TODO: Move etau to solver options
            etau = 10e-8
            # R_kkt = cs.vertcat(cs.transpose(dL_dw), nlp.g, nlp.lam.cat * nlp.h + etau)

            # Concatenate primal-dual variables
            z = cs.vertcat(nlp.w.sym, nlp.pi.sym, nlp.lam.sym)

            # Build KKT matrix
            R = cs.vertcat(cs.transpose(dL_dw), nlp.g.sym, nlp.lam.sym * nlp.h.sym + etau)

            # Generate sensitivity of the KKT matrix with respect to primal-dual variables
            dR_dz = cs.jacobian(R, z)

            # Generate sensitivity of the KKT matrix with respect to parameters
            dR_dp = cs.jacobian(R, nlp.p.sym)

            fun = dict()
            fun["L"] = cs.Function(
                "L",
                [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym],
                [L],
                ["w", "lbw", "ubw", "pi", "lam", "p"],
                ["L"],
            )
            fun["dL_dp"] = cs.Function(
                "dL_dp", [nlp.w.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym], [dL_dp], ["w", "pi", "lam", "p"], ["dL_dp"]
            )
            fun["dL_dw"] = cs.Function("dL_dw", [nlp.w.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym], [dL_dw])

            # fun["cost"] = cs.Function("cost", [nlp.w], [nlp.cost])
            fun["g"] = cs.Function("g", [nlp.w.sym, nlp.p.sym], [nlp.g.sym], ["w", "p"], ["g"])

            fun["h"] = cs.Function("h", [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym], [nlp.h.sym], ["w", "lbw", "ubw"], ["h"])
            # fun["casadi"]["dg_dw"] = cs.Function("dg_dw", [nlp.w, nlp.p], [cs.jacobian(nlp.g, nlp.w)])
            # fun["casadi"]["dh_dw"] = cs.Function("dh_dw", [nlp.w, nlp.p], [cs.jacobian(nlp.h, nlp.w)])
            # fun["casadi"]["dR_dz"] = cs.Function(
            #     "dR_dz",
            #     [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym, nlp.g.sym, nlp.h.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym],
            #     [dR_dz],
            # )

            fun["R"] = cs.Function("R", [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym], [R])

            fun["z"] = cs.Function("z", [nlp.w.sym, nlp.pi.sym, nlp.lam.sym], [z], ["w", "pi", "lam"], ["z"])

            fun["dR_dz"] = cs.Function(
                "dR_dz",
                [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym],
                [dR_dz],
                ["w", "lbw", "ubw", "pi", "lam", "p"],
                ["dR_dz"],
            )

            fun["dR_dp"] = cs.Function(
                "dR_dp",
                [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym],
                [dR_dp],
                ["w", "lbw", "ubw", "pi", "lam", "p"],
                ["dR_dz"],
            )

            # fun["casadi"]["dR_dp"] = cs.Function("dR_dp", [z, nlp.p], [dR_dp])

            if build:
                # Generate  and compile c-code for the functions
                for key, val in fun.items():
                    val.generate(f"{name}_{key}.c", {"mex": False})

                # Compile using os.system
                status = os.system(f"gcc -fPIC -shared {name}_{key}.c -o {name}_{key}.so")

                if status != 0:
                    raise Exception(f"Error compiling {name}_{key}.c")

                fun[key] = cs.external(key, f"{name}_{key}.so")

                if False:  # Do this outside
                    # Test use of the compiled function

                    # test_z = np.zeros((nlp.w.cat.shape[0] + pi.cat.shape[0] + lam.cat.shape[0], 1))
                    w_test = np.zeros((self.nlp.w.cat.shape[0], 1))
                    lam_test = np.zeros((self.nlp.lam.cat.shape[0], 1))
                    pi_test = np.zeros((self.nlp.pi.cat.shape[0], 1))
                    p_test = np.ones((self.nlp.p_solver.shape[0], 1))

                    test = dict.fromkeys(fun.keys())

                    for key in ["R_kkt", "dL_dw"]:
                        test[key] = fun[key](w_test, pi_test, lam_test, p_test)
                    for key in ["g", "h", "dg_dw", "dh_dw"]:
                        test[key] = fun[key](w_test, p_test)
                    for key in ["cost"]:
                        test[key] = fun[key](w_test)

                    # test_res = fun["R_kkt"](w_test, pi_test, lam_test, p_test)

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
            {"f": self.nlp.cost, "x": self.nlp.w.sym, "p": self.nlp.p_solver.sym, "g": self.nlp.g_solver},
            {"ipopt": {"max_iter": 100, "print_level": 0}, "jit": build, "verbose": False, "print_time": False},
        )

        # build_lagrange_function(out, self.ocp)

        self.nlp_solution = None

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

    def compute_policy(self, state: np.ndarray, solve: bool = True) -> np.ndarray:
        """
        Compute the policy.

        Args:
            state: State.
            solve: Solve the OCP.

        Returns:
            Policy.
        """
        if solve:
            self.constraints_set(stage_=0, field_="lbx", value_=state)
            self.constraints_set(stage_=0, field_="ubx", value_=state)
            self.set(stage_=0, field_="x", value_=state)
            self.solve()

        return self.nlp.w.val["u", 0].full()

    def compute_state_action_value_function_value(
        self, state: np.ndarray, action: np.ndarray, type=cs.DM
    ) -> Union[np.ndarray, cs.DM]:
        """
        Compute the value of the state-action value function.

        Returns:
            Value of the state-action value function.
        """

        self.constraints_set(stage_=0, field_="lbx", value_=state)
        self.constraints_set(stage_=0, field_="ubx", value_=state)
        self.set(stage_=0, field_="x", value_=state)

        self.constraints_set(stage_=0, field_="lbu", value_=action)
        self.constraints_set(stage_=0, field_="ubu", value_=action)
        self.set(stage_=0, field_="u", value_=action)

        self.solve()

        return self.nlp_solution["f"]

    def compute_policy_parametric_sensitivity(self, type=cs.DM) -> Union[np.ndarray, cs.DM]:
        """
        Compute the gradient of the policy.

        Returns:
            Gradient of the policy.
        """

        dR_dz = self.fun["dR_dz"](
            self.nlp.w.val, self.nlp.lbw.val, self.nlp.ubw.val, self.nlp.pi.val, self.nlp.lam.val, self.nlp.p.val
        )
        dR_dp = self.fun["dR_dp"](
            self.nlp.w.val, self.nlp.lbw.val, self.nlp.ubw.val, self.nlp.pi.val, self.nlp.lam.val, self.nlp.p.val
        )

        # dR_dz_inv = np.linalg.inv(dR_dz)

        dpi_dp_all = -np.linalg.inv(dR_dz) @ dR_dp
        dpi_dp = dpi_dp_all[: self.ocp.dims.nu]

        return dpi_dp

    def compute_state_action_value_function_parametric_sensitivity(self, type=cs.DM) -> Union[np.ndarray, cs.DM]:
        """
        Compute the gradient of the state value function.

        Returns:
            Gradient of the state value function.
        """
        return self.compute_lagrange_function_parametric_sensitivity(type=type)

    def compute_state_value_function_value(self, state: np.ndarray, type=cs.DM) -> Union[np.ndarray, cs.DM]:
        """
        Compute the value of the state.

        Returns:
            Value of the state.
        """
        # if type == cs.DM:
        #     return self.nlp_solution["f"]
        # else:
        #     raise NotImplementedError()

        self.constraints_set(stage_=0, field_="lbx", value_=state)
        self.constraints_set(stage_=0, field_="ubx", value_=state)
        self.set(stage_=0, field_="x", value_=state)

        self.solve()

        return self.nlp_solution["f"]

    def compute_state_value_function_parametric_sensitivity(self, type=cs.DM) -> Union[np.ndarray, cs.DM]:
        """
        Compute the gradient of the state value function.

        Returns:
            Gradient of the state value function.
        """
        return self.compute_lagrange_function_parametric_sensitivity(type=type)

    def compute_lagrange_function_value(self, type=cs.DM) -> float:
        """
        Compute the value of the Lagrange function.

        Returns:
            Value of the Lagrange function.
        """
        if type == cs.DM:
            return self.fun["L"](
                self.nlp.w.val, self.nlp.lbw.val, self.nlp.ubw.val, self.nlp.pi.val, self.nlp.lam.val, self.nlp.p.val
            )
        else:
            raise NotImplementedError()

    def compute_lagrange_function_parametric_sensitivity(self, type=cs.DM) -> Union[np.ndarray, cs.DM]:
        """
        Compute the gradient of the Lagrange function.

        Returns:
            Gradient of the Lagrange function.
        """
        if type == cs.DM:
            return self.fun["dL_dp"](self.nlp.w.val, self.nlp.pi.val, self.nlp.lam.val, self.nlp.p.val)
        else:
            raise NotImplementedError()

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

        if field_ == "p":
            self.nlp.p_solver.val["p", stage_] = value_
            self.nlp.p.val["p", stage_] = value_

        # treat parameters separately
        # if field_ == "p":
        #     self.p = value_

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
            if stage_ == 0:
                self.nlp.lbw.val["lbx", stage_] = value_
            else:
                # TODO: Not tested yet.
                self.nlp.lbw.val["lbx", stage_][self.nlp.idxhbx] = value_[self.nlp.idxhbx]
                self.nlp.p_solver.val["lsbx", stage_] = value_[self.nlp.idxsbx]
        elif field_ == "ubx":
            if stage_ == 0:
                self.nlp.ubw.val["ubx", stage_] = value_
            else:
                # TODO: Not tested yet.
                self.nlp.ubw.val["ubx", stage_][self.nlp.idxhbx] = value_[self.nlp.idxhbx]
                self.nlp.p_solver.val["lsbx", stage_] = value_[self.nlp.idxsbx]
        elif field_ == "lbu":
            self.nlp.lbw.val["lbu", stage_] = value_
        elif field_ == "ubu":
            self.nlp.ubw.val["ubu", stage_] = value_
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
            return self.nlp.w.val["x", stage_].full()

        if field_ == "u":
            return self.nlp.w.val["u", stage_].full()

        if field_ == "slbx":
            return self.nlp.w.val["slbx", stage_].full()

        if field_ == "subx":
            return self.nlp.w.val["subx", stage_].full()

        if field_ == "pi":
            raise NotImplementedError()

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
            x0=self.nlp.w.val,
            lbx=self.nlp.lbw.val,
            ubx=self.nlp.ubw.val,
            lbg=self.nlp.lbg_solver,
            ubg=self.nlp.ubg_solver,
            p=self.nlp.p_solver.val,
        )

        self.nlp.w.val = self.nlp.w.sym(self.nlp_solution["x"])

        ### Build the multipliers for the equality and inequality constraints we use in our NLP formulation (different to solver)
        # TODO: Move the multiplier treatment to a separate function

        g_idx = [item for sublist in self.idx["g"]["pi"] for item in sublist]

        self.nlp.g.val = self.nlp_solution["g"][g_idx]

        self.nlp.pi.val = self.nlp_solution["lam_g"][g_idx]

        self.nlp.h.val = self.fun["h"](self.nlp.w.val, self.nlp.lbw.val, self.nlp.ubw.val)

        lam_x = self.nlp.w.sym(self.nlp_solution["lam_x"])

        # TODO: Use idxsbx, etc. Otherwise possible LICQ problems with redundant hard and soft constraints

        # Multipliers are positive if upper bound is active and negative if lower bound is active
        # https://groups.google.com/g/casadi-users/c/fcjt-JX5BIc/m/cKJGV9h9BwAJ
        for stage_ in range(self.ocp.dims.N - 1):
            self.nlp.lam.val["lbu", stage_] = -cs.fmin(lam_x["u", stage_], 0.0)
            self.nlp.lam.val["ubu", stage_] = +cs.fmax(lam_x["u", stage_], 0.0)

            self.nlp.lam.val["lbx", stage_] = -cs.fmin(lam_x["x", stage_], 0.0)
            self.nlp.lam.val["ubx", stage_] = +cs.fmax(lam_x["x", stage_], 0.0)

            self.nlp.lam.val["lslbx", stage_] = -cs.fmin(lam_x["slbx", stage_], 0.0)
            self.nlp.lam.val["lsubx", stage_] = -cs.fmin(lam_x["subx", stage_], 0.0)

            self.nlp.lam.val["lsbx", stage_] = -cs.fmin(self.nlp_solution["lam_g"][self.idx["g"]["lsbx"][stage_]], 0.0)
            self.nlp.lam.val["usbx", stage_] = +cs.fmax(self.nlp_solution["lam_g"][self.idx["g"]["usbx"][stage_]], 0.0)

        stage_ = self.ocp.dims.N - 1

        self.nlp.lam.val["lbx", stage_] = -cs.fmin(lam_x["x", stage_], 0.0)
        self.nlp.lam.val["ubx", stage_] = +cs.fmax(lam_x["x", stage_], 0.0)

        self.nlp.lam.val["lslbx", stage_] = -cs.fmin(lam_x["slbx", stage_], 0.0)
        self.nlp.lam.val["lsubx", stage_] = -cs.fmin(lam_x["subx", stage_], 0.0)

        self.nlp.lam.val["lsbx", stage_] = -cs.fmin(self.nlp_solution["lam_g"][self.idx["g"]["lsbx"][stage_]], 0.0)
        self.nlp.lam.val["usbx", stage_] = +cs.fmax(self.nlp_solution["lam_g"][self.idx["g"]["usbx"][stage_]], 0.0)

        self.nlp.g.val = self.fun["g"](self.nlp.w.val, self.nlp.p.val)
        self.nlp.h.val = self.fun["h"](self.nlp.w.val, self.nlp.lbw.val, self.nlp.ubw.val)

        # Test if g includes inf or nan
        if np.any(np.isinf(self.nlp.g.val)) or np.any(np.isnan(self.nlp.g.val)):
            raise RuntimeError("g includes inf or nan")

        # Test if h includes inf or nan
        if np.any(np.isinf(self.nlp.h.val)) or np.any(np.isnan(self.nlp.h.val)):
            raise RuntimeError("h includes inf or nan")

        return self.nlp_solution


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


def build_nlp(ocp: CasadiOcp) -> (CasadiNLP, dict, dict):
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
    # TODO: Check if we are converting back and forth between list and np.ndarray too much
    idxbx = constraints.idxbx.tolist()
    idxbu = constraints.idxbu.tolist()

    # Index of soft box constraints
    idxsbx = constraints.idxsbx.tolist()
    idxsbu = constraints.idxsbu.tolist()

    # Index of hard box constraints
    idxhbx = [idx for idx in idxbx if idx not in idxsbx]
    idxhbu = [idx for idx in idxbu if idx not in idxsbu]

    nlp.idxsbx = idxsbx
    nlp.idxhbx = idxhbx

    nlp.idxhbu = idxhbu
    nlp.idxsbu = idxsbu

    # Add states to decision variables
    states = struct_symSX([tuple([entry(label) for label in state_labels])])

    # State at each stage
    x_entry = entry("x", repeat=ocp.dims.N, struct=states)

    # Lower bound of state box constraint
    lbx_entry = entry("lbx", repeat=ocp.dims.N, struct=states)

    # Upper bound of state box constraint
    ubx_entry = entry("ubx", repeat=ocp.dims.N, struct=states)

    # Add inputs to decision variables
    inputs = struct_symSX([tuple([entry(label) for label in input_labels])])

    # Input at each stage
    u_entry = entry("u", repeat=ocp.dims.N - 1, struct=inputs)

    # Lower bound of input box constraint
    lbu_entry = entry("lbu", repeat=ocp.dims.N - 1, struct=inputs)

    # Upper bound of input box constraint
    ubu_entry = entry("ubu", repeat=ocp.dims.N - 1, struct=inputs)

    # Add slack variables for relaxed state box constraints to decision variables
    # TODO: Handle the case when there are no relaxed state box constraints
    # TODO: Add support for relaxed input box constraints
    # TODO: Add support for other types of relaxed constraints

    # Slack variable for lower state box constraint
    slbx_entry = entry("slbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))

    # Lower bound of slack variable for lower state box constraint
    lslbx_entry = entry("lslbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))

    # Upper bound of slack variable for lower state box constraint
    uslbx_entry = entry("uslbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))

    # Slack variable for upper state box constraint
    subx_entry = entry("subx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))

    # Lower bound of slack variable for upper state box constraint
    lsubx_entry = entry("lsubx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))

    # Upper bound of slack variable for upper state box constraint
    usubx_entry = entry("usubx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))

    # Hard lower bound of state box constraint
    hlbx_entry = entry("hlbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxhbx]))

    # Hard upper bound of state box constraint
    hubx_entry = entry("hubx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxhbx]))

    # TODO: Add support for relaxed input box constraints etc.
    nlp.w.sym = struct_symSX([(x_entry, u_entry, slbx_entry, subx_entry)])
    nlp.lbw.sym = struct_symSX([(lbx_entry, lbu_entry, lslbx_entry, lsubx_entry)])
    nlp.ubw.sym = struct_symSX([(ubx_entry, ubu_entry, uslbx_entry, usubx_entry)])

    ############# Parameter vector #############

    # Parameter vector
    # TODO: Add support for multivariable parameters

    p_model_labels = ocp.model.p.str().strip("[]").split(", ")

    p_model_entry = entry("p", repeat=ocp.dims.N, struct=struct_symSX([entry(label) for label in p_model_labels]))
    p_lbx_entry = entry("lsbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in constraints.idxsbx]))
    p_ubx_entry = entry("usbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in constraints.idxsbx]))

    nlp.p_solver.sym = struct_symSX([(p_model_entry, p_lbx_entry, p_ubx_entry)])

    nlp.p.sym = struct_symSX([(p_model_entry)])

    # nlp.p = ocp.model.p

    ############ Build the NLP ############

    ############ Equality constraints ############

    pi_entries = entry("pi", repeat=ocp.dims.N - 1, struct=states)
    nlp.pi.sym = struct_symSX([pi_entries])

    g = []
    lbg = []
    ubg = []

    x, u, slbx, subx = nlp.w.sym[...]
    lbx, lbu, lslbx, lsubx = nlp.lbw.sym[...]
    ubx, ubu, uslbx, usubx = nlp.ubw.sym[...]
    # _, _, _ = nlp.p_solver.sym[...]

    for stage_ in range(ocp.dims.N - 1):
        g.append(
            nlp.w.sym["x", stage_ + 1] - nlp.f_disc(nlp.w.sym["x", stage_], nlp.w.sym["u", stage_], nlp.p.sym["p", stage_])
        )
        lbg.append([0 for _ in range(ocp.dims.nx)])
        ubg.append([0 for _ in range(ocp.dims.nx)])

    nlp.g.sym = cs.vertcat(*g)
    nlp.g.fun = cs.Function("g", [nlp.w.sym, nlp.p.sym], [nlp.g.sym], ["w", "p"], ["g"])

    print(f"pi.shape = {nlp.pi.sym.shape}")
    print(f"g.fun: {nlp.g.fun}")

    ############ Inequality constraints ############

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

    # Build inequality constraint
    lam_entries = []
    h = []

    # Initial conditions x0
    # h += [lbx[0] - x[0]]
    # lam_entries += [entry("lbx_0", struct=struct_symSX([state_labels[idx] for idx in range(ocp.dims.nx)]))]

    # h += [x[0] - ubx[0]]
    # lam_entries += [entry("ubx_0", struct=struct_symSX([state_labels[idx] for idx in range(ocp.dims.nx)]))]

    # # Initial conditions u0
    # h += [lbu[0] - u[0]]
    # lam_entries += [entry("lbu_0", struct=struct_symSX([input_labels[idx] for idx in range(ocp.dims.nu)]))]

    # h += [u[0] - ubu[0]]
    # lam_entries += [entry("ubu_0", struct=struct_symSX([input_labels[idx] for idx in range(ocp.dims.nu)]))]

    # test_entries = []
    # for stage_ in range(3):
    #     test_entries += [entry("lbu", struct=struct_symSX([input_labels[idx] for idx in idxhbu]))]

    # test = struct_symSX([tuple(test_entries)])

    # TODO: Causes singular matrix error due to redundant slack and hard hard constraints in KKT matrix. Remove there for now

    idxhbu = [idx for idx in range(ocp.dims.nu)]
    idxhbx = [idx for idx in range(ocp.dims.nx)]

    for stage_ in range(0, ocp.dims.N - 1):
        if idxhbu:
            h += [lbu[stage_][idxhbu] - u[stage_][idxhbu]]
            if stage_ == 0:
                lam_entries += [
                    entry("lbu", repeat=ocp.dims.N - 1, struct=struct_symSX([input_labels[idx] for idx in idxhbu]))
                ]
        if idxhbx:
            h += [lbx[stage_][idxhbx] - x[stage_][idxhbx]]
            if stage_ == 0:
                lam_entries += [entry("lbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxhbx]))]
        if idxhbu:
            h += [u[stage_][idxhbu] - ubu[stage_][idxhbu]]
            if stage_ == 0:
                lam_entries += [
                    entry("ubu", repeat=ocp.dims.N - 1, struct=struct_symSX([input_labels[idx] for idx in idxhbu]))
                ]
        if idxhbx:
            h += [x[stage_][idxhbx] - ubx[stage_][idxhbx]]
            if stage_ == 0:
                lam_entries += [entry("ubx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxhbx]))]
        if idxsbx:
            h += [+lbx[stage_][idxsbx] - x[stage_][idxsbx] - slbx[stage_]]
            if stage_ == 0:
                lam_entries += [entry("lsbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))]
        if idxsbx:
            h += [x[stage_][idxsbx] - ubx[stage_][idxsbx] - subx[stage_]]
            if stage_ == 0:
                lam_entries += [entry("usbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))]
        if idxsbx:  # s_lbx > 0
            h += [-slbx[stage_]]
            if stage_ == 0:
                lam_entries += [entry("lslbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))]
        if idxsbx:  # s_ubx > 0
            h += [-subx[stage_]]
            if stage_ == 0:
                lam_entries += [entry("lsubx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))]

    if idxhbx:  # lbx_e <= x_e
        h += [lbx[stage_][idxhbx] - x[stage_][idxhbx]]
    if idxhbx:  # x_e <= ubx_e
        h += [x[stage_][idxhbx] - ubx[stage_][idxhbx]]
    if idxsbx:  # lbx_e <= x_e + s_lbx_e
        h += [+lbx[stage_][idxsbx] - x[stage_][idxsbx] - slbx[stage_]]
    if idxsbx:  # x_e - s_ubx_e <= ubx_e
        h += [-ubx[stage_][idxsbx] + x[stage_][idxsbx] - subx[stage_]]
    if idxsbx:  # s_lbx_e > 0
        h += [-slbx[stage_]]
    if idxsbx:  # s_ubx_e > 0
        h += [-subx[stage_]]

    nlp.lam.sym = struct_symSX([tuple(lam_entries)])
    nlp.h.sym = cs.vertcat(*h)
    nlp.h.fun = cs.Function("h", [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym], [nlp.h.sym], ["w", "lbw", "ubw"], ["h"])

    print(f"lam.sym.shape = {nlp.lam.sym.shape}")
    print(f"h.fun: {nlp.h.fun}")

    nlp.g_solver = []
    nlp.lbg_solver = []
    nlp.ubg_solver = []

    idx = dict()
    idx["g"] = dict()
    idx["g"]["pi"] = []
    idx["g"]["lsbx"] = []
    idx["g"]["usbx"] = []

    running_index = 0

    (p, lsbx, usbx) = nlp.p_solver.sym[...]

    for stage_ in range(ocp.dims.N - 1):
        # Add dynamics constraints
        nlp.g_solver.append(x[stage_ + 1] - nlp.f_disc(x[stage_], u[stage_], p[stage_]))
        nlp.lbg_solver.append([0 for _ in range(ocp.dims.nx)])
        nlp.ubg_solver.append([0 for _ in range(ocp.dims.nx)])

        # Add indices for the added elements to g
        idx["g"]["pi"].append([running_index + i for i in range(ocp.dims.nx)])
        running_index = idx["g"]["pi"][-1][-1] + 1

        # Add relaxed box constraints for lower bounds
        nlp.g_solver.append(lsbx[stage_] - x[stage_][idxsbx] - slbx[stage_])
        nlp.lbg_solver.append([-cs.inf for _ in idxsbx])
        nlp.ubg_solver.append([0 for _ in idxsbx])

        # Add indices for the added elements to g
        idx["g"]["lsbx"].append([running_index + i for i in range(len(idxsbx))])
        running_index = idx["g"]["lsbx"][-1][-1] + 1

        # Add relaxed box constraints for upper bounds
        nlp.g_solver.append(-usbx[stage_] + x[stage_][idxsbx] - subx[stage_])
        nlp.lbg_solver.append([-cs.inf for _ in idxsbx])
        nlp.ubg_solver.append([0 for _ in idxsbx])

        # Add indices for the added elements to g
        idx["g"]["usbx"].append([running_index + i for i in range(len(idxsbx))])
        running_index = idx["g"]["usbx"][-1][-1] + 1

    # Add terminal constraints
    stage_ = ocp.dims.N - 1
    # Add relaxed box constraints for lower bounds
    nlp.g_solver.append(lsbx[stage_] - x[stage_][idxsbx] - slbx[stage_])
    nlp.lbg_solver.append([-cs.inf for _ in idxsbx])
    nlp.ubg_solver.append([0 for _ in idxsbx])

    # Add indices for the added elements to g
    idx["g"]["lsbx"].append([running_index + i for i in range(len(idxsbx))])
    running_index = idx["g"]["lsbx"][-1][-1] + 1

    # Add relaxed box constraints for upper bounds
    nlp.g_solver.append(-usbx[stage_] + x[stage_][idxsbx] - subx[stage_])
    nlp.lbg_solver.append([-cs.inf for _ in idxsbx])
    nlp.ubg_solver.append([0 for _ in idxsbx])

    # Add indices for the added elements to g
    idx["g"]["usbx"].append([running_index + i for i in range(len(idxsbx))])
    running_index = idx["g"]["usbx"][-1][-1] + 1

    nlp.g_solver = cs.vertcat(*nlp.g_solver)
    nlp.lbg_solver = cs.vertcat(*nlp.lbg_solver)
    nlp.ubg_solver = cs.vertcat(*nlp.ubg_solver)

    stage_cost_function = define_stage_cost_function(
        x=ocp.model.x,
        u=ocp.model.u,
        sl=cs.SX.sym("sl", ocp.dims.ns),
        su=cs.SX.sym("su", ocp.dims.ns),
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
        sl_e=cs.SX.sym("sl_e", ocp.dims.ns_e),
        su_e=cs.SX.sym("su_e", ocp.dims.ns_e),
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
        nlp.cost += stage_cost_function(x[stage_], u[stage_], slbx[stage_], subx[stage_])

    # Add terminal cost
    stage_ = ocp.dims.N - 1
    nlp.cost += terminal_cost_function(x[stage_], slbx[stage_], subx[stage_])

    # Keep for reference on how to initialize the hard bounds
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

    nlp.lbw.val = nlp.lbw.sym(0)
    nlp.lbw.val["lbx", lambda x: cs.vertcat(*x)] = np.tile(lhbx, (1, ocp.dims.N))
    nlp.lbw.val["lbu", lambda x: cs.vertcat(*x)] = np.tile(lhbu, (1, ocp.dims.N - 1))
    for stage_ in range(ocp.dims.N):
        nlp.lbw.val["lslbx", stage_] = [0 for _ in constraints.idxsbx]
        nlp.lbw.val["lsubx", stage_] = [0 for _ in constraints.idxsbx]

    nlp.ubw.val = nlp.ubw.sym(0)
    nlp.ubw.val["ubx", lambda x: cs.vertcat(*x)] = np.tile(uhbx, (1, ocp.dims.N))
    nlp.ubw.val["ubu", lambda x: cs.vertcat(*x)] = np.tile(uhbu, (1, ocp.dims.N - 1))
    for stage_ in range(ocp.dims.N):
        nlp.ubw.val["uslbx", stage_] = [np.inf for _ in constraints.idxsbx]
        nlp.ubw.val["usubx", stage_] = [np.inf for _ in constraints.idxsbx]

    # Parameter vector
    nlp.p_solver.val = nlp.p_solver.sym(0)
    nlp.p_solver.val["p", lambda x: cs.vertcat(*x)] = np.tile(ocp.parameter_values, (1, ocp.dims.N))
    nlp.p_solver.val["lsbx", lambda x: cs.vertcat(*x)] = np.tile([constraints.lbx[i] for i in idxsbx], (1, ocp.dims.N))
    nlp.p_solver.val["usbx", lambda x: cs.vertcat(*x)] = np.tile([constraints.ubx[i] for i in idxsbx], (1, ocp.dims.N))

    nlp.p.val = nlp.p.sym(0)
    nlp.p.val["p", lambda x: cs.vertcat(*x)] = np.tile(ocp.parameter_values, (1, ocp.dims.N))

    # Initial guess
    x0 = ocp.constraints.lbx_0.tolist()
    u0 = 0
    nlp.w.val = nlp.w.sym(0)
    nlp.w.val["x", lambda x: cs.vertcat(*x)] = np.tile(x0, (1, ocp.dims.N))
    nlp.w.val["u", lambda x: cs.vertcat(*x)] = np.tile(u0, (1, ocp.dims.N - 1))
    for stage_ in range(ocp.dims.N):
        nlp.w.val["slbx", stage_] = [0 for _ in constraints.idxsbx]
        nlp.w.val["subx", stage_] = [0 for _ in constraints.idxsbx]

    # Set multiplier values later after solution
    nlp.lam.val = nlp.lam.sym(0)

    assert nlp.g.fun.size_out(0)[0] == nlp.pi.sym.shape[0], "Dimension mismatch between g (constraints) and pi (multipliers)"
    assert (
        nlp.h.fun.size_out(0)[0] == nlp.lam.sym.shape[0]
    ), "Dimension mismatch between h (inequalities) and lam (multipliers)"
    assert (
        nlp.w.sym.shape[0] == nlp.lbw.sym.shape[0]
    ), "Dimension mismatch between w (decision variables) and lbw (lower bounds)"
    assert (
        nlp.w.sym.shape[0] == nlp.ubw.sym.shape[0]
    ), "Dimension mismatch between w (decision variables) and ubw (upper bounds)"

    return nlp, idx


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

        self.ocp.cost = define_acados_cost(ocp=self.ocp, config=config)

        self.ocp.constraints = define_acados_constraints(ocp=self.ocp, config=config)

        self.ocp.dims = define_acados_dims(ocp=self.ocp, config=config)

        self.ocp.dims.nsbx = self.ocp.constraints.idxsbx.shape[0]
        self.ocp.dims.nsbu = self.ocp.constraints.idxsbu.shape[0]

        self.ocp.solver_options = config.ocp_options

        # self.ocp_solver = CasadiOcpSolver(self.ocp, build=build)

        self.v_ocp_solver = CasadiOcpSolver(self.ocp, build=build, name="v_ocp_solver")
        self.q_ocp_solver = CasadiOcpSolver(self.ocp, build=build, name="q_ocp_solver")

        self.parameter_values = self.ocp.parameter_values

        # TODO: At the moment we only support one parameter for all stages. Add support for stage-wise parameters.
        for stage_ in range(self.ocp.dims.N):
            self.q_ocp_solver.set(stage_, "p", self.ocp.parameter_values)
            self.v_ocp_solver.set(stage_, "p", self.ocp.parameter_values)

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

    def get_action(self, x0: np.ndarray, solve: bool = True) -> np.ndarray:
        """
        Update the solution of the OCP solver.

        Args:
            x0: Initial state.

        Returns:
            action: Scaled optimal control action.
        """

        if solve:
            # Set initial state
            self.v_ocp_solver.constraints_set(0, "lbx", x0)
            self.v_ocp_solver.constraints_set(0, "ubx", x0)

            # Solve the optimization problem
            self.v_ocp_solver.solve()

        # Get solution
        action = self.v_ocp_solver.get(0, "u")

        # Scale to [-1, 1] for gym
        action = self.scale_action(action)

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
