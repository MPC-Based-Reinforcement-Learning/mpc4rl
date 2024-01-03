from typing import Union
import numpy as np
import casadi as cs
from casadi.tools import struct_symMX, struct_MX, struct_symSX, struct_SX, entry
from casadi.tools import *

from rlmpc.common.mpc import MPC
from rlmpc.mpc.cartpole.common import define_parameter_values, define_discrete_dynamics_function
from rlmpc.mpc.utils import ERK4

import matplotlib.pyplot as plt

from rlmpc.mpc.cartpole.common import (
    Config,
    define_model_expressions,
    define_dimensions,
    define_cost,
    define_constraints,
    define_stage_cost_function,
    define_terminal_cost_function,
    CasadiNLP,
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


class CasadiOcpSolver:
    """docstring for CasadiOcp."""

    # _model: dict
    # _cost: cs.Function
    # _constraints: cs.Function

    ocp: AcadosOcp
    nlp: CasadiNLP
    p: np.ndarray
    nlp_solution: dict

    # Idx to keep track of constraints and multipliers
    idx: dict

    # Use generate and build mehods to implement jit compilation
    @classmethod
    def generate(cls, acados_ocp: AcadosOcp):
        pass

    @classmethod
    def build(cls, acados_ocp: AcadosOcp):
        pass

    def __init__(self, _ocp: AcadosOcp, build, name: str = "ocp_solver", code_export_dir: str = "c_generated_code"):
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

        # Print nlp.h_licq.sym and nlp.lam_licq.sym side by side for each element
        # for i in range(nlp.h_licq.sym.shape[0]):
        #     print(f"{nlp.h_licq.sym[i]} \t {nlp.lam_licq.sym[i]}")

        if True:
            # Define the Lagrangian
            nlp.L.sym = nlp.cost + cs.dot(nlp.pi.sym, nlp.g.sym) + cs.dot(nlp.lam_licq.sym, nlp.h.sym)

            # L += cs.dot(nlp.lam.sym[0])

            # a = cs.DM.ones(3, 1)
            # b = cs.DM.ones(3, 1)
            # b[0] = 1
            # b[1] = 2
            # b[2] = 3

            # print(cs.mtimes([a.T, b]))
            # print(cs.dot(a, b))

            # Define the Lagrangian gradient with respect to the decision variables
            nlp.dL_dw.sym = cs.jacobian(nlp.L.sym, nlp.w.sym)

            # nlp.ddL_dwdw.sym = cs.jacobian(nlp.dL_dw.sym, nlp.w.sym)
            # nlp.ddL_dwpi.sym = cs.jacobian(nlp.dL_dw.sym, nlp.pi.sym)
            # nlp.ddL_dwlam.sym = cs.jacobian(nlp.dL_dw.sym, nlp.lam_licq.sym)

            # Define the Lagrangian gradient with respect to the parameters
            # TODO: Add support for multivariable parameters
            nlp.dL_dp.sym = cs.jacobian(nlp.L.sym, nlp.p.sym)

            # TODO: Move etau to solver options
            etau = 10e-8
            # R_kkt = cs.vertcat(cs.transpose(dL_dw), nlp.g, nlp.lam.cat * nlp.h + etau)

            # Concatenate primal-dual variables
            z = cs.vertcat(nlp.w.sym, nlp.pi.sym, nlp.lam_licq.sym)

            # Build KKT matrix
            nlp.R.sym = cs.vertcat(cs.transpose(nlp.dL_dw.sym), nlp.g.sym, nlp.lam_licq.sym * nlp.h.sym + etau)

            # Generate sensitivity of the KKT matrix with respect to primal-dual variables
            dR_dz = cs.jacobian(nlp.R.sym, z)

            fun = dict()
            fun["dR_dw"] = cs.Function(
                "dR_dw", [nlp.w.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym], [cs.jacobian(nlp.R.sym, nlp.w.sym)]
            )
            fun["dR_dpi"] = cs.Function(
                "dR_dpi", [nlp.w.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym], [cs.jacobian(nlp.R.sym, nlp.pi.sym)]
            )
            fun["dR_dlam"] = cs.Function(
                "dR_dlam",
                [nlp.w.sym, nlp.pi.sym, nlp.lam.sym, nlp.lbw.sym, nlp.ubw.sym, nlp.p.sym, nlp.p_solver.sym],
                [cs.jacobian(nlp.R.sym, nlp.lam_licq.sym)],
            )

            # Generate sensitivity of the KKT matrix with respect to parameters
            nlp.dR_dp.sym = cs.jacobian(nlp.R.sym, nlp.p.sym)

            fun["L"] = cs.Function(
                "L",
                [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym, nlp.p_solver.sym],
                [nlp.L.sym],
                ["w", "lbw", "ubw", "pi", "lam", "p", "p_solver"],
                ["L"],
            )
            fun["dL_dp"] = cs.Function(
                "dL_dp",
                [nlp.w.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym, nlp.p_solver.sym],
                [nlp.dL_dp.sym],
                ["w", "pi", "lam", "p", "p_solver"],
                ["dL_dp"],
            )
            fun["dL_dw"] = cs.Function("dL_dw", [nlp.w.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym], [nlp.dL_dw.sym])

            # fun["cost"] = cs.Function("cost", [nlp.w], [nlp.cost])
            fun["g"] = cs.Function("g", [nlp.w.sym, nlp.p.sym], [nlp.g.sym], ["w", "p"], ["g"])

            fun["h"] = cs.Function(
                "h",
                [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym, nlp.p_solver.sym],
                [nlp.h.sym],
                ["w", "lbw", "ubw", "p_solver"],
                ["h"],
            )
            # fun["casadi"]["dg_dw"] = cs.Function("dg_dw", [nlp.w, nlp.p], [cs.jacobian(nlp.g, nlp.w)])
            # fun["casadi"]["dh_dw"] = cs.Function("dh_dw", [nlp.w, nlp.p], [cs.jacobian(nlp.h, nlp.w)])
            # fun["casadi"]["dR_dz"] = cs.Function(
            #     "dR_dz",
            #     [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym, nlp.g.sym, nlp.h.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym],
            #     [dR_dz],
            # )

            fun["R"] = cs.Function(
                "R", [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym, nlp.p_solver.sym], [nlp.R.sym]
            )

            fun["z"] = cs.Function("z", [nlp.w.sym, nlp.pi.sym, nlp.lam.sym], [z], ["w", "pi", "lam"], ["z"])

            fun["dR_dz"] = cs.Function(
                "dR_dz",
                [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym, nlp.p_solver.sym],
                [dR_dz],
                ["w", "lbw", "ubw", "pi", "lam", "p", "p_solver"],
                ["dR_dz"],
            )

            fun["dR_dp"] = cs.Function(
                "dR_dp",
                [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym],
                [nlp.dR_dp.sym],
                ["w", "lbw", "ubw", "pi", "lam", "p"],
                ["dR_dp"],
            )

            # fun["casadi"]["dR_dp"] = cs.Function("dR_dp", [z, nlp.p], [dR_dp])

            if build:
                # Make code_export_dir if it does not exist
                if not os.path.exists(code_export_dir):
                    os.makedirs(code_export_dir)

                # Generate  and compile c-code for the functions
                for key, val in fun.items():
                    # val.generate(f"{code_export_dir}/{name}_{key}.c", {"mex": False})
                    val.generate(f"{name}_{key}.c", {"mex": False})

                    # Move the generated c-code to code_export_dir
                    os.rename(f"{name}_{key}.c", f"{code_export_dir}/{name}_{key}.c")

                    # Compile using os.system
                    status = os.system(
                        f"gcc -fPIC -shared {code_export_dir}/{name}_{key}.c -o {code_export_dir}/{name}_{key}.so"
                    )

                    if status != 0:
                        raise Exception(f"Error compiling {code_export_dir}/{name}_{key}.c")

                    fun[key] = cs.external(key, f"{code_export_dir}/{name}_{key}.so")

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

            self.nlp = nlp

            self.nlp.L.fun = fun["L"]
            self.nlp.dL_dp.fun = fun["dL_dp"]
            self.nlp.dL_dw.fun = fun["dL_dw"]
            self.nlp.R.fun = fun["R"]
            self.nlp.dR_dw.fun = fun["dR_dw"]
            self.nlp.dR_dp.fun = fun["dR_dp"]

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
            self.nlp.w.val,
            self.nlp.lbw.val,
            self.nlp.ubw.val,
            self.nlp.pi.val,
            self.nlp.lam.val,
            self.nlp.p.val,
            self.nlp.p_solver.val,
        )

        dR_dp = self.fun["dR_dp"](
            self.nlp.w.val,
            self.nlp.lbw.val,
            self.nlp.ubw.val,
            self.nlp.pi.val,
            self.nlp.lam.val,
            self.nlp.p.val,
        )

        # dR_dz_inv = np.linalg.inv(dR_dz)

        R = self.fun["R"](
            self.nlp.w.val,
            self.nlp.lbw.val,
            self.nlp.ubw.val,
            self.nlp.pi.val,
            self.nlp.lam.val,
            self.nlp.p.val,
            self.nlp.p_solver.val,
        )

        # Check where R is zero

        # Test if dR_dp is singular
        if np.linalg.matrix_rank(dR_dp) != dR_dp.shape[0]:
            print("dR_dp is singular")

        # Test if dR_dz is singular
        if np.linalg.matrix_rank(dR_dz) != dR_dz.shape[0]:
            print("dR_dz is singular")

        exit(0)

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
                self.nlp.w.val,
                self.nlp.lbw.val,
                self.nlp.ubw.val,
                self.nlp.pi.val,
                self.nlp.lam.val,
                self.nlp.p.val,
                self.nlp.p_solver.val,
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
            return self.fun["dL_dp"](self.nlp.w.val, self.nlp.pi.val, self.nlp.lam.val, self.nlp.p.val, self.nlp.p_solver.val)
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

    def constraints_get(self, stage_, field_, value_):
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
                return self.nlp.lbw.val["lbx", stage_]
            else:
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

        self.nlp.h.val = self.fun["h"](self.nlp.w.val, self.nlp.lbw.val, self.nlp.ubw.val, self.nlp.p_solver.val)

        lam_x = self.nlp.w.sym(self.nlp_solution["lam_x"])

        # TODO: Use idxsbx, etc. Otherwise possible LICQ problems with redundant hard and soft constraints

        # Multipliers are positive if upper bound is active and negative if lower bound is active
        # https://groups.google.com/g/casadi-users/c/fcjt-JX5BIc/m/cKJGV9h9BwAJ
        for stage_ in range(self.ocp.dims.N - 1):
            self.nlp.lam.val["lbu", stage_] = -cs.fmin(lam_x["u", stage_], 0.0)
            self.nlp.lam.val["ubu", stage_] = +cs.fmax(lam_x["u", stage_], 0.0)

            self.nlp.lam.val["lbx", stage_] = -cs.fmin(lam_x["x", stage_], 0.0)
            self.nlp.lam.val["ubx", stage_] = +cs.fmax(lam_x["x", stage_], 0.0)

        # Check if lslbx is in self.nlp.lam.val.keys()
        if "lslbx" in self.nlp.lam.val.keys():
            for stage_ in range(self.ocp.dims.N - 1):
                self.nlp.lam.val["lslbx", stage_] = -cs.fmin(lam_x["slbx", stage_], 0.0)
                self.nlp.lam.val["lsubx", stage_] = -cs.fmin(lam_x["subx", stage_], 0.0)

                self.nlp.lam.val["lsbx", stage_] = -cs.fmin(self.nlp_solution["lam_g"][self.idx["g"]["lsbx"][stage_]], 0.0)
                self.nlp.lam.val["usbx", stage_] = +cs.fmax(self.nlp_solution["lam_g"][self.idx["g"]["usbx"][stage_]], 0.0)

        stage_ = self.ocp.dims.N - 1

        self.nlp.lam.val["lbx", stage_] = -cs.fmin(lam_x["x", stage_], 0.0)
        self.nlp.lam.val["ubx", stage_] = +cs.fmax(lam_x["x", stage_], 0.0)

        if "lslbx" in self.nlp.lam.val.keys():
            self.nlp.lam.val["lslbx", stage_] = -cs.fmin(lam_x["slbx", stage_], 0.0)
            self.nlp.lam.val["lsubx", stage_] = -cs.fmin(lam_x["subx", stage_], 0.0)

            self.nlp.lam.val["lsbx", stage_] = -cs.fmin(self.nlp_solution["lam_g"][self.idx["g"]["lsbx"][stage_]], 0.0)
            self.nlp.lam.val["usbx", stage_] = +cs.fmax(self.nlp_solution["lam_g"][self.idx["g"]["usbx"][stage_]], 0.0)

        # Test if g includes inf or nan
        if np.any(np.isinf(self.nlp.g.val)) or np.any(np.isnan(self.nlp.g.val)):
            raise RuntimeError("g includes inf or nan")

        # Test if h includes inf or nan
        if np.any(np.isinf(self.nlp.h.val)) or np.any(np.isnan(self.nlp.h.val)):
            # Check which h is nan or inf
            # for i in range(self.nlp.h.val.shape[0]):
            #     if np.any(np.isinf(self.nlp.h.val[i])) or np.any(np.isnan(self.nlp.h.val[i])):
            #         print(
            #             f"i: {i}, h: {self.nlp.h.val[i]}, expression: {self.nlp.h.sym[i]}, lbw: {self.nlp.lbw.val.cat[i]}, ubw: {self.nlp.ubw.val.cat[i]}"
            #         )
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


def build_nlp(ocp: AcadosOcp) -> (CasadiNLP, dict, dict):
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

    entries = {"w": [], "lbw": [], "ubw": []}

    # entries["w"].append(u_entry)

    # State at each stage
    entries["w"].append(entry("x", repeat=ocp.dims.N, struct=states))

    # Lower bound of state box constraint
    # lbx_entry = entry("lbx", repeat=ocp.dims.N, struct=states)
    entries["lbw"].append(entry("lbx", repeat=ocp.dims.N, struct=states))

    # Upper bound of state box constraint
    # ubx_entry = entry("ubx", repeat=ocp.dims.N, struct=states)
    entries["ubw"].append(entry("ubx", repeat=ocp.dims.N, struct=states))

    # Add inputs to decision variables
    inputs = struct_symSX([tuple([entry(label) for label in input_labels])])

    # Input at each stage
    # u_entry = entry("u", repeat=ocp.dims.N - 1, struct=inputs)
    entries["w"].append(entry("u", repeat=ocp.dims.N - 1, struct=inputs))

    # Lower bound of input box constraint
    # lbu_entry = entry("lbu", repeat=ocp.dims.N - 1, struct=inputs)
    entries["lbw"].append(entry("lbu", repeat=ocp.dims.N - 1, struct=inputs))

    # Upper bound of input box constraint
    # ubu_entry = entry("ubu", repeat=ocp.dims.N - 1, struct=inputs)
    entries["ubw"].append(entry("ubu", repeat=ocp.dims.N - 1, struct=inputs))

    # Add slack variables for relaxed state box constraints to decision variables
    # TODO: Handle the case when there are no relaxed state box constraints
    # TODO: Add support for relaxed input box constraints
    # TODO: Add support for other types of relaxed constraints

    # Slack variable for lower state box constraint
    if idxsbx:
        # slbx_entry = entry("slbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))
        entries["w"].append(entry("slbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx])))

        # Lower bound of slack variable for lower state box constraint
        # lslbx_entry = entry("lslbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))
        entries["lbw"].append(entry("lslbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx])))

        # Upper bound of slack variable for lower state box constraint
        # uslbx_entry = entry("uslbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))
        entries["ubw"].append(entry("uslbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx])))

        # Slack variable for upper state box constraint
        # subx_entry = entry("subx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))
        entries["w"].append(entry("subx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx])))

        # Lower bound of slack variable for upper state box constraint
        # lsubx_entry = entry("lsubx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))
        entries["lbw"].append(entry("lsubx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx])))

        # Upper bound of slack variable for upper state box constraint
        # usubx_entry = entry("usubx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))
        entries["ubw"].append(entry("usubx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx])))

    # Hard lower bound of state box constraint
    hlbx_entry = entry("hlbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxhbx]))

    # Hard upper bound of state box constraint
    hubx_entry = entry("hubx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxhbx]))

    # TODO: Add support for relaxed input box constraints etc.
    # nlp.w.sym = struct_symSX([(x_entry, u_entry, slbx_entry, subx_entry)])

    # Convert w_entries to tuple

    nlp.w.sym = struct_symSX([tuple(entries["w"])])
    # nlp.lbw.sym = struct_symSX([(lbx_entry, lbu_entry, lslbx_entry, lsubx_entry)])
    nlp.lbw.sym = struct_symSX([tuple(entries["lbw"])])
    # nlp.ubw.sym = struct_symSX([(ubx_entry, ubu_entry, uslbx_entry, usubx_entry)])
    nlp.ubw.sym = struct_symSX([tuple(entries["ubw"])])

    ############# Parameter vector #############

    # Parameter vector
    # TODO: Add support for multivariable parameters

    p_model_labels = ocp.model.p.str().strip("[]").split(", ")

    entries["p_solver"] = []
    entries["p"] = []

    # p_model_entry = entry("p", repeat=ocp.dims.N, struct=struct_symSX([entry(label) for label in p_model_labels]))
    entries["p_solver"].append(entry("p", repeat=ocp.dims.N, struct=struct_symSX([entry(label) for label in p_model_labels])))
    entries["p"].append(entry("p", repeat=ocp.dims.N, struct=struct_symSX([entry(label) for label in p_model_labels])))

    # p_lbx_entry = entry("lsbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in constraints.idxsbx]))
    entries["p_solver"].append(
        entry("lsbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in constraints.idxsbx]))
    )

    # p_ubx_entry = entry("usbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in constraints.idxsbx]))
    entries["p_solver"].append(
        entry("usbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in constraints.idxsbx]))
    )

    nlp.p_solver.sym = struct_symSX([tuple(entries["p_solver"])])

    # nlp.p.sym = struct_symSX([(p_model_entry)])
    nlp.p.sym = struct_symSX([tuple(entries["p"])])

    # nlp.p = ocp.model.p

    ############ Build the NLP ############

    ############ Equality constraints ############

    # pi_entries = entry("pi", repeat=ocp.dims.N - 1, struct=states)
    nlp.pi.sym = struct_symSX([entry("pi", repeat=ocp.dims.N - 1, struct=states)])

    g = []
    lbg = []
    ubg = []

    # x, u, slbx, subx = nlp.w.sym[...]
    # lbx, lbu, lslbx, lsubx = nlp.lbw.sym[...]
    # ubx, ubu, uslbx, usubx = nlp.ubw.sym[...]
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

    running_index = 0

    idx = dict()
    idx["h"] = dict()
    idx["h"]["lbu"] = []
    idx["h"]["lbx"] = []
    idx["h"]["ubu"] = []
    idx["h"]["ubx"] = []
    idx["h"]["lsbx"] = []
    idx["h"]["usbx"] = []
    idx["h"]["lslbx"] = []
    idx["h"]["lsubx"] = []

    lam = []

    for stage_ in range(0, ocp.dims.N):
        print(stage_)
        if idxhbu:
            if stage_ == 0:
                h += [nlp.lbw.sym["lbu", stage_] - nlp.w.sym["u", stage_]]

                idx["h"]["lbu"].append([running_index + i for i in range(ocp.dims.nu)])
                lam_entries += [entry("lbu", repeat=ocp.dims.N - 1, struct=struct_symSX(input_labels))]
                running_index = idx["h"]["lbu"][-1][-1] + 1
            elif 0 < stage_ < ocp.dims.N - 1:
                h += [nlp.lbw.sym["lbu", stage_][idxhbu] - nlp.w.sym["u", stage_][idxhbu]]
                idx["h"]["lbu"].append([(running_index + i) for i in range(len(idxhbu))])
                running_index = idx["h"]["lbu"][-1][-1] + 1
            else:
                pass

            print(f"Running index = {running_index}")
        if idxhbx:
            if stage_ == 0:
                h += [nlp.lbw.sym["lbx", stage_] - nlp.w.sym["x", stage_]]
                idx["h"]["lbx"].append([running_index + i for i in range(ocp.dims.nx)])
                lam_entries += [entry("lbx", repeat=ocp.dims.N, struct=struct_symSX(state_labels))]
            else:
                h += [nlp.lbw.sym["lbx", stage_][idxhbx] - nlp.w.sym["x", stage_][idxhbx]]
                idx["h"]["lbx"].append([running_index + i for i in range(len(idxhbx))])

            running_index = idx["h"]["lbx"][-1][-1] + 1
            print(f"Running index = {running_index}")
        if idxhbu:
            if stage_ == 0:
                h += [nlp.ubw.sym["ubu", stage_] - nlp.w.sym["u", stage_]]
                idx["h"]["ubu"].append([running_index + i for i in range(ocp.dims.nu)])
                lam_entries += [entry("ubu", repeat=ocp.dims.N - 1, struct=struct_symSX(input_labels))]
                running_index = idx["h"]["ubu"][-1][-1] + 1
            elif 0 < stage_ < ocp.dims.N - 1:
                h += [nlp.ubw.sym["ubu", stage_][idxhbu] - nlp.w.sym["u", stage_][idxhbu]]
                idx["h"]["ubu"].append([running_index + i for i in range(len(idxhbu))])
                running_index = idx["h"]["ubu"][-1][-1] + 1
            else:
                pass

            print(f"Running index = {running_index}")
        if idxhbx:
            if stage_ == 0:
                h += [nlp.ubw.sym["ubx", stage_] - nlp.w.sym["x", stage_]]
                idx["h"]["ubx"].append([running_index + i for i in range(ocp.dims.nx)])
                lam_entries += [entry("ubx", repeat=ocp.dims.N, struct=struct_symSX(state_labels))]
            else:
                # h += [ubx[stage_][idxhbx] - x[stage_][idxhbx]]
                h += [nlp.ubw.sym["ubx", stage_][idxhbx] - nlp.w.sym["x", stage_][idxhbx]]
                idx["h"]["ubx"].append([running_index + i for i in range(len(idxhbx))])

            running_index = idx["h"]["ubx"][-1][-1] + 1
            print(f"Running index = {running_index}")

        if idxsbx:
            if stage_ == 0:
                lam_entries += [entry("lsbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))]
            else:
                # h += [+lbx[stage_][idxsbx] - x[stage_][idxsbx] - slbx[stage_]]
                h += [nlp.p_solver.sym["lsbx", stage_] - nlp.w.sym["x", stage_][idxsbx] - nlp.w.sym["slbx", stage_]]
                idx["h"]["lsbx"].append([running_index + i for i in range(len(idxsbx))])
                running_index = idx["h"]["lsbx"][-1][-1] + 1

            print(f"Running index = {running_index}")

        if idxsbx:
            if stage_ == 0:
                lam_entries += [entry("usbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))]
            else:
                # h += [x[stage_][idxsbx] - ubx[stage_][idxsbx] - subx[stage_]]
                h += [nlp.w.sym["x", stage_][idxsbx] - nlp.p_solver.sym["usbx", stage_] - nlp.w.sym["subx", stage_]]
                idx["h"]["usbx"].append([running_index + i for i in range(len(idxsbx))])

                running_index = idx["h"]["usbx"][-1][-1] + 1

            print(f"Running index = {running_index}")

        if idxsbx:  # s_lbx > 0
            if stage_ == 0:
                lam_entries += [entry("lslbx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))]
            else:
                # h += [-slbx[stage_]]
                h += [-nlp.w.sym["slbx", stage_]]
                idx["h"]["lslbx"].append([running_index + i for i in range(len(idxsbx))])
                running_index = idx["h"]["lslbx"][-1][-1] + 1

            print(f"Running index = {running_index}")
        if idxsbx:  # s_ubx > 0
            if stage_ == 0:
                lam_entries += [entry("lsubx", repeat=ocp.dims.N, struct=struct_symSX([state_labels[idx] for idx in idxsbx]))]
            else:
                # h += [-subx[stage_]]
                h += [-nlp.w.sym["subx", stage_]]
                idx["h"]["lsubx"].append([running_index + i for i in range(len(idxsbx))])
                running_index = idx["h"]["lsubx"][-1][-1] + 1

            print(f"Running index = {running_index}")

    # if idxhbx:  # lbx_e <= x_e
    #     h += [lbx[stage_] - x[stage_]]
    #     idx["h"]["lbx"].append([running_index + i for i in range(ocp.dims.nx)])
    #     # idx["h_licq"]["lbx"].append([idx["h"]["lbx"][-1][i] for i in idxhbx])

    #     running_index = idx["h"]["lbx"][-1][-1] + 1
    # if idxhbx:  # x_e <= ubx_e
    #     h += [x[stage_] - ubx[stage_]]
    #     idx["h"]["ubx"].append([running_index + i for i in range(ocp.dims.nx)])
    #     # idx["h_licq"]["ubx"].append([idx["h"]["ubx"][-1][i] for i in idxhbx])
    #     running_index = idx["h"]["ubx"][-1][-1] + 1
    # if idxsbx:  # lbx_e <= x_e + s_lbx_e
    #     h += [+lbx[stage_][idxsbx] - x[stage_][idxsbx] - slbx[stage_]]
    #     idx["h"]["lsbx"].append([running_index + i for i in range(len(idxsbx))])
    #     # idx["h_licq"]["lsbx"].append([running_index + i for i in range(len(idxsbx))])
    #     running_index = idx["h"]["lsbx"][-1][-1] + 1
    # if idxsbx:  # x_e - s_ubx_e <= ubx_e
    #     h += [-ubx[stage_][idxsbx] + x[stage_][idxsbx] - subx[stage_]]
    #     idx["h"]["usbx"].append([running_index + i for i in range(len(idxsbx))])
    #     # idx["h_licq"]["usbx"].append([running_index + i for i in range(len(idxsbx))])
    #     running_index = idx["h"]["usbx"][-1][-1] + 1
    # if idxsbx:  # s_lbx_e > 0
    #     h += [-slbx[stage_]]
    #     idx["h"]["lslbx"].append([running_index + i for i in range(len(idxsbx))])
    #     # idx["h_licq"]["lslbx"].append([running_index + i for i in range(len(idxsbx))])
    #     running_index = idx["h"]["lslbx"][-1][-1] + 1
    # if idxsbx:  # s_ubx_e > 0
    #     h += [-subx[stage_]]
    #     idx["h"]["lsubx"].append([running_index + i for i in range(len(idxsbx))])
    #     # idx["h_licq"]["lsubx"].append([running_index + i for i in range(len(idxsbx))])
    #     running_index = idx["h"]["lsubx"][-1][-1] + 1

    nlp.lam.sym = struct_symSX([tuple(lam_entries)])
    nlp.h.sym = cs.vertcat(*h)

    # Remove constraints in lbx, ubx, lbu, ubu that are in idxsbx, idxsbu but not in idxhbx, idxhbu from nlp.lam.sym and nlp.h.sym
    # TODO: Add if conditions for idxhbx, idxhbu, idxsbx, idxsbu
    lam_licq = []
    lam_licq += [nlp.lam.sym["lbu"][0]]
    lam_licq += [nlp.lam.sym["lbx"][0]]
    lam_licq += [nlp.lam.sym["ubu"][0]]
    lam_licq += [nlp.lam.sym["ubx"][0]]

    for stage_ in range(1, ocp.dims.N - 1):
        lam_licq += [nlp.lam.sym["lbu", stage_][i] for i in idxhbu]
        lam_licq += [nlp.lam.sym["lbx", stage_][i] for i in idxhbx]
        lam_licq += [nlp.lam.sym["ubu", stage_][i] for i in idxhbu]
        lam_licq += [nlp.lam.sym["ubx", stage_][i] for i in idxhbx]
        if idxsbx:
            lam_licq += [nlp.lam.sym["lsbx", stage_]]
            lam_licq += [nlp.lam.sym["usbx", stage_]]
            lam_licq += [nlp.lam.sym["lslbx", stage_]]
            lam_licq += [nlp.lam.sym["lsubx", stage_]]

    stage_ = ocp.dims.N - 1

    lam_licq += [nlp.lam.sym["lbx", stage_][i] for i in idxhbx]
    lam_licq += [nlp.lam.sym["ubx", stage_][i] for i in idxhbx]
    if idxsbx:
        lam_licq += [nlp.lam.sym["lsbx", stage_]]
        lam_licq += [nlp.lam.sym["usbx", stage_]]
        lam_licq += [nlp.lam.sym["lslbx", stage_]]
        lam_licq += [nlp.lam.sym["lsubx", stage_]]

    # assert nlp.lam_licq.sym.shape[0] == nlp.h_licq.sym.shape[0], "Error in building the NLP h(x, u, p) function"
    assert running_index == nlp.h.sym.shape[0], "Error in building the NLP h(x, u, p) function"

    print(f"lam.sym.shape = {nlp.lam.sym.shape}")
    print(f"h.fun: {nlp.h.fun}")

    nlp.h.fun = cs.Function(
        "h", [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym, nlp.p_solver.sym], [nlp.h.sym], ["w", "lbw", "ubw", "p_solver"], ["h"]
    )
    # nlp.h_licq.fun = cs.Function(
    #     "h_licq", [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym], [nlp.h_licq.sym], ["w", "lbw", "ubw"], ["h_licq"]
    # )
    nlp.lam_licq.sym = cs.vertcat(*lam_licq)

    nlp.lam_licq.fun = cs.Function("lam_licq", [nlp.lam.sym], [nlp.lam_licq.sym], ["lam_all"], ["lam_licq"])

    nlp.g_solver = []
    nlp.lbg_solver = []
    nlp.ubg_solver = []

    running_index = 0

    # (p, lsbx, usbx) = nlp.p_solver.sym[...]

    idx["g"] = dict()
    idx["g"]["pi"] = []
    idx["g"]["lsbx"] = []
    idx["g"]["usbx"] = []

    for stage_ in range(ocp.dims.N - 1):
        # Add dynamics constraints
        # nlp.g_solver.append(x[stage_ + 1] - nlp.f_disc(x[stage_], u[stage_], p[stage_]))
        nlp.g_solver.append(
            nlp.w.sym["x", stage_ + 1]
            - nlp.f_disc(nlp.w.sym["x", stage_], nlp.w.sym["u", stage_], nlp.p_solver.sym["p", stage_])
        )
        nlp.lbg_solver.append([0 for _ in range(ocp.dims.nx)])
        nlp.ubg_solver.append([0 for _ in range(ocp.dims.nx)])

        # Add indices for the added elements to g
        idx["g"]["pi"].append([running_index + i for i in range(ocp.dims.nx)])
        running_index = idx["g"]["pi"][-1][-1] + 1

        if idxsbx:
            # Add relaxed box constraints for lower bounds
            # nlp.g_solver.append(lsbx[stage_] - x[stage_][idxsbx] - slbx[stage_])
            nlp.g_solver.append(nlp.p_solver.sym["lsbx", stage_] - nlp.w.sym["x", stage_][idxsbx] - nlp.w.sym["slbx", stage_])
            nlp.lbg_solver.append([-cs.inf for _ in idxsbx])
            nlp.ubg_solver.append([0 for _ in idxsbx])

            # Add indices for the added elements to g
            idx["g"]["lsbx"].append([running_index + i for i in range(len(idxsbx))])
            running_index = idx["g"]["lsbx"][-1][-1] + 1

            # Add relaxed box constraints for upper bounds
            # nlp.g_solver.append(-usbx[stage_] + x[stage_][idxsbx] - subx[stage_])
            nlp.g_solver.append(-nlp.p_solver.sym["usbx", stage_] + nlp.w.sym["x", stage_][idxsbx] - nlp.w.sym["subx", stage_])
            nlp.lbg_solver.append([-cs.inf for _ in idxsbx])
            nlp.ubg_solver.append([0 for _ in idxsbx])

            # Add indices for the added elements to g
            idx["g"]["usbx"].append([running_index + i for i in range(len(idxsbx))])
            running_index = idx["g"]["usbx"][-1][-1] + 1

    # Add terminal constraints
    stage_ = ocp.dims.N - 1

    if idxsbx:
        # Add relaxed box constraints for lower bounds
        # nlp.g_solver.append(lsbx[stage_] - x[stage_][idxsbx] - slbx[stage_])
        nlp.g_solver.append(nlp.p_solver["lsbx", stage_] - nlp.w.sym["x", stage_][idxsbx] - nlp.w.sym["slbx", stage_])
        nlp.lbg_solver.append([-cs.inf for _ in idxsbx])
        nlp.ubg_solver.append([0 for _ in idxsbx])

        # Add indices for the added elements to g
        idx["g"]["lsbx"].append([running_index + i for i in range(len(idxsbx))])
        running_index = idx["g"]["lsbx"][-1][-1] + 1

        # Add relaxed box constraints for upper bounds
        # nlp.g_solver.append(-usbx[stage_] + x[stage_][idxsbx] - subx[stage_])
        nlp.g_solver.append(-nlp.p_solver.sym["usbx", stage_] + nlp.w.sym["x", stage_][idxsbx] - nlp.w.sym["subx", stage_])
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
        if idxsbx:
            nlp.cost += stage_cost_function(
                nlp.w.sym["x", stage_], nlp.w.sym["u", stage_], nlp.w.sym["slbx", stage_], nlp.w.sym["subx", stage_]
            )
        else:
            nlp.cost += stage_cost_function(nlp.w.sym["x", stage_], nlp.w.sym["u", stage_])

    # Add terminal cost
    stage_ = ocp.dims.N - 1
    if idxsbx:
        nlp.cost += terminal_cost_function(nlp.w.sym["x", stage_], nlp.w.sym["slbx", stage_], nlp.w.sym["subx", stage_])
    else:
        nlp.cost += terminal_cost_function(nlp.w.sym["x", stage_])

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
    if idxsbx:
        for stage_ in range(ocp.dims.N):
            nlp.lbw.val["lslbx", stage_] = [0 for _ in constraints.idxsbx]
            nlp.lbw.val["lsubx", stage_] = [0 for _ in constraints.idxsbx]

    nlp.ubw.val = nlp.ubw.sym(0)
    nlp.ubw.val["ubx", lambda x: cs.vertcat(*x)] = np.tile(uhbx, (1, ocp.dims.N))
    nlp.ubw.val["ubu", lambda x: cs.vertcat(*x)] = np.tile(uhbu, (1, ocp.dims.N - 1))
    if idxsbx:
        for stage_ in range(ocp.dims.N):
            nlp.ubw.val["uslbx", stage_] = [np.inf for _ in constraints.idxsbx]
            nlp.ubw.val["usubx", stage_] = [np.inf for _ in constraints.idxsbx]

    # Parameter vector
    nlp.p_solver.val = nlp.p_solver.sym(0)
    nlp.p_solver.val["p", lambda x: cs.vertcat(*x)] = np.tile(ocp.parameter_values, (1, ocp.dims.N))
    if idxsbx:
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
    if idxsbx:
        for stage_ in range(ocp.dims.N):
            nlp.w.val["slbx", stage_] = [0 for _ in constraints.idxsbx]
            nlp.w.val["subx", stage_] = [0 for _ in constraints.idxsbx]

    # Set multiplier values later after solution
    nlp.lam.val = nlp.lam.sym(0)

    assert nlp.g.fun.size_out(0)[0] == nlp.pi.sym.shape[0], "Dimension mismatch between g (constraints) and pi (multipliers)"
    assert (
        nlp.w.sym.shape[0] == nlp.lbw.sym.shape[0]
    ), "Dimension mismatch between w (decision variables) and lbw (lower bounds)"
    assert (
        nlp.w.sym.shape[0] == nlp.ubw.sym.shape[0]
    ), "Dimension mismatch between w (decision variables) and ubw (upper bounds)"

    return nlp, idx


def build_lagrange_function(nlp: CasadiNLP, ocp: AcadosOcp) -> cs.Function:
    pass


def build_kkt_residual_function(ocp: AcadosOcp) -> cs.Function:
    pass


def build_policy_gradient_function(ocp: AcadosOcp) -> cs.Function:
    pass


def build_state_action_value_function(ocp: AcadosOcp) -> cs.Function:
    pass


def build_state_value_function(ocp: AcadosOcp) -> cs.Function:
    pass


class CasadiMPC(MPC):
    """docstring for CartpoleMPC."""

    ocp: AcadosOcp
    ocp_solver: CasadiOcpSolver
    parameter_values: np.ndarray

    parameter_values: np.ndarray

    def __init__(self, config: Config, build: bool = False):
        super().__init__()

        self.ocp = AcadosOcp()

        self.ocp.model = define_acados_model(ocp=self.ocp, config=config)

        self.ocp.parameter_values = define_parameter_values(ocp=self.ocp, config=config)

        self.ocp.cost = define_acados_cost(ocp=self.ocp, config=config)

        self.ocp.constraints = define_acados_constraints(ocp=self.ocp, config=config)

        self.ocp.dims = define_acados_dims(ocp=self.ocp, config=config)

        self.ocp.dims.nsbx = self.ocp.constraints.idxsbx.shape[0]
        self.ocp.dims.nsbu = self.ocp.constraints.idxsbu.shape[0]

        self.ocp.solver_options = config.ocp_options

        # self.ocp_solver = CasadiOcpSolver(self.ocp, build=build)

        self.v_ocp_solver = CasadiOcpSolver(
            self.ocp, build=build, name="v_ocp_solver", code_export_dir=config.meta.code_export_dir
        )

        self.q_ocp_solver = CasadiOcpSolver(
            self.ocp, build=build, name="q_ocp_solver", code_export_dir=config.meta.code_export_dir
        )

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
