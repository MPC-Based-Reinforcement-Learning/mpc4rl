import casadi as cs
import numpy as np
from typing import Union
from casadi.tools import struct_symSX, entry
from acados_template import AcadosOcp, AcadosOcpSolver

from rlmpc.common.utils import ACADOS_MULTIPLIER_ORDER

from matplotlib import pyplot as plt


def assign_slices_to_idx(stage: int, idx: dict[slice], dims: dict[int]) -> dict[slice]:
    keys = list(dims.keys())
    for i, key in enumerate(dims.keys()):
        if i == 0:
            idx[f"{key}_{stage}"] = slice(0, dims[key])
        else:
            idx[f"{key}_{stage}"] = slice(idx[f"{keys[i-1]}_{stage}"].stop, idx[f"{keys[i-1]}_{stage}"].stop + dims[key])

    return idx


class LagrangeMultiplierMap(object):
    """
    Class to store dimensions of constraints
    """

    order: list = ACADOS_MULTIPLIER_ORDER

    idx_at_stage: list

    def __init__(self, ocp: AcadosOcp):
        super().__init__()

        lam_idx = dict()

        # Initial stage
        lam_idx = assign_slices_to_idx(
            stage=0,
            idx=lam_idx,
            dims={
                "lbu": ocp.dims.nu,
                "lbx": ocp.dims.nx,
                "lh": ocp.dims.nh,
                "ubu": ocp.dims.nu,
                "ubx": ocp.dims.nx,
                "uh": ocp.dims.nh,
                "lsh": ocp.dims.nsh,
                "ush": ocp.dims.nsh,
            },
        )

        # Middle stages
        for stage in range(1, ocp.dims.N):
            lam_idx = assign_slices_to_idx(
                stage=stage,
                idx=lam_idx,
                dims={
                    "lbu": ocp.dims.nbu,
                    "lbx": ocp.dims.nbx,
                    "lh": ocp.dims.nh,
                    "ubu": ocp.dims.nbu,
                    "ubx": ocp.dims.nbx,
                    "uh": ocp.dims.nh,
                    "lsbu": ocp.dims.nsbu,
                    "lsbx": ocp.dims.nsbx,
                    "lsh": ocp.dims.nsh,
                    "usbu": ocp.dims.nsbu,
                    "usbx": ocp.dims.nsbx,
                    "ush": ocp.dims.nsh,
                },
            )

        # Final stage
        lam_idx = assign_slices_to_idx(
            stage=ocp.dims.N,
            idx=lam_idx,
            dims={
                "lbx": ocp.dims.nbx_e,
                "lh": ocp.dims.nh_e,
                "ubx": ocp.dims.nbx_e,
                "uh": ocp.dims.nh_e,
                "lsbx": ocp.dims.nsbx_e,
                "lsh": ocp.dims.nsh_e,
                "usbx": ocp.dims.nsbx_e,
                "ush": ocp.dims.nsh_e,
            },
        )

        # self.idx_at_stage = idx_at_stage
        self.lam_idx = lam_idx

        s_idx = dict()

        # Initial stage
        s_idx = assign_slices_to_idx(stage=0, idx=s_idx, dims={"slbx": ocp.dims.nsbx, "slh": ocp.dims.nsh})
        s_idx = assign_slices_to_idx(stage=0, idx=s_idx, dims={"subx": ocp.dims.nsbx, "suh": ocp.dims.nsh})

        # Middle stages
        for stage in range(1, ocp.dims.N):
            s_idx = assign_slices_to_idx(
                stage=stage, idx=s_idx, dims={"slbu": ocp.dims.nsbu, "slbx": ocp.dims.nsbx, "slh": ocp.dims.nsh}
            )
            s_idx = assign_slices_to_idx(
                stage=stage, idx=s_idx, dims={"subu": ocp.dims.nsbu, "subx": ocp.dims.nsbx, "suh": ocp.dims.nsh}
            )

            # s_idx[f"subu_{stage}"] = s_idx[f"slbu_{stage}"]
            # s_idx[f"subx_{stage}"] = s_idx[f"slbx_{stage}"]
            # s_idx[f"suh_{stage}"] = s_idx[f"slh_{stage}"]

        # for stage in range(1, ocp.dims.N):
        #     s_idx[f"slbu_{stage}"] = slice(0, len(ocp.constraints.idxsbu))
        #     s_idx[f"slbx_{stage}"] = slice(
        #         s_idx[f"slbu_{stage}"].stop, s_idx[f"slbu_{stage}"].stop + len(ocp.constraints.idxsbx)
        #     )

        #     # sl and su are the same for the middle stages
        #     s_idx[f"subu_{stage}"] = s_idx[f"slbu_{stage}"]
        #     s_idx[f"subx_{stage}"] = s_idx[f"slbx_{stage}"]

        s_idx = assign_slices_to_idx(stage=ocp.dims.N, idx=s_idx, dims={"slbx": ocp.dims.nsbx, "slh": ocp.dims.nsh})
        s_idx = assign_slices_to_idx(stage=ocp.dims.N, idx=s_idx, dims={"subx": ocp.dims.nsbx, "suh": ocp.dims.nsh})
        # s_idx[f"subx_{stage}"] = s_idx[f"slbx_{stage}"]
        # s_idx[f"suh_{stage}"] = s_idx[f"slh_{stage}"]

        # s_idx[f"slbx_{stage}"] = slice(0, len(ocp.constraints.idxsbx_e))
        # s_idx = assign_slices_to_idx(stage=stage, idx=s_idx, dims=dims)

        self.s_idx = s_idx

        # print("hallo")

    # def get_idx_at_stage(self, stage: int, field: str) -> slice:
    #     """
    #     Get the indices of the constraints of the given type at the given stage.

    #     Parameters:
    #         stage: stage index
    #         field: constraint type

    #     Returns:
    #         indices: slice object
    #     """
    #     return self.idx_at_stage[stage][field]

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
        # return lam[self.get_idx_at_stage(stage, field)]
        if field == "lbx_0":
            return lam[self.lam_idx["lbx_0"]]
        elif field == "ubx_0":
            return lam[self.lam_idx["ubx_0"]]
        if field == "lbu_0":
            return lam[self.lam_idx["lbu_0"]]
        elif field == "ubu_0":
            return lam[self.lam_idx["ubu_0"]]
        elif field == "lbu_k":
            return lam[self.lam_idx["lbu_k"][stage]]
        elif field == "ubu_k":
            return lam[self.lam_idx["ubu_k"][stage]]
        elif field == "lbx_k":
            return lam[self.lam_idx["lbx_k"][stage]]
        elif field == "ubx_k":
            return lam[self.lam_idx["ubx_k"][stage]]
        elif field == "lbx_e":
            return lam[self.lam_idx["lbx_e"]]
        elif field == "ubx_e":
            return lam[self.lam_idx["ubx_e"]]


class NLPEntry:
    """docstring for CasadiNLPEntry."""

    sym: Union[cs.SX, cs.MX]
    val: Union[list, np.ndarray]
    fun: cs.Function

    def __init__(self):
        super().__init__()

        self.sym = None
        self.val = None
        self.fun = None


class NLP:
    """docstring for CasadiNLP."""

    cost: NLPEntry
    vars: NLPEntry
    f_disc: cs.Function
    g: NLPEntry  # Dynamics equality constraints
    pi: NLPEntry  # Lange multiplier for dynamics equality constraints
    h: NLPEntry  # Inequality constraints
    lam: NLPEntry  # Lange multiplier for inequality constraints
    h_dict: dict
    lam_dict: dict
    L: NLPEntry
    dL_dw: NLPEntry
    dL_du: NLPEntry
    dL_dx: NLPEntry
    dL_dp: NLPEntry
    w: NLPEntry
    x: NLPEntry
    u: NLPEntry
    z: NLPEntry
    # R: NLPEntry
    # dR_dw: NLPEntry
    # dR_dp: NLPEntry
    # dR_dz: NLPEntry
    # dT = NLPEntry

    def __init__(self, ocp: AcadosOcp):
        super().__init__()

        self.dims = ocp.dims

        self.multiplier_map = LagrangeMultiplierMap(ocp)

        self.cost = NLPEntry()
        self.vars = NLPEntry()
        self.w = NLPEntry()
        self.x = NLPEntry()
        self.u = NLPEntry()
        self.z = NLPEntry()
        self.f_disc = None
        self.g = NLPEntry()
        self.pi = NLPEntry()
        self.h = NLPEntry()
        self.lam = NLPEntry()
        self.L = NLPEntry()
        self.dL_dw = NLPEntry()
        self.dL_dp = NLPEntry()
        self.dL_du = NLPEntry()
        self.dL_dx = NLPEntry()
        self.h_dict = dict()
        self.lam_dict = dict()
        # self.ddL_dwdw = NLPEntry()
        # self.ddL_dwdpi = NLPEntry()
        # self.ddL_dwdlam = NLPEntry()
        self.R = NLPEntry()
        # self.dR_dw = NLPEntry()
        self.dR_dp = NLPEntry()
        self.dR_dz = NLPEntry()
        # self.dT = NLPEntry()

    def assert_kkt_residual(self) -> np.ndarray:
        return test_nlp_kkt_residual(self)

    def set(self, stage_, field_, value_):
        if len(value_) == 0:
            return

        if field_ in ["x", "u", "p", "dT"]:
            self.vars.val[field_, stage_] = value_
            return 0
        elif field_ in ["lbx", "ubx", "lbu", "ubu", "lsbx", "usbx", "lh", "uh", "lsh", "ush"]:
            self.vars.val[f"{field_}_{stage_}"] = value_
            return 0
        # elif field_ in ["lbx_0", "ubx_0", "lbu_0", "ubu_0", "lbx_e", "ubx_e"]:
        #     self.vars.val[field_] = value_
        #     return 0
        elif field_ == "sl":
            return self.set_sl(stage_, value_)
        elif field_ == "su":
            return self.set_su(stage_, value_)
        elif field_ == "pi":
            return self.set_pi(stage_, value_)
        elif field_ == "lam":
            return self.set_lam(stage_, value_)
        else:
            raise Exception(f"Field {field_} not supported.")

    def get_existing_sl(self, stage_) -> list[str]:
        return list(
            set([f"s{key}_{stage_}" for key in ["lbu", "lbx", "lg", "lh", "lphi"]]).intersection(set(self.vars.val.keys()))
        )

    def get_existing_su(self, stage_) -> list[str]:
        return list(
            set([f"s{key}_{stage_}" for key in ["ubu", "ubx", "ug", "uh", "uphi"]]).intersection(set(self.vars.val.keys()))
        )

    def set_sl(self, stage_, value_):
        for key in self.get_existing_sl(stage_):
            self.vars.val[key] = value_[self.multiplier_map.s_idx[key]]

        return 0

    def set_su(self, stage_, value_):
        for key in self.get_existing_su(stage_):
            self.vars.val[key] = value_[self.multiplier_map.s_idx[key]]

        return 0

    def get_sl(self, stage_):
        return cs.vertcat(*[self.vars.val[key] for key in self.get_existing_sl(stage_)]).full().flatten()

    def get_su(self, stage_):
        return cs.vertcat(*[self.vars.val[key] for key in self.get_existing_su(stage_)]).full().flatten()

    def set_pi(self, stage_, value_):
        self.pi.val["pi", stage_] = value_
        return 0

    def set_lam(self, stage_, value_):
        keys = ["lbu", "lbx", "lh", "ubu", "ubx", "uh", "lsbx", "lsh", "usbx", "ush"]
        # print(self.multiplier_map.lam_idx.keys())
        for key in keys:
            if f"{key}_{stage_}" in self.lam_dict.keys():
                # print(f"{key}_{stage_}: {self.multiplier_map.lam_idx[f'{key}_{stage_}']}")
                self.lam_dict[f"{key}_{stage_}"] = value_[self.multiplier_map.lam_idx[f"{key}_{stage_}"]]

        # if f"lbu_{stage_}" in self.lam_dict.keys():
        #     self.lam_dict[f"lbu_{stage_}"] = value_[self.multiplier_map.lam_idx[f"lbu_{stage_}"]]
        # if f"lbx_{stage_}" in self.lam_dict.keys():
        #     self.lam_dict[f"lbx_{stage_}"] = value_[self.multiplier_map.lam_idx[f"lbx_{stage_}"]]
        # if f"ubu_{stage_}" in self.lam_dict.keys():
        #     self.lam_dict[f"ubu_{stage_}"] = value_[self.multiplier_map.lam_idx[f"ubu_{stage_}"]]
        # if f"ubx_{stage_}" in self.lam_dict.keys():
        #     self.lam_dict[f"ubx_{stage_}"] = value_[self.multiplier_map.lam_idx[f"ubx_{stage_}"]]
        # if f"lsbx_{stage_}" in self.lam_dict.keys():
        #     self.lam_dict[f"lsbx_{stage_}"] = value_[self.multiplier_map.lam_idx[f"lsbx_{stage_}"]]
        # if f"usbx_{stage_}" in self.lam_dict.keys():
        #     self.lam_dict[f"usbx_{stage_}"] = value_[self.multiplier_map.lam_idx[f"usbx_{stage_}"]]

        return 0

    def get(self, stage_, field_):
        if field_ in ["x", "u", "p", "dT"]:
            return self.vars.val[field_, stage_]
        elif field_ in ["lbx", "ubx", "lbu", "ubu", "lsbx", "usbx", "lh", "uh", "lsh", "ush"]:
            return self.vars.val[f"{field_}_{stage_}"]
        elif field_ == "sl":
            return self.get_sl(stage_)
        elif field_ == "su":
            return self.get_su(stage_)
        elif field_ == "pi":
            return self.pi.val["pi", stage_]
        elif field_ == "lam":
            # return cs.vertcat(*[self.lam_dict[key] for key in self.lam_dict.keys()])
            raise NotImplementedError("Not implemented yet.")
        else:
            raise Exception(f"Field {field_} not supported.")

    def get_cost(self):
        return self.cost.val

    def get_residuals(self):
        return [
            self.get_stationarity_residual(),
            self.get_equality_residual(),
            self.get_inequality_residual(),
            self.get_complementarity_residual(),
        ]

    def get_stationarity_residual(self):
        return np.linalg.norm(self.dL_dw.val)

    def get_equality_residual(self):
        return np.linalg.norm(self.g.val)

    def get_inequality_residual(self):
        # Find where self.lam.val > 1e-6
        idx = np.where(self.lam.val > 1e-8)[0]
        return np.linalg.norm(self.h.val[idx])

    def get_complementarity_residual(self):
        return np.linalg.norm(self.lam.val * self.h.val)

    def print_inequality_constraints(self):
        for i in range(self.h.val.shape[0]):
            print(f"{self.h.sym[i]}: {self.h.val[i]}")

    def plot_stationarity(self):
        plt.figure()
        plt.plot(self.dL_dw.val.full().flatten())
        plt.grid(True)
        plt.title("Stationarity residual")
        plt.show()


def define_discrete_dynamics_function(ocp: AcadosOcp) -> cs.Function:
    # Step size.

    x = ocp.model.x
    u = ocp.model.u
    p = ocp.model.p

    if ocp.solver_options.integrator_type == "ERK":
        f_expl = ocp.model.f_expl_expr

        # Continuous dynamics function.
        f = cs.Function("f", [x, u, p], [f_expl])

        # TODO: Add support for other integrator types
        # Integrate given amount of steps over the interval with Runge-Kutta 4 scheme
        h = ocp.solver_options.tf / ocp.dims.N / ocp.solver_options.sim_method_num_stages

        for _ in range(ocp.solver_options.sim_method_num_steps):
            k1 = f(x, u, p)
            k2 = f(x + h / 2 * k1, u, p)
            k3 = f(x + h / 2 * k2, u, p)
            k4 = f(x + h * k3, u, p)

            xnext = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return cs.Function("F", [x, u, p], [xnext])
    elif ocp.solver_options.integrator_type == "DISCRETE":
        xnext = ocp.model.disc_dyn_expr
        return cs.Function("F", [x, u, p], [xnext])
    else:
        raise NotImplementedError("Only ERK integrator types are supported at the moment.")


def define_y_0_function(ocp: AcadosOcp) -> cs.Function:
    model = ocp.model

    x = model.x
    u = model.u

    if ocp.cost.cost_type_0 == "NONLINEAR_LS":
        y_0_fun = cs.Function("y_0_fun", [x, u], [model.cost_y_expr_0], ["x", "u"], ["y_0"])

    return y_0_fun


def define_y_function(ocp: AcadosOcp) -> cs.Function:
    model = ocp.model

    x = model.x
    u = model.u

    if ocp.cost.cost_type == "NONLINEAR_LS":
        y_fun = cs.Function("y_fun", [x, u], [model.cost_y_expr], ["x", "u"], ["y"])

    return y_fun


def define_y_e_function(ocp: AcadosOcp) -> cs.Function:
    model = ocp.model

    x = model.x

    if ocp.cost.cost_type_e == "NONLINEAR_LS":
        y_e_fun = cs.Function("y_e_fun", [x], [model.cost_y_expr_e], ["x"], ["y_e"])

    return y_e_fun


def define_nls_cost_function(ocp: AcadosOcp) -> cs.Function:
    model = ocp.model

    x = model.x
    u = model.u
    y_fun = define_y_function(ocp)
    yref = ocp.cost.yref
    W = ocp.cost.W

    nls_cost_function = cs.Function(
        "l",
        [x, u],
        [0.5 * cs.mtimes([(y_fun(x, u) - yref).T, W, (y_fun(x, u) - yref)])],
        ["x", "u"],
        ["out"],
    )

    return nls_cost_function


def Jac_idx(idx: np.ndarray, n: int) -> np.ndarray:
    J = np.zeros((n, idx.shape[0]))
    for i, j in enumerate(idx):
        J[i, j] = 1

    return J


def define_nls_cost_function_e(ocp: AcadosOcp) -> cs.Function:
    model = ocp.model

    x = model.x
    y_e_fun = define_y_e_function(ocp)
    yref_e = ocp.cost.yref_e.reshape(-1, 1)
    W_e = ocp.cost.W_e

    nls_cost_function_e = cs.Function(
        "l_e",
        [x],
        [0.5 * cs.mtimes([(y_e_fun(x) - yref_e).T, W_e, (y_e_fun(x) - yref_e)])],
        ["x"],
        ["out"],
    )

    return nls_cost_function_e


def define_nls_cost_function_0(ocp: AcadosOcp) -> cs.Function:
    model = ocp.model

    x = model.x
    u = model.u
    y_0_fun = define_y_0_function(ocp)
    yref_0 = ocp.cost.yref_0
    W_0 = ocp.cost.W_0

    nls_cost_function_0 = cs.Function(
        "l",
        [x, u],
        [0.5 * cs.mtimes([(y_0_fun(x, u) - yref_0).T, W_0, (y_0_fun(x, u) - yref_0)])],
        ["x", "u"],
        ["out"],
    )

    return nls_cost_function_0


def define_external_cost_function_0(ocp: AcadosOcp) -> cs.Function:
    return cs.Function("l_0", [ocp.model.x, ocp.model.u, ocp.model.p], [ocp.model.cost_expr_ext_cost_0])


def define_external_cost_function(ocp: AcadosOcp) -> cs.Function:
    return cs.Function("l", [ocp.model.x, ocp.model.u, ocp.model.p], [ocp.model.cost_expr_ext_cost])


def define_external_cost_function_e(ocp: AcadosOcp) -> cs.Function:
    return cs.Function("l_e", [ocp.model.x, ocp.model.p], [ocp.model.cost_expr_ext_cost_e])


def get_state_labels(ocp: AcadosOcp) -> list[str]:
    return ocp.model.x.str().strip("[]").split(", ")


def get_input_labels(ocp: AcadosOcp) -> list[str]:
    return ocp.model.u.str().strip("[]").split(", ")


def get_sbx_labels(ocp: AcadosOcp) -> list[str]:
    print("hallo")

    state_labels = get_state_labels(ocp)

    constraint_labels = [state_labels[idx] for idx in ocp.constraints.idxbx]
    sbx_labels = [constraint_labels[idx] for idx in ocp.constraints.idxsbx]
    return sbx_labels


def get_parameter_labels(ocp: AcadosOcp) -> list[str]:
    return ocp.model.p.str().strip("[]").split(", ")


def append_inequality_constraint_entries(entries: dict, field: str, labels: list[str], idx: np.ndarray, repeat=None) -> dict:
    if not len(idx) > 0:
        return entries
    # if idx:
    sub_labels = [labels[i] for i in idx]

    stage = field.split("_")[-1]

    if repeat:
        entries["variables"].append(entry(field, repeat=repeat, struct=struct_symSX(sub_labels)))
        # entries["multipliers"]["lam"].append(
        #     entry(field, repeat=repeat, struct=struct_symSX([f"lam_{label}" for label in sub_labels]))
        # )
    else:
        entries["variables"].append(entry(field, struct=struct_symSX(sub_labels)))
        # entries["multipliers"]["lam"].append(entry(field, struct=struct_symSX([f"lam_{label}" for label in sub_labels])))

    # Add slacks in addtion to lower/upper constraint
    if "lsbx" in field:
        entries["variables"].append(entry(f"slbx_{stage}", struct=struct_symSX(sub_labels)))
    elif "lsh" in field:
        entries["variables"].append(entry(f"slh_{stage}", struct=struct_symSX(sub_labels)))
    elif "usbx" in field:
        entries["variables"].append(entry(f"subx_{stage}", struct=struct_symSX(sub_labels)))
    elif "ush" in field:
        entries["variables"].append(entry(f"suh_{stage}", struct=struct_symSX(sub_labels)))

    return entries


def is_lower_constraints(field: str) -> bool:
    return field[0] == "l"


def is_upper_constraints(field: str) -> bool:
    return field[0] == "u"


def append_relaxed_inequality_constraint_entries(
    entries: dict, field: str, labels: list[str], idx: np.ndarray, s_idx: np.ndarray
) -> dict:
    # if idx:
    sub_labels = [labels[i] for i in idx]

    if len(idx) > 0:
        entries["variables"].append(entry(field, struct=struct_symSX(sub_labels)))
        entries["multipliers"]["lam"].append(entry(field, struct=struct_symSX([f"lam_{label}" for label in sub_labels])))
        if is_lower_constraints(field):
            entries["slacks"]["sl"].append(entry(field, struct=struct_symSX([f"sl_{label}" for label in sub_labels])))
        elif is_upper_constraints(field):
            entries["slacks"]["su"].append(entry(field, struct=struct_symSX([f"su_{label}" for label in sub_labels])))

    return entries

    # "lbx_k", repeat=ocp.dims.N - 1, struct=struct_symSX([f"lam_{labels['x'][k]}" for k in ocp.constraints.idxbx])


def append_inequality_constraint(key: str, expr, h, lam):
    h[key] = expr
    lam[key] = cs.SX.sym(f"lam_{key}", h[key].shape)

    return h, lam


def append_lbu(stage, h, lam, vars, ocp: AcadosOcp):
    if f"lbu_{stage}" not in vars.keys():
        return h, lam

    if stage == 0:
        idxbu = np.arange(ocp.dims.nu)
    else:
        idxbu = ocp.constraints.idxbu

    return append_inequality_constraint(f"lbu_{stage}", vars[f"lbu_{stage}"] - vars["u", stage][idxbu], h, lam)


def append_lbx(stage, h, lam, vars, ocp: AcadosOcp):
    if f"lbx_{stage}" not in vars.keys():
        return h, lam

    if stage == 0:
        idxbx = np.arange(ocp.dims.nx)
        idxsbx = np.array([])
    elif 0 < stage < ocp.dims.N:
        idxbx = ocp.constraints.idxbx
        idxsbx = ocp.constraints.idxsbx
    else:
        idxbx = ocp.constraints.idxbx_e
        idxsbx = ocp.constraints.idxsbx_e

    Jbx = Jac_idx(idxbx, ocp.dims.nx)

    if idxsbx.shape[0] > 0:
        Jsbx = Jac_idx(idxsbx, idxbx.shape[0])
        h, lam = append_inequality_constraint(
            f"lbx_{stage}", vars[f"lbx_{stage}"] - Jbx @ vars["x", stage] - Jsbx @ vars[f"slbx_{stage}"], h, lam
        )
    else:
        h, lam = append_inequality_constraint(f"lbx_{stage}", vars[f"lbx_{stage}"] - Jbx @ vars["x", stage], h, lam)

    return h, lam


def append_lh(stage, h, lam, vars, ocp: AcadosOcp):
    if f"lh_{stage}" not in vars.keys():
        return h, lam

    if stage < ocp.dims.N:
        idxsh = ocp.constraints.idxsh
        h_fun = cs.Function("h", [ocp.model.x, ocp.model.u, ocp.model.p], [ocp.model.con_h_expr])
    else:
        idxsh = ocp.constraints.idxsh_e
        h_fun = cs.Function("h", [ocp.model.x, ocp.model.u, ocp.model.p], [ocp.model.con_h_expr_e])

    h_val = h_fun(vars["x", stage], vars["u", stage], vars["p"])

    if idxsh.shape[0] > 0:
        Jsh = Jac_idx(idxsh, ocp.dims.nh)
        h, lam = append_inequality_constraint(f"lh_{stage}", vars[f"lh_{stage}"] - Jsh @ vars[f"slh_{stage}"] - h_val, h, lam)

    else:
        h, lam = append_inequality_constraint(f"lh_{stage}", vars[f"lh_{stage}"] - h_val, h, lam)

    return h, lam


def append_ubu(stage, h, lam, vars, ocp: AcadosOcp):
    if f"ubu_{stage}" not in vars.keys():
        return h, lam

    if stage == 0:
        idxbu = np.arange(ocp.dims.nu)
    else:
        idxbu = ocp.constraints.idxbu

    return append_inequality_constraint(f"ubu_{stage}", vars["u", stage][idxbu] - vars[f"ubu_{stage}"], h, lam)


def append_ubx(stage, h, lam, vars, ocp: AcadosOcp):
    if f"ubx_{stage}" not in vars.keys():
        return h, lam

    if stage == 0:
        idxbx = np.arange(ocp.dims.nx)
        idxsbx = np.array([])
    elif 0 < stage < ocp.dims.N:
        idxbx = ocp.constraints.idxbx
        idxsbx = ocp.constraints.idxsbx
    else:
        idxbx = ocp.constraints.idxbx_e
        idxsbx = ocp.constraints.idxsbx_e

    Jbx = Jac_idx(idxbx, ocp.dims.nx)

    if idxsbx.shape[0] > 0:
        Jsbx = Jac_idx(idxsbx, idxbx.shape[0])
        return append_inequality_constraint(
            f"ubx_{stage}", Jbx @ vars["x", stage] - vars[f"ubx_{stage}"] - Jsbx @ vars[f"subx_{stage}"], h, lam
        )
    else:
        return append_inequality_constraint(f"ubx_{stage}", Jbx @ vars["x", stage] - vars[f"ubx_{stage}"], h, lam)


def append_uh(stage, h, lam, vars, ocp: AcadosOcp):
    if f"uh_{stage}" not in vars.keys():
        return h, lam

    if stage < ocp.dims.N:
        idxsh = ocp.constraints.idxsh
        h_fun = cs.Function("h", [ocp.model.x, ocp.model.u, ocp.model.p], [ocp.model.con_h_expr])
    else:
        idxsh = ocp.constraints.idxsh_e
        h_fun = cs.Function("h", [ocp.model.x, ocp.model.u, ocp.model.p], [ocp.model.con_h_expr_e])

    h_val = h_fun(vars["x", stage], vars["u", stage], vars["p"])

    if idxsh.shape[0] > 0:
        Jsh = Jac_idx(idxsh, ocp.dims.nh)
        return append_inequality_constraint(f"uh_{stage}", h_val - vars[f"uh_{stage}"] - Jsh @ vars[f"suh_{stage}"], h, lam)
    else:
        return append_inequality_constraint(f"uh_{stage}", h_val - vars[f"uh_{stage}"], h, lam)


def append_lsbu(stage, h, lam, vars, ocp: AcadosOcp):
    if f"lsbu_{stage}" not in vars.keys():
        return h, lam


def append_lsbx(stage, h, lam, vars, ocp: AcadosOcp):
    if f"lsbx_{stage}" not in vars.keys():
        return h, lam

    return append_inequality_constraint(f"lsbx_{stage}", -vars[f"slbx_{stage}"], h, lam)


def append_lsh(stage, h, lam, vars, ocp: AcadosOcp):
    if f"lsh_{stage}" not in vars.keys():
        return h, lam

    return append_inequality_constraint(f"lsh_{stage}", -vars[f"slh_{stage}"], h, lam)


def append_usbu(stage, h, lam, vars, ocp: AcadosOcp):
    if f"usbu_{stage}" not in vars.keys():
        return h, lam


def append_usbx(stage, h, lam, vars, ocp: AcadosOcp):
    if f"usbx_{stage}" not in vars.keys():
        return h, lam

    return append_inequality_constraint(f"usbx_{stage}", -vars[f"subx_{stage}"], h, lam)


def append_ush(stage, h, lam, vars, ocp: AcadosOcp):
    if f"ush_{stage}" not in vars.keys():
        return h, lam

    return append_inequality_constraint(f"ush_{stage}", -vars[f"suh_{stage}"], h, lam)


def append_lbu_0(h, lam, vars, ocp: AcadosOcp):
    if "lbu_0" not in vars.keys():
        return h, lam

    return append_inequality_constraint("lbu_0", vars["lbu_0"] - vars["u", 0], h, lam)


def append_lbx_0(h, lam, vars, ocp: AcadosOcp):
    if "lbx_0" not in vars.keys():
        return h, lam

    return append_inequality_constraint("lbx_0", vars["lbx_0"] - vars["x", 0], h, lam)


def append_ubu_0(h, lam, vars, ocp: AcadosOcp):
    if "ubu_0" in vars.keys():
        return append_inequality_constraint("ubu_0", vars["u", 0] - vars["ubu_0"], h, lam)


def append_ubx_0(h, lam, vars, ocp: AcadosOcp):
    if "ubx_0" in vars.keys():
        return append_inequality_constraint("ubx_0", vars["x", 0] - vars["ubx_0"], h, lam)


def define_inequality_constraints(vars: struct_symSX, ocp: AcadosOcp) -> tuple[dict, dict]:
    h = dict()
    lam = dict()

    # Initial stage
    h, lam = append_lbu_0(h, lam, vars, ocp)

    h, lam = append_lbx_0(h, lam, vars, ocp)

    h, lam = append_lh(0, h, lam, vars, ocp)

    h, lam = append_ubu_0(h, lam, vars, ocp)

    h, lam = append_ubx_0(h, lam, vars, ocp)

    h, lam = append_uh(0, h, lam, vars, ocp)

    h, lam = append_lsbu(0, h, lam, vars, ocp)

    h, lam = append_lsbx(0, h, lam, vars, ocp)

    h, lam = append_lsh(0, h, lam, vars, ocp)

    h, lam = append_usbu(0, h, lam, vars, ocp)

    h, lam = append_usbx(0, h, lam, vars, ocp)

    h, lam = append_ush(0, h, lam, vars, ocp)

    # Middle stages
    for stage in range(1, ocp.dims.N):
        h, lam = append_lbu(stage, h, lam, vars, ocp)

        h, lam = append_lbx(stage, h, lam, vars, ocp)

        h, lam = append_lh(stage, h, lam, vars, ocp)

        h, lam = append_ubu(stage, h, lam, vars, ocp)

        h, lam = append_ubx(stage, h, lam, vars, ocp)

        h, lam = append_uh(stage, h, lam, vars, ocp)

        h, lam = append_lsbu(stage, h, lam, vars, ocp)

        h, lam = append_lsbx(stage, h, lam, vars, ocp)

        h, lam = append_lsh(stage, h, lam, vars, ocp)

        h, lam = append_usbu(stage, h, lam, vars, ocp)

        h, lam = append_usbx(stage, h, lam, vars, ocp)

        h, lam = append_ush(stage, h, lam, vars, ocp)

    # Last stage
    stage = ocp.dims.N

    h, lam = append_lbx(stage, h, lam, vars, ocp)

    h, lam = append_lh(stage, h, lam, vars, ocp)

    h, lam = append_ubx(stage, h, lam, vars, ocp)

    h, lam = append_uh(stage, h, lam, vars, ocp)

    h, lam = append_lsbx(stage, h, lam, vars, ocp)

    h, lam = append_lsh(stage, h, lam, vars, ocp)

    h, lam = append_usbx(stage, h, lam, vars, ocp)

    h, lam = append_ush(stage, h, lam, vars, ocp)

    return h, lam


def define_equality_constraints(vars: struct_symSX, ocp: AcadosOcp) -> cs.SX:
    g = list()
    f_disc = define_discrete_dynamics_function(ocp)
    for stage_ in range(ocp.dims.N):
        g.append(f_disc(vars["x", stage_], vars["u", stage_], vars["p"]) - vars["x", stage_ + 1])

    g = cs.vertcat(*g)

    return g


def append_lbx_constraint(h: list, vars: struct_symSX, ocp: AcadosOcp, stage: int) -> list:
    h.append(vars["lbx_k", stage - 1] - vars["x", stage])

    return h


def build_nlp(ocp: AcadosOcp) -> NLP:
    """
    Build the NLP for the OCP.

    TODO: Add support for other cost types
    TODO: Adapt to SX/MX depending on the provided model
    TODO: Add support for different parameters at each stage
    TODO: Add support for varying/learning reference trajectories, i.e. set as parameters
    TODO: Add support for varying/learning cost weights, i.e. set as parameters
    TODO: Add support for varying/learning constraints, i.e. set as parameters
    """

    nlp = NLP(ocp)

    entries = {"variables": [], "multipliers": {"lam": [], "pi": []}, "slacks": {"sl": [], "su": []}}

    # State at each stage
    entries["variables"].append(entry("u", struct=struct_symSX(get_input_labels(ocp)), repeat=ocp.dims.N))
    entries["variables"].append(entry("x", struct=struct_symSX(get_state_labels(ocp)), repeat=ocp.dims.N + 1))
    # entries["variables"].append(entry("slbx", struct=struct_symSX(get_sbx_labels(ocp)), repeat=ocp.dims.N + 1))
    # entries["variables"].append(entry("subx", struct=struct_symSX(get_sbx_labels(ocp)), repeat=ocp.dims.N + 1))
    entries["variables"].append(entry("p", struct=struct_symSX(get_parameter_labels(ocp))))

    # Inequality constraints
    h_labels = [f"h_{i}" for i in range(ocp.constraints.lh.shape[0])]

    # Initial stage

    entries = append_inequality_constraint_entries(
        entries, field="lbu_0", labels=get_input_labels(ocp), idx=np.arange(ocp.dims.nu)
    )

    entries = append_inequality_constraint_entries(
        entries, field="lbx_0", labels=get_state_labels(ocp), idx=ocp.constraints.idxbx_0
    )

    entries = append_inequality_constraint_entries(
        entries, field="lh_0", labels=h_labels, idx=np.arange(ocp.constraints.lh.shape[0])
    )

    entries = append_inequality_constraint_entries(
        entries, field="ubu_0", labels=get_input_labels(ocp), idx=np.arange(ocp.dims.nu)
    )
    entries = append_inequality_constraint_entries(
        entries, field="ubx_0", labels=get_state_labels(ocp), idx=ocp.constraints.idxbx_0
    )

    entries = append_inequality_constraint_entries(
        entries, field="uh_0", labels=h_labels, idx=np.arange(ocp.constraints.uh.shape[0])
    )

    entries = append_inequality_constraint_entries(
        entries,
        field="lsh_0",
        labels=[h_labels[i] for i in ocp.constraints.idxsh],
        idx=ocp.constraints.idxsh,
    )

    entries = append_inequality_constraint_entries(
        entries,
        field="ush_0",
        labels=[h_labels[i] for i in ocp.constraints.idxsh],
        idx=ocp.constraints.idxsh,
    )

    # Middle stages
    for stage in range(1, ocp.dims.N):
        entries = append_inequality_constraint_entries(
            entries,
            field=f"lbu_{stage}",
            labels=get_input_labels(ocp),
            idx=ocp.constraints.idxbu,
        )

        entries = append_inequality_constraint_entries(
            entries,
            field=f"lbx_{stage}",
            labels=get_state_labels(ocp),
            idx=ocp.constraints.idxbx,
        )

        entries = append_inequality_constraint_entries(
            entries,
            field=f"lh_{stage}",
            labels=h_labels,
            idx=np.arange(ocp.constraints.lh.shape[0]),
        )

        entries = append_inequality_constraint_entries(
            entries,
            field=f"ubu_{stage}",
            labels=get_input_labels(ocp),
            idx=ocp.constraints.idxbu,
        )

        entries = append_inequality_constraint_entries(
            entries,
            field=f"ubx_{stage}",
            labels=get_state_labels(ocp),
            idx=ocp.constraints.idxbx,
        )

        entries = append_inequality_constraint_entries(
            entries,
            field=f"uh_{stage}",
            labels=[f"h_{i}" for i in range(ocp.constraints.uh.shape[0])],
            idx=np.arange(ocp.constraints.lh.shape[0]),
        )

        entries = append_inequality_constraint_entries(
            entries,
            field=f"lsbu_{stage}",
            labels=get_input_labels(ocp),
            idx=ocp.constraints.idxbx[ocp.constraints.idxsbu.tolist()],
        )

        entries = append_inequality_constraint_entries(
            entries,
            field=f"lsbx_{stage}",
            labels=get_state_labels(ocp),
            idx=ocp.constraints.idxbx[ocp.constraints.idxsbx.tolist()],
        )

        entries = append_inequality_constraint_entries(
            entries,
            field=f"lsh_{stage}",
            labels=[h_labels[i] for i in ocp.constraints.idxsh],
            idx=ocp.constraints.idxsh,
        )

        entries = append_inequality_constraint_entries(
            entries,
            field=f"usbu_{stage}",
            labels=get_input_labels(ocp),
            idx=ocp.constraints.idxbu[ocp.constraints.idxsbu.tolist()],
        )

        entries = append_inequality_constraint_entries(
            entries,
            field=f"usbx_{stage}",
            labels=get_state_labels(ocp),
            idx=ocp.constraints.idxbx[ocp.constraints.idxsbx.tolist()],
        )

        entries = append_inequality_constraint_entries(
            entries,
            field=f"ush_{stage}",
            labels=[h_labels[i] for i in ocp.constraints.idxsh],
            idx=ocp.constraints.idxsh,
        )

    stage = ocp.dims.N

    entries = append_inequality_constraint_entries(
        entries, field=f"lbx_{stage}", labels=get_state_labels(ocp), idx=ocp.constraints.idxbx_e
    )

    entries = append_inequality_constraint_entries(
        entries,
        field=f"lh_{stage}",
        labels=[f"h_{i}" for i in range(ocp.constraints.lh_e.shape[0])],
        idx=np.arange(ocp.constraints.lh_e.shape[0]),
    )

    entries = append_inequality_constraint_entries(
        entries, field=f"ubx_{stage}", labels=get_state_labels(ocp), idx=ocp.constraints.idxbx_e
    )

    entries = append_inequality_constraint_entries(
        entries,
        field=f"uh_{stage}",
        labels=[f"h_{i}" for i in range(ocp.constraints.uh_e.shape[0])],
        idx=np.arange(ocp.constraints.lh_e.shape[0]),
    )

    entries = append_inequality_constraint_entries(
        entries,
        field=f"lsbx_{stage}",
        labels=get_state_labels(ocp),
        idx=ocp.constraints.idxbx_e[ocp.constraints.idxsbx_e.tolist()],
    )

    entries = append_inequality_constraint_entries(
        entries,
        field=f"lsh_{stage}",
        labels=[f"h_{stage}_{i}" for i in range(ocp.constraints.lh_e.shape[0])],
        idx=ocp.constraints.idxsh_e,
    )

    entries = append_inequality_constraint_entries(
        entries,
        field=f"usbx_{stage}",
        labels=get_state_labels(ocp),
        idx=ocp.constraints.idxbx_e[ocp.constraints.idxsbx_e.tolist()],
    )

    entries = append_inequality_constraint_entries(
        entries,
        field=f"ush_{stage}",
        labels=[f"h_{stage}_{i}" for i in range(ocp.constraints.uh_e.shape[0])],
        idx=ocp.constraints.idxsh_e,
    )

    # Varying interval length
    entries["variables"].append(entry("dT", repeat=ocp.dims.N, struct=struct_symSX([entry("dT")])))

    # Lagrange multipliers for equality constraints
    entries["multipliers"]["pi"].append(entry("pi", repeat=ocp.dims.N, struct=struct_symSX(get_state_labels(ocp))))

    # Equality constraints
    vars = struct_symSX([tuple(entries["variables"])])

    # lam = struct_symSX([tuple(entries["multipliers"]["lam"])])

    pi = struct_symSX([tuple(entries["multipliers"]["pi"])])

    g = define_equality_constraints(vars, ocp)

    assert g.shape[0] == pi.cat.shape[0], "Dimension mismatch between g (equality constraints) and pi (multipliers)"

    # Inequality constraints
    h_dict, lam_dict = define_inequality_constraints(vars, ocp)

    nlp.lam.sym = cs.vertcat(*list(lam_dict.values()))
    nlp.h.sym = cs.vertcat(*list(h_dict.values()))

    nlp.lam.val = np.zeros(nlp.lam.sym.shape[0])
    nlp.h.val = np.zeros(nlp.h.sym.shape[0])

    nlp.lam_dict = lam_dict
    nlp.h_dict = h_dict

    assert (
        nlp.h.sym.shape[0] == nlp.lam.sym.shape[0]
    ), f"Dimension mismatch between h (inequality constraints, shape {nlp.h.sym.shape[0]}) and \
        lam (multipliers, shape {nlp.lam.sym.shape[0]})"

    # Build inequality constraint

    if ocp.cost.cost_type == "NONLINEAR_LS":
        cost_function = define_nls_cost_function(ocp)
        cost_function_e = define_nls_cost_function_e(ocp)
        cost_function_0 = define_nls_cost_function_0(ocp)

        cost = 0

        # Initial stage
        stage_ = 0
        cost += vars["dT", stage_] * cost_function_0(vars["x", stage_], vars["u", stage_])

        # Middle stages
        for stage_ in range(1, ocp.dims.N):
            cost += vars["dT", stage_] * cost_function(vars["x", stage_], vars["u", stage_])

        # # Add terminal cost
        stage_ = ocp.dims.N
        cost += cost_function_e(vars["x", stage_])

    elif ocp.cost.cost_type == "EXTERNAL":
        cost_function = define_external_cost_function(ocp)
        cost_function_e = define_external_cost_function_e(ocp)
        cost_function_0 = define_external_cost_function_0(ocp)

        cost = 0
        cost += vars["dT", 0] * cost_function_0(vars["x", 0], vars["u", 0], vars["p"])
        cost += sum(
            [
                vars["dT", stage] * cost_function(vars["x", stage], vars["u", stage], vars["p"])
                for stage in range(1, ocp.dims.N)
            ]
        )
        cost += cost_function_e(vars["x", ocp.dims.N], vars["p"])

        # TODO: Add support for mixing relaxed constraints
        if len(ocp.constraints.idxsh) > 0 and len(ocp.constraints.idxsbx) > 0:
            raise NotImplementedError(
                "Not implemented yet. Can either have soft state or soft inequality constraints, not both"
            )

        # Add cost for slack variables
        if len(ocp.constraints.idxsh) > 0:
            cost += sum([vars["dT", stage] * vars[f"slh_{stage}"].T @ ocp.cost.zl for stage in range(ocp.dims.N)])
            cost += sum([vars["dT", stage] * vars[f"suh_{stage}"].T @ ocp.cost.zu for stage in range(ocp.dims.N)])

        if len(ocp.constraints.idxsh_e) > 0:
            cost += vars[f"slh_{ocp.dims.N}"].T @ ocp.cost.zl_e
            cost += vars[f"suh_{ocp.dims.N}"].T @ ocp.cost.zu_e

        if len(ocp.constraints.idxsbx) > 0:
            cost += sum([vars[f"slbx_{stage}"] @ ocp.cost.zl for stage in range(1, ocp.dims.N)])
            cost += sum([vars[f"subx_{stage}"] @ ocp.cost.zu for stage in range(1, ocp.dims.N)])

        if len(ocp.constraints.idxsbx_e) > 0:
            cost += vars[f"slbx_{ocp.dims.N}"] @ ocp.cost.zl_e
            cost += vars[f"subx_{ocp.dims.N}"] @ ocp.cost.zu_e

    # nlp.cost.sym = 0

    # nlp.cost.fun = cs.Function("cost", [nlp.w.sym, nlp.dT.sym], [nlp.cost.sym], ["w", "dT"], ["cost"])
    # nlp.cost.val = 0

    # nlp.dT.val = nlp.dT.sym(0)
    # nlp.dT.val["dT", lambda x: cs.vertcat(*x)] = np.tile(ocp.solver_options.tf / ocp.dims.N, (1, ocp.dims.N))

    nlp.vars.sym = vars
    nlp.vars.val = vars(0)

    if len(ocp.constraints.lbu) > 0:
        for stage in range(0, ocp.dims.N - 1):
            nlp.vars.val[f"lbu_{stage}"] = ocp.constraints.lbu
            nlp.vars.val[f"ubu_{stage}"] = ocp.constraints.ubu

    # vars_val["dT", lambda x: cs.vertcat(*x)] = np.tile(ocp.solver_options.tf / ocp.dims.N, (1, ocp.dims.N))
    for stage in range(ocp.dims.N):
        nlp.vars.val["dT", stage] = ocp.solver_options.tf / ocp.dims.N

    nlp.vars.val["p"] = ocp.parameter_values

    nlp.cost.sym = cost
    nlp.cost.val = 0
    nlp.cost.fun = cs.Function("cost", [nlp.vars.sym], [nlp.cost.sym])

    nlp.g.sym = g
    nlp.g.fun = cs.Function("g", [nlp.vars.sym], [nlp.g.sym])

    # nlp.h.sym = cs.vertcat(*list(h.values()))
    nlp.h.fun = cs.Function("h", [nlp.vars.sym], [nlp.h.sym])

    # nlp.lam.sym = lam
    # nlp.lam.val = lam(0)

    # for i in range(nlp.lam.sym.shape[0]):
    #     print(f"{nlp.h.sym[i]} * {nlp.lam.sym[i]}")

    nlp.pi.sym = pi
    nlp.pi.val = pi(0)

    # nlp.L.sym = nlp.cost.sym + cs.dot(nlp.lam.sym, nlp.h.sym) + cs.dot(nlp.pi.sym, nlp.g.sym)
    nlp.L.sym = nlp.cost.sym + cs.dot(nlp.lam.sym, nlp.h.sym) + cs.dot(nlp.pi.sym, nlp.g.sym)

    nlp.L.fun = cs.Function("L", [nlp.vars.sym, nlp.pi.sym, nlp.lam.sym], [nlp.L.sym])

    w = []
    for stage in range(ocp.dims.N):
        w.append(nlp.vars.sym["u", stage])
        w.append(nlp.vars.sym["x", stage])
    w.append(nlp.vars.sym["x", ocp.dims.N])

    u = cs.vertcat(*nlp.vars.sym["u", :])
    x = cs.vertcat(*nlp.vars.sym["x", :])

    lam = nlp.lam.sym

    pi = nlp.pi.sym.cat

    w = cs.vertcat(u, x)
    nlp.dL_dw.sym = cs.jacobian(nlp.L.sym, w)
    nlp.dL_dw.fun = cs.Function("dL_dw", [nlp.vars.sym, nlp.pi.sym, nlp.lam.sym], [nlp.dL_dw.sym])

    nlp.dL_dx.sym = cs.jacobian(nlp.L.sym, x)
    nlp.dL_dx.fun = cs.Function("dL_dx", [nlp.vars.sym, nlp.pi.sym, nlp.lam.sym], [nlp.dL_dx.sym])

    nlp.dL_du.sym = cs.jacobian(nlp.L.sym, u)
    nlp.dL_du.fun = cs.Function("dL_du", [nlp.vars.sym, nlp.pi.sym, nlp.lam.sym], [nlp.dL_du.sym])

    nlp.dL_dp.sym = cs.jacobian(nlp.L.sym, nlp.vars.sym["p"])
    nlp.dL_dp.fun = cs.Function("dL_dp", [nlp.vars.sym, nlp.pi.sym, nlp.lam.sym], [nlp.dL_dp.sym])

    nlp.R.sym = cs.vertcat(cs.transpose(nlp.dL_dw.sym), nlp.g.sym, nlp.lam.sym * nlp.h.sym)
    nlp.R.fun = cs.Function("R", [nlp.vars.sym, nlp.pi.sym, nlp.lam.sym], [nlp.R.sym])

    nlp.dR_dp.sym = cs.jacobian(nlp.R.sym, nlp.vars.sym["p"])
    nlp.dR_dp.fun = cs.Function("dR_dp", [nlp.vars.sym, nlp.pi.sym, nlp.lam.sym], [nlp.dR_dp.sym])

    nlp.z.sym = cs.vertcat(u, x, pi, lam)

    nlp.z.fun = cs.Function("z", [nlp.vars.sym, nlp.pi.sym, nlp.lam.sym], [nlp.z.sym])

    nlp.dR_dz.sym = cs.jacobian(nlp.R.sym, nlp.z.sym)
    nlp.dR_dz.fun = cs.Function("dR_dz", [nlp.vars.sym, nlp.pi.sym, nlp.lam.sym], [nlp.dR_dz.sym])

    nlp.x.sym = cs.horzcat(*nlp.vars.sym["x", :]).T
    nlp.x.fun = cs.Function("x", [nlp.vars.sym], [nlp.x.sym])

    nlp.u.sym = cs.horzcat(*nlp.vars.sym["u", :]).T
    nlp.u.fun = cs.Function("u", [nlp.vars.sym], [nlp.u.sym])

    # Initial stage
    # TODO: This needs checks if the constraints actually exist
    # lbu_0 = np.ones((ocp.dims.nu,)) * ocp.constraints.lbu[0]
    for key, value in {
        "lbu": ocp.constraints.lbu,
        "lbx": ocp.constraints.lbx_0,
        "lh": ocp.constraints.lh,
        "ubu": ocp.constraints.ubu,
        "ubx": ocp.constraints.ubx_0,
        "uh": ocp.constraints.uh,
    }.items():
        nlp.set(0, key, value)

    # Middle stages
    for stage in range(1, ocp.dims.N):
        for key, value in {
            "lbu": ocp.constraints.lbu,
            "lbx": ocp.constraints.lbx,
            "lh": ocp.constraints.lh,
            "ubu": ocp.constraints.ubu,
            "ubx": ocp.constraints.ubx,
            "uh": ocp.constraints.uh,
            "lsbx": ocp.constraints.lsbx,
            "lsh": ocp.constraints.lsh,
            "usbx": ocp.constraints.usbx,
            "ush": ocp.constraints.ush,
        }.items():
            nlp.set(stage, key, value)

    # Final stage
    for key, value in {
        "lbx": ocp.constraints.lbx_e,
        "lh": ocp.constraints.lh_e,
        "ubx": ocp.constraints.ubx_e,
        "uh": ocp.constraints.uh_e,
        "lsbx": ocp.constraints.lsbx_e,
        "lsh": ocp.constraints.lsh_e,
        "usbx": ocp.constraints.usbx_e,
        "ush": ocp.constraints.ush_e,
    }.items():
        nlp.set(ocp.dims.N, key, value)

    # nlp.set(stage, "lbx", ocp.constraints.lbx_e)
    # nlp.set(stage, "ubx", ocp.constraints.ubx_e)
    # nlp.set(stage, "lsbx", ocp.constraints.lsbx_e)
    # nlp.set(stage, "usbx", ocp.constraints.usbx_e)

    # z = cs.vertcat(nlp.w.sym, nlp.pi.sym, nlp.lam.sym)

    # L = nlp.cost.sym + cs.mtimes([nlp.lam.sym.T, nlp.h.sym]) + cs.mtimes([nlp.pi.sym.T, nlp.g.sym])

    return nlp


def find_nlp_entry_expr_dependencies(nlp: NLP, nlp_entry: str, vars: list[str]) -> tuple[list[cs.SX], list[str]]:
    """
    Find dependencies of expr on var.
    """

    # Loop over a list of attributes in nlp
    arg_list = []
    name_list = []
    for attr in vars:
        # Check if nlp.dL_dp.sym is function of nlp.attr.sym
        dL_dp_depends_on_attr = any(cs.which_depends(getattr(nlp, nlp_entry).sym, getattr(nlp, attr).sym))
        if dL_dp_depends_on_attr:
            print(f"{nlp_entry} depends on", attr)
            arg_list.append(getattr(nlp, attr).sym)
            name_list.append(attr)

    return arg_list, name_list


def update_nlp_w(nlp: NLP, ocp_solver: AcadosOcpSolver) -> NLP:
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


def update_nlp_h(nlp: NLP):
    """
    Update the inequality constraints.

    Args:
        nlp: NLP to update.
        ocp_solver: OCP solver to get the solution from.

    Returns:
        h: Updated inequality constraints.
    """

    return nlp.h.fun(w=nlp.w.val, lbw=nlp.lbw.val, ubw=nlp.ubw.val)["h"]


def update_nlp_g(nlp: NLP):
    """
    Update the equality constraints.

    Args:
        nlp: NLP to update.
        ocp_solver: OCP solver to get the solution from.

    Returns:
        g: Updated equality constraints.
    """

    return nlp.g.fun(w=nlp.w.val, p=nlp.p.val)["g"]


def update_nlp_pi(nlp: NLP, ocp_solver: AcadosOcpSolver) -> NLP:
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


def update_nlp_lam(nlp: NLP, ocp_solver: AcadosOcpSolver, multiplier_map: LagrangeMultiplierMap) -> NLP:
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


def update_nlp_R(nlp: NLP):
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


def update_nlp_L(nlp: NLP):
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


def update_nlp_dL_dw(nlp: NLP):
    """
    Update the sensitivity of the Lagrangian with respect to the primal variables.

    Args:
        nlp: NLP to update.

    Returns:
        nlp: Updated NLP.
    """
    return nlp.dL_dw.fun(w=nlp.w.val, pi=nlp.pi.val, lam=nlp.lam.val, p=nlp.p.val, dT=nlp.dT.val)["dL_dw"]


def update_nlp_dL_dp(nlp: NLP):
    """
    Update the sensitivity of the Lagrangian with respect to the parameters.

    Args:
        nlp: NLP to update.

    Returns:
        nlp: Updated NLP.
    """
    return nlp.dL_dp.fun(w=nlp.w.val, pi=nlp.pi.val, p=nlp.p.val)["dL_dp"]


def update_nlp_dR_dz(nlp: NLP):
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


def update_nlp_dR_dp(nlp: NLP) -> NLP:
    """
    Update the sensitivity of the KKT matrix with respect to the parameters.

    Args:
        nlp: NLP to update.

    Returns:
        nlp: Updated NLP.
    """
    return nlp.dR_dp.fun(w=nlp.w.val, pi=nlp.pi.val, p=nlp.p.val)["dR_dp"]


def test_nlp_is_primal_feasible(nlp: NLP, tol: float = 1e-6) -> bool:
    """
    Check if the primal variables are feasible.
    """

    # Find where nlp.h.val > 0.0
    for idx in range(nlp.h.val.shape[0]):
        if nlp.h.val[idx] > 0.0:
            print(idx, nlp.h.sym[idx], nlp.h.val[idx])

    # TODO: Add message to assert. Detail which constraint is violated.
    assert np.allclose(nlp.g.val, 0.0, atol=tol)
    assert np.all(nlp.h.val < tol)

    return True


def test_nlp_kkt_residual(nlp: NLP, tol: float = 1e-6) -> bool:
    # KKT residual check
    assert np.allclose(nlp.R.val, 0.0, atol=tol)

    return True


def test_nlp_stationarity(nlp: NLP, tol: float = 1e-6) -> bool:
    # Stationarity check
    assert np.allclose(nlp.dL_dw.val, 0.0, atol=tol)

    return True


def test_nlp_is_dual_feasible(nlp: NLP) -> bool:
    # Dual feasibility check
    assert np.all(nlp.lam.val.cat >= 0.0)

    return True


def test_nlp_satisfies_complementarity(nlp: NLP, tol: float = 1e-6) -> bool:
    # Complementary slackness check
    assert np.allclose(nlp.lam.val * nlp.h.val, 0.0, atol=tol)

    return True


def test_nlp_sanity(nlp: NLP, tol: float = 1e-6) -> bool:
    """
    Check if the NLP is feasible and satisfies the KKT conditions.
    """
    test_nlp_is_primal_feasible(nlp=nlp, tol=tol)
    test_nlp_is_dual_feasible(nlp=nlp)
    test_nlp_stationarity(nlp=nlp, tol=tol)
    test_nlp_kkt_residual(nlp=nlp, tol=tol)
    test_nlp_satisfies_complementarity(nlp=nlp, tol=tol)

    return True


def set_nlp_x(nlp: NLP, ocp_solver: AcadosOcpSolver) -> NLP:
    """
    Update the primal variables.

    Args:
        nlp: NLP to update.
        ocp_solver: OCP solver to get the solution from.

    Returns:
        nlp: Updated NLP.
    """

    for stage in range(ocp_solver.acados_ocp.dims.N + 1):
        nlp.vars.val["x", stage] = ocp_solver.get(stage, "x")

    # nlp.set(stage, "x") = ocp_solver.get(stage, "x")

    # nlp.vars.val["x", ocp_solver.acados_ocp.dims.N] = ocp_solver.get(ocp_solver.acados_ocp.dims.N, "x")

    return nlp


def set_nlp_u(nlp: NLP, ocp_solver: AcadosOcpSolver) -> NLP:
    """
    Update the primal variables.

    Args:
        nlp: NLP to update.
        ocp_solver: OCP solver to get the solution from.

    Returns:
        nlp: Updated NLP.
    """

    for stage in range(ocp_solver.acados_ocp.dims.N):
        nlp.vars.val["u", stage] = ocp_solver.get(stage, "u")

    return nlp


def set_nlp_pi(nlp: NLP, ocp_solver: AcadosOcpSolver) -> NLP:
    """
    Update the primal variables.

    Args:
        nlp: NLP to update.
        ocp_solver: OCP solver to get the solution from.

    Returns:
        nlp: Updated NLP.
    """

    for stage in range(ocp_solver.acados_ocp.dims.N):
        nlp.pi.val["pi", stage] = ocp_solver.get(stage, "pi")

    return nlp


def set_nlp_lam(nlp: NLP, ocp_solver: AcadosOcpSolver, multiplier_map: LagrangeMultiplierMap) -> NLP:
    """
    Update the primal variables.

    Args:
        nlp: NLP to update.
        ocp_solver: OCP solver to get the solution from.

    Returns:
        nlp: Updated NLP.
    """

    lam = ocp_solver.get(0, "lam")
    nlp.lam.val["lbx_0"] = multiplier_map(0, "lbx_0", lam)
    nlp.lam.val["ubx_0"] = multiplier_map(0, "ubx_0", lam)
    nlp.lam.val["lbu_0"] = multiplier_map(0, "lbu_0", lam)
    nlp.lam.val["ubu_0"] = multiplier_map(0, "ubu_0", lam)

    running_index = 10
    for stage in range(1, ocp_solver.acados_ocp.dims.N):
        lam = ocp_solver.get(stage, "lam")
        if len(ocp_solver.acados_ocp.constraints.idxbx) > 0:
            nlp.lam.val["lbx_k", stage - 1] = multiplier_map(stage, "lbx_k", lam)
            nlp.lam.val["ubx_k", stage - 1] = multiplier_map(stage, "ubx_k", lam)
        if len(ocp_solver.acados_ocp.constraints.idxbu) > 0:
            nlp.lam.val["lbu_k", stage - 1] = multiplier_map(stage, "lbu_k", lam)
            nlp.lam.val["ubu_k", stage - 1] = multiplier_map(stage, "ubu_k", lam)

        running_index += 2

    # stage = ocp_solver.acados_ocp.dims.N - 1
    # if len(ocp_solver.acados_ocp.constraints.idxbu) > 0:
    #     nlp.lam.val["lbu_k", stage - 1] = multiplier_map(stage, "lbu_k", lam)
    #     nlp.lam.val["ubu_k", stage - 1] = multiplier_map(stage, "ubu_k", lam)

    # stage = ocp_solver.acados_ocp.dims.N - 1
    # nlp.lam.val["lbx_k", stage - 1] = multiplier_map(stage, "lbx", ocp_solver.get(stage, "lam"))
    # nlp.lam.val["ubx_k", stage - 1] = multiplier_map(stage, "ubx", ocp_solver.get(stage, "lam"))

    if len(ocp_solver.acados_ocp.constraints.idxbx_e > 0):
        nlp.lam.val["lbx", stage] = multiplier_map(stage, "lbx", ocp_solver.get(stage, "lam"))
        nlp.lam.val["ubx", stage] = multiplier_map(stage, "ubx", ocp_solver.get(stage, "lam"))

    return nlp


def print_nlp_vars(nlp: NLP):
    for i in range(nlp.vars.val.cat.shape[0]):
        print(f"{nlp.vars.val.cat[i]} <-- {nlp.vars.sym.cat[i]}")


# def print_nlp_vars(nlp: NLP):
#     max_len = max(len(str(val)) for val in nlp.vars.val.cat)

#     for i in range(nlp.vars.val.cat.shape[0]):
#         val = nlp.vars.val.cat[i]
#         sym = nlp.vars.sym.cat[i]

#         # Adjust the spacing between val and sym based on the longest val
#         print(f"{val:<{max_len + 2}}{sym}")


def update_nlp(nlp: NLP, ocp_solver: AcadosOcpSolver) -> NLP:
    """
    Update the NLP with the solution of the OCP solver.

    Args:
        nlp: NLP to update.
        ocp_solver: OCP solver to get the solution from.
        multiplier_map: Map of multipliers.

    Returns:
        nlp: Updated NLP.
    """

    for stage in range(ocp_solver.acados_ocp.dims.N):
        nlp.set(stage, "u", ocp_solver.get(stage, "u"))
    for stage in range(ocp_solver.acados_ocp.dims.N + 1):
        nlp.set(stage, "x", ocp_solver.get(stage, "x"))
    for stage in range(ocp_solver.acados_ocp.dims.N):
        nlp.set(stage, "pi", ocp_solver.get(stage, "pi"))
    for stage in range(ocp_solver.acados_ocp.dims.N + 1):
        nlp.set(stage, "lam", ocp_solver.get(stage, "lam"))
    for stage in range(ocp_solver.acados_ocp.dims.N + 1):
        nlp.set(stage, "sl", ocp_solver.get(stage, "sl"))
    for stage in range(ocp_solver.acados_ocp.dims.N + 1):
        nlp.set(stage, "su", ocp_solver.get(stage, "su"))

    nlp.lam.val = cs.vertcat(*list(nlp.lam_dict.values()))

    nlp.x.val = nlp.x.fun(nlp.vars.val)
    nlp.u.val = nlp.u.fun(nlp.vars.val)

    nlp.cost.val = nlp.cost.fun(nlp.vars.val)
    nlp.h.val = nlp.h.fun(nlp.vars.val)
    nlp.g.val = nlp.g.fun(nlp.vars.val)

    nlp.L.val = nlp.L.fun(nlp.vars.val, nlp.pi.val, nlp.lam.val)
    nlp.dL_dw.val = nlp.dL_dw.fun(nlp.vars.val, nlp.pi.val, nlp.lam.val)
    nlp.dL_dp.val = nlp.dL_dp.fun(nlp.vars.val, nlp.pi.val, nlp.lam.val)
    nlp.dL_du.val = nlp.dL_du.fun(nlp.vars.val, nlp.pi.val, nlp.lam.val)
    nlp.dL_dx.val = nlp.dL_dx.fun(nlp.vars.val, nlp.pi.val, nlp.lam.val)

    nlp.R.val = nlp.R.fun(nlp.vars.val, nlp.pi.val, nlp.lam.val)
    nlp.dR_dz.val = nlp.dR_dz.fun(nlp.vars.val, nlp.pi.val, nlp.lam.val)
    nlp.dR_dp.val = nlp.dR_dp.fun(nlp.vars.val, nlp.pi.val, nlp.lam.val)

    assert np.allclose(
        nlp.u.val, np.vstack([ocp_solver.get(stage, "u") for stage in range(ocp_solver.acados_ocp.dims.N)]), atol=1e-6
    ), "Control mismatch between NLP and OCP solver."

    assert np.allclose(
        nlp.x.val, np.vstack([ocp_solver.get(stage, "x") for stage in range(ocp_solver.acados_ocp.dims.N + 1)]), atol=1e-6
    ), "State mismatch between NLP and OCP solver."

    assert np.allclose(
        nlp.pi.val.cat.full().flatten(),
        np.concatenate([ocp_solver.get(stage, "pi") for stage in range(ocp_solver.acados_ocp.dims.N)]),
        atol=1e-6,
    ), "Lagrange multipliers (equality constraints) mismatch between NLP and OCP solver."

    assert np.allclose(
        nlp.lam.val.full().flatten(),
        np.concatenate([ocp_solver.get(stage, "lam") for stage in range(ocp_solver.acados_ocp.dims.N + 1)]),
        atol=1e-6,
    ), "Lagrange multipliers (inequality constraints) mismatch between NLP and OCP solver."

    assert np.allclose(
        np.concatenate([nlp.get(stage, "sl") for stage in range(ocp_solver.acados_ocp.dims.N + 1)]),
        np.concatenate([ocp_solver.get(stage, "sl") for stage in range(ocp_solver.acados_ocp.dims.N + 1)]),
        atol=1e-6,
    ), "Slack variables mismatch between NLP and OCP solver."

    assert np.allclose(
        np.concatenate([nlp.get(stage, "su") for stage in range(ocp_solver.acados_ocp.dims.N + 1)]),
        np.concatenate([ocp_solver.get(stage, "su") for stage in range(ocp_solver.acados_ocp.dims.N + 1)]),
        atol=1e-6,
    ), "Slack variables mismatch between NLP and OCP solver."

    assert abs(nlp.cost.val - ocp_solver.get_cost()) < 1e-1, "Cost mismatch between NLP and OCP solver."

    assert np.allclose(nlp.g.val, 0.0, atol=1e-6), "Equality constraints are not satisfied."

    assert np.all(nlp.h.val < 1e-6), "Inequality constraints are not satisfied."

    assert np.allclose(nlp.h.val * nlp.lam.val, 0.0, atol=1e-6), "Complementary slackness not satisfied."

    assert np.allclose(nlp.dL_du.val, 0.0, atol=1e-6), "Stationarity wrt u not satisfied."

    assert np.allclose(nlp.dL_dx.val, 0.0, atol=1e-6), "Stationarity wrt x not satisfied."

    # if not np.all(nlp.h.val < 1e-6):
    #     nlp.print_inequality_constraints()

    # Print complementary slackness where it is not satisfied
    # for i in range(nlp.lam.val.shape[0]):
    #     if abs(nlp.h.val[i] * nlp.lam.val[i]) > 1e-6:
    #         print(
    #             f"{nlp.h.sym[i]} * {nlp.lam.sym[i]} = {nlp.h.val[i] * nlp.lam.val[i]}; {nlp.h.sym[i]} = {nlp.h.val[i]} ; {nlp.lam.sym[i]} = {nlp.lam.val[i]}"  # noqa: E501
    #         )

    # if not np.allclose(nlp.h.val * nlp.lam.val, 0.0, atol=1e-6):
    #     print("Complete slackness not satisfied.")
    #     print(nlp.h.val * nlp.lam.val)
    #     print(nlp.vars.val["lbu_0"])
    #     print(nlp.vars.val["ubu_0"])
    #     print(nlp.vars.val["u", 0])
    #     print("")

    # print(nlp.dL_dx.val[:2])

    # if not np.allclose(nlp.dL_dx.val, 0.0, atol=1e-6):
    #     print("Stationarity wrt w not satisfied.")
    #     print(nlp.dL_dx.val[nlp.dL_dx.val != 0.0])
    #     print("")

    return nlp
