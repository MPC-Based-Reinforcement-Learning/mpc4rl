import casadi as cs
from dataclasses import asdict, dataclass, field
import numpy as np
from acados_template import AcadosOcpOptions, AcadosOcp, AcadosOcpCost
from typing import Optional, Union
import scipy
from casadi.tools import struct_symSX, entry


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


class CasadiNLP:
    """docstring for CasadiNLP."""

    cost: CasadiNLPEntry
    w: CasadiNLPEntry
    lbw: CasadiNLPEntry
    ubw: CasadiNLPEntry
    lbw_solver: CasadiNLPEntry
    ubw_solver: CasadiNLPEntry
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
    h_licq: CasadiNLPEntry  # Inequality constraints
    lam: CasadiNLPEntry  # Lange multiplier for inequality constraints
    lam_licq: CasadiNLPEntry  # Lange multiplier for inequality constraints
    idxhbx: list
    idxsbx: list
    idxhbu: list
    idxsbu: list
    L: CasadiNLPEntry
    dL_dw: CasadiNLPEntry
    dL_dp: CasadiNLPEntry
    R: CasadiNLPEntry
    dR_dw: CasadiNLPEntry
    dR_dp: CasadiNLPEntry
    dR_dz: CasadiNLPEntry
    dT = CasadiNLPEntry

    def __init__(self):
        super().__init__()

        self.cost = CasadiNLPEntry()
        self.w = CasadiNLPEntry()
        self.lbw = CasadiNLPEntry()
        self.ubw = CasadiNLPEntry()
        self.lbw_solver = CasadiNLPEntry()
        self.ubw_solver = CasadiNLPEntry()
        self.g_solver = None
        self.lbg_solver = None
        self.ubg_solver = None
        self.p_solver = CasadiNLPEntry()
        self.p_val = None
        self.p = CasadiNLPEntry()
        self.f_disc = None
        self.shooting = None
        self.g = CasadiNLPEntry()
        self.dg_dw = CasadiNLPEntry()
        self.dg_dpi = CasadiNLPEntry()
        self.dg_dlam = CasadiNLPEntry()
        self.pi = CasadiNLPEntry()
        self.h = CasadiNLPEntry()
        self.h_licq = CasadiNLPEntry()
        self.lam = CasadiNLPEntry()
        self.lam_licq = CasadiNLPEntry()
        self.idxhbx = None
        self.idxsbx = None
        self.idxhbu = None
        self.idxsbu = None
        self.L = CasadiNLPEntry()
        self.dL_dw = CasadiNLPEntry()
        self.ddL_dwdw = CasadiNLPEntry()
        self.ddL_dwdpi = CasadiNLPEntry()
        self.ddL_dwdlam = CasadiNLPEntry()
        self.dL_dp = CasadiNLPEntry()
        self.R = CasadiNLPEntry()
        self.dR_dw = CasadiNLPEntry()
        self.dR_dp = CasadiNLPEntry()
        self.dR_dz = CasadiNLPEntry()
        self.dT = CasadiNLPEntry()

    def set(self, stage_, field_, val_):
        if field_ == "p":
            self.p.val["p", stage_] = val_


def build_nlp(ocp: AcadosOcp) -> tuple[CasadiNLP, dict]:
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

    labels = {
        "x": ocp.model.x.str().strip("[]").split(", "),
        "u": ocp.model.u.str().strip("[]").split(", "),
        "p": ocp.model.p.str().strip("[]").split(", "),
    }

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
    states = struct_symSX([tuple([entry(label) for label in labels["x"]])])

    entries = {"w": [], "lbw": [], "ubw": [], "p": [], "pi": []}

    # State at each stage
    entries["w"].append(entry("x", repeat=ocp.dims.N + 1, struct=states))

    # Lower bound of state box constraint
    entries["lbw"].append(entry("lbx", repeat=ocp.dims.N + 1, struct=states))

    # Upper bound of state box constraint
    entries["ubw"].append(entry("ubx", repeat=ocp.dims.N + 1, struct=states))

    # Add inputs to decision variables
    inputs = struct_symSX([tuple([entry(label) for label in labels["u"]])])

    # Input at each stage
    entries["w"].append(entry("u", repeat=ocp.dims.N, struct=inputs))

    # Lower bound of input box constraint
    entries["lbw"].append(entry("lbu", repeat=ocp.dims.N, struct=inputs))

    # Upper bound of input box constraint
    entries["ubw"].append(entry("ubu", repeat=ocp.dims.N, struct=inputs))

    nlp.w.sym = struct_symSX([tuple(entries["w"])])
    nlp.lbw.sym = struct_symSX([tuple(entries["lbw"])])
    nlp.ubw.sym = struct_symSX([tuple(entries["ubw"])])

    # Parameter vector
    entries["p"].append(entry("p", repeat=ocp.dims.N + 1, struct=struct_symSX([entry(label) for label in labels["p"]])))

    nlp.p.sym = struct_symSX([tuple(entries["p"])])

    nlp.dT.sym = struct_symSX([entry("dT", repeat=ocp.dims.N, struct=struct_symSX([entry("dT")]))])

    # Lagrange multipliers for equality constraints

    entries["pi"].append(entry("pi", repeat=ocp.dims.N, struct=states))

    # Equality constraints
    nlp.pi.sym = struct_symSX([tuple(entries["pi"])])

    g = []
    lbg = []
    ubg = []

    for stage_ in range(ocp.dims.N):
        print("g stage: ", stage_)
        g.append(
            nlp.f_disc(nlp.w.sym["x", stage_], nlp.w.sym["u", stage_], nlp.p.sym["p", stage_]) - nlp.w.sym["x", stage_ + 1]
        )
        lbg.append([0 for _ in range(ocp.dims.nx)])
        ubg.append([0 for _ in range(ocp.dims.nx)])

    nlp.g.sym = cs.vertcat(*g)
    nlp.g.fun = cs.Function("g", [nlp.w.sym, nlp.p.sym], [nlp.g.sym], ["w", "p"], ["g"])

    print(f"pi.shape = {nlp.pi.sym.shape}")
    print(f"g.fun: {nlp.g.fun}")

    # Inequality constraints

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

    # inequality_constraint = {"lbu": [], "lbx": [], "ubu": [], "ubx": []}

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

    for stage_ in range(0, ocp.dims.N + 1):
        print(stage_)
        if idxhbu:
            if stage_ == 0:
                h += [nlp.lbw.sym["lbu", stage_] - nlp.w.sym["u", stage_]]

                idx["h"]["lbu"].append([running_index + i for i in range(ocp.dims.nu)])
                lam_entries += [entry("lbu", repeat=ocp.dims.N, struct=struct_symSX(labels["u"]))]
                running_index = idx["h"]["lbu"][-1][-1] + 1
            elif 0 < stage_ < ocp.dims.N:
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
                lam_entries += [entry("lbx", repeat=ocp.dims.N + 1, struct=struct_symSX(labels["x"]))]
            else:
                h += [nlp.lbw.sym["lbx", stage_][idxhbx] - nlp.w.sym["x", stage_][idxhbx]]
                idx["h"]["lbx"].append([running_index + i for i in range(len(idxhbx))])

            running_index = idx["h"]["lbx"][-1][-1] + 1
            print(f"Running index = {running_index}")
        if idxhbu:
            if stage_ == 0:
                h += [nlp.w.sym["u", stage_] - nlp.ubw.sym["ubu", stage_]]
                idx["h"]["ubu"].append([running_index + i for i in range(ocp.dims.nu)])
                lam_entries += [entry("ubu", repeat=ocp.dims.N, struct=struct_symSX(labels["u"]))]
                running_index = idx["h"]["ubu"][-1][-1] + 1
            elif 0 < stage_ < ocp.dims.N:
                h += [nlp.w.sym["u", stage_][idxhbu] - nlp.ubw.sym["ubu", stage_][idxhbu]]
                idx["h"]["ubu"].append([running_index + i for i in range(len(idxhbu))])
                running_index = idx["h"]["ubu"][-1][-1] + 1
            else:
                pass

            print(f"Running index = {running_index}")
        if idxhbx:
            if stage_ == 0:
                h += [nlp.w.sym["x", stage_] - nlp.ubw.sym["ubx", stage_]]
                idx["h"]["ubx"].append([running_index + i for i in range(ocp.dims.nx)])
                lam_entries += [entry("ubx", repeat=ocp.dims.N + 1, struct=struct_symSX(labels["x"]))]
            else:
                h += [nlp.w.sym["x", stage_][idxhbx] - nlp.ubw.sym["ubx", stage_][idxhbx]]
                idx["h"]["ubx"].append([running_index + i for i in range(len(idxhbx))])

            running_index = idx["h"]["ubx"][-1][-1] + 1
            print(f"Running index = {running_index}")

    nlp.lam.sym = struct_symSX([tuple(lam_entries)])
    nlp.h.sym = cs.vertcat(*h)

    # assert nlp.lam_licq.sym.shape[0] == nlp.h_licq.sym.shape[0], "Error in building the NLP h(x, u, p) function"
    assert running_index == nlp.h.sym.shape[0], "Error in building the NLP h(x, u, p) function"

    print(f"lam.sym.shape = {nlp.lam.sym.shape}")
    print(f"h.fun: {nlp.h.fun}")

    nlp.h.fun = cs.Function("h", [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym], [nlp.h.sym], ["w", "lbw", "ubw"], ["h"])

    # y0_fun = cs.Function(
    #     "y_fun",
    #     [ocp.model.x, ocp.model.u, ocp.model.z, ocp.model.p],
    #     [ocp.model.cost_y_expr_0],
    #     ["x", "u", "z", "p"],
    #     ["y"],
    # )

    # y_fun = cs.Function(
    #     "y_fun", [ocp.model.x, ocp.model.u, ocp.model.z, ocp.model.p], [ocp.model.cost_y_expr], ["x", "u", "z", "p"], ["y"]
    # )

    # ye_fun = cs.Function("y_fun", [ocp.model.x, ocp.model.p], [ocp.model.cost_y_expr_e], ["x", "p"], ["y"])

    # W_0 = ocp.cost.W_0
    # W = ocp.cost.W
    # W_e = ocp.cost.W_e

    cost_function = define_nls_cost_function(ocp)
    cost_function_e = define_nls_cost_function_e(ocp)
    cost_function_0 = define_nls_cost_function_0(ocp)

    nlp.cost.sym = 0

    stage_ = 0
    nlp.cost.sym += nlp.dT.sym["dT", stage_] * cost_function_0(nlp.w.sym["x", stage_], nlp.w.sym["u", stage_])

    # Build the cost function
    for stage_ in range(1, ocp.dims.N):
        nlp.cost.sym += nlp.dT.sym["dT", stage_] * cost_function(nlp.w.sym["x", stage_], nlp.w.sym["u", stage_])

    # Add terminal cost
    stage_ = ocp.dims.N
    nlp.cost.sym += cost_function_e(nlp.w.sym["x", stage_])

    nlp.cost.fun = cs.Function("cost", [nlp.w.sym, nlp.dT.sym], [nlp.cost.sym], ["w", "dT"], ["cost"])
    nlp.cost.val = 0

    nlp.dT.val = nlp.dT.sym(0)
    nlp.dT.val["dT", lambda x: cs.vertcat(*x)] = np.tile(ocp.solver_options.tf / ocp.dims.N, (1, ocp.dims.N))

    # Keep for reference on how to initialize the hard bounds
    # Hard box constraints
    lhbu = [constraints.lbu[i] if i in idxhbu else -np.inf for i in range(ocp.dims.nu)]
    lhbx = [constraints.lbx[i] if i in idxhbx else -np.inf for i in range(ocp.dims.nx)]
    uhbu = [constraints.ubu[i] if i in idxhbu else np.inf for i in range(ocp.dims.nu)]
    uhbx = [constraints.ubx[i] if i in idxhbx else np.inf for i in range(ocp.dims.nx)]

    # Soft box constraints
    # lsbx = cs.vertcat(*[constraints.lbx[i] for i in idxsbx])
    # lsbu = cs.vertcat(*[constraints.lbu[i] for i in idxsbu])
    # usbx = cs.vertcat(*[constraints.ubx[i] for i in idxsbx])
    # usbu = cs.vertcat(*[constraints.ubu[i] for i in idxsbu])

    nlp.lbw.val = nlp.lbw.sym(0)
    nlp.lbw.val["lbu", lambda x: cs.vertcat(*x)] = np.tile(lhbu, (1, ocp.dims.N))
    nlp.lbw.val["lbx", lambda x: cs.vertcat(*x)] = np.tile(lhbx, (1, ocp.dims.N + 1))
    if idxsbx:
        for stage_ in range(ocp.dims.N + 1):
            nlp.lbw.val["lslbx", stage_] = [0 for _ in constraints.idxsbx]
            nlp.lbw.val["lsubx", stage_] = [0 for _ in constraints.idxsbx]

    nlp.ubw.val = nlp.ubw.sym(0)
    nlp.ubw.val["ubu", lambda x: cs.vertcat(*x)] = np.tile(uhbu, (1, ocp.dims.N))
    nlp.ubw.val["ubx", lambda x: cs.vertcat(*x)] = np.tile(uhbx, (1, ocp.dims.N + 1))
    if idxsbx:
        for stage_ in range(ocp.dims.N + 1):
            nlp.ubw.val["uslbx", stage_] = [np.inf for _ in constraints.idxsbx]
            nlp.ubw.val["usubx", stage_] = [np.inf for _ in constraints.idxsbx]

    # Parameter vector
    nlp.p.val = nlp.p.sym(0)
    nlp.p.val["p", lambda x: cs.vertcat(*x)] = np.tile(ocp.parameter_values, (1, ocp.dims.N + 1))

    # Initial guess
    x0 = ocp.constraints.lbx_0.tolist()
    u0 = 0
    nlp.w.val = nlp.w.sym(0)
    nlp.w.val["u", lambda x: cs.vertcat(*x)] = np.tile(u0, (1, ocp.dims.N))
    nlp.w.val["x", lambda x: cs.vertcat(*x)] = np.tile(x0, (1, ocp.dims.N + 1))
    # if idxsbx:
    #     for stage_ in range(ocp.dims.N):
    #         nlp.w.val["slbx", stage_] = [0 for _ in constraints.idxsbx]
    #         nlp.w.val["subx", stage_] = [0 for _ in constraints.idxsbx]

    # Set multiplier values later after solution
    nlp.pi.val = nlp.pi.sym(0)
    nlp.lam.val = nlp.lam.sym(0)

    assert nlp.g.fun.size_out(0)[0] == nlp.pi.sym.shape[0], "Dimension mismatch between g (constraints) and pi (multipliers)"
    assert (
        nlp.w.sym.shape[0] == nlp.lbw.sym.shape[0]
    ), "Dimension mismatch between w (decision variables) and lbw (lower bounds)"
    assert (
        nlp.w.sym.shape[0] == nlp.ubw.sym.shape[0]
    ), "Dimension mismatch between w (decision variables) and ubw (upper bounds)"

    return nlp, idx


def find_nlp_entry_expr_dependencies(nlp: CasadiNLP, nlp_entry: str, vars: list[str]) -> tuple[list[cs.SX], list[str]]:
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
    cost: AcadosOcpCost,
    slack: bool = False,
) -> cs.SX:
    if cost.cost_type == "LINEAR_LS":
        y_e = cs.mtimes([cost.Vx_e, x_e])

        cost = 0
        cost += cs.mtimes([(y_e - yref_e).T, W_e, (y_e - yref_e)])

        if slack:
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
        else:
            terminal_cost = cs.Function(
                "m",
                # [x, u, sl, su, yref, W, Zl, Zu, zl, zu],
                [x_e],
                [cost],
                ["x_e"],
                ["out"],
            )

        return terminal_cost
    else:
        raise NotImplementedError("Only LINEAR_LS cost types are supported at the moment.")


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
    cost: AcadosOcpCost,
    slack: bool = False,
) -> cs.SX:
    if cost.cost_type == "LINEAR_LS":
        y = cs.mtimes([cost.Vx, x]) + cs.mtimes([cost.Vu, u])

        cost = 0
        cost += cs.mtimes([(y - yref).T, W, (y - yref)])
        if slack:
            cost += cs.mtimes([sl.T, Zl, sl])
            cost += cs.mtimes([su.T, Zu, su])
            cost += cs.mtimes([sl.T, zl])
            cost += cs.mtimes([su.T, zu])

        if slack:
            stage_cost = cs.Function(
                "l",
                # [x, u, sl, su, yref, W, Zl, Zu, zl, zu],
                [x, u, sl, su],
                [cost],
                ["x", "u", "sl", "su"],
                ["out"],
            )
        else:
            stage_cost = cs.Function(
                "l",
                # [x, u, sl, su, yref, W, Zl, Zu, zl, zu],
                [x, u],
                [cost],
                ["x", "u"],
                ["out"],
            )

        return stage_cost
    else:
        raise NotImplementedError("Only LINEAR_LS cost types are supported at the moment.")


def define_y_0_function(ocp: AcadosOcp) -> cs.Function:
    model = ocp.model

    x = model.x
    u = model.u

    y_0_fun = cs.Function("y_0_fun", [x, u], [model.cost_y_expr_0], ["x", "u"], ["y_0"])

    return y_0_fun


def define_y_function(ocp: AcadosOcp) -> cs.Function:
    model = ocp.model

    x = model.x
    u = model.u

    y_fun = cs.Function("y_fun", [x, u], [model.cost_y_expr], ["x", "u"], ["y"])

    return y_fun


def define_y_e_function(ocp: AcadosOcp) -> cs.Function:
    model = ocp.model

    x = model.x

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


def define_nls_cost_function_e(ocp: AcadosOcp) -> cs.Function:
    model = ocp.model

    x = model.x
    y_e_fun = define_y_e_function(ocp)
    yref_e = ocp.cost.yref_e
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


def define_discrete_dynamics_function(ocp: AcadosOcp) -> cs.Function:
    # Step size.

    x = ocp.model.x
    u = ocp.model.u
    p = ocp.model.p
    f_expl = ocp.model.f_expl_expr

    # Continuous dynamics function.
    f = cs.Function("f", [x, u, p], [f_expl])

    # TODO: Add support for other integrator types
    # Integrate given amount of steps over the interval with Runge-Kutta 4 scheme
    if ocp.solver_options.integrator_type == "ERK":
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


@dataclass
class Param:
    value: float
    fixed: bool = True

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class ModelParams:
    """
    Parameter class for Cartpole model in MPC.
    """

    M: Param  # mass of the cart
    m: Param  # mass of the pole
    l: Param  # length of the pole
    g: Param  # gravity

    @classmethod
    def from_dict(cls, config_dict: dict):
        # return ModelParams(**config_dict)
        return cls(
            M=Param.from_dict(config_dict["M"]),
            m=Param.from_dict(config_dict["m"]),
            l=Param.from_dict(config_dict["l"]),
            g=Param.from_dict(config_dict["g"]),
        )

    def to_dict(self):
        return asdict(self)


@dataclass
class CostParams:
    """
    Parameter class for Cartpole cost in MPC.
    """

    cost_type: str
    cost_type_e: str
    Q: np.ndarray
    R: np.ndarray
    Q_e: np.ndarray
    # Zl: np.ndarray
    # Zu: np.ndarray
    # zl: np.ndarray
    # zu: np.ndarray
    # Zl_e: np.ndarray
    # Zu_e: np.ndarray
    # zl_e: np.ndarray
    # zu_e: np.ndarray

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            cost_type=config_dict["cost_type"],
            cost_type_e=config_dict["cost_type_e"],
            Q=np.diag(config_dict["Q"]),
            R=np.diag(config_dict["R"]),
            Q_e=np.diag(config_dict["Q_e"]),
            # Zl=np.diag(config_dict["Zl"]),
            # Zu=np.diag(config_dict["Zu"]),
            # zl=np.array(config_dict["zl"]),
            # zu=np.array(config_dict["zu"]),
            # Zl_e=np.diag(config_dict["Zl_e"]),
            # Zu_e=np.diag(config_dict["Zu_e"]),
            # zl_e=np.array(config_dict["zl_e"]),
            # zu_e=np.array(config_dict["zu_e"]),
        )

    def to_dict(self):
        return asdict(self)


@dataclass
class ConstraintParams:
    """
    Parameter class for Cartpole constraints in MPC.
    """

    constraint_type: str
    x0: np.ndarray
    lbu: np.ndarray
    ubu: np.ndarray
    lbx: np.ndarray
    ubx: np.ndarray
    lbx_e: np.ndarray
    ubx_e: np.ndarray
    idxbx: np.ndarray
    idxbx_e: np.ndarray
    idxbu: np.ndarray
    idxsbx: np.ndarray = field(default_factory=lambda: np.array([]))
    idxsbu: np.ndarray = field(default_factory=lambda: np.array([]))
    idxsbx_e: np.ndarray = field(default_factory=lambda: np.array([]))

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            constraint_type=config_dict["constraint_type"],
            x0=np.array(config_dict["x0"]),
            lbu=np.array(config_dict["lbu"]),
            ubu=np.array(config_dict["ubu"]),
            lbx=np.array(config_dict["lbx"]),
            ubx=np.array(config_dict["ubx"]),
            lbx_e=np.array(config_dict["lbx_e"]),
            ubx_e=np.array(config_dict["ubx_e"]),
            idxbx=np.array(config_dict["idxbx"]),
            idxbx_e=np.array(config_dict["idxbx_e"]),
            idxbu=np.array(config_dict["idxbu"]),
            idxsbx=np.array(config_dict["idxsbx"]),
            idxsbu=np.array(config_dict["idxsbu"]),
            idxsbx_e=np.array(config_dict["idxsbx_e"]),
        )

    def to_dict(self):
        return asdict(self)


@dataclass
class Dimensions:
    """
    Parameter class for Cartpole dimensions in MPC.
    """

    nx: int  # number of states
    nu: int  # number of inputs
    N: int  # horizon length

    @classmethod
    def from_dict(cls, config_dict: dict):
        return Dimensions(**config_dict)

    def to_dict(self):
        return asdict(self)


class OcpOptions(AcadosOcpOptions):
    """
    Parameter class for Cartpole solver options in MPC.
    """

    def __init__(self):
        super().__init__()

    # tf: float
    # integrator_type: Optional[str]
    # nlp_solver_type: Optional[str]
    # qp_solver: Optional[str]
    # hessian_approx: Optional[str]
    # nlp_solver_max_iter: Optional[int]
    # qp_solver_iter_max: Optional[int]

    # TODO: Add more options to cover all AcadosOcpOptions. See
    # https://docs.acados.org/interfaces/acados_python_interface/#acadosocpoptions
    # for more info. Reconsider this solution, it requires more work to maintain
    # when the AcadosOcpOptions class changes.

    @classmethod
    def from_dict(cls, config_dict: dict):
        instance = cls()
        for key, value in config_dict.items():
            setattr(instance, key, value)

        return instance

    # def to_dict(self):
    #     return asdict(self)


@dataclass
class Meta:
    """
    Parameter class for Cartpole meta parameters in MPC.
    """

    json_file: str = "acados_ocp.json"
    code_export_dir: str = "c_generated_code"

    @classmethod
    def from_dict(cls, config_dict: dict):
        return Meta(**config_dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class Config:
    """Configuration class for managing mpc parameters."""

    model_name: Optional[str]
    model_params: Optional[ModelParams]
    cost: Optional[CostParams]
    constraints: Optional[ConstraintParams]
    dimensions: Optional[Dimensions]
    ocp_options: Optional[OcpOptions]
    meta: Optional[Meta]

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(
            model_name=config_dict["model"]["name"],
            model_params=ModelParams.from_dict(config_dict["model"]["params"]),
            cost=CostParams.from_dict(config_dict["cost"]),
            constraints=ConstraintParams.from_dict(config_dict["constraints"]),
            dimensions=Dimensions.from_dict(config_dict["dimensions"]),
            ocp_options=OcpOptions.from_dict(config_dict["ocp_options"]),
            meta=Meta.from_dict(config_dict["meta"]),
        )

    def to_dict(self) -> dict:
        config_dict = {}

        if self.model_name is not None:
            config_dict["model_name"] = self.model_name

        if self.model_params is not None:
            config_dict["model_params"] = self.model_params.to_dict()

        if self.cost is not None:
            config_dict["cost"] = self.cost.to_dict()

        if self.constraints is not None:
            config_dict["constraints"] = self.constraints.to_dict()

        if self.dimensions is not None:
            config_dict["dimensions"] = self.dimensions.to_dict()

        if self.ocp_options is not None:
            config_dict["ocp_options"] = self.ocp_options.to_dict()

        if self.meta is not None:
            config_dict["meta"] = self.meta.to_dict()

        return config_dict


def define_parameter_values(ocp: AcadosOcp, config: Config) -> (cs.SX, np.ndarray):
    # Set up parameters to nominal values
    # p = {key: param["value"] for key, param in config.model_params.to_dict().items()}

    parameter_values = []
    # Set up parameters to symbolic variables if not fixed
    for _, param in config.model_params.to_dict().items():
        if not param["fixed"]:
            # p_sym += [cs.SX.sym(key)]
            # p[key] = p_sym[-1]
            parameter_values += [param["value"]]

    # p_sym = cs.vertcat(*p_sym)
    parameter_values = np.array(parameter_values)

    return parameter_values


def define_model_expressions(config: Config) -> (dict, np.ndarray):
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

    model = dict()
    model["f_impl_expr"] = f_impl
    model["f_expl_expr"] = f_expl
    model["x"] = x
    model["xdot"] = x_dot
    model["p"] = p_sym
    model["u"] = u
    model["z"] = z
    model["name"] = name

    return model, parameter_values


def define_dimensions(config: Config) -> dict:
    dims = dict()
    dims["nx"] = config.dimensions.nx
    dims["nu"] = config.dimensions.nu
    dims["N"] = config.dimensions.N
    dims["ny"] = dims["nx"] + dims["nu"]
    dims["ny_e"] = dims["nx"]

    return dims


def define_cost(config: Config) -> dict:
    cost = dict()

    dims = define_dimensions(config)

    cost["cost_type"] = config.cost.cost_type
    cost["cost_type_e"] = config.cost.cost_type_e

    cost["W"] = scipy.linalg.block_diag(config.cost.Q, config.cost.R)
    cost["W_e"] = config.cost.Q_e

    cost["yref"] = np.zeros((dims["ny"],))
    cost["yref_e"] = np.zeros((dims["ny_e"],))
    cost["cost_type"] = config.cost.cost_type
    cost["cost_type_e"] = config.cost.cost_type_e

    cost["Vx"] = np.zeros((dims["ny"], dims["nx"]), dtype=np.float32)
    cost["Vx"][: dims["nx"], : dims["nx"]] = np.eye(dims["nx"])

    cost["Vu"] = np.zeros((dims["ny"], dims["nu"]))
    cost["Vu"][-1, 0] = 1.0

    cost["Vx_e"] = np.eye(dims["nx"])

    # cost["Zl"] = config.cost.Zl
    # cost["Zu"] = config.cost.Zu
    # cost["zl"] = config.cost.zl
    # cost["zu"] = config.cost.zu
    # cost["Zl_e"] = config.cost.Zl_e
    # cost["Zu_e"] = config.cost.Zu_e
    # cost["zl_e"] = config.cost.zl_e
    # cost["zu_e"] = config.cost.zu_e

    return cost


def define_constraints(config: Config) -> dict:
    constraints = dict()

    constraints["constr_type"] = config.constraints.constraint_type
    constraints["x0"] = config.constraints.x0.reshape(-1)
    constraints["lbu"] = config.constraints.lbu.reshape(-1)
    constraints["ubu"] = config.constraints.ubu.reshape(-1)
    constraints["lbx"] = config.constraints.lbx.reshape(-1)
    constraints["ubx"] = config.constraints.ubx.reshape(-1)
    constraints["lbx_e"] = config.constraints.lbx_e.reshape(-1)
    constraints["ubx_e"] = config.constraints.ubx_e.reshape(-1)
    constraints["idxbx"] = config.constraints.idxbx.reshape(-1)
    constraints["idxbx_e"] = config.constraints.idxbx_e.reshape(-1)
    constraints["idxbu"] = config.constraints.idxbu.reshape(-1)
    constraints["idxsbx"] = config.constraints.idxsbx.reshape(-1)
    constraints["idxsbu"] = config.constraints.idxsbu.reshape(-1)
    constraints["idxsbx_e"] = config.constraints.idxsbx_e.reshape(-1)

    return constraints


# def define_parameters(config: Config) -> np.array:
#     return np.array(
#         [
#             config.model.M,
#             config.model.m,
#             config.model.l,
#             config.model.g,
#         ]
#     )
