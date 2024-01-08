import casadi as cs
import numpy as np
from typing import Union
from casadi.tools import struct_symSX, entry
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosOcpConstraints

from rlmpc.common.utils import ACADOS_MULTIPLIER_ORDER, rename_key_in_dict


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
    w: NLPEntry
    lbw: NLPEntry
    ubw: NLPEntry
    lbw_solver: NLPEntry
    ubw_solver: NLPEntry
    g_solver: Union[cs.SX, cs.MX]
    lbg_solver: Union[list, np.ndarray]
    ubg_solver: Union[list, np.ndarray]
    p: NLPEntry
    p_solver: NLPEntry
    p_val: Union[list, np.ndarray]
    f_disc: cs.Function
    shooting: struct_symSX
    # g: Union[cs.SX, cs.MX]  # Dynamics equality constraints
    g: NLPEntry  # Dynamics equality constraints
    pi: NLPEntry  # Lange multiplier for dynamics equality constraints
    h: NLPEntry  # Inequality constraints
    h_licq: NLPEntry  # Inequality constraints
    lam: NLPEntry  # Lange multiplier for inequality constraints
    lam_licq: NLPEntry  # Lange multiplier for inequality constraints
    idxhbx: list
    idxsbx: list
    idxhbu: list
    idxsbu: list
    L: NLPEntry
    dL_dw: NLPEntry
    dL_dp: NLPEntry
    R: NLPEntry
    dR_dw: NLPEntry
    dR_dp: NLPEntry
    dR_dz: NLPEntry
    dT = NLPEntry

    def __init__(self):
        super().__init__()

        self.cost = NLPEntry()
        self.w = NLPEntry()
        self.lbw = NLPEntry()
        self.ubw = NLPEntry()
        self.lbw_solver = NLPEntry()
        self.ubw_solver = NLPEntry()
        self.g_solver = None
        self.lbg_solver = None
        self.ubg_solver = None
        self.p_solver = NLPEntry()
        self.p_val = None
        self.p = NLPEntry()
        self.f_disc = None
        self.shooting = None
        self.g = NLPEntry()
        self.dg_dw = NLPEntry()
        self.dg_dpi = NLPEntry()
        self.dg_dlam = NLPEntry()
        self.pi = NLPEntry()
        self.h = NLPEntry()
        self.h_licq = NLPEntry()
        self.lam = NLPEntry()
        self.lam_licq = NLPEntry()
        self.idxhbx = None
        self.idxsbx = None
        self.idxhbu = None
        self.idxsbu = None
        self.L = NLPEntry()
        self.dL_dw = NLPEntry()
        self.ddL_dwdw = NLPEntry()
        self.ddL_dwdpi = NLPEntry()
        self.ddL_dwdlam = NLPEntry()
        self.dL_dp = NLPEntry()
        self.R = NLPEntry()
        self.dR_dw = NLPEntry()
        self.dR_dp = NLPEntry()
        self.dR_dz = NLPEntry()
        self.dT = NLPEntry()

    def assert_kkt_residual(self) -> np.ndarray:
        return test_nlp_kkt_residual(self)


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


def build_nlp(ocp: AcadosOcp) -> tuple[NLP, dict]:
    """
    Build the NLP for the OCP.

    TODO: Add support for other cost types
    TODO: Adapt to SX/MX depending on the provided model
    TODO: Add support for different parameters at each stage
    TODO: Add support for varying/learning reference trajectories, i.e. set as parameters
    TODO: Add support for varying/learning cost weights, i.e. set as parameters
    TODO: Add support for varying/learning constraints, i.e. set as parameters
    """

    nlp = NLP()

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

    if False:  # No stage-varying parameters for now.
        entries["p"].append(entry("p", repeat=ocp.dims.N + 1, struct=struct_symSX([entry(label) for label in labels["p"]])))
        nlp.p.sym = struct_symSX([tuple(entries["p"])])
    else:
        nlp.p.sym = struct_symSX(labels["p"])

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
        g.append(nlp.f_disc(nlp.w.sym["x", stage_], nlp.w.sym["u", stage_], nlp.p.sym.cat) - nlp.w.sym["x", stage_ + 1])
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

    if False:
        nlp.p.val["p", lambda x: cs.vertcat(*x)] = np.tile(ocp.parameter_values, (1, ocp.dims.N + 1))
    else:
        nlp.p.val = ocp.parameter_values

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
    test_nlp_kkt_residual(nlp=nlp, tol=tol)
    test_nlp_stationarity(nlp=nlp, tol=tol)
    test_nlp_is_dual_feasible(nlp=nlp)
    test_nlp_satisfies_complementarity(nlp=nlp, tol=tol)

    return True


def update_nlp(nlp: NLP, ocp_solver: AcadosOcpSolver, multiplier_map: LagrangeMultiplierMap) -> NLP:
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
