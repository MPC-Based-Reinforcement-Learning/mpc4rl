from typing import Union
import numpy as np
import casadi as cs

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

    def __init__(self):
        super().__init__()


class CasadiOcpSolver:
    """docstring for CasadiOcp."""

    # _model: dict
    # _cost: cs.Function
    # _constraints: cs.Function

    ocp: CasadiOcp
    p: np.ndarray

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

        (
            self.nlp_solver,
            self.w0,
            self.lbw,
            self.ubw,
            self.lbg,
            self.ubg,
        ) = build_nlp_solver(self.ocp)

        # self._model = define_model_expressions(config=config)

        # self._cost = define_cost(config=config)

        # self._constraints = define_constraints(config=config)

    # def set(self, stage: int, field: str, value: np.ndarray) -> None:
    #     """

    #     The following is a modified implementation of parts of acados_template/acados_ocp_solver.py

    #     Set a field of the OCP solver.

    #     Args:
    #         stage: Stage index.
    #         field: Field name.
    #         value: Field value.
    #     """
    #     raise NotImplementedError()

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
            start = stage_ * (self.ocp.dims.nx + self.ocp.dims.nu)
            end = start + self.ocp.dims.nx
            self.lbw[start:end] = value_
        elif field_ == "ubx":
            start = stage_ * (self.ocp.dims.nx + self.ocp.dims.nu)
            end = start + self.ocp.dims.nx
            self.ubw[start:end] = value_
        elif field_ == "lbu":
            raise Exception("lbu is not implemented yet.")
        elif field_ == "ubu":
            raise Exception("ubu is not implemented yet.")
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

        if stage_ < 0 or stage_ > self.ocp.dims.N:
            raise Exception(
                f"AcadosOcpSolver.get(stage={stage_}, field={field_}): stage index must be in [0, {self.ocp.dims.N}], got: {stage_}."
            )

        if stage_ == self.ocp.dims.N and field_ == "pi":
            raise Exception(
                f"AcadosOcpSolver.get(stage={stage_}, field={field_}): field '{field_}' does not exist at final stage {stage_}."
            )

        if field_ == "x":
            start_idx = stage_ * (self.ocp.dims.nx + self.ocp.dims.nu)
            end_idx = start_idx + self.ocp.dims.nx

            return self.nlp_solution["x"][start_idx:end_idx].full().flatten()

        if field_ == "u":
            start_idx = (
                stage_ * (self.ocp.dims.nx + self.ocp.dims.nu) + self.ocp.dims.nx
            )
            end_idx = start_idx + self.ocp.dims.nu

            return self.nlp_solution["x"][start_idx:end_idx].full().flatten()

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
            x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=self.p
        )

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
        raise NotImplementedError(
            "Only ERK4 integrator types are supported at the moment."
        )

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
        raise NotImplementedError(
            "Only ERK integrator types are supported at the moment."
        )


def define_stage_cost_function(ocp: CasadiOcp) -> cs.Function:
    model = ocp.model
    cost = ocp.cost

    # TODO: Add support for other cost types
    # TODO: Add yref as a parameter
    if cost.cost_type == "LINEAR_LS":
        Vx = cost.Vx
        Vu = cost.Vu
        yref = cost.yref.reshape(-1, 1)
        W = cost.W

        x = model.x
        u = model.u
        y = cs.mtimes([Vx, x]) + cs.mtimes([Vu, u])

        stage_cost_function = cs.Function(
            "stage_cost",
            [x, u],
            [cs.mtimes([(y - yref).T, W, (y - yref)])],
        )

        return stage_cost_function


def define_terminal_cost_function(ocp: CasadiOcp) -> cs.Function:
    model = ocp.model
    cost = ocp.cost

    # TODO: Add support for other cost types
    # TODO: Add yref as a parameter
    if cost.cost_type == "LINEAR_LS":
        Vx_e = cost.Vx_e
        yref_e = cost.yref_e.reshape(-1, 1)
        W_e = cost.W_e

        x = model.x
        y_e = cs.mtimes([Vx_e, x])

        terminal_cost_function = cs.Function(
            "stage_cost",
            [x],
            [cs.mtimes([(y_e - yref_e).T, W_e, (y_e - yref_e)])],
        )

        return terminal_cost_function


def build_nlp_solver(ocp: CasadiOcp) -> cs.nlpsol:
    # F = define_discrete_dynamics_function(model=model, h=h, ocp_options=ocp.solver_options)
    F = define_discrete_dynamics_function(ocp)

    constraints = ocp.constraints

    stage_cost_function = define_stage_cost_function(ocp)

    terminal_cost_function = define_terminal_cost_function(ocp)

    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []

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
    w0 += x0

    # Formulate the NLP
    for k in range(ocp.dims.N):
        # New NLP variable for the control
        uk = sym("u_" + str(k))
        w += [uk]
        lbw += constraints.lbu.tolist()
        ubw += constraints.ubu.tolist()
        w0 += [0]

        J = J + stage_cost_function(xk, uk)

        # Integrate till the end of the interval
        xk_end = F(xk, uk, p)

        # New NLP variable for state at end of interval
        xk = sym("x_" + str(k + 1), ocp.dims.nx)
        w += [xk]
        lbw += constraints.lbx.tolist()
        ubw += constraints.ubx.tolist()
        w0 += x0

        # Add equality constraint
        g += [xk_end - xk]
        lbg += np.zeros((ocp.dims.nx,)).tolist()
        ubg += np.zeros((ocp.dims.nx,)).tolist()

    # Add terminal cost
    J = J + terminal_cost_function(xk_end)

    # NLP
    prob = {"f": J, "x": cs.vertcat(*w), "g": cs.vertcat(*g), "p": p}

    # Create an NLP solver
    solver = cs.nlpsol("solver", "ipopt", prob)

    return solver, w0, lbw, ubw, lbg, ubg


class CasadiMPC(MPC):
    """docstring for CartpoleMPC."""

    parameter_values: np.ndarray

    def __init__(self, config: Config, build: bool = True):
        super().__init__()

        self.ocp = CasadiOcp()

        self.ocp.model, self.ocp.parameter_values = define_acados_model(
            ocp=self.ocp, config=config
        )

        self.ocp.dims = define_acados_dims(ocp=self.ocp, config=config)

        self.ocp.cost = define_acados_cost(ocp=self.ocp, config=config)

        self.ocp.constraints = define_acados_constraints(ocp=self.ocp, config=config)

        self.ocp.solver_options = config.ocp_options

        self.ocp_solver = CasadiOcpSolver(self.ocp)

        self.parameter_values = self.ocp.parameter_values

        for stage_ in range(self.ocp.dims.N):
            self.ocp_solver.set(stage_, "p", self.ocp.parameter_values)

        # _ = self.ocp_solver.solve(
        #     x0=self.ocp.constraints.x0, p=self.ocp.parameter_values
        # )

    def plot_solution(self) -> (plt.figure, plt.axes):
        X = np.vstack(
            [self.ocp_solver.get(stage_, "x") for stage_ in range(self.ocp.dims.N + 1)]
        )
        U = np.vstack(
            [self.ocp_solver.get(stage_, "u") for stage_ in range(self.ocp.dims.N)]
        )

        fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(10, 10))
        for k, ax in enumerate(axes[:-1]):
            ax.step(range(self.ocp.dims.N + 1), X[:, k], color="k")
            ax.grid(True)
        axes[-1].step(range(self.ocp.dims.N), U[:, 0], color="k")
        axes[-1].grid(True)

        axes[0].set_ylabel("x")
        axes[1].set_ylabel("v")
        axes[2].set_ylabel("theta")
        axes[3].set_ylabel("omega")

        axes[-1].set_xlabel("Step [-]")

        return (fig, axes)

    def get_parameters(self) -> np.ndarray:
        return self.parameter_values

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def get_action(self, x0: np.ndarray) -> np.ndarray:
        """
        Update the solution of the OCP solver.

        Args:
            x0: Initial state.

        Returns:
            u: Optimal control action.
        """

        # Set initial state
        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)
        # self.ocp_solver.set(0, "x", x0)

        # Solve the optimization problem
        self.ocp_solver.solve()

        # Get solution
        u = self.ocp_solver.get(0, "u")

        # Scale to [-1, 1] for gym
        action = (
            2.0
            * (
                (u - self.ocp.constraints.lbu)
                / (self.ocp.constraints.ubu - self.ocp.constraints.lbu)
            )
            - 1.0
        )

        return action
